//! NotchLC encoder and decoder.
//!
//! Licensed LGPL-2.1-or-later, because the `decode` module is ported from FFmpeg's
//! `libavcodec/notchlc.c` and `libavcodec/lzf.c` (LGPL-2.1+, (c) 2020 Paul B
//! Mahol). See COPYING.LESSER.
//!
//! The encoder below, and the `clip` and `gpu` modules, are original work: the encoder was
//! written from a bitstream description rather than ported. They carry the
//! crate's licence by virtue of shipping alongside the decoder, not because
//! they are derived from it.
//!
//! The encoder is a spike: quality-A luma, coarse chroma.
//!
//! Packet layout (what goes in a mov sample):
//! ```text
//! 'NLC1' magic, uncompressed_size le32, compressed_size le32, format le32
//! format 2 => uncompressed payload follows immediately.
//! ```
//!
//! Payload layout: a 40-byte header of ten le32 fields, then sections.
//! Offsets in the header are stored as offset/4, except y_data_size and
//! data_end which are plain byte counts.
//!
//! ```text
//!  0  texture_size_x
//!  4  texture_size_y
//!  8  uv_offset_data_offset /4
//! 12  y_control_data_offset /4
//! 16  a_control_word_offset /4
//! 20  uv_data_offset /4
//! 24  y_data_size (bytes)
//! 28  a_data_offset /4
//! 32  a_count_size /4
//! 36  data_end (bytes)
//! 40  y row-offset table (one le32 per 4-row band) ...
//! ```

#[cfg(feature = "clip")]
mod clip;
mod decode;
#[cfg(feature = "gpu")]
mod gpu;
mod movmux;

#[cfg(feature = "clip")]
pub use clip::{Clip, ClipError};
#[cfg(feature = "gpu")]
pub use gpu::GpuDecoder;
pub use decode::{
    bit_offsets, decode_packet, decode_payload, decompress_packet, parse_header,
    Error as DecodeError, Frame, Header, Payload,
};
pub use movmux::MovWriter;

// The decoder compares a little-endian u32 read against MKBETAG('N','L','C','1'),
// which packs the fourcc big-endian-style — so on disk the magic is "1CLN".
const MAGIC: [u8; 4] = u32::from_be_bytes(*b"NLC1").to_le_bytes();
const FORMAT_LZ4: u32 = 1;

/// Encodes single frames. Planes are 12-bit (0..=4095) full-range, stored as
/// u16. `y` carries luma (or G), `u`/`v` the chroma (or B/R) channels.
pub struct FrameEncoder {
    width: u32,
    height: u32,
}

fn put32(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

impl FrameEncoder {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Encode one frame into a complete NLC1 packet (ready to mux as a mov
    /// sample). `y`, `u`, `v` are `width * height` 12-bit samples.
    pub fn encode_packet(&self, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        assert_eq!(y.len(), w * h);
        assert_eq!(u.len(), w * h);
        assert_eq!(v.len(), w * h);

        let cols4 = w.div_ceil(4);
        let rows4 = h.div_ceil(4);
        let cols16 = w.div_ceil(16);
        let rows16 = h.div_ceil(16);

        let hdr_size = 40usize;
        let row_table_size = rows4 * 4;
        let y_control_size = rows4 * cols4 * 4;
        let uv_table_size = rows16 * cols16 * 4;
        let uv_data_size = rows16 * cols16 * 12; // coarse mode: 12 bytes/block
        let y_row_size = cols4 * 8; // 16 px * 4 bits = 8 bytes per 4x4 block
        let y_data_size = rows4 * y_row_size;

        // Section layout (all 4-byte aligned by construction).
        let row_table_off = hdr_size;
        let y_control_off = row_table_off + row_table_size;
        let uv_table_off = y_control_off + y_control_size;
        let uv_data_off = uv_table_off + uv_table_size;
        // Alpha: we emit no alpha data. The decoder treats the frame as
        // fully opaque when uv_count_offset == a_control_word_offset, where
        // uv_count_offset = y_data_offset - a_data_offset. Point a_data at the
        // uv data start (must be strictly < y_data_offset) and set
        // a_control_word_offset to the resulting difference.
        let a_data_off = uv_data_off;
        let y_data_off = uv_data_off + uv_data_size;
        let a_control_word_off = y_data_off - a_data_off;
        let data_end = y_data_off + y_data_size;

        let mut buf = vec![0u8; data_end];

        // Header (offsets stored /4).
        put32(&mut buf, 0, self.width);
        put32(&mut buf, 4, self.height);
        put32(&mut buf, 8, (uv_table_off / 4) as u32);
        put32(&mut buf, 12, (y_control_off / 4) as u32);
        put32(&mut buf, 16, (a_control_word_off / 4) as u32);
        put32(&mut buf, 20, (uv_data_off / 4) as u32);
        put32(&mut buf, 24, y_data_size as u32);
        put32(&mut buf, 28, (a_data_off / 4) as u32);
        put32(&mut buf, 32, 0); // a_count_size
        put32(&mut buf, 36, data_end as u32);

        // Y plane: per 4-row band, a row-offset entry then per 4x4 block a
        // control word (12-bit min/max + four 2-bit row controls, all 3 =
        // 4 bits per index) and 16 x 4-bit indices packed LSB-first.
        for band in 0..rows4 {
            put32(&mut buf, row_table_off + band * 4, (band * y_row_size) as u32);
            let y0 = band * 4;

            // Pass 1: gather blocks and write their control words.
            let mut blocks = vec![[0u16; 16]; cols4];
            for (bx, block) in blocks.iter_mut().enumerate() {
                let x0 = bx * 4;
                for i in 0..4 {
                    for j in 0..4 {
                        let yy = (y0 + i).min(h - 1);
                        let xx = (x0 + j).min(w - 1);
                        block[i * 4 + j] = y[yy * w + xx].min(4095);
                    }
                }
                let y_min = *block.iter().min().unwrap();
                let y_max = *block.iter().max().unwrap();
                let control = 0xFF00_0000u32 | y_min as u32 | ((y_max as u32) << 12);
                put32(&mut buf, y_control_off + (band * cols4 + bx) * 4, control);
            }

            // Pass 2: pack the 4-bit indices LSB-first across the band.
            let row_start = y_data_off + band * y_row_size;
            let mut bits = BitWriter::new(&mut buf[row_start..row_start + y_row_size]);
            for block in &blocks {
                let y_min = *block.iter().min().unwrap();
                let y_max = *block.iter().max().unwrap();
                let diff = (y_max - y_min) as u32;
                for i in 0..4 {
                    for j in 0..4 {
                        let idx = best_index(block[i * 4 + j], y_min, diff, 15);
                        bits.write(idx, 4);
                    }
                }
            }
        }

        // UV planes: per 16x16 block, coarse mode (is8x8=0, escape=0): one
        // pair of 8-bit endpoints per channel and a 2-bit index per 4x4
        // sub-block.
        for by in 0..rows16 {
            for bx in 0..cols16 {
                let block_idx = by * cols16 + bx;
                let data = encode_uv_block(u, v, w, h, bx * 16, by * 16);
                put32(
                    &mut buf,
                    uv_table_off + block_idx * 4,
                    ((block_idx * 12) / 4) as u32,
                );
                buf[uv_data_off + block_idx * 12..uv_data_off + block_idx * 12 + 12]
                    .copy_from_slice(&data);
            }
        }

        // Wrap in the NLC1 packet header. We use format 1 (LZ4) with a
        // literal-only stream: the decoder re-bases its reader to the
        // decompressed payload for formats 0/1, whereas format 2 leaves
        // offsets relative to the packet start (an FFmpeg quirk real files
        // never exercise, since they are always compressed).
        let compressed = lz4_literal_only(&buf);
        let mut pkt = Vec::with_capacity(16 + compressed.len());
        pkt.extend_from_slice(&MAGIC);
        pkt.extend_from_slice(&(buf.len() as u32).to_le_bytes()); // uncompressed_size
        pkt.extend_from_slice(&(compressed.len() as u32).to_le_bytes()); // compressed_size
        pkt.extend_from_slice(&FORMAT_LZ4.to_le_bytes());
        pkt.extend_from_slice(&compressed);
        pkt
    }
}

/// Emit `data` as a single literal-only LZ4 block sequence:
/// token 0xF0, length extension bytes (255s + remainder), then the literals.
/// The decoder stops when input is exhausted, so no match offset follows.
fn lz4_literal_only(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 255 + 16);
    let len = data.len();
    if len < 15 {
        out.push((len as u8) << 4);
    } else {
        out.push(0xF0);
        let mut rem = len - 15;
        while rem >= 255 {
            out.push(255);
            rem -= 255;
        }
        out.push(rem as u8);
    }
    out.extend_from_slice(data);
    out
}

/// Pick the index whose decoded value is nearest to `v`.
/// Decoded = min + (diff * idx + div - 1) / div.
fn best_index(v: u16, min: u16, diff: u32, div: u32) -> u32 {
    if diff == 0 {
        return 0;
    }
    let target = v as u32;
    let mut best = 0u32;
    let mut best_err = u32::MAX;
    for idx in 0..=div {
        let dec = min as u32 + (diff * idx + (div - 1)) / div;
        let err = dec.abs_diff(target);
        if err < best_err {
            best_err = err;
            best = idx;
        }
    }
    best
}

/// Encode one 16x16 chroma block in coarse mode. Returns the 12 bytes to
/// place in the uv data section: le16 is8x8=0, le16 escape=0, u0,v0,u1,v1,
/// le32 loc.
fn encode_uv_block(u: &[u16], v: &[u16], w: usize, h: usize, x0: usize, y0: usize) -> [u8; 12] {
    // Mean of each 4x4 sub-block, for both channels.
    let mut mu = [0u32; 16];
    let mut mv = [0u32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut su = 0u64;
            let mut sv = 0u64;
            for ii in 0..4 {
                for jj in 0..4 {
                    let yy = (y0 + i * 4 + ii).min(h - 1);
                    let xx = (x0 + j * 4 + jj).min(w - 1);
                    su += u[yy * w + xx].min(4095) as u64;
                    sv += v[yy * w + xx].min(4095) as u64;
                }
            }
            mu[i * 4 + j] = (su / 16) as u32;
            mv[i * 4 + j] = (sv / 16) as u32;
        }
    }

    // 8-bit endpoints from the 12-bit mean range; the decoder expands them
    // with nibble replication: e = (b << 4) | (b & 0xF).
    let expand = |b: u32| (b << 4) | (b & 0xF);
    let u0 = expand(*mu.iter().min().unwrap() >> 4);
    let u1 = expand(*mu.iter().max().unwrap() >> 4);
    let v0 = expand(*mv.iter().min().unwrap() >> 4);
    let v1 = expand(*mv.iter().max().unwrap() >> 4);

    // 2-bit index per sub-block: decoded = e0 + (dif * idx + 2) / 3. A single
    // 2-bit slot in loc drives both channels, so pick the best compromise.
    let pick = |m: u32, e0: u32, e1: u32| -> u32 {
        let dif = e1 as i64 - e0 as i64;
        let mut best = 0u32;
        let mut best_err = u64::MAX;
        for idx in 0..4u32 {
            let dec = (e0 as i64 + (dif * idx as i64 + 2) / 3) as u32;
            let err = (dec as u64).abs_diff(m as u64);
            if err < best_err {
                best_err = err;
                best = idx;
            }
        }
        best
    };

    // Decoder consumes loc in (i, j) order, 2 bits per sub-block.
    let mut loc = 0u32;
    for i in 0..4 {
        for j in 0..4 {
            let iu = pick(mu[i * 4 + j], u0, u1);
            let iv = pick(mv[i * 4 + j], v0, v1);
            let slot = ((iu + iv) / 2) & 3;
            loc |= slot << ((i * 4 + j) * 2);
        }
    }

    let mut out = [0u8; 12];
    // is8x8 = 0, escape = 0 already zeroed.
    out[4] = (u0 >> 4) as u8;
    out[5] = (v0 >> 4) as u8;
    out[6] = (u1 >> 4) as u8;
    out[7] = (v1 >> 4) as u8;
    out[8..12].copy_from_slice(&loc.to_le_bytes());
    out
}

/// LSB-first bit writer (matches get_bits with BITSTREAM_READER_LE).
struct BitWriter<'a> {
    buf: &'a mut [u8],
    bit_pos: usize,
}

impl<'a> BitWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    fn write(&mut self, value: u32, nbits: usize) {
        for k in 0..nbits {
            let bit = (value >> k) & 1;
            let byte = self.bit_pos / 8;
            let off = self.bit_pos % 8;
            self.buf[byte] |= (bit as u8) << off;
            self.bit_pos += 1;
        }
    }
}
