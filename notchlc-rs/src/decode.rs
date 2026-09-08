//! NotchLC frame decoder.
//!
//! Ported from FFmpeg's `libavcodec/notchlc.c` (Copyright (c) 2020 Paul B
//! Mahol) and `libavcodec/lzf.c`, both LGPL-2.1-or-later. That provenance is
//! why this crate is LGPL rather than MIT/Apache.
//!
//! Along with `decode.wgsl`, this is the derived code in the crate. Keeping it
//! confined to two files means a future clean-room rewrite — should the crate
//! ever need to be MIT/Apache, to sit beside hap-wgpu on crates.io — is a
//! rewrite of two files, not of the crate.
//!
//! Output is YUVA444P12 with an identity ("RGB") colorspace and full range, so
//! the planes are GBR: `y` is G, `u` is B, `v` is R.

use std::fmt;

const HISTORY_SIZE: usize = 64 * 1024;
const LZF_LITERAL_MAX: u8 = 1 << 5;
const LZF_LONG_BACKREF: usize = 7 + 2;

/// The decoder compares a little-endian u32 read against MKBETAG('N','L','C','1'),
/// so on disk the magic reads "1CLN".
const MAGIC: [u8; 4] = u32::from_be_bytes(*b"NLC1").to_le_bytes();

#[derive(Debug)]
pub enum Error {
    NotNotchLc,
    /// A format byte the reference decoder itself rejects (`AVERROR_PATCHWELCOME`).
    UnsupportedFormat(u32),
    Truncated,
    /// Header offsets that don't fit the payload, or a block mode that has no
    /// meaning — the same cases FFmpeg answers with AVERROR_INVALIDDATA.
    Invalid(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotNotchLc => write!(f, "not a NotchLC packet"),
            Error::UnsupportedFormat(n) => write!(f, "unsupported compression format {n}"),
            Error::Truncated => write!(f, "packet truncated"),
            Error::Invalid(what) => write!(f, "invalid bitstream: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// A decoded frame. Planes are 12-bit in `u16`, padded out to a multiple of 16
/// in both axes so the block loops never need bounds checks — same trick
/// FFmpeg gets from its padded frame buffers. Use [`Frame::to_rgba8`] to get
/// cropped, packed pixels.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Row stride of every plane, in samples (not bytes).
    pub stride: usize,
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub a: Vec<u16>,
}

impl Frame {
    /// Pack to 8-bit RGBA, cropped to the real dimensions. GBR identity:
    /// R comes from `v`, G from `y`, B from `u`.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = Vec::with_capacity(w * h * 4);
        for row in 0..h {
            let base = row * self.stride;
            for col in 0..w {
                let i = base + col;
                out.push((self.v[i] >> 4) as u8);
                out.push((self.y[i] >> 4) as u8);
                out.push((self.u[i] >> 4) as u8);
                out.push((self.a[i] >> 4) as u8);
            }
        }
        out
    }
}

/// Little-endian reader over the decompressed payload.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    fn u8(&mut self) -> Result<u8, Error> {
        let v = *self.buf.get(self.pos).ok_or(Error::Truncated)?;
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16, Error> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        let s = self.buf.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(s)
    }
}

/// LSB-first bit reader, used only for the luma residuals.
struct BitReader<'a> {
    buf: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn at(buf: &'a [u8], byte: usize) -> Self {
        Self { buf, bit: byte * 8 }
    }

    /// Reads up to 4 bits. Past the end reads as zero, matching what FFmpeg's
    /// padded input buffer yields.
    fn bits(&mut self, n: usize) -> u32 {
        let mut v = 0u32;
        for k in 0..n {
            let byte = self.buf.get(self.bit >> 3).copied().unwrap_or(0);
            v |= (((byte >> (self.bit & 7)) & 1) as u32) << k;
            self.bit += 1;
        }
        v
    }
}

/// One packet's payload, ready for block decoding — on the CPU via
/// [`decode_payload`], or by uploading `bytes` to the GPU.
#[derive(Debug, Clone)]
pub struct Payload {
    pub bytes: Vec<u8>,
    /// Where the 40-byte frame header starts. Nonzero only for uncompressed
    /// packets, which keep the packet header in front while their section
    /// offsets stay relative to the packet start.
    pub header_at: usize,
    pub uncompressed_size: usize,
}

/// Undo the packet's compression wrapper. Inherently serial, so this is the
/// part that wants to happen off the render thread.
pub fn decompress_packet(pkt: &[u8]) -> Result<Payload, Error> {
    if pkt.len() <= 40 {
        return Err(Error::Truncated);
    }
    let mut r = Reader::at(pkt, 0);
    if r.take(4)? != MAGIC {
        return Err(Error::NotNotchLc);
    }
    let uncompressed_size = r.u32()? as usize;
    let compressed_size = r.u32()? as usize;
    let format = r.u32()?;

    let bytes = match format {
        0 => {
            let out = lzf_uncompress(&pkt[16..])?;
            if uncompressed_size > out.len() {
                return Err(Error::Invalid("lzf output shorter than header claims"));
            }
            out
        }
        1 => {
            let end = 16usize
                .checked_add(compressed_size)
                .filter(|e| *e <= pkt.len())
                .ok_or(Error::Truncated)?;
            let out = lz4_uncompress(&pkt[16..end], uncompressed_size)?;
            if out.len() != uncompressed_size {
                return Err(Error::Invalid("lz4 output size mismatch"));
            }
            out
        }
        2 => pkt.to_vec(),
        n => return Err(Error::UnsupportedFormat(n)),
    };

    Ok(Payload {
        bytes,
        header_at: if format == 2 { 16 } else { 0 },
        uncompressed_size,
    })
}

/// Decode an already-decompressed payload on the CPU.
pub fn decode_payload(payload: &Payload) -> Result<Frame, Error> {
    decode_blocks(&payload.bytes, payload.header_at, payload.uncompressed_size)
}

/// Decode one complete NLC1 packet.
pub fn decode_packet(pkt: &[u8]) -> Result<Frame, Error> {
    decode_payload(&decompress_packet(pkt)?)
}

/// The 40-byte frame header, with every offset resolved to an absolute byte
/// position in the payload. Shared by the CPU decoder and the GPU path, which
/// needs the same offsets as shader parameters.
#[derive(Debug, Clone)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub uv_offset_data_offset: usize,
    pub y_control_data_offset: usize,
    pub a_control_word_offset: usize,
    pub uv_data_offset: usize,
    pub y_data_offset: usize,
    pub a_data_offset: usize,
    pub data_end: usize,
    /// Where the per-4-row-band luma row-offset table starts.
    pub row_offsets_at: usize,
    /// True when the alpha section is absent and every pixel is 4095.
    pub opaque: bool,
}

impl Header {
    /// 4x4 luma blocks across and down.
    pub fn blocks4(&self) -> (usize, usize) {
        (
            (self.width as usize).div_ceil(4),
            (self.height as usize).div_ceil(4),
        )
    }

    /// 16x16 chroma/alpha blocks across and down.
    pub fn blocks16(&self) -> (usize, usize) {
        (
            (self.width as usize).div_ceil(16),
            (self.height as usize).div_ceil(16),
        )
    }
}

pub fn parse_header(payload: &Payload) -> Result<Header, Error> {
    parse_header_at(&payload.bytes, payload.header_at, payload.uncompressed_size)
}

fn parse_header_at(buf: &[u8], header_at: usize, uncompressed_size: usize) -> Result<Header, Error> {
    let mut r = Reader::at(buf, header_at);
    let width = r.u32()?;
    let height = r.u32()?;
    if width == 0 || height == 0 {
        return Err(Error::Invalid("zero dimensions"));
    }

    // Offsets stored as offset/4, except y_data_size and data_end.
    let scaled = |what: &'static str, r: &mut Reader| -> Result<usize, Error> {
        let v = r.u32()? as usize;
        let v = v.checked_mul(4).ok_or(Error::Invalid(what))?;
        if v >= uncompressed_size {
            return Err(Error::Invalid(what));
        }
        Ok(v)
    };
    let uv_offset_data_offset = scaled("uv_offset_data_offset", &mut r)?;
    let y_control_data_offset = scaled("y_control_data_offset", &mut r)?;
    let a_control_word_offset = scaled("a_control_word_offset", &mut r)?;
    let uv_data_offset = scaled("uv_data_offset", &mut r)?;
    let y_data_size = r.u32()? as usize;
    let a_data_offset = scaled("a_data_offset", &mut r)?;
    let _a_count_size = scaled("a_count_size", &mut r)?;
    let data_end = r.u32()? as usize;
    if data_end > uncompressed_size {
        return Err(Error::Invalid("data_end past payload"));
    }

    let row_offsets_at = r.pos;
    if data_end <= y_data_size {
        return Err(Error::Invalid("y_data_size exceeds data_end"));
    }
    let y_data_offset = data_end - y_data_size;
    if y_data_offset <= a_data_offset {
        return Err(Error::Invalid("y_data_offset below a_data_offset"));
    }

    Ok(Header {
        width,
        height,
        uv_offset_data_offset,
        y_control_data_offset,
        a_control_word_offset,
        uv_data_offset,
        y_data_offset,
        a_data_offset,
        data_end,
        row_offsets_at,
        opaque: y_data_offset - a_data_offset == a_control_word_offset,
    })
}

/// Absolute bit position, within the payload, of every 4x4 luma block's
/// residuals — the running total the reference decoder accumulates as it walks
/// a band left to right.
///
/// The GPU decodes blocks in parallel and so cannot accumulate anything, but
/// each block's length is fixed by its own control word, so the whole table is
/// a cheap prefix sum here (~130k adds for 1080p).
pub fn bit_offsets(payload: &Payload, header: &Header) -> Result<Vec<u32>, Error> {
    let buf = &payload.bytes[..];
    let (cols4, rows4) = header.blocks4();
    let mut out = vec![0u32; cols4 * rows4];
    let mut rows = Reader::at(buf, header.row_offsets_at);

    for band in 0..rows4 {
        let row_offset = rows.u32()? as usize;
        let mut bit = (header
            .y_data_offset
            .checked_add(row_offset)
            .ok_or(Error::Invalid("luma row offset"))?
            * 8) as u32;
        let mut ctrl = Reader::at(buf, header.y_control_data_offset + band * cols4 * 4);
        for col in 0..cols4 {
            out[band * cols4 + col] = bit;
            let item = ctrl.u32()?;
            // Four sub-rows of four samples, each sample (ctrl+1) bits wide.
            for i in 0..4 {
                bit += 4 * (((item >> (24 + 2 * i)) & 3) + 1);
            }
        }
    }
    Ok(out)
}

fn decode_blocks(buf: &[u8], header_at: usize, uncompressed_size: usize) -> Result<Frame, Error> {
    let h = parse_header_at(buf, header_at, uncompressed_size)?;
    let (width, height) = (h.width as usize, h.height as usize);

    // Pad both axes to 16 so the 4x4 luma and 16x16 chroma/alpha writes stay in
    // bounds without a check in the inner loop.
    let stride = width.next_multiple_of(16);
    let rows = height.next_multiple_of(16);
    let plane = stride * rows;
    let mut frame = Frame {
        width: h.width,
        height: h.height,
        stride,
        y: vec![0; plane],
        u: vec![0; plane],
        v: vec![0; plane],
        a: vec![0; plane],
    };

    decode_luma(buf, &mut frame, h.row_offsets_at, h.y_control_data_offset, h.y_data_offset)?;
    if h.opaque {
        frame.a.fill(4095);
    } else {
        decode_alpha(buf, &mut frame, h.a_control_word_offset, h.uv_data_offset, h.a_data_offset, h.data_end)?;
    }
    decode_chroma(buf, &mut frame, h.uv_offset_data_offset, h.uv_data_offset)?;
    Ok(frame)
}

fn decode_luma(
    buf: &[u8],
    frame: &mut Frame,
    row_offsets_at: usize,
    y_control_data_offset: usize,
    y_data_offset: usize,
) -> Result<(), Error> {
    let (w, h, stride) = (frame.width as usize, frame.height as usize, frame.stride);
    let mut rows = Reader::at(buf, row_offsets_at);
    let mut ctrl = Reader::at(buf, y_control_data_offset);

    let mut band = 0usize;
    while band < h {
        let row_offset = rows.u32()? as usize;
        let base = y_data_offset
            .checked_add(row_offset)
            .ok_or(Error::Invalid("luma row offset"))?;
        let mut bits = BitReader::at(buf, base);

        let mut x = 0usize;
        while x < w {
            let item = ctrl.u32()?;
            let y_min = item & 4095;
            let y_max = (item >> 12) & 4095;
            let y_diff = y_max.wrapping_sub(y_min);

            for i in 0..4 {
                let nb_bits = ((item >> (24 + 2 * i)) & 3) as usize + 1;
                let div = (1u32 << nb_bits) - 1;
                let add = div - 1;
                let row = (band + i) * stride + x;
                for j in 0..4 {
                    let idx = bits.bits(nb_bits);
                    let val = y_min.wrapping_add((y_diff.wrapping_mul(idx) + add) / div);
                    frame.y[row + j] = val.min(4095) as u16;
                }
            }
            x += 4;
        }
        band += 4;
    }
    Ok(())
}

fn decode_alpha(
    buf: &[u8],
    frame: &mut Frame,
    a_control_word_offset: usize,
    uv_data_offset: usize,
    a_data_offset: usize,
    data_end: usize,
) -> Result<(), Error> {
    let (w, h, stride) = (frame.width as usize, frame.height as usize, frame.stride);
    let mut ctrl = Reader::at(buf, a_control_word_offset);

    let mut by16 = 0usize;
    while by16 < h {
        let mut x = 0usize;
        while x < w {
            let mut m = ctrl.u32()?;
            let offset = ctrl.u32()? as usize;
            let offset = offset
                .checked_mul(4)
                .and_then(|o| o.checked_add(uv_data_offset))
                .and_then(|o| o.checked_add(a_data_offset))
                .ok_or(Error::Invalid("alpha block offset"))?;
            if offset >= data_end {
                return Err(Error::Invalid("alpha block past data_end"));
            }

            let mut d = Reader::at(buf, offset);
            let control = d.u64()?;
            let alpha0 = (control & 0xFF) as i32;
            let alpha1 = ((control >> 8) & 0xFF) as i32;
            let mut control = control >> 16;

            for by in 0..4 {
                for bx in 0..4 {
                    let value = match m & 3 {
                        0 => 0u16,
                        1 => 4095u16,
                        2 => (((alpha0 + (alpha1 - alpha0) * (control & 7) as i32) << 4) & 0xFFFF) as u16,
                        _ => return Err(Error::Invalid("alpha block mode 3")),
                    };
                    for i in 0..4 {
                        let row = (by16 + i + by * 4) * stride + x + bx * 4;
                        frame.a[row..row + 4].fill(value);
                    }
                    control >>= 3;
                    m >>= 2;
                }
            }
            x += 16;
        }
        by16 += 16;
    }
    Ok(())
}

fn decode_chroma(
    buf: &[u8],
    frame: &mut Frame,
    uv_offset_data_offset: usize,
    uv_data_offset: usize,
) -> Result<(), Error> {
    let (w, h, stride) = (frame.width as usize, frame.height as usize, frame.stride);
    let mut table = Reader::at(buf, uv_offset_data_offset);

    let mut by16 = 0usize;
    while by16 < h {
        let mut x = 0usize;
        while x < w {
            let offset = table.u32()? as usize;
            let offset = offset
                .checked_mul(4)
                .and_then(|o| o.checked_add(uv_data_offset))
                .ok_or(Error::Invalid("chroma block offset"))?;
            let mut d = Reader::at(buf, offset);

            let mut u = [[0i32; 16]; 16];
            let mut v = [[0i32; 16]; 16];
            let mut is8x8 = d.u16()?;
            let escape = d.u16()?;

            if escape == 0 && is8x8 == 0 {
                let (u0, v0, udif, vdif, mut loc) = read_endpoints(&mut d)?;
                for i in (0..16).step_by(4) {
                    for j in (0..16).step_by(4) {
                        let (uu, vv) = lerp(u0, v0, udif, vdif, loc);
                        for ii in 0..4 {
                            for jj in 0..4 {
                                u[i + ii][j + jj] = uu;
                                v[i + ii][j + jj] = vv;
                            }
                        }
                        loc >>= 2;
                    }
                }
            } else {
                for i in (0..16).step_by(8) {
                    for j in (0..16).step_by(8) {
                        if is8x8 & 1 != 0 {
                            let (u0, v0, udif, vdif, mut loc) = read_endpoints(&mut d)?;
                            for ii in (0..8).step_by(2) {
                                for jj in (0..8).step_by(2) {
                                    let (uu, vv) = lerp(u0, v0, udif, vdif, loc);
                                    for iii in 0..2 {
                                        for jjj in 0..2 {
                                            u[i + ii + iii][j + jj + jjj] = uu;
                                            v[i + ii + iii][j + jj + jjj] = vv;
                                        }
                                    }
                                    loc >>= 2;
                                }
                            }
                        } else if escape != 0 {
                            for ii in (0..8).step_by(4) {
                                for jj in (0..8).step_by(4) {
                                    let (u0, v0, udif, vdif, mut loc) = read_endpoints(&mut d)?;
                                    for iii in 0..4 {
                                        for jjj in 0..4 {
                                            let (uu, vv) = lerp(u0, v0, udif, vdif, loc);
                                            u[i + ii + iii][j + jj + jjj] = uu;
                                            v[i + ii + iii][j + jj + jjj] = vv;
                                            loc >>= 2;
                                        }
                                    }
                                }
                            }
                        }
                        is8x8 >>= 1;
                    }
                }
            }

            for i in 0..16 {
                let row = (by16 + i) * stride + x;
                for j in 0..16 {
                    frame.u[row + j] = u[i][j] as u16;
                    frame.v[row + j] = v[i][j] as u16;
                }
            }
            x += 16;
        }
        by16 += 16;
    }
    Ok(())
}

/// Two 8-bit endpoints per channel, nibble-replicated to 12-bit, plus the
/// packed 2-bit weight selectors.
fn read_endpoints(d: &mut Reader) -> Result<(i32, i32, i32, i32, u32), Error> {
    let rep = |b: u8| -> i32 { (((b as i32) << 4) | (b as i32 & 0xF)) as i32 };
    let u0 = rep(d.u8()?);
    let v0 = rep(d.u8()?);
    let u1 = rep(d.u8()?);
    let v1 = rep(d.u8()?);
    let loc = d.u32()?;
    Ok((u0, v0, u1 - u0, v1 - v0, loc))
}

fn lerp(u0: i32, v0: i32, udif: i32, vdif: i32, loc: u32) -> (i32, i32) {
    let w = (loc & 3) as i32;
    (u0 + (udif * w + 2) / 3, v0 + (vdif * w + 2) / 3)
}

/// LZ4 as the reference decoder implements it: a 64 KiB circular history
/// window, and a zero match offset ends the stream rather than being an error.
/// Stock LZ4 crates reject streams this accepts, so it is ported rather than
/// delegated.
fn lz4_uncompress(src: &[u8], hint: usize) -> Result<Vec<u8>, Error> {
    let mut history = vec![0u8; HISTORY_SIZE];
    let mut out = Vec::with_capacity(hint);
    let mut pos = 0usize;
    let mut r = Reader::at(src, 0);

    while r.pos < src.len() {
        let token = r.u8()?;
        let mut num_literals = (token >> 4) as usize;
        if num_literals == 15 {
            loop {
                let c = r.u8()?;
                num_literals += c as usize;
                if c != 255 {
                    break;
                }
            }
        }

        let literals = r.take(num_literals)?;
        for &b in literals {
            history[pos] = b;
            pos += 1;
            if pos == HISTORY_SIZE {
                out.extend_from_slice(&history);
                pos = 0;
            }
        }

        if r.pos >= src.len() {
            break;
        }

        let delta = r.u16()? as usize;
        if delta == 0 {
            return Ok(out);
        }
        let mut match_length = 4 + (token & 0x0F) as usize;
        if match_length == 4 + 0x0F {
            loop {
                let c = r.u8()?;
                match_length += c as usize;
                if c != 255 {
                    break;
                }
            }
        }

        let mut reference_pos = if pos >= delta {
            pos - delta
        } else {
            HISTORY_SIZE + pos - delta
        };
        for _ in 0..match_length {
            history[pos] = history[reference_pos];
            pos += 1;
            reference_pos = (reference_pos + 1) % HISTORY_SIZE;
            if pos == HISTORY_SIZE {
                out.extend_from_slice(&history);
                pos = 0;
            }
        }
    }

    out.extend_from_slice(&history[..pos]);
    Ok(out)
}

/// LZF, per `libavcodec/lzf.c`. Back-references may overlap the bytes they
/// produce, so the copy is byte-at-a-time.
fn lzf_uncompress(src: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out: Vec<u8> = Vec::new();
    let mut r = Reader::at(src, 0);

    while src.len() - r.pos > 2 {
        let s = r.u8()?;
        if s < LZF_LITERAL_MAX {
            let n = s as usize + 1;
            out.extend_from_slice(r.take(n)?);
        } else {
            let mut l = 2 + (s >> 5) as usize;
            let mut off = (((s & 0x1f) as usize) << 8) + 1;
            if l == LZF_LONG_BACKREF {
                l += r.u8()? as usize;
            }
            off += r.u8()? as usize;
            if off > out.len() {
                return Err(Error::Invalid("lzf back-reference before output"));
            }
            let mut back = out.len() - off;
            for _ in 0..l {
                let b = out[back];
                out.push(b);
                back += 1;
            }
        }
    }
    Ok(out)
}

