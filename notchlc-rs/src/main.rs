//! Spike test: generate a known RGB pattern, encode as NotchLC, mux to mov.
//! Also dumps frame 0 as raw rgb24 for comparison against an ffmpeg decode.
//!
//! Usage: nlc_encode [out.mov]

use notchlc_rs::{FrameEncoder, MovWriter};

const W: usize = 256;
const H: usize = 256;
const FRAMES: usize = 30;

/// 8-bit source pattern: moving red sweep, green vertical gradient,
/// blue diagonal gradient.
fn pattern(f: usize) -> Vec<[u8; 3]> {
    let mut px = vec![[0u8; 3]; W * H];
    for y in 0..H {
        for x in 0..W {
            px[y * W + x] = [
                ((x + f * 4) % 256) as u8,
                y as u8,
                ((x + y) / 2) as u8,
            ];
        }
    }
    px
}

/// 8-bit -> 12-bit via nibble replication.
fn to12(v: u8) -> u16 {
    ((v as u16) << 4) | ((v as u16) >> 4)
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "test.mov".into());

    let enc = FrameEncoder::new(W as u32, H as u32);
    let mut mov = MovWriter::new(W as u32, H as u32, 30);

    for f in 0..FRAMES {
        let px = pattern(f);
        // Decoder outputs YUVA444P12 with identity ("RGB") colorspace, so the
        // planes are GBR: Y<-G, U<-B, V<-R. Verified against ffmpeg decode.
        let mut y = vec![0u16; W * H];
        let mut u = vec![0u16; W * H];
        let mut v = vec![0u16; W * H];
        for i in 0..W * H {
            y[i] = to12(px[i][1]);
            u[i] = to12(px[i][2]);
            v[i] = to12(px[i][0]);
        }
        mov.add_sample(enc.encode_packet(&y, &u, &v));

        if f == 0 {
            let mut rgb = Vec::with_capacity(W * H * 3);
            for p in &px {
                rgb.extend_from_slice(p);
            }
            std::fs::write("expected0.rgb", &rgb).unwrap();
        }
    }

    let mut file = std::fs::File::create(&out).unwrap();
    mov.write_to(&mut file).unwrap();
    println!("wrote {out}");
}
