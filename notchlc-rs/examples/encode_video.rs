//! Encode any video ffmpeg can read into a NotchLC .mov.
//!
//! Usage: cargo run --release --example encode_video -- <in> <out.mov> [max_frames]
//!
//! Note the spike encoder's LZ4 wrapper is literal-only, so output is
//! effectively uncompressed — expect roughly 700 KB per 1080p-ish frame.

use notchlc_rs::{FrameEncoder, MovWriter};
use std::io::Read;
use std::process::{Command, Stdio};

fn probe(path: &str, key: &str) -> Option<String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate", "-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let (name, value) = l.split_once('=')?;
        (name.trim() == key).then(|| value.trim().to_string())
    })
}

/// 8-bit -> 12-bit via nibble replication, matching the decoder's expansion.
fn to12(v: u8) -> u16 {
    ((v as u16) << 4) | ((v as u16) >> 4)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: encode_video <in> <out.mov> [max_frames]");
    let output = args.next().expect("usage: encode_video <in> <out.mov> [max_frames]");
    let max_frames: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);

    let w: usize = probe(&input, "width").and_then(|v| v.parse().ok()).expect("no width");
    let h: usize = probe(&input, "height").and_then(|v| v.parse().ok()).expect("no height");
    let fps = probe(&input, "r_frame_rate")
        .and_then(|v| {
            let (n, d) = v.split_once('/')?;
            Some((n.parse::<f64>().ok()? / d.parse::<f64>().ok()?).round() as u32)
        })
        .unwrap_or(30);
    println!("{input}: {w}x{h} @ {fps}fps");

    let mut child = Command::new("ffmpeg")
        .arg("-i").arg(&input)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-v", "quiet", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("ffmpeg not runnable");
    let mut stdout = child.stdout.take().unwrap();

    let enc = FrameEncoder::new(w as u32, h as u32);
    let mut mov = MovWriter::new(w as u32, h as u32, fps);
    let mut rgb = vec![0u8; w * h * 3];
    // Decoder output is GBR identity: Y carries G, U carries B, V carries R.
    let (mut y, mut u, mut v) = (vec![0u16; w * h], vec![0u16; w * h], vec![0u16; w * h]);
    let mut n = 0usize;

    while n < max_frames && stdout.read_exact(&mut rgb).is_ok() {
        for i in 0..w * h {
            y[i] = to12(rgb[i * 3 + 1]);
            u[i] = to12(rgb[i * 3 + 2]);
            v[i] = to12(rgb[i * 3]);
        }
        mov.add_sample(enc.encode_packet(&y, &u, &v));
        n += 1;
        if n % 50 == 0 {
            println!("  {n} frames");
        }
    }
    let _ = child.kill();

    let mut file = std::fs::File::create(&output).unwrap();
    mov.write_to(&mut file).unwrap();
    let size = std::fs::metadata(&output).unwrap().len();
    println!("wrote {output}: {n} frames, {:.1} MB", size as f64 / 1e6);
}
