//! Decode every frame of test.mov and compare against ffmpeg's own decode.
//!
//! ffmpeg is the reference implementation, so it is the oracle. Comparison is
//! against yuva444p12le, the decoder's *native* format. Do not "simplify" this
//! to rgb24 or gbrp12le: swscale ignores the AVCOL_SPC_RGB identity colorspace
//! the decoder declares and applies a BT.601 matrix, so those outputs disagree
//! with a correct decode.
//!
//! Regenerate the input with `cargo run` if it goes missing.

#![cfg(feature = "clip")]
use hap_qt::QtReader;
use std::path::Path;
use std::process::Command;

#[test]
fn matches_ffmpeg() {
    // Defaults to the spike's own output; point NLC_TEST_MOV at any NotchLC
    // file to check the decoder against ffmpeg on real content.
    let mov_path = std::env::var("NLC_TEST_MOV")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("test.mov"));
    if !mov_path.exists() {
        eprintln!("skipping: test.mov missing — run `cargo run` to generate it");
        return;
    }

    let reference = match Command::new("ffmpeg")
        .arg("-i")
        .arg(&mov_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuva444p12le", "-v", "quiet", "-"])
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        Ok(o) => panic!("ffmpeg failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(e) => {
            eprintln!("skipping: ffmpeg not runnable ({e})");
            return;
        }
    };

    let reader = QtReader::open_codec(&mov_path, &["nclc"]).expect("demux failed");
    let (width, height) = reader.resolution();
    let frame_count = reader.frame_count();
    assert!(frame_count > 0, "no samples found");

    let mov = std::fs::read(&mov_path).unwrap();
    let plane = width as usize * height as usize;
    let frame_bytes = plane * 4 * 2;
    assert_eq!(
        reference.len(),
        frame_bytes * frame_count as usize,
        "ffmpeg returned {} frames, demux found {frame_count}",
        reference.len() / frame_bytes
    );

    for n in 0..frame_count {
        let (offset, size) = reader.sample_range(n).unwrap();
        let packet = &mov[offset as usize..offset as usize + size as usize];
        let frame = notchlc_rs::decode_packet(packet).unwrap_or_else(|e| panic!("frame {n}: {e}"));
        assert_eq!((frame.width, frame.height), (width, height));

        let base = n as usize * frame_bytes;
        for (p, (name, ours)) in [
            ("Y", &frame.y),
            ("U", &frame.u),
            ("V", &frame.v),
            ("A", &frame.a),
        ]
        .into_iter()
        .enumerate()
        {
            for row in 0..height as usize {
                for col in 0..width as usize {
                    let off = base + p * plane * 2 + (row * width as usize + col) * 2;
                    let want = u16::from_le_bytes([reference[off], reference[off + 1]]);
                    let got = ours[row * frame.stride + col];
                    assert_eq!(
                        got, want,
                        "frame {n} plane {name} at ({col},{row}): got {got}, ffmpeg says {want}"
                    );
                }
            }
        }
    }

    eprintln!("{frame_count} frames match ffmpeg exactly on all four 12-bit planes");
}
