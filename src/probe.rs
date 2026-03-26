use crate::job::FileInfo;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Find ffprobe binary: check next to our executable first (bundled),
/// then fall back to PATH.
fn find_ffprobe() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
            if bundled.exists() {
                return bundled;
            }
        }
    }
    PathBuf::from("ffprobe")
}

/// Probe a video file for metadata. Tries native MP4 parsing first,
/// falls back to ffprobe.
pub fn probe_video(path: &Path) -> Result<FileInfo> {
    probe_native_mp4(path).or_else(|_| probe_ffprobe(path))
}

/// Native MP4 probe using the `mp4` crate — no ffmpeg needed for H.264/MP4.
fn probe_native_mp4(path: &Path) -> Result<FileInfo> {
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let reader = mp4::Mp4Reader::read_header(std::io::BufReader::new(file), size)?;

    let track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track found")?;

    let width = track.width();
    let height = track.height();
    let duration_ms = track.duration().as_millis() as f32;
    let duration_secs = duration_ms / 1000.0;
    let frame_count = track.sample_count();
    let fps = if duration_secs > 0.0 {
        frame_count as f32 / duration_secs
    } else {
        30.0
    };

    let codec_name = format!("{:?}", track.media_type().unwrap_or(mp4::MediaType::H264));

    Ok(FileInfo {
        width: width as u32,
        height: height as u32,
        fps,
        frame_count,
        duration_secs,
        codec_name,
    })
}

/// Probe using ffprobe subprocess — handles any format ffmpeg supports.
fn probe_ffprobe(path: &Path) -> Result<FileInfo> {
    let output = Command::new(find_ffprobe())
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-select_streams", "v:0",
        ])
        .arg(path)
        .output()
        .context("ffprobe not found — install ffmpeg for non-MP4 input files")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("failed to parse ffprobe JSON")?;

    let stream = json["streams"]
        .as_array()
        .and_then(|s| s.first())
        .context("no video stream in ffprobe output")?;

    let width = stream["width"].as_u64().unwrap_or(0) as u32;
    let height = stream["height"].as_u64().unwrap_or(0) as u32;
    let codec_name = stream["codec_name"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    // Parse frame rate from r_frame_rate "30/1" or "30000/1001"
    let fps = parse_fps(stream["r_frame_rate"].as_str().unwrap_or("30/1"));

    // Duration from format or stream
    let duration_secs = json["format"]["duration"]
        .as_str()
        .or(stream["duration"].as_str())
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);

    let frame_count = stream["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or_else(|| (fps * duration_secs).round() as u32);

    Ok(FileInfo {
        width,
        height,
        fps,
        frame_count,
        duration_secs,
        codec_name,
    })
}

fn parse_fps(rate_str: &str) -> f32 {
    if let Some((num, den)) = rate_str.split_once('/') {
        let n: f32 = num.parse().unwrap_or(30.0);
        let d: f32 = den.parse().unwrap_or(1.0);
        if d > 0.0 { n / d } else { 30.0 }
    } else {
        rate_str.parse().unwrap_or(30.0)
    }
}
