use crate::job::{EncodeSettings, GpuMode, FileInfo};
use anyhow::{Context, Result};
use hap_qt::{CompressionMode, HapFrameEncoder, QtHapWriter, VideoConfig};
use hap_wgpu::GpuDxtCompressor;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::Arc;

/// Find ffmpeg binary: check next to our executable first (bundled),
/// then fall back to PATH.
fn find_ffmpeg() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent() {
            let bundled = dir.join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });
            if bundled.exists() {
                return bundled;
            }
        }
    PathBuf::from("ffmpeg")
}

/// Progress updates sent from the encoder thread to the UI.
#[derive(Debug, Clone)]
pub enum EncodeProgress {
    Encoding { frame: u32, total: u32 },
    Complete { duration_secs: f32, output_size: u64 },
    Failed(String),
}

/// Shared GPU resources, created once and reused across jobs.
pub struct GpuResources {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuResources {
    /// Create headless wgpu device for encoding (no window surface needed).
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .ok()?;

        // Check BC texture compression support
        let features = adapter.features();
        if !features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            log::warn!("GPU adapter does not support BC texture compression");
            return None;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("hap-convert"),
                required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
                ..Default::default()
            },
        ))
        .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

/// Encode a single video file to HAP.
/// Runs synchronously — call from a worker thread.
/// Sends progress updates through `progress_tx`.
pub fn encode_file(
    input: &Path,
    output: &Path,
    info: &FileInfo,
    settings: EncodeSettings,
    gpu: Option<&GpuResources>,
    progress_tx: &mpsc::Sender<EncodeProgress>,
) -> Result<()> {
    let EncodeSettings {
        codec,
        quality,
        scale,
        gpu_mode,
    } = settings;
    // Everything downstream — the encoder, the writer, the frame buffer — works
    // in output pixels. ffmpeg does the resampling.
    let (width, height) = scale.apply(info.width, info.height);
    if (width, height) != (info.width, info.height) {
        log::info!(
            "scaling {}x{} to {}x{}",
            info.width,
            info.height,
            width,
            height
        );
    }
    let fps = info.fps;
    let total_frames = info.frame_count;
    let hap_format = codec.to_hap_format();
    let dxt_quality = quality.to_dxt_quality();

    // Decide GPU vs CPU
    let use_gpu = match gpu_mode {
        GpuMode::ForceCpu => false,
        GpuMode::ForceGpu => {
            if gpu.is_none() {
                anyhow::bail!("GPU mode forced but no GPU available");
            }
            true
        }
        GpuMode::Auto => gpu.is_some() && GpuDxtCompressor::supports_format(hap_format),
    };

    // Set up the GPU compressor if we're using it
    let gpu_compressor = if use_gpu {
        let g = gpu.unwrap();
        GpuDxtCompressor::try_new(
            Arc::clone(&g.device),
            Arc::clone(&g.queue),
            width,
            height,
        )
    } else {
        None
    };

    // Create frame encoder (CPU path, also used for Snappy wrapping in GPU path)
    let mut frame_encoder = HapFrameEncoder::new(hap_format, width, height)
        .context("failed to create HAP frame encoder")?;
    frame_encoder.set_compression(CompressionMode::Snappy);
    frame_encoder.set_quality(dxt_quality);

    // Create QuickTime writer
    let video_config = VideoConfig::new(width, height, fps, hap_format);
    let mut writer =
        QtHapWriter::create(output, video_config).context("failed to create output file")?;

    // Spawn ffmpeg to decode input to raw RGBA frames on stdout
    let mut ffmpeg_cmd = Command::new(find_ffmpeg());
    ffmpeg_cmd.args(["-y", "-i"]).arg(input);
    if (width, height) != (info.width, info.height) {
        // Lanczos: this is a downscale that will be looked at on a big screen,
        // and it happens once at encode time rather than every frame at
        // showtime, so spend the quality here.
        ffmpeg_cmd.args(["-vf", &format!("scale={width}:{height}:flags=lanczos")]);
    }
    let mut ffmpeg = ffmpeg_cmd
        .args([
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-an",
            "-v", "quiet",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn ffmpeg — is it installed and on your PATH?")?;

    let stdout = ffmpeg.stdout.take().unwrap();
    let mut reader = std::io::BufReader::new(stdout);

    let frame_size = (width as usize) * (height as usize) * 4;
    let mut frame_buf = vec![0u8; frame_size];
    let mut frame_idx: u32 = 0;

    loop {
        // Read one full RGBA frame
        match reader.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        // Encode the frame
        let hap_frame = if let Some(ref gpu_comp) = gpu_compressor {
            // GPU path: compress on GPU, then wrap with Snappy + HAP header
            let (pw, ph) = gpu_comp.dimensions();
            let input_data = if width != pw || height != ph {
                hap_wgpu::pad_rgba(&frame_buf, width, height, pw, ph)
            } else {
                frame_buf.clone()
            };
            let dxt_data = gpu_comp
                .compress(&input_data, hap_format, dxt_quality)
                .context("GPU DXT compression failed")?;
            frame_encoder
                .encode_from_dxt(&dxt_data)
                .context("HAP frame encoding failed")?
        } else {
            // CPU path: full DXT + Snappy + header
            frame_encoder
                .encode(&frame_buf)
                .context("CPU HAP encoding failed")?
        };

        writer
            .write_frame(&hap_frame)
            .context("failed to write frame")?;

        frame_idx += 1;

        // Send progress every 5 frames to avoid channel congestion
        if frame_idx.is_multiple_of(5) || frame_idx == total_frames {
            let _ = progress_tx.send(EncodeProgress::Encoding {
                frame: frame_idx,
                total: total_frames,
            });
        }
    }

    writer.finalize().context("failed to finalize output")?;

    // Wait for ffmpeg to exit
    let _ = ffmpeg.wait();

    // Report completion
    Ok(())
}

/// Spawn the encoder on a background thread for a single job.
/// Returns a receiver for progress updates.
pub fn spawn_encode(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    info: FileInfo,
    settings: EncodeSettings,
    gpu: Option<Arc<GpuResources>>,
) -> mpsc::Receiver<EncodeProgress> {
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let start = std::time::Instant::now();

        match encode_file(
            &input,
            &output,
            &info,
            settings,
            gpu.as_deref(),
            &tx,
        ) {
            Ok(()) => {
                let duration_secs = start.elapsed().as_secs_f32();
                let output_size = std::fs::metadata(&output)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let _ = tx.send(EncodeProgress::Complete {
                    duration_secs,
                    output_size,
                });
            }
            Err(e) => {
                let _ = tx.send(EncodeProgress::Failed(format!("{e:#}")));
            }
        }
    });

    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{HapCodec, Quality, Scale};

    /// The scale option has to survive the whole pipeline — ffmpeg's filter, the
    /// frame buffer size, the encoder and the QuickTime header all have to agree
    /// on the output dimensions, and a mismatch shows up as a torn image rather
    /// than an error. So encode a real file and read the size back.
    #[test]
    fn encodes_at_the_requested_size() {
        let dir = std::env::temp_dir().join("hap_convert_scale_test");
        let _ = std::fs::create_dir_all(&dir);
        let input = dir.join("src.mp4");

        // Skip rather than fail where ffmpeg is absent; it is a runtime
        // dependency of the converter, not of the build.
        let made_input = Command::new(find_ffmpeg())
            .args([
                "-y", "-f", "lavfi",
                "-i", "testsrc=size=3840x2160:rate=25:duration=1",
                "-pix_fmt", "yuv420p", "-v", "quiet",
            ])
            .arg(&input)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !made_input {
            eprintln!("ffmpeg unavailable — skipping");
            return;
        }

        let info = FileInfo {
            width: 3840,
            height: 2160,
            fps: 25.0,
            frame_count: 25,
            duration_secs: 1.0,
        };
        let gpu = GpuResources::try_new();

        for (scale, expected) in [
            (Scale::Original, (3840, 2160)),
            (Scale::Hd1080, (1920, 1080)),
            (Scale::Hd720, (1280, 720)),
        ] {
            let output = dir.join(format!("out_{}.mov", scale.label()));
            let (tx, rx) = mpsc::channel();
            encode_file(
                &input,
                &output,
                &info,
                EncodeSettings {
                    codec: HapCodec::Hap1,
                    quality: Quality::Fast,
                    scale,
                    gpu_mode: GpuMode::Auto,
                },
                gpu.as_ref(),
                &tx,
            )
            .unwrap_or_else(|e| panic!("{} encode failed: {e}", scale.label()));
            drop(rx);

            let reader = hap_qt::QtReader::open(&output)
                .unwrap_or_else(|e| panic!("{} output unreadable: {e:?}", scale.label()));
            assert_eq!(
                reader.resolution(),
                expected,
                "{} produced the wrong size",
                scale.label()
            );
            assert!(reader.frame_count() > 0, "{} made no frames", scale.label());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
