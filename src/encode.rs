use crate::job::{FileInfo, GpuMode, HapCodec};
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
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });
            if bundled.exists() {
                return bundled;
            }
        }
    }
    PathBuf::from("ffmpeg")
}

/// Progress updates sent from the encoder thread to the UI.
#[derive(Debug, Clone)]
pub enum EncodeProgress {
    Probing,
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
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
    codec: HapCodec,
    gpu_mode: GpuMode,
    gpu: Option<&GpuResources>,
    progress_tx: &mpsc::Sender<EncodeProgress>,
) -> Result<()> {
    let width = info.width;
    let height = info.height;
    let fps = info.fps;
    let total_frames = info.frame_count;
    let hap_format = codec.to_hap_format();

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

    // Create QuickTime writer
    let video_config = VideoConfig::new(width, height, fps, hap_format);
    let mut writer =
        QtHapWriter::create(output, video_config).context("failed to create output file")?;

    // Spawn ffmpeg to decode input to raw RGBA frames on stdout
    let mut ffmpeg = Command::new(find_ffmpeg())
        .args([
            "-y",
            "-i",
        ])
        .arg(input)
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
                .compress(&input_data, hap_format)
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
        if frame_idx % 5 == 0 || frame_idx == total_frames {
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
    codec: HapCodec,
    gpu_mode: GpuMode,
    gpu: Option<Arc<GpuResources>>,
) -> mpsc::Receiver<EncodeProgress> {
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let start = std::time::Instant::now();

        match encode_file(
            &input,
            &output,
            &info,
            codec,
            gpu_mode,
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
