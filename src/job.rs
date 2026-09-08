use std::path::PathBuf;

/// All supported HAP codec variants for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapCodec {
    /// DXT1 / BC1 — fast, smallest file, RGB only (no alpha)
    Hap1,
    /// DXT5 / BC3 — RGBA with full alpha channel
    Hap5,
    /// DXT5-YCoCg — high-quality colour, no alpha
    HapY,
    /// BC7 — highest quality RGBA (Hap R)
    Hap7,
    /// BC4 — alpha channel only
    HapA,
}

impl HapCodec {
    pub const ALL: &[HapCodec] = &[
        HapCodec::Hap1,
        HapCodec::Hap5,
        HapCodec::HapY,
        HapCodec::Hap7,
        HapCodec::HapA,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            HapCodec::Hap1 => "HAP (DXT1)",
            HapCodec::Hap5 => "HAP Alpha (DXT5)",
            HapCodec::HapY => "HAP Q (YCoCg)",
            HapCodec::Hap7 => "HAP R (BC7)",
            HapCodec::HapA => "HAP Alpha-Only (BC4)",
        }
    }

    pub fn short_label(&self) -> &'static str {
        match self {
            HapCodec::Hap1 => "HAP1",
            HapCodec::Hap5 => "HAP5",
            HapCodec::HapY => "HAPY",
            HapCodec::Hap7 => "HAPR",
            HapCodec::HapA => "HAPA",
        }
    }

    pub fn file_suffix(&self) -> &'static str {
        match self {
            HapCodec::Hap1 => "_hap1",
            HapCodec::Hap5 => "_hap5",
            HapCodec::HapY => "_hapq",
            HapCodec::Hap7 => "_hapr",
            HapCodec::HapA => "_hapa",
        }
    }

    /// Convert to hap-qt HapFormat.
    pub fn to_hap_format(self) -> hap_qt::HapFormat {
        match self {
            HapCodec::Hap1 => hap_qt::HapFormat::Hap1,
            HapCodec::Hap5 => hap_qt::HapFormat::Hap5,
            HapCodec::HapY => hap_qt::HapFormat::HapY,
            HapCodec::Hap7 => hap_qt::HapFormat::Hap7,
            HapCodec::HapA => hap_qt::HapFormat::HapA,
        }
    }
}

/// Encoding quality preset.
///
/// Applies to every codec on both the GPU and CPU paths: it sets the number of
/// endpoint-refinement rounds in the compute shaders, the texpresso algorithm
/// for CPU DXT, and the BC7 refit count for Hap R.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Bounding-box / range fit only. Fastest.
    Fast,
    #[default]
    Balanced,
    /// Most refinement rounds. Still far faster than CPU on the GPU path.
    Best,
}

impl Quality {
    pub const ALL: &[Quality] = &[Quality::Fast, Quality::Balanced, Quality::Best];

    pub fn label(&self) -> &'static str {
        match self {
            Quality::Fast => "Fast",
            Quality::Balanced => "Balanced",
            Quality::Best => "Best",
        }
    }

    pub fn to_dxt_quality(self) -> hap_qt::DxtQuality {
        match self {
            Quality::Fast => hap_qt::DxtQuality::Fast,
            Quality::Balanced => hap_qt::DxtQuality::Balanced,
            Quality::Best => hap_qt::DxtQuality::Best,
        }
    }
}

/// Output resolution, as a fraction of the source or a fixed height.
///
/// Playback cost is per pixel and nothing else: HAP frames decompress at a
/// fixed rate per byte, and a 4K frame is four times the bytes of a 1080p one.
/// Encoding a 4K master for a 1080p output buys nothing and costs four times
/// the CPU at showtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scale {
    #[default]
    Original,
    /// Fit inside 1920x1080, keeping aspect.
    Hd1080,
    /// Fit inside 1280x720, keeping aspect.
    Hd720,
    Half,
}

impl Scale {
    pub const ALL: &[Scale] = &[Scale::Original, Scale::Hd1080, Scale::Hd720, Scale::Half];

    pub fn label(&self) -> &'static str {
        match self {
            Scale::Original => "Original",
            Scale::Hd1080 => "1080p",
            Scale::Hd720 => "720p",
            Scale::Half => "Half",
        }
    }

    /// Output size for a `width` x `height` source.
    ///
    /// Never upscales — a 720p source asked for 1080p stays 720p. Both axes are
    /// rounded to a multiple of 4 so DXT blocks land whole and the encoder has
    /// nothing to pad.
    pub fn apply(self, width: u32, height: u32) -> (u32, u32) {
        let (w, h) = match self {
            Scale::Original => (width, height),
            Scale::Half => (width / 2, height / 2),
            Scale::Hd1080 => fit_within(width, height, 1920, 1080),
            Scale::Hd720 => fit_within(width, height, 1280, 720),
        };
        (round_to_block(w), round_to_block(h))
    }
}

/// Scale down to fit inside `max_w` x `max_h`, keeping aspect. Never enlarges.
fn fit_within(width: u32, height: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if width <= max_w && height <= max_h {
        return (width, height);
    }
    let ratio = (max_w as f64 / width as f64).min(max_h as f64 / height as f64);
    (
        (width as f64 * ratio).round() as u32,
        (height as f64 * ratio).round() as u32,
    )
}

/// Round to a multiple of 4, the DXT block size, staying at least one block.
fn round_to_block(n: u32) -> u32 {
    (n / 4).max(1) * 4
}

/// GPU vs CPU encoding preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMode {
    Auto,
    ForceGpu,
    ForceCpu,
}

impl GpuMode {
    pub fn label(&self) -> &'static str {
        match self {
            GpuMode::Auto => "Auto",
            GpuMode::ForceGpu => "GPU",
            GpuMode::ForceCpu => "CPU",
        }
    }
}

/// Everything that decides how a file is encoded.
///
/// These four always travel together from the UI to the encoder; passing them
/// as one keeps the next option from having to thread through three signatures.
#[derive(Debug, Clone, Copy)]
pub struct EncodeSettings {
    pub codec: HapCodec,
    pub quality: Quality,
    pub scale: Scale,
    pub gpu_mode: GpuMode,
}

/// Probed metadata about an input file.
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frame_count: u32,
    pub duration_secs: f32,
}

impl FileInfo {
    pub fn resolution_label(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    pub fn duration_label(&self) -> String {
        let mins = (self.duration_secs / 60.0).floor() as u32;
        let secs = (self.duration_secs % 60.0).floor() as u32;
        format!("{mins}:{secs:02}")
    }
}

/// Status of a single conversion job.
#[derive(Debug, Clone)]
pub enum JobStatus {
    Queued,
    Encoding { frame: u32, total: u32 },
    Complete { duration_secs: f32, output_size: u64 },
    Failed(String),
}

impl JobStatus {
    pub fn is_finished(&self) -> bool {
        matches!(self, JobStatus::Complete { .. } | JobStatus::Failed(_))
    }
}

/// A single file conversion job.
#[derive(Debug, Clone)]
pub struct ConvertJob {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub codec: HapCodec,
    pub status: JobStatus,
    pub file_info: Option<FileInfo>,
}

impl ConvertJob {
    pub fn new(input_path: PathBuf, codec: HapCodec, output_dir: Option<&PathBuf>) -> Self {
        let stem = input_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let out_name = format!("{}{}.mov", stem, codec.file_suffix());
        let output_path = match output_dir {
            Some(dir) => dir.join(&out_name),
            None => input_path.parent().unwrap_or(std::path::Path::new(".")).join(&out_name),
        };
        Self {
            input_path,
            output_path,
            codec,
            status: JobStatus::Queued,
            file_info: None,
        }
    }

    pub fn file_name(&self) -> String {
        self.input_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into()
    }
}

/// Manages the batch job queue.
pub struct JobQueue {
    pub jobs: Vec<ConvertJob>,
}

impl JobQueue {
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    pub fn add(&mut self, job: ConvertJob) {
        self.jobs.push(job);
    }

    pub fn clear(&mut self) {
        self.jobs.clear();
    }

    pub fn remove_finished(&mut self) {
        self.jobs.retain(|j| !j.status.is_finished());
    }

    pub fn next_queued(&self) -> Option<usize> {
        self.jobs.iter().position(|j| matches!(j.status, JobStatus::Queued))
    }

    pub fn count_complete(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Complete { .. }))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_keeps_aspect_and_never_enlarges() {
        // 4K 16:9 down to 1080p.
        assert_eq!(Scale::Hd1080.apply(3840, 2160), (1920, 1080));
        // Already smaller than the target: left alone.
        assert_eq!(Scale::Hd1080.apply(1280, 720), (1280, 720));
        assert_eq!(Scale::Original.apply(3840, 2160), (3840, 2160));
        assert_eq!(Scale::Half.apply(3840, 2160), (1920, 1080));

        // A tall 4:3 source fits by height, not width.
        let (w, h) = Scale::Hd720.apply(1600, 1200);
        assert_eq!(h, 720);
        assert!((w as i32 - 960).abs() <= 4, "expected ~960 wide, got {w}");

        // Every result is a whole number of DXT blocks.
        for scale in Scale::ALL {
            for (w, h) in [(3840, 2160), (1920, 1080), (1366, 768), (640, 482)] {
                let (ow, oh) = scale.apply(w, h);
                assert_eq!((ow % 4, oh % 4), (0, 0), "{scale:?} on {w}x{h} -> {ow}x{oh}");
                assert!(ow > 0 && oh > 0);
            }
        }
    }
}
