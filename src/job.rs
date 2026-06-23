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
    /// BC7 — highest quality RGBA
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
            HapCodec::Hap7 => "HAP Q (BC7)",
            HapCodec::HapA => "HAP Alpha-Only (BC4)",
        }
    }

    pub fn short_label(&self) -> &'static str {
        match self {
            HapCodec::Hap1 => "HAP1",
            HapCodec::Hap5 => "HAP5",
            HapCodec::HapY => "HAPY",
            HapCodec::Hap7 => "HAP7",
            HapCodec::HapA => "HAPA",
        }
    }

    pub fn file_suffix(&self) -> &'static str {
        match self {
            HapCodec::Hap1 => "_hap1",
            HapCodec::Hap5 => "_hap5",
            HapCodec::HapY => "_hapq",
            HapCodec::Hap7 => "_hap7",
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
