use crate::encode::{self, EncodeProgress, GpuResources};
use crate::job::{ConvertJob, GpuMode, HapCodec, JobQueue, JobStatus};
use crate::probe;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};

/// Main application state.
pub struct HapConvertApp {
    /// Job queue holding all files to convert.
    queue: JobQueue,
    /// Global codec selection (applied to new files).
    codec: HapCodec,
    /// GPU/CPU preference.
    gpu_mode: GpuMode,
    /// Custom output directory (None = same as input).
    output_dir: Option<PathBuf>,
    /// Shared GPU resources.
    gpu: Option<Arc<GpuResources>>,
    /// Whether a job is currently encoding.
    encoding_active: bool,
    /// Progress channel for the currently active job.
    active_rx: Option<mpsc::Receiver<EncodeProgress>>,
    /// Index of the currently encoding job.
    active_job_idx: Option<usize>,
    /// Whether GPU init has been attempted.
    gpu_checked: bool,
}

impl HapConvertApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            queue: JobQueue::new(),
            codec: HapCodec::Hap1,
            gpu_mode: GpuMode::Auto,
            output_dir: None,
            gpu: None,
            encoding_active: false,
            active_rx: None,
            active_job_idx: None,
            gpu_checked: false,
        }
    }

    /// Try to initialize GPU resources (once).
    fn ensure_gpu(&mut self) {
        if !self.gpu_checked {
            self.gpu_checked = true;
            match GpuResources::try_new() {
                Some(g) => {
                    log::info!("GPU encoding available");
                    self.gpu = Some(Arc::new(g));
                }
                None => {
                    log::warn!("GPU encoding not available, using CPU");
                }
            }
        }
    }

    /// Add files to the queue.
    fn add_files(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            // Skip non-video extensions
            if !is_video_file(&path) {
                continue;
            }
            // Avoid duplicates
            if self.queue.jobs.iter().any(|j| j.input_path == path) {
                continue;
            }
            let mut job = ConvertJob::new(path.clone(), self.codec, self.output_dir.as_ref());
            // Probe the file for metadata
            match probe::probe_video(&path) {
                Ok(info) => {
                    job.file_info = Some(info);
                }
                Err(e) => {
                    log::warn!("Failed to probe {}: {e}", path.display());
                }
            }
            self.queue.add(job);
        }
    }

    /// Start encoding the next queued job.
    fn start_next_job(&mut self) {
        if self.encoding_active {
            return;
        }
        self.ensure_gpu();

        if let Some(idx) = self.queue.next_queued() {
            let job = &mut self.queue.jobs[idx];

            // Need file info to encode
            let info = match &job.file_info {
                Some(i) => i.clone(),
                None => {
                    job.status = JobStatus::Failed("No file info available".into());
                    return;
                }
            };

            job.status = JobStatus::Encoding {
                frame: 0,
                total: info.frame_count,
            };

            let rx = encode::spawn_encode(
                job.input_path.clone(),
                job.output_path.clone(),
                info,
                job.codec,
                self.gpu_mode,
                self.gpu.clone(),
            );

            self.active_rx = Some(rx);
            self.active_job_idx = Some(idx);
            self.encoding_active = true;
        }
    }

    /// Poll the active encoder for progress.
    fn poll_progress(&mut self) {
        let Some(rx) = &self.active_rx else { return };
        let Some(idx) = self.active_job_idx else { return };

        // Drain all available messages
        while let Ok(msg) = rx.try_recv() {
            match msg {
                EncodeProgress::Encoding { frame, total } => {
                    if let Some(job) = self.queue.jobs.get_mut(idx) {
                        job.status = JobStatus::Encoding { frame, total };
                    }
                }
                EncodeProgress::Complete {
                    duration_secs,
                    output_size,
                } => {
                    if let Some(job) = self.queue.jobs.get_mut(idx) {
                        job.status = JobStatus::Complete {
                            duration_secs,
                            output_size,
                        };
                    }
                    self.encoding_active = false;
                    self.active_rx = None;
                    self.active_job_idx = None;
                    // Auto-start the next job
                    self.start_next_job();
                    return;
                }
                EncodeProgress::Failed(e) => {
                    if let Some(job) = self.queue.jobs.get_mut(idx) {
                        job.status = JobStatus::Failed(e);
                    }
                    self.encoding_active = false;
                    self.active_rx = None;
                    self.active_job_idx = None;
                    // Continue with next job despite failure
                    self.start_next_job();
                    return;
                }
                EncodeProgress::Probing => {}
            }
        }
    }

    /// Start encoding all queued jobs.
    fn convert_all(&mut self) {
        self.start_next_job();
    }
}

impl eframe::App for HapConvertApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll encoder progress
        self.poll_progress();

        // Request repaint while encoding
        if self.encoding_active {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Handle dropped files
        let dropped: Vec<PathBuf> = ctx
            .input(|i| {
                i.raw.dropped_files
                    .iter()
                    .filter_map(|f| f.path.clone())
                    .collect()
            });
        if !dropped.is_empty() {
            self.add_files(dropped);
        }

        // Main panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("rustjay-hap-convert");
            ui.add_space(4.0);

            // --- Settings row ---
            ui.horizontal(|ui| {
                ui.label("Codec:");
                egui::ComboBox::from_id_salt("codec_select")
                    .selected_text(self.codec.label())
                    .show_ui(ui, |ui| {
                        for c in HapCodec::ALL {
                            ui.selectable_value(&mut self.codec, *c, c.label());
                        }
                    });

                ui.separator();

                ui.label("GPU:");
                egui::ComboBox::from_id_salt("gpu_select")
                    .selected_text(self.gpu_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.gpu_mode, GpuMode::Auto, "Auto");
                        ui.selectable_value(&mut self.gpu_mode, GpuMode::ForceGpu, "GPU");
                        ui.selectable_value(&mut self.gpu_mode, GpuMode::ForceCpu, "CPU");
                    });

                ui.separator();

                ui.label("Output:");
                if let Some(dir) = &self.output_dir {
                    let dir_name = dir
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    ui.label(format!("{dir_name}/"));
                } else {
                    ui.label("Same as input");
                }
                if ui.button("Browse...").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(dir);
                    }
                }
                if self.output_dir.is_some() && ui.button("Reset").clicked() {
                    self.output_dir = None;
                }
            });

            ui.add_space(8.0);

            // --- File list ---
            let available = ui.available_size();
            let list_height = (available.y - 40.0).max(100.0);

            egui::Frame::new()
                .fill(ui.visuals().extreme_bg_color)
                .corner_radius(4.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    if self.queue.is_empty() {
                        ui.allocate_space(egui::vec2(available.x - 24.0, list_height));
                        ui.centered_and_justified(|ui| {
                            ui.label("Drop video files here or click Add Files");
                        });
                    } else {
                        egui::ScrollArea::vertical()
                            .max_height(list_height)
                            .show(ui, |ui| {
                                ui.set_min_width(available.x - 24.0);
                                let mut remove_idx = None;

                                for (i, job) in self.queue.jobs.iter_mut().enumerate() {
                                    ui.horizontal(|ui| {
                                        // File name
                                        let name = job.file_name();
                                        ui.label(
                                            egui::RichText::new(&name)
                                                .monospace()
                                                .strong(),
                                        );

                                        // Resolution + fps
                                        if let Some(info) = &job.file_info {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{}  {:.0}fps  {}",
                                                    info.resolution_label(),
                                                    info.fps,
                                                    info.duration_label(),
                                                ))
                                                .weak(),
                                            );
                                        }

                                        // Per-job codec override
                                        egui::ComboBox::from_id_salt(format!("codec_{i}"))
                                            .width(100.0)
                                            .selected_text(job.codec.short_label())
                                            .show_ui(ui, |ui| {
                                                for c in HapCodec::ALL {
                                                    ui.selectable_value(
                                                        &mut job.codec,
                                                        *c,
                                                        c.short_label(),
                                                    );
                                                }
                                            });

                                        // Progress / status
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                // Remove button (only if not encoding this one)
                                                if !matches!(job.status, JobStatus::Encoding { .. })
                                                    && ui.small_button("x").clicked()
                                                {
                                                    remove_idx = Some(i);
                                                }

                                                match &job.status {
                                                    JobStatus::Queued => {
                                                        ui.label("Queued");
                                                    }
                                                    JobStatus::Probing => {
                                                        ui.spinner();
                                                        ui.label("Probing...");
                                                    }
                                                    JobStatus::Encoding { frame, total } => {
                                                        let pct = if *total > 0 {
                                                            *frame as f32 / *total as f32
                                                        } else {
                                                            0.0
                                                        };
                                                        ui.add(
                                                            egui::ProgressBar::new(pct)
                                                                .desired_width(120.0)
                                                                .text(format!(
                                                                    "{frame}/{total}"
                                                                )),
                                                        );
                                                    }
                                                    JobStatus::Complete {
                                                        duration_secs,
                                                        output_size,
                                                    } => {
                                                        let size_mb =
                                                            *output_size as f32 / (1024.0 * 1024.0);
                                                        ui.label(
                                                            egui::RichText::new(format!(
                                                                "Done ({duration_secs:.1}s, {size_mb:.1}MB)"
                                                            ))
                                                            .color(egui::Color32::from_rgb(100, 200, 100)),
                                                        );
                                                    }
                                                    JobStatus::Failed(e) => {
                                                        ui.label(
                                                            egui::RichText::new(format!("Error: {e}"))
                                                                .color(egui::Color32::from_rgb(200, 80, 80)),
                                                        );
                                                    }
                                                }
                                            },
                                        );
                                    });
                                    ui.separator();
                                }

                                if let Some(idx) = remove_idx {
                                    self.queue.jobs.remove(idx);
                                    // Adjust active job index if needed
                                    if let Some(active) = self.active_job_idx {
                                        if idx < active {
                                            self.active_job_idx = Some(active - 1);
                                        } else if idx == active {
                                            // Removed the active job — shouldn't happen since
                                            // we block removal during encoding, but handle it
                                            self.encoding_active = false;
                                            self.active_rx = None;
                                            self.active_job_idx = None;
                                        }
                                    }
                                }
                            });
                    }
                });

            ui.add_space(8.0);

            // --- Bottom bar ---
            ui.horizontal(|ui| {
                if ui.button("Add Files").clicked() {
                    let files = rfd::FileDialog::new()
                        .add_filter(
                            "Video",
                            &["mp4", "mov", "avi", "mkv", "webm", "m4v", "mxf", "ts"],
                        )
                        .pick_files();
                    if let Some(paths) = files {
                        self.add_files(paths);
                    }
                }

                let has_queued = self.queue.next_queued().is_some();
                let convert_btn = ui.add_enabled(
                    has_queued && !self.encoding_active,
                    egui::Button::new("Convert All"),
                );
                if convert_btn.clicked() {
                    self.convert_all();
                }

                if ui
                    .add_enabled(!self.queue.is_empty(), egui::Button::new("Clear"))
                    .clicked()
                {
                    if !self.encoding_active {
                        self.queue.clear();
                    } else {
                        self.queue.remove_finished();
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let total = self.queue.jobs.len();
                    let done = self.queue.count_complete();
                    if total > 0 {
                        ui.label(format!("{done}/{total} complete"));
                    }
                    if self.gpu.is_some() {
                        ui.label(
                            egui::RichText::new("GPU")
                                .small()
                                .color(egui::Color32::from_rgb(100, 200, 100)),
                        );
                    }
                });
            });
        });
    }
}

fn is_video_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_lowercase().as_str(),
            "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "mxf" | "ts" | "wmv" | "flv"
        ),
        None => false,
    }
}
