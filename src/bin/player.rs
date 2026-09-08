//! Minimal NotchLC / HAP player. ffmpeg decodes to raw RGBA on a pipe; egui shows it.
//!
//! ponytail: CPU decode via the ffmpeg pipe — same trick encode.rs already uses, and it
//! plays every codec ffmpeg knows. For GPU-native HAP (no CPU touch) swap in
//! hap_wgpu::HapPlayer, which already exists; do that when 4K playback drops frames.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::time::{Duration, Instant};

// ponytail: copied from encode.rs rather than promoting the crate to a lib for 8 lines.
fn tool(name: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let bundled = dir.join(if cfg!(windows) { format!("{name}.exe") } else { name.into() });
        if bundled.exists() {
            return bundled;
        }
    }
    PathBuf::from(name)
}

/// A running ffmpeg process feeding decoded frames over a bounded channel.
struct Source {
    path: PathBuf,
    child: Child,
    rx: Receiver<Vec<u8>>,
    w: usize,
    h: usize,
    fps: f32,
}

impl Source {
    fn open(path: &Path) -> Result<Self, String> {
        let probe = Command::new(tool("ffprobe"))
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height,r_frame_rate,codec_name",
                   // key=value, because ffprobe emits fields in its own order,
                   // not the order they are requested in.
                   "-of", "default=noprint_wrappers=1"])
            .arg(path)
            .output()
            .map_err(|e| format!("ffprobe: {e}"))?;
        let out = String::from_utf8_lossy(&probe.stdout);
        let field = |key: &str| -> Option<&str> {
            out.lines().find_map(|l| {
                let (name, value) = l.split_once('=')?;
                (name.trim() == key).then(|| value.trim())
            })
        };
        let w: usize = field("width").and_then(|v| v.parse().ok()).ok_or("no video stream")?;
        let h: usize = field("height").and_then(|v| v.parse().ok()).ok_or("no video stream")?;
        let fps = field("r_frame_rate").and_then(parse_rational).unwrap_or(30.0);
        // swscale ignores the identity ("RGB") colorspace NotchLC declares and
        // applies a BT.601 matrix, so asking it for rgba here would show the
        // wrong colours. Take the decoder's native planes and pack them below.
        let planar12 = field("codec_name") == Some("notchlc");
        let pix_fmt = if planar12 { "yuva444p12le" } else { "rgba" };

        let mut child = Command::new(tool("ffmpeg"))
            .arg("-i").arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-an", "-v", "quiet", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;

        let mut stdout = child.stdout.take().unwrap();
        let (tx, rx) = sync_channel::<Vec<u8>>(3); // backpressure: decode stays ~3 frames ahead
        std::thread::spawn(move || {
            let mut buf = vec![0u8; w * h * if planar12 { 8 } else { 4 }];
            while stdout.read_exact(&mut buf).is_ok() {
                let frame = if planar12 { pack_gbra12(&buf, w * h) } else { buf.clone() };
                if tx.send(frame).is_err() {
                    break; // player dropped the source
                }
            }
        });

        Ok(Source { path: path.to_path_buf(), child, rx, w, h, fps })
    }

    fn restart(&mut self) -> Result<(), String> {
        let path = self.path.clone();
        let _ = self.child.kill();
        *self = Source::open(&path)?;
        Ok(())
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// yuva444p12le -> rgba8. NotchLC's planes are GBR identity, full range:
/// G is plane 0, B is plane 1, R is plane 2.
fn pack_gbra12(planes: &[u8], px: usize) -> Vec<u8> {
    let sample = |plane: usize, i: usize| -> u8 {
        let o = (plane * px + i) * 2;
        (u16::from_le_bytes([planes[o], planes[o + 1]]) >> 4) as u8
    };
    let mut out = Vec::with_capacity(px * 4);
    for i in 0..px {
        out.push(sample(2, i));
        out.push(sample(0, i));
        out.push(sample(1, i));
        out.push(sample(3, i));
    }
    out
}

fn parse_rational(s: &str) -> Option<f32> {
    let (n, d) = s.split_once('/')?;
    let (n, d): (f32, f32) = (n.parse().ok()?, d.parse().ok()?);
    (d != 0.0).then(|| n / d)
}

struct Player {
    source: Option<Source>,
    texture: Option<egui::TextureHandle>,
    next_frame_at: Instant,
    playing: bool,
    error: Option<String>,
}

impl Player {
    fn load(&mut self, path: &Path) {
        match Source::open(path) {
            Ok(s) => {
                self.source = Some(s);
                self.texture = None;
                self.next_frame_at = Instant::now();
                self.playing = true;
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }
}

impl eframe::App for Player {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        let dropped = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(file) = dropped {
            self.load(&file);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
            self.next_frame_at = Instant::now();
        }

        if self.playing
            && let Some(src) = self.source.as_mut()
        {
            let period = Duration::from_secs_f32(1.0 / src.fps.max(1.0));
            let now = Instant::now();
            if now >= self.next_frame_at {
                match src.rx.try_recv() {
                    Ok(rgba) => {
                        let img = egui::ColorImage::from_rgba_unmultiplied([src.w, src.h], &rgba);
                        match self.texture.as_mut() {
                            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                            None => {
                                self.texture =
                                    Some(ctx.load_texture("frame", img, egui::TextureOptions::LINEAR))
                            }
                        }
                        // Skip ahead instead of drifting if we fell behind.
                        self.next_frame_at = (self.next_frame_at + period).max(now);
                    }
                    Err(TryRecvError::Disconnected) => {
                        if let Err(e) = src.restart() {
                            self.error = Some(e);
                            self.playing = false;
                        }
                    }
                    Err(TryRecvError::Empty) => {} // decoder behind; try again next repaint
                }
            }
            ctx.request_repaint_after(self.next_frame_at.saturating_duration_since(Instant::now()));
        }

        egui::Panel::bottom("bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("Video", &["mov", "mp4", "avi", "mkv"])
                        .pick_file()
                {
                    self.load(&p);
                }
                let label = if self.playing { "Pause" } else { "Play" };
                if ui.button(label).clicked() {
                    self.playing = !self.playing;
                    self.next_frame_at = Instant::now();
                }
                if let Some(src) = &self.source {
                    ui.label(format!("{}×{} @ {:.2} fps", src.w, src.h, src.fps));
                }
                if let Some(e) = &self.error {
                    ui.colored_label(egui::Color32::RED, e);
                }
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::BLACK))
            .show(ui, |ui| match &self.texture {
                Some(tex) => {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Image::from_texture(tex).shrink_to_fit());
                    });
                }
                None => {
                    ui.centered_and_justified(|ui| ui.label("Drop a NotchLC or HAP file here"));
                }
            });
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    let mut player = Player {
        source: None,
        texture: None,
        next_frame_at: Instant::now(),
        playing: false,
        error: None,
    };
    if let Some(arg) = std::env::args().nth(1) {
        player.load(Path::new(&arg));
    }

    eframe::run_native(
        "Rustjay Player",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([960.0, 600.0])
                .with_drag_and_drop(true),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(player))),
    )
}
