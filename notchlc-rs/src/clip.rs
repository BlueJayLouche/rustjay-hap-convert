//! A NotchLC movie opened for playback: demux plus decompression, with a
//! worker running ahead of the playhead so the render thread never pays for
//! LZ4.
//!
//! The file is mmap'd rather than read, so a clip costs no resident memory of
//! its own — the page cache keeps hot ones around. Only the decompressed
//! payloads live in the cache, a handful of frames at a time.

use crate::decode::{decode_payload, decompress_packet, Error as DecodeError, Frame, Payload};
use hap_qt::{QtError, QtReader};
use memmap2::Mmap;
use std::collections::VecDeque;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::{self, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

/// How far ahead of the playhead the worker decompresses.
const READAHEAD: u32 = 3;
/// Payloads kept decompressed. Enough for the read-ahead plus the frame in
/// flight, with room for a little jitter.
const CACHE_FRAMES: usize = 8;

#[derive(Debug)]
pub enum ClipError {
    Io(std::io::Error),
    Container(QtError),
    Decode(DecodeError),
    /// The sample table points outside the file.
    BadSampleRange,
    FrameOutOfRange(u32),
}

impl fmt::Display for ClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClipError::Io(e) => write!(f, "{e}"),
            ClipError::Container(e) => write!(f, "{e}"),
            ClipError::Decode(e) => write!(f, "{e}"),
            ClipError::BadSampleRange => write!(f, "sample range outside the file"),
            ClipError::FrameOutOfRange(n) => write!(f, "frame {n} out of range"),
        }
    }
}

impl std::error::Error for ClipError {}

impl From<std::io::Error> for ClipError {
    fn from(e: std::io::Error) -> Self {
        ClipError::Io(e)
    }
}
impl From<QtError> for ClipError {
    fn from(e: QtError) -> Self {
        ClipError::Container(e)
    }
}
impl From<DecodeError> for ClipError {
    fn from(e: DecodeError) -> Self {
        ClipError::Decode(e)
    }
}

/// Recently decompressed payloads, newest last.
#[derive(Default)]
struct Cache {
    entries: VecDeque<(u32, Arc<Payload>)>,
}

impl Cache {
    fn get(&self, frame: u32) -> Option<Arc<Payload>> {
        self.entries
            .iter()
            .find(|(n, _)| *n == frame)
            .map(|(_, p)| Arc::clone(p))
    }

    fn insert(&mut self, frame: u32, payload: Arc<Payload>) {
        if self.entries.iter().any(|(n, _)| *n == frame) {
            return;
        }
        self.entries.push_back((frame, payload));
        while self.entries.len() > CACHE_FRAMES {
            self.entries.pop_front();
        }
    }
}

pub struct Clip {
    mmap: Arc<Mmap>,
    /// Byte offset and size of every sample, resolved once at open so the
    /// worker doesn't need the reader's file handle.
    ranges: Arc<Vec<(u64, u32)>>,
    width: u32,
    height: u32,
    fps: f32,
    frame_count: u32,
    cache: Arc<Mutex<Cache>>,
    playhead: Sender<u32>,
}

impl Clip {
    pub fn open(path: &Path) -> Result<Self, ClipError> {
        let reader = QtReader::open_codec(path, &["nclc"])?;
        let (width, height) = reader.resolution();
        let frame_count = reader.frame_count();
        let fps = reader.fps();
        let ranges: Vec<(u64, u32)> = (0..frame_count)
            .map(|i| reader.sample_range(i))
            .collect::<Result<_, _>>()?;
        let ranges = Arc::new(ranges);

        // SAFETY: the usual mmap caveat — the decoder will see garbage if the
        // file is modified underneath us. Playback sources are not written to.
        let mmap = Arc::new(unsafe { Mmap::map(&File::open(path)?)? });

        let cache = Arc::new(Mutex::new(Cache::default()));
        let (playhead, rx) = mpsc::channel::<u32>();

        std::thread::spawn({
            let mmap = Arc::clone(&mmap);
            let ranges = Arc::clone(&ranges);
            let cache = Arc::clone(&cache);
            move || {
                while let Ok(mut at) = rx.recv() {
                    // Only the newest position matters; anything older is a
                    // playhead we have already passed.
                    loop {
                        match rx.try_recv() {
                            Ok(newer) => at = newer,
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    for frame in at + 1..=(at + READAHEAD).min(frame_count.saturating_sub(1)) {
                        if cache.lock().unwrap().get(frame).is_some() {
                            continue;
                        }
                        if let Ok(payload) = decompress_at(&mmap, &ranges, frame) {
                            cache.lock().unwrap().insert(frame, Arc::new(payload));
                        }
                    }
                }
            }
        });

        Ok(Self {
            mmap,
            ranges,
            width,
            height,
            fps,
            frame_count,
            cache,
            playhead,
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// The decompressed payload for `frame`, and a nudge to the worker to run
    /// on from here. A hit costs a lock; a miss — a seek, normally — pays the
    /// decompression on the calling thread.
    pub fn payload(&self, frame: u32) -> Result<Arc<Payload>, ClipError> {
        if frame >= self.frame_count {
            return Err(ClipError::FrameOutOfRange(frame));
        }
        let _ = self.playhead.send(frame);

        if let Some(payload) = self.cache.lock().unwrap().get(frame) {
            return Ok(payload);
        }
        let payload = Arc::new(decompress_at(&self.mmap, &self.ranges, frame)?);
        // ponytail: a fresh allocation per miss. Recycle buffers only if the
        // allocator shows up in a profile — the cache already bounds them.
        self.cache.lock().unwrap().insert(frame, Arc::clone(&payload));
        Ok(payload)
    }

    /// Decode a frame on the CPU. The GPU path uploads [`Payload::bytes`]
    /// instead; this is the reference, and what non-GPU callers use.
    pub fn decode(&self, frame: u32) -> Result<Frame, ClipError> {
        let payload = self.payload(frame)?;
        Ok(decode_payload(&payload)?)
    }
}

fn decompress_at(
    mmap: &Mmap,
    ranges: &[(u64, u32)],
    frame: u32,
) -> Result<Payload, ClipError> {
    let (offset, size) = ranges[frame as usize];
    let start = offset as usize;
    let packet = mmap
        .get(start..start + size as usize)
        .ok_or(ClipError::BadSampleRange)?;
    Ok(decompress_packet(packet)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mov() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("test.mov")
    }

    /// The mmap'd sample ranges must address exactly what a plain file read
    /// would — an off-by-one in the sample table would decode garbage here.
    #[test]
    fn frames_match_a_plain_file_read() {
        let path = test_mov();
        if !path.exists() {
            eprintln!("skipping: test.mov missing — run `cargo run` to generate it");
            return;
        }
        let clip = Clip::open(&path).expect("open failed");
        assert_eq!(clip.resolution(), (256, 256));
        assert_eq!(clip.frame_count(), 30);

        let whole = std::fs::read(&path).unwrap();
        for frame in 0..clip.frame_count() {
            let (offset, size) = clip.ranges[frame as usize];
            let packet = &whole[offset as usize..offset as usize + size as usize];
            let want = crate::decode_packet(packet).unwrap();
            let got = clip.decode(frame).unwrap();
            assert_eq!(got.y, want.y, "frame {frame} luma");
            assert_eq!(got.u, want.u, "frame {frame} u");
            assert_eq!(got.v, want.v, "frame {frame} v");
        }
    }

    #[test]
    fn worker_reads_ahead_of_the_playhead() {
        let path = test_mov();
        if !path.exists() {
            return;
        }
        let clip = Clip::open(&path).expect("open failed");
        clip.payload(0).unwrap();

        // The worker is asynchronous, so poll rather than assume a deadline.
        for _ in 0..100 {
            let ready = (1..=READAHEAD)
                .all(|n| clip.cache.lock().unwrap().get(n).is_some());
            if ready {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("worker never cached frames 1..={READAHEAD} after requesting frame 0");
    }
}
