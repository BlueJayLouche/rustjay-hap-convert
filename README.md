# rustjay-hap-convert

Standalone batch video-to-HAP converter with a drag-and-drop GUI. Built with [egui](https://github.com/emilk/egui) and GPU-accelerated DXT compression via [hap-rs](https://github.com/BlueJayLouche/hap-rs).

## Features

- **All HAP codecs** — HAP1 (DXT1), HAP Alpha (DXT5), HAP Q (YCoCg), HAP Q (BC7), HAP HDR (BC6H), HAP Alpha-Only (BC4)
- **Drag and drop** — drop video files directly onto the window
- **Batch export** — queue multiple files, converts sequentially with per-file progress
- **GPU acceleration** — auto-detects GPU with BC compression support, falls back to CPU
- **Per-file codec override** — set a global default or change individual files before converting
- **Custom output directory** — or output alongside the input file

## Requirements

- [FFmpeg](https://ffmpeg.org/) on your PATH (used to decode input videos)
- GPU with BC texture compression support (optional, for accelerated encoding)

## Build

```bash
git clone https://github.com/BlueJayLouche/rustjay-hap-convert.git
cd rustjay-hap-convert
cargo run --release
```

## Supported Input Formats

Any format FFmpeg can decode: MP4, MOV, AVI, MKV, WebM, MXF, and more.

## Output

QuickTime `.mov` files with HAP video codec — compatible with any HAP-enabled player (VDMX, Resolume, MadMapper, etc.).

## License

MIT OR Apache-2.0
