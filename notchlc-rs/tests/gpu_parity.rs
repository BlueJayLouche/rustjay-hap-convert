//! The compute shader must agree with the CPU decoder exactly — the CPU path
//! is itself checked against ffmpeg, so this transitively pins the shader to
//! the reference decoder.

#![cfg(all(feature = "gpu", feature = "clip"))]
use notchlc_rs::{bit_offsets, decode_payload, parse_header, Clip, GpuDecoder};
use std::path::Path;
use std::sync::Arc;

fn gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("notchlc test"),
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Pull an RGBA8 texture back to the CPU.
fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    // copy_texture_to_buffer needs rows padded to 256 bytes.
    let row = (width * 4).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mapped = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let start = (y * row) as usize;
        out.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

#[test]
fn shader_matches_cpu_decoder() {
    let path = std::env::var("NLC_TEST_MOV")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("test.mov"));
    if !path.exists() {
        eprintln!("skipping: test.mov missing — run `cargo run` to generate it");
        return;
    }
    let Some((device, queue)) = gpu() else {
        eprintln!("skipping: no usable wgpu adapter");
        return;
    };

    let clip = Clip::open(&path).expect("open failed");
    let (width, height) = clip.resolution();
    let mut decoder = GpuDecoder::new(&device, width, height);

    for frame in 0..clip.frame_count() {
        let payload = clip.payload(frame).unwrap();
        let header = parse_header(&payload).unwrap();
        let offsets = bit_offsets(&payload, &header).unwrap();

        let want = decode_payload(&payload).unwrap().to_rgba8();
        let texture = decoder.decode(&queue, &payload, &header, &offsets);
        let got = readback(&device, &queue, &texture, width, height);

        assert_eq!(got.len(), want.len(), "frame {frame} size");
        if got != want {
            let bad = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap();
            let px = (bad / 4) % width as usize;
            let py = (bad / 4) / width as usize;
            panic!(
                "frame {frame} differs at ({px},{py}) channel {}: gpu {} cpu {}",
                bad % 4,
                got[bad],
                want[bad]
            );
        }
    }

    eprintln!(
        "{} frames: shader matches the CPU decoder exactly",
        clip.frame_count()
    );
}
