//! Measure payload supply, CPU decode and (with --features gpu) GPU decode on
//! a real clip. The yardstick for "is the shader worth it".
use notchlc_rs::Clip;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let clip = Clip::open(std::path::Path::new(&path)).unwrap();
    let (w, h) = clip.resolution();
    println!("{w}x{h}, {} frames @ {:.0}fps", clip.frame_count(), clip.fps());

    // cold: nothing cached, so this pays the decompress inline
    let t = Instant::now();
    let p = clip.payload(0).unwrap();
    println!("cold payload(0):     {:>7.3} ms  ({} KB)", t.elapsed().as_secs_f64() * 1e3, p.bytes.len() / 1024);

    std::thread::sleep(std::time::Duration::from_millis(200)); // let the worker run
    let t = Instant::now();
    let _ = clip.payload(1).unwrap();
    println!("warm payload(1):     {:>7.3} ms", t.elapsed().as_secs_f64() * 1e3);

    // sequential playback: how fast can we pull payloads with the worker ahead
    let t = Instant::now();
    for n in 0..clip.frame_count() {
        let _ = clip.payload(n).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / clip.frame_count() as f64;
    println!("sequential payloads: {:>7.3} ms/frame ({:.0} fps, {:.0} clips at 30fps)",
             per * 1e3, 1.0 / per, 1.0 / (per * 30.0));

    // full CPU decode, for comparison with the GPU path later
    let t = Instant::now();
    for n in 0..clip.frame_count().min(30) {
        let _ = clip.decode(n).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / 30.0;
    println!("cpu decode:          {:>7.3} ms/frame ({:.1} clips at 30fps)", per * 1e3, 1.0 / (per * 30.0));

    gpu_bench(&clip);
}

#[cfg(feature = "gpu")]
fn gpu_bench(clip: &Clip) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        println!("gpu decode:          no adapter");
        return;
    };
    let Ok((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
    else {
        println!("gpu decode:          no device");
        return;
    };
    let (w, h) = clip.resolution();
    let mut dec = notchlc_rs::GpuDecoder::new(&device, w, h);

    let frames = clip.frame_count().min(200);
    // Warm up: first dispatch pays pipeline and buffer creation.
    for n in 0..frames.min(5) {
        let p = clip.payload(n).unwrap();
        let hdr = notchlc_rs::parse_header(&p).unwrap();
        let off = notchlc_rs::bit_offsets(&p, &hdr).unwrap();
        let _ = dec.decode(&queue, &p, &hdr, &off);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // How much of the GPU path is CPU-side prep that belongs on the worker?
    let t = Instant::now();
    for n in 0..frames {
        let p = clip.payload(n).unwrap();
        let hdr = notchlc_rs::parse_header(&p).unwrap();
        let _ = notchlc_rs::bit_offsets(&p, &hdr).unwrap();
    }
    let prep = t.elapsed().as_secs_f64() / frames as f64;

    let t = Instant::now();
    for n in 0..frames {
        let p = clip.payload(n).unwrap();
        let hdr = notchlc_rs::parse_header(&p).unwrap();
        let off = notchlc_rs::bit_offsets(&p, &hdr).unwrap();
        let _ = dec.decode(&queue, &p, &hdr, &off);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let per = t.elapsed().as_secs_f64() / frames as f64;
    println!("  of which prep:     {:>7.3} ms/frame (header + bit offsets, CPU)", prep * 1e3);
    println!("gpu decode:          {:>7.3} ms/frame ({:.0} clips at 30fps)", per * 1e3, 1.0 / (per * 30.0));
    println!("  upload + dispatch: {:>7.3} ms/frame", (per - prep) * 1e3);
}

#[cfg(not(feature = "gpu"))]
fn gpu_bench(_: &Clip) {
    println!("gpu decode:          (build with --features gpu)");
}
