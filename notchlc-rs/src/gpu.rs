//! GPU block decode: upload a decompressed payload, dispatch one invocation
//! per 4x4 luma block, get an RGBA8 texture back.
//!
//! The output format is `Rgba8Unorm` — deliberately not the sRGB variant, to
//! match what hap-wgpu hands its consumers, so NotchLC and HAP clips land in a
//! renderer looking the same.

use crate::decode::{Header, Payload};
use std::sync::Arc;

/// Textures in flight. The consumer samples the returned texture during the
/// same frame it asks for it, so a small ring is enough to keep a decode from
/// overwriting a texture still being read.
const RING: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    cols4: u32,
    rows4: u32,
    cols16: u32,
    y_control: u32,
    uv_table: u32,
    uv_data: u32,
    a_control: u32,
    a_data: u32,
    opaque: u32,
    _pad: u32,
}

pub struct GpuDecoder {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    payload: wgpu::Buffer,
    offsets: wgpu::Buffer,
    textures: Vec<Arc<wgpu::Texture>>,
    binds: Vec<wgpu::BindGroup>,
    next: usize,
    width: u32,
    height: u32,
    /// Dispatch size for the payload most recently prepared.
    dispatch: (u32, u32),
}

impl GpuDecoder {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("notchlc decode"),
            source: wgpu::ShaderSource::Wgsl(include_str!("decode.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("notchlc decode"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("notchlc decode"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("notchlc decode"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("notchlc params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let textures: Vec<Arc<wgpu::Texture>> = (0..RING)
            .map(|i| {
                Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("notchlc frame {i}")),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                }))
            })
            .collect();

        // Sized on the first decode, once the payload length is known.
        let payload = empty_storage(&device, "notchlc payload");
        let offsets = empty_storage(&device, "notchlc bit offsets");

        let mut decoder = Self {
            device: device.clone(),
            pipeline,
            layout,
            params,
            payload,
            offsets,
            textures,
            binds: Vec::new(),
            next: 0,
            width,
            height,
            dispatch: (0, 0),
        };
        decoder.rebuild_binds();
        decoder
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Stage one frame's data for the GPU. Buffer writes are queued, not
    /// submitted, so they land with whatever submit the caller makes next.
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        payload: &Payload,
        header: &Header,
        bit_offsets: &[u32],
    ) {
        let (cols4, rows4) = header.blocks4();
        let (cols16, _) = header.blocks16();

        self.ensure_capacity(payload.bytes.len(), bit_offsets.len());
        self.write_payload(queue, &payload.bytes);
        queue.write_buffer(&self.offsets, 0, bytemuck::cast_slice(bit_offsets));
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                width: header.width,
                height: header.height,
                cols4: cols4 as u32,
                rows4: rows4 as u32,
                cols16: cols16 as u32,
                y_control: header.y_control_data_offset as u32,
                uv_table: header.uv_offset_data_offset as u32,
                uv_data: header.uv_data_offset as u32,
                a_control: header.a_control_word_offset as u32,
                a_data: header.a_data_offset as u32,
                opaque: header.opaque as u32,
                _pad: 0,
            }),
        );
        self.dispatch = (cols4.div_ceil(8) as u32, rows4.div_ceil(8) as u32);
    }

    /// Record the decode into a caller-supplied encoder and hand back the
    /// texture it will fill.
    ///
    /// Nothing is submitted here: a host that paces its own frames — batching
    /// several passes into one submit, or gating submits on a fence — keeps
    /// full control of when this work reaches the GPU. Call [`Self::prepare`]
    /// first, and submit on the same queue the texture is later sampled from,
    /// so ordering alone makes it ready.
    pub fn record(&mut self, encoder: &mut wgpu::CommandEncoder) -> Arc<wgpu::Texture> {
        let slot = self.next;
        self.next = (self.next + 1) % RING;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("notchlc decode"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.binds[slot], &[]);
        pass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, 1);
        drop(pass);

        Arc::clone(&self.textures[slot])
    }

    /// Prepare, record and submit in one call, for hosts with no submit
    /// schedule of their own.
    pub fn decode(
        &mut self,
        queue: &wgpu::Queue,
        payload: &Payload,
        header: &Header,
        bit_offsets: &[u32],
    ) -> Arc<wgpu::Texture> {
        self.prepare(queue, payload, header, bit_offsets);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("notchlc decode"),
            });
        let texture = self.record(&mut encoder);
        queue.submit([encoder.finish()]);
        texture
    }

    /// Payload lengths are not multiples of 4, and buffer writes must be. Send
    /// the aligned body directly and pad only the last few bytes, rather than
    /// copying the whole payload to pad it.
    fn write_payload(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        let body = bytes.len() & !3;
        queue.write_buffer(&self.payload, 0, &bytes[..body]);
        if body < bytes.len() {
            let mut tail = [0u8; 4];
            tail[..bytes.len() - body].copy_from_slice(&bytes[body..]);
            queue.write_buffer(&self.payload, body as u64, &tail);
        }
    }

    fn ensure_capacity(&mut self, payload_len: usize, offsets_len: usize) {
        let want_payload = (payload_len.next_multiple_of(4)) as u64;
        let want_offsets = (offsets_len * 4) as u64;
        if self.payload.size() >= want_payload && self.offsets.size() >= want_offsets {
            return;
        }
        if self.payload.size() < want_payload {
            self.payload = storage(&self.device, "notchlc payload", want_payload);
        }
        if self.offsets.size() < want_offsets {
            self.offsets = storage(&self.device, "notchlc bit offsets", want_offsets);
        }
        self.rebuild_binds();
    }

    fn rebuild_binds(&mut self) {
        self.binds = self
            .textures
            .iter()
            .map(|texture| {
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("notchlc decode"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.payload.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.offsets.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                    ],
                })
            })
            .collect();
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn empty_storage(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    storage(device, label, 4)
}
