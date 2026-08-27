// NotchLC block decode, LGPL-2.1-or-later: the block layout and arithmetic
// here follow FFmpeg's libavcodec/notchlc.c (Copyright (c) 2020 Paul B Mahol),
// restructured for parallel execution. See COPYING.LESSER.
//
// One invocation per 4x4 luma block: it decodes its own
// 16 luma samples, works out the chroma and alpha covering those pixels, and
// writes 16 RGBA texels.
//
// Parallel decode is possible because every block's position in the bitstream
// is known up front: chroma and alpha blocks carry explicit offsets, and the
// luma residuals' bit positions arrive in `bit_offsets`, prefix-summed on the
// CPU. Planes are GBR identity, so R comes from V, G from Y, B from U.

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

@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> bit_offsets: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba8unorm, write>;

fn load_u8(byte: u32) -> u32 {
    return (payload[byte >> 2u] >> ((byte & 3u) * 8u)) & 0xffu;
}

fn load_u16(byte: u32) -> u32 {
    return load_u8(byte) | (load_u8(byte + 1u) << 8u);
}

// Unaligned by necessity: luma row offsets are plain byte counts.
fn load_u32(byte: u32) -> u32 {
    if ((byte & 3u) == 0u) {
        return payload[byte >> 2u];
    }
    return load_u8(byte)
        | (load_u8(byte + 1u) << 8u)
        | (load_u8(byte + 2u) << 16u)
        | (load_u8(byte + 3u) << 24u);
}

/// Up to 4 bits at an arbitrary bit position; spans at most two bytes.
fn load_bits(bitpos: u32, n: u32) -> u32 {
    let byte = bitpos >> 3u;
    let pair = load_u8(byte) | (load_u8(byte + 1u) << 8u);
    return (pair >> (bitpos & 7u)) & ((1u << n) - 1u);
}

/// Endpoints are stored 8-bit and expanded by nibble replication.
fn expand(b: u32) -> i32 {
    return i32((b << 4u) | (b & 0xfu));
}

/// The reference decoder's weighted step between two endpoints.
fn lerp3(e0: i32, e1: i32, w: u32) -> i32 {
    return e0 + ((e1 - e0) * i32(w) + 2) / 3;
}

/// Chroma for one pixel, given the 16x16 block it falls in.
fn chroma_at(block_byte: u32, px: u32, py: u32) -> vec2<i32> {
    let is8x8 = load_u16(block_byte);
    let escape = load_u16(block_byte + 2u);
    let lx = px & 15u;
    let ly = py & 15u;

    if (escape == 0u && is8x8 == 0u) {
        // One endpoint pair for the whole 16x16, a 2-bit weight per 4x4.
        let u0 = expand(load_u8(block_byte + 4u));
        let v0 = expand(load_u8(block_byte + 5u));
        let u1 = expand(load_u8(block_byte + 6u));
        let v1 = expand(load_u8(block_byte + 7u));
        let loc = load_u32(block_byte + 8u);
        let w = (loc >> (2u * ((ly / 4u) * 4u + (lx / 4u)))) & 3u;
        return vec2<i32>(lerp3(u0, u1, w), lerp3(v0, v1, w));
    }

    // Otherwise the block is four 8x8 quadrants, each either an 8x8 with 2x2
    // weights or, under escape, four 4x4s with per-pixel weights. Quadrants are
    // variable length, so sum the ones before this pixel's.
    let qi = ly / 8u;
    let qj = lx / 8u;
    let q = qi * 2u + qj;
    var byte = block_byte + 4u;
    for (var k = 0u; k < q; k = k + 1u) {
        if (((is8x8 >> k) & 1u) != 0u) {
            byte = byte + 8u;
        } else if (escape != 0u) {
            byte = byte + 32u;
        }
    }

    if (((is8x8 >> q) & 1u) != 0u) {
        let u0 = expand(load_u8(byte));
        let v0 = expand(load_u8(byte + 1u));
        let u1 = expand(load_u8(byte + 2u));
        let v1 = expand(load_u8(byte + 3u));
        let loc = load_u32(byte + 4u);
        let w = (loc >> (2u * (((ly & 7u) / 2u) * 4u + ((lx & 7u) / 2u)))) & 3u;
        return vec2<i32>(lerp3(u0, u1, w), lerp3(v0, v1, w));
    }
    if (escape != 0u) {
        // Four 4x4 sub-blocks of 8 bytes each, in row-major order.
        let sub = ((ly & 7u) / 4u) * 2u + ((lx & 7u) / 4u);
        let sb = byte + sub * 8u;
        let u0 = expand(load_u8(sb));
        let v0 = expand(load_u8(sb + 1u));
        let u1 = expand(load_u8(sb + 2u));
        let v1 = expand(load_u8(sb + 3u));
        let loc = load_u32(sb + 4u);
        let w = (loc >> (2u * ((ly & 3u) * 4u + (lx & 3u)))) & 3u;
        return vec2<i32>(lerp3(u0, u1, w), lerp3(v0, v1, w));
    }
    return vec2<i32>(0, 0);
}

/// Alpha for one pixel. The 4x4 sub-block weights sit in a 64-bit control
/// word, so the extraction straddles two u32s.
fn alpha_at(bx16: u32, by16: u32, px: u32, py: u32) -> u32 {
    if (params.opaque != 0u) {
        return 4095u;
    }
    let entry = params.a_control + (by16 * params.cols16 + bx16) * 8u;
    let m = load_u32(entry);
    let offset = load_u32(entry + 4u) * 4u + params.uv_data + params.a_data;

    let lo = load_u32(offset);
    let hi = load_u32(offset + 4u);
    let alpha0 = i32(lo & 0xffu);
    let alpha1 = i32((lo >> 8u) & 0xffu);

    let sub = ((py & 15u) / 4u) * 4u + ((px & 15u) / 4u);
    let mode = (m >> (2u * sub)) & 3u;
    if (mode == 0u) {
        return 0u;
    }
    if (mode == 1u) {
        return 4095u;
    }
    // Weights start at bit 16 of the 64-bit word, three bits per sub-block.
    let p = 16u + 3u * sub;
    var w: u32;
    if (p >= 32u) {
        w = (hi >> (p - 32u)) & 7u;
    } else if (p > 29u) {
        w = ((lo >> p) | (hi << (32u - p))) & 7u;
    } else {
        w = (lo >> p) & 7u;
    }
    // Reproduces the reference's wrap into a 16-bit plane.
    return u32((alpha0 + (alpha1 - alpha0) * i32(w)) << 4) & 0xffffu;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= params.cols4 || by >= params.rows4) {
        return;
    }

    let block = by * params.cols4 + bx;
    let item = load_u32(params.y_control + block * 4u);
    let y_min = item & 4095u;
    let y_max = (item >> 12u) & 4095u;
    let y_diff = y_max - y_min;
    var bitpos = bit_offsets[block];

    let x0 = bx * 4u;
    let y0 = by * 4u;
    let uv_entry = load_u32(params.uv_table + ((y0 / 16u) * params.cols16 + (x0 / 16u)) * 4u);
    let uv_block = uv_entry * 4u + params.uv_data;

    for (var i = 0u; i < 4u; i = i + 1u) {
        let nb = ((item >> (24u + 2u * i)) & 3u) + 1u;
        let div = (1u << nb) - 1u;
        let add = div - 1u;
        let py = y0 + i;

        for (var j = 0u; j < 4u; j = j + 1u) {
            let idx = load_bits(bitpos, nb);
            bitpos = bitpos + nb;
            let luma = min(y_min + (y_diff * idx + add) / div, 4095u);
            let px = x0 + j;
            if (px >= params.width || py >= params.height) {
                continue;
            }

            let uv = chroma_at(uv_block, px, py);
            let a = alpha_at(x0 / 16u, y0 / 16u, px, py);
            textureStore(
                out_tex,
                vec2<i32>(i32(px), i32(py)),
                vec4<f32>(
                    f32((u32(uv.y) >> 4u) & 0xffu) / 255.0,
                    f32((luma >> 4u) & 0xffu) / 255.0,
                    f32((u32(uv.x) >> 4u) & 0xffu) / 255.0,
                    f32((a >> 4u) & 0xffu) / 255.0,
                ),
            );
        }
    }
}
