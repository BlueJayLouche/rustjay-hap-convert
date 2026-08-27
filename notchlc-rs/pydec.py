#!/usr/bin/env python3
"""Faithful port of FFmpeg's notchlc.c decode path, for validating the spike encoder."""
import struct, sys

def main(path, frame_idx=0):
    d = open(path, 'rb').read()
    # locate mdat
    mdat = d.find(b'mdat') - 4
    pos = mdat + 8
    # read stsz? simpler: packet size from its own header (uncompressed_size+16)
    for _ in range(frame_idx):
        usize = struct.unpack_from('<I', d, pos+4)[0]
        pos += 16 + usize
    pkt = d[pos:]
    assert pkt[:4] == b'NLC1', pkt[:4]
    usize, csize, fmt = struct.unpack_from('<III', pkt, 4)
    assert fmt <= 2
    buf = pkt[16:16+usize]
    assert fmt == 2, 'python port only handles uncompressed'

    class R:  # LE byte reader over buf
        def __init__(s, off=0): s.o = off
        def u32(s): v = struct.unpack_from('<I', buf, s.o)[0]; s.o += 4; return v
        def u16(s): v = struct.unpack_from('<H', buf, s.o)[0]; s.o += 2; return v
        def u8(s):  v = buf[s.o]; s.o += 1; return v
        def seek(s, o): s.o = o

    gb = R(0)
    tex_x, tex_y = gb.u32(), gb.u32()
    uv_off = gb.u32()*4; y_ctrl = gb.u32()*4; a_ctrl = gb.u32()*4
    uv_data = gb.u32()*4; y_size = gb.u32(); a_data = gb.u32()*4
    a_count = gb.u32()*4; data_end = gb.u32()
    row_tab = gb.o
    y_data_off = data_end - y_size
    uv_count_off = y_data_off - a_data
    W, H = tex_x, tex_y
    print(f'{W}x{H} y_data_off={y_data_off} opaque={uv_count_off == a_ctrl}')

    # Y plane
    yplane = [[0]*W for _ in range(H)]
    rgb_r = R(row_tab)
    gb.seek(y_ctrl)
    for y0 in range(0, H, 4):
        row_off = rgb_r.u32()
        dgb = R(y_data_off + row_off)
        bitpos = 0
        def getbits(n):
            nonlocal bitpos
            v = 0
            base = y_data_off + row_off
            for k in range(n):
                b = buf[base + bitpos//8]
                v |= ((b >> (bitpos % 8)) & 1) << k
                bitpos += 1
            return v
        for x0 in range(0, W, 4):
            item = gb.u32()
            y_min = item & 4095; y_max = (item >> 12) & 4095
            y_diff = y_max - y_min
            ctrl = [(item >> (24 + 2*i)) & 3 for i in range(4)]
            for i in range(4):
                nb = ctrl[i] + 1
                div = (1 << nb) - 1
                add = div - 1
                for j in range(4):
                    val = min(y_min + (y_diff*getbits(nb) + add)//div, 4095)
                    if y0+i < H and x0+j < W:
                        yplane[y0+i][x0+j] = val

    # alpha
    aplane = [[4095]*W for _ in range(H)]

    # UV
    uplane = [[0]*W for _ in range(H)]
    vplane = [[0]*W for _ in range(H)]
    rgb_r.seek(uv_off)
    for y0 in range(0, H, 16):
        for x0 in range(0, W, 16):
            off = rgb_r.u32()*4
            dgb = R(uv_data + off)
            is8x8 = dgb.u16(); escape = dgb.u16()
            assert is8x8 == 0 and escape == 0, (is8x8, escape)
            u0, v0, u1, v1 = dgb.u8(), dgb.u8(), dgb.u8(), dgb.u8()
            loc = dgb.u32()
            u0 = (u0 << 4) | (u0 & 0xF); v0 = (v0 << 4) | (v0 & 0xF)
            u1 = (u1 << 4) | (u1 & 0xF); v1 = (v1 << 4) | (v1 & 0xF)
            udif, vdif = u1-u0, v1-v0
            for i in range(0, 16, 4):
                for j in range(0, 16, 4):
                    uu = u0 + (udif*(loc & 3) + 2)//3
                    vv = v0 + (vdif*(loc & 3) + 2)//3
                    loc >>= 2
                    for ii in range(4):
                        for jj in range(4):
                            if y0+i+ii < H and x0+j+jj < W:
                                uplane[y0+i+ii][x0+j+jj] = uu
                                vplane[y0+i+ii][x0+j+jj] = vv

    # GBR identity: R=v, G=y, B=u
    out = bytearray()
    for yy in range(H):
        for xx in range(W):
            out += bytes((vplane[yy][xx] >> 4, yplane[yy][xx] >> 4, uplane[yy][xx] >> 4))
    open('pydec0.rgb', 'wb').write(bytes(out))
    print('wrote pydec0.rgb')

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
