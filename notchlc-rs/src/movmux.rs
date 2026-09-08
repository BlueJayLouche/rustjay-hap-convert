//! Minimal QuickTime mov muxer for NotchLC samples (fourcc `nclc`).

use std::io::Write;

/// Writes a .mov with a single video track, one sample per frame.
pub struct MovWriter {
    width: u32,
    height: u32,
    fps: u32,
    samples: Vec<Vec<u8>>,
}

impl MovWriter {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self { width, height, fps, samples: Vec::new() }
    }

    pub fn add_sample(&mut self, packet: Vec<u8>) {
        self.samples.push(packet);
    }

    /// Serialize the whole movie. Samples are held in memory, so this is for
    /// the spike only.
    pub fn finish(&self) -> Vec<u8> {
        let ftyp = make_ftyp();
        let mdat_payload_size: usize = self.samples.iter().map(|s| s.len()).sum();
        let mdat_data_off = (ftyp.len() + 8) as u32; // single chunk offset

        let mut mdat = Vec::with_capacity(8 + mdat_payload_size);
        put_box_header(&mut mdat, 8 + mdat_payload_size as u32, b"mdat");
        for s in &self.samples {
            mdat.extend_from_slice(s);
        }

        let moov = self.make_moov(mdat_data_off);

        let mut out = Vec::new();
        out.extend_from_slice(&ftyp);
        out.extend_from_slice(&mdat);
        out.extend_from_slice(&moov);
        out
    }

    /// Write straight to a file.
    pub fn write_to<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        w.write_all(&self.finish())
    }

    fn make_moov(&self, chunk_offset: u32) -> Vec<u8> {
        let n = self.samples.len() as u32;
        let frame_dur_ts = 1u32; // in track timescale units
        let track_timescale = self.fps;
        let media_duration = n * frame_dur_ts;
        let movie_duration = n * 1000 / self.fps; // in movie timescale (1000)

        // stsd with one nclc video sample entry.
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0; 6]); // reserved
        entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        entry.extend_from_slice(&[0; 16]); // pre_defined + reserved
        entry.extend_from_slice(&(self.width as u16).to_be_bytes());
        entry.extend_from_slice(&(self.height as u16).to_be_bytes());
        entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horiz 72dpi
        entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vert 72dpi
        entry.extend_from_slice(&0u32.to_be_bytes()); // reserved
        entry.extend_from_slice(&1u16.to_be_bytes()); // frame_count
        let mut name = [0u8; 32];
        name[1..8].copy_from_slice(b"NotchLC");
        name[0] = 7;
        entry.extend_from_slice(&name); // compressorname (pascal)
        entry.extend_from_slice(&24u16.to_be_bytes()); // depth
        entry.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined
        let mut entry_box = Vec::new();
        put_box_header(&mut entry_box, 8 + entry.len() as u32, b"nclc");
        entry_box.extend_from_slice(&entry);

        let mut stsd = full_box_header(0, 0);
        stsd.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stsd.extend_from_slice(&entry_box);
        let stsd = wrap_box(b"stsd", &stsd);

        let mut stts = full_box_header(0, 0);
        stts.extend_from_slice(&1u32.to_be_bytes());
        stts.extend_from_slice(&n.to_be_bytes());
        stts.extend_from_slice(&frame_dur_ts.to_be_bytes());
        let stts = wrap_box(b"stts", &stts);

        let mut stsc = full_box_header(0, 0);
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        stsc.extend_from_slice(&n.to_be_bytes()); // samples_per_chunk
        stsc.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        let stsc = wrap_box(b"stsc", &stsc);

        let mut stsz = full_box_header(0, 0);
        stsz.extend_from_slice(&0u32.to_be_bytes()); // sample_size (variable)
        stsz.extend_from_slice(&n.to_be_bytes());
        for s in &self.samples {
            stsz.extend_from_slice(&(s.len() as u32).to_be_bytes());
        }
        let stsz = wrap_box(b"stsz", &stsz);

        let mut stco = full_box_header(0, 0);
        stco.extend_from_slice(&1u32.to_be_bytes());
        stco.extend_from_slice(&chunk_offset.to_be_bytes());
        let stco = wrap_box(b"stco", &stco);

        // Sync samples: every frame is a keyframe (intra codec).
        let mut stss = full_box_header(0, 0);
        stss.extend_from_slice(&n.to_be_bytes());
        for i in 1..=n {
            stss.extend_from_slice(&i.to_be_bytes());
        }
        let stss = wrap_box(b"stss", &stss);

        let mut stbl_content = Vec::new();
        for b in [&stsd, &stts, &stsc, &stsz, &stss, &stco] {
            stbl_content.extend_from_slice(b);
        }
        let stbl = wrap_box(b"stbl", &stbl_content);

        let mut dref = full_box_header(0, 0);
        dref.extend_from_slice(&1u32.to_be_bytes());
        let mut url = full_box_header(0, 1); // self-contained
        let url_box = wrap_box(b"url ", &{
            let mut v = Vec::new();
            v.append(&mut url);
            v
        });
        dref.extend_from_slice(&url_box);
        let dref = wrap_box(b"dref", &dref);
        let dinf = wrap_box(b"dinf", &dref);

        let mut vmhd = full_box_header(0, 1);
        vmhd.extend_from_slice(&0u16.to_be_bytes()); // graphicsmode
        vmhd.extend_from_slice(&[0; 6]); // opcolor
        let vmhd = wrap_box(b"vmhd", &vmhd);

        let mut minf_content = Vec::new();
        minf_content.extend_from_slice(&vmhd);
        minf_content.extend_from_slice(&dinf);
        minf_content.extend_from_slice(&stbl);
        let minf = wrap_box(b"minf", &minf_content);

        let mut mdhd = full_box_header(0, 0);
        mdhd.extend_from_slice(&0u32.to_be_bytes()); // creation
        mdhd.extend_from_slice(&0u32.to_be_bytes()); // modification
        mdhd.extend_from_slice(&track_timescale.to_be_bytes());
        mdhd.extend_from_slice(&media_duration.to_be_bytes());
        mdhd.extend_from_slice(&0x55C4u16.to_be_bytes()); // language 'und'
        mdhd.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        let mdhd = wrap_box(b"mdhd", &mdhd);

        let mut hdlr = full_box_header(0, 0);
        hdlr.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        hdlr.extend_from_slice(b"vide");
        hdlr.extend_from_slice(&[0; 12]); // reserved
        hdlr.extend_from_slice(b"VideoHandler\0");
        let hdlr = wrap_box(b"hdlr", &hdlr);

        let mut mdia_content = Vec::new();
        mdia_content.extend_from_slice(&mdhd);
        mdia_content.extend_from_slice(&hdlr);
        mdia_content.extend_from_slice(&minf);
        let mdia = wrap_box(b"mdia", &mdia_content);

        let mut tkhd = full_box_header(0, 3); // enabled | in_movie
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // creation
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // modification
        tkhd.extend_from_slice(&1u32.to_be_bytes()); // track_id
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // reserved
        tkhd.extend_from_slice(&movie_duration.to_be_bytes());
        tkhd.extend_from_slice(&[0; 8]); // reserved
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // layer
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // volume
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // reserved
        tkhd.extend_from_slice(&MATRIX);
        tkhd.extend_from_slice(&(self.width << 16).to_be_bytes());
        tkhd.extend_from_slice(&(self.height << 16).to_be_bytes());
        let tkhd = wrap_box(b"tkhd", &tkhd);

        let mut trak_content = Vec::new();
        trak_content.extend_from_slice(&tkhd);
        trak_content.extend_from_slice(&mdia);
        let trak = wrap_box(b"trak", &trak_content);

        let mut mvhd = full_box_header(0, 0);
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // creation
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // modification
        mvhd.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        mvhd.extend_from_slice(&movie_duration.to_be_bytes());
        mvhd.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate
        mvhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume
        mvhd.extend_from_slice(&[0; 10]); // reserved
        mvhd.extend_from_slice(&MATRIX);
        mvhd.extend_from_slice(&[0; 24]); // pre_defined
        mvhd.extend_from_slice(&2u32.to_be_bytes()); // next_track_id
        let mvhd = wrap_box(b"mvhd", &mvhd);

        let mut moov_content = Vec::new();
        moov_content.extend_from_slice(&mvhd);
        moov_content.extend_from_slice(&trak);
        wrap_box(b"moov", &moov_content)
    }
}

const MATRIX: [u8; 36] = [
    0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0x40, 0x00, 0x00, 0x00,
];

fn make_ftyp() -> Vec<u8> {
    let mut v = Vec::new();
    put_box_header(&mut v, 20, b"ftyp");
    v.extend_from_slice(b"qt  ");
    v.extend_from_slice(&0x0000_0200u32.to_be_bytes()); // minor version
    v.extend_from_slice(b"qt  ");
    v
}

fn full_box_header(version: u8, flags: u32) -> Vec<u8> {
    let mut v = vec![version];
    v.extend_from_slice(&(flags & 0x00FF_FFFF).to_be_bytes()[1..]);
    v
}

fn put_box_header(out: &mut Vec<u8>, size: u32, tag: &[u8; 4]) {
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(tag);
}

fn wrap_box(tag: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + content.len());
    put_box_header(&mut v, 8 + content.len() as u32, tag);
    v.extend_from_slice(content);
    v
}
