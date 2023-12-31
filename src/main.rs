use std::fs::File;
use std::io::{BufReader, Cursor};
use std::iter::repeat;
use std::thread;
use std::time::{Duration, Instant};

use bitstream_io::{BigEndian, BitRead, BitReader};
use matroska_demuxer::{Frame, TrackType};

use nv_video_codec_sdk::NvDecoder;
use video_common::VideoCodec;

fn main() {
    let mut matroska = matroska_demuxer::MatroskaFile::open(BufReader::new(
        File::open("data/chainsaw_man_s01e01_v.mkv").unwrap(),
    ))
    .unwrap();
    for (i, track) in matroska.tracks().iter().enumerate() {
        println!(
            "{i} {:?}({}): {}",
            track.track_type(),
            track.codec_id(),
            track.language().unwrap_or("en")
        );
    }

    let extradata = read_avc_config_record(matroska.tracks()[0].codec_private().unwrap());

    nv_video_codec_sdk::init();

    let (mut decoder, frames) = NvDecoder::new(VideoCodec::H264);
    decoder.feed_packet(&extradata, 0);
    let mut frame = Frame::default();
    let mut num_packets = 0;
    let mut packet = vec![0, 0, 0, 1];
    let mut num_frames = 0;
    let mut last_frame = Instant::now();
    while let Ok(end) = matroska.next_frame(&mut frame) {
        let track = &matroska.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            // Keep only NALU header
            packet.truncate(4);
            // Ignore first 4 bytes (slice len)
            packet.extend_from_slice(&frame.data[4..]);
            decoder.feed_packet(&packet, frame.timestamp as i64);
            num_packets += 1;
            if num_packets > 1000 {
                break;
            }
        }
        if let Ok(frame) = frames.try_recv() {
            // let fps = 1000000 / last_frame.elapsed().as_micros();
            // println!("{} fps", fps);
            last_frame = Instant::now();
            num_frames += 1;
        }
    }
    nv_video_codec_sdk::sync();

    thread::sleep(Duration::from_secs_f32(2.0));
}

fn read_avc_config_record(codec_private: &[u8]) -> Vec<u8> {
    let mut reader = BitReader::endian(Cursor::new(codec_private), BigEndian);
    let version: u8 = reader.read(8).unwrap();
    let profile: u8 = reader.read(8).unwrap();
    let profile_compat: u8 = reader.read(8).unwrap();
    let level: u8 = reader.read(8).unwrap();
    reader.read::<u8>(6).unwrap(); // Reserved
    let nal_size: u8 = reader.read::<u8>(2).unwrap() + 1;
    reader.read::<u8>(3).unwrap(); // Reserved
    let num_sps: u8 = reader.read(5).unwrap();

    let mut nalus = Vec::new();
    for _ in 0..num_sps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }
    let num_pps: u8 = reader.read(8).unwrap();
    for _ in 0..num_pps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }

    nalus
}
