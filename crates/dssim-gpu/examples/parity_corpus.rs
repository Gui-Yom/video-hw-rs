//! Corpus parity demo: score every JPEG in `dssim-cuda/test_data/`
//! against `source.png` using both the published `dssim-core` v3.4
//! crate and `dssim-gpu`, and print a side-by-side table.
//!
//! Run:
//! ```bash
//! CUDA_PATH=/usr/local/cuda cargo run --release -p dssim-gpu --example parity_corpus
//! cargo run --release -p dssim-gpu --example parity_corpus --no-default-features --features wgpu
//! ```
//!
//! The integration tests in `tests/parity_lock.rs` lock the q70/q90
//! cases; this example surfaces the full quality range so a reader can
//! eyeball low-q behavior at a glance.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use dssim_core::{Dssim as DssimCpu, ToRGBAPLU};
use dssim_gpu::Dssim;
use imgref::ImgVec;
use rgb::RGB;

fn cpu_dssim(ref_data: &[u8], dis_data: &[u8], w: usize, h: usize) -> f64 {
    let dssim = DssimCpu::new();
    let to_rgb = |buf: &[u8]| -> Vec<RGB<u8>> {
        buf.chunks_exact(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect()
    };
    let ref_rgb = to_rgb(ref_data).to_rgblu();
    let dis_rgb = to_rgb(dis_data).to_rgblu();
    let ref_img = ImgVec::new(ref_rgb, w, h);
    let dis_img = ImgVec::new(dis_rgb, w, h);
    let ref_prep = dssim.create_image(&ref_img).unwrap();
    let dis_prep = dssim.create_image(&dis_img).unwrap();
    let (score, _) = dssim.compare(&ref_prep, dis_prep);
    score.into()
}

fn main() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dssim-cuda/test_data");
    let src_path = dir.join("source.png");
    if !src_path.exists() {
        eprintln!("corpus missing: {}", src_path.display());
        std::process::exit(1);
    }

    let img = image::open(&src_path).expect("decode source.png").to_rgb8();
    let (w, h) = img.dimensions();
    let ref_data = img.into_raw();

    let client = Backend::client(&Default::default());
    let mut d = Dssim::<Backend>::new(client, w, h).expect("Dssim::new");

    println!(
        "corpus parity: dssim-core (CPU) vs dssim-gpu ({} backend) on {}×{} reference",
        std::any::type_name::<Backend>().rsplit("::").next().unwrap(),
        w,
        h
    );
    println!(
        "{:<8} {:>14} {:>14} {:>10}",
        "case", "cpu", "gpu", "rel %"
    );

    let mut quals = vec!["q1.jpg", "q5.jpg", "q20.jpg", "q45.jpg", "q70.jpg", "q90.jpg"];
    quals.retain(|q| dir.join(q).exists());

    for q in quals {
        let dis = image::open(dir.join(q)).expect("decode jpeg").to_rgb8();
        assert_eq!(dis.dimensions(), (w, h));
        let dis_data = dis.into_raw();
        let cpu = cpu_dssim(&ref_data, &dis_data, w as usize, h as usize);
        let gpu = d.compute(&ref_data, &dis_data).expect("compute").score;
        let rel = if cpu > 1e-6 {
            (gpu - cpu).abs() / cpu * 100.0
        } else {
            (gpu - cpu).abs() * 100.0
        };
        println!("{q:<8} {cpu:>14.6} {gpu:>14.6} {rel:>9.3}%");
    }
}
