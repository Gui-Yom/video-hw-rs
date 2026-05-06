//! dssim-gpu vs dssim-core (CPU) wall-clock comparison.
//!
//! Run:
//! ```bash
//! CUDA_PATH=/usr/local/cuda cargo run --release -p dssim-gpu --example bench
//! ```

use std::time::Instant;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use dssim_core::{Dssim as DssimCpu, ToRGBAPLU};
use dssim_gpu::Dssim;
use imgref::ImgVec;
use rgb::RGB;

fn make_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        for _ in 0..3 {
            s = s * Wrapping(1664525u32) + Wrapping(1013904223u32);
            v.push(((s.0 >> 16) & 0xFF) as u8);
        }
    }
    v
}

fn cpu_dssim(ref_data: &[u8], dis_data: &[u8], w: usize, h: usize) -> f64 {
    let dssim = DssimCpu::new();
    let to_rgb = |buf: &[u8]| -> Vec<RGB<u8>> {
        buf.chunks_exact(3).map(|c| RGB::new(c[0], c[1], c[2])).collect()
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

fn bench_size(w: u32, h: u32) {
    let img_a = make_image(w as usize, h as usize, 42);
    let img_b = make_image(w as usize, h as usize, 137);
    let n_warmup = 4;
    let n_measure = 16;

    // GPU: cached-reference path.
    let client = Backend::client(&Default::default());
    let mut d = Dssim::<Backend>::new(client.clone(), w, h).unwrap();
    d.set_reference(&img_a).unwrap();
    for _ in 0..n_warmup {
        let _ = d.compute_with_reference(&img_b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = d.compute_with_reference(&img_b).unwrap();
    }
    let gpu_cwr = t.elapsed().as_secs_f64() / n_measure as f64;

    // GPU: full compute (set_reference + compute_with_reference).
    let mut d2 = Dssim::<Backend>::new(Backend::client(&Default::default()), w, h).unwrap();
    for _ in 0..n_warmup {
        let _ = d2.compute(&img_a, &img_b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = d2.compute(&img_a, &img_b).unwrap();
    }
    let gpu_full = t.elapsed().as_secs_f64() / n_measure as f64;

    // CPU.
    for _ in 0..n_warmup {
        let _ = cpu_dssim(&img_a, &img_b, w as usize, h as usize);
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = cpu_dssim(&img_a, &img_b, w as usize, h as usize);
    }
    let cpu = t.elapsed().as_secs_f64() / n_measure as f64;

    let mp = (w as f64 * h as f64) / 1e6;
    let speed_cwr = cpu / gpu_cwr;
    let speed_full = cpu / gpu_full;
    println!(
        "{:>5}x{:<5}  cpu {:>8.2} ms  gpu_cwr {:>8.2} ms ({:>5.1}× faster)  gpu_full {:>8.2} ms ({:>5.1}×)  ({:.2} MP)",
        w, h, cpu * 1e3, gpu_cwr * 1e3, speed_cwr, gpu_full * 1e3, speed_full, mp
    );
}

fn main() {
    println!("dssim-gpu vs dssim-core (RTX 5070 + Ryzen 9 7950X)");
    for (w, h) in [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        bench_size(w, h);
    }
}
