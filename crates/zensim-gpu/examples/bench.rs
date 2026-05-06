//! Quick wall-clock comparison of `Zensim::compute_features` at various
//! resolutions. Not a rigorous benchmark — just enough signal to guide
//! optimisation. Compares against CPU `zensim::Zensim::compute(...).score()`.

use std::time::Instant;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{Zensim, score_from_features};

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

fn bench_size_phases(w: u32, h: u32) {
    use cubecl::server::ComputeServer;
    let img_a = make_image(w as usize, h as usize, 42);
    let img_b = make_image(w as usize, h as usize, 137);
    let client = Backend::client(&Default::default());
    let mut z = Zensim::<Backend>::new(client.clone(), w, h).unwrap();
    z.set_reference(&img_a).unwrap();

    let n_warmup = 4;
    let n_measure = 16;
    for _ in 0..n_warmup {
        let _ = z.compute_with_reference(&img_b).unwrap();
    }

    // Total wall.
    let t0 = Instant::now();
    for _ in 0..n_measure {
        let _ = z.compute_with_reference(&img_b).unwrap();
    }
    let total = t0.elapsed().as_secs_f64() / n_measure as f64;

    // Just the upload and host-fold parts (run set_reference each iter,
    // which is the upload + run_xyb_pyramid path; minus that we get
    // an estimate of the kernel pipeline + read).
    let t0 = Instant::now();
    for _ in 0..n_measure {
        z.set_reference(&img_a).unwrap();
    }
    let setref = t0.elapsed().as_secs_f64() / n_measure as f64;

    println!(
        "phases {:>5}x{:<5}  total {:>7.2} ms  set_ref {:>7.2} ms  cwr-only ~{:>7.2} ms",
        w,
        h,
        total * 1e3,
        setref * 1e3,
        (total - setref) * 1e3
    );
}

fn bench_size(w: u32, h: u32) {
    let img_a = make_image(w as usize, h as usize, 42);
    let img_b = make_image(w as usize, h as usize, 137);
    let n_warmup = 4;
    let n_measure = 16;

    // GPU: cached-reference path (set_reference once, compute many).
    let client = Backend::client(&Default::default());
    let mut z = Zensim::<Backend>::new(client, w, h).unwrap();
    z.set_reference(&img_a).unwrap();
    for _ in 0..n_warmup {
        let _ = z.compute_with_reference(&img_b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = z.compute_with_reference(&img_b).unwrap();
    }
    let gpu_cwr = t.elapsed().as_secs_f64() / n_measure as f64;

    // GPU: full compute_features (set_reference + compute_with_reference).
    let mut z2 = Zensim::<Backend>::new(Backend::client(&Default::default()), w, h).unwrap();
    for _ in 0..n_warmup {
        let _ = z2.compute_features(&img_a, &img_b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = z2.compute_features(&img_a, &img_b).unwrap();
    }
    let gpu_full = t.elapsed().as_secs_f64() / n_measure as f64;

    // CPU
    let zcpu = ZensimCpu::new(ZensimProfile::latest());
    let pix_a: Vec<[u8; 3]> = img_a.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let pix_b: Vec<[u8; 3]> = img_b.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let s = RgbSlice::new(&pix_a, w as usize, h as usize);
    let d = RgbSlice::new(&pix_b, w as usize, h as usize);
    for _ in 0..n_warmup {
        let _ = zcpu.compute(&s, &d).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n_measure {
        let _ = zcpu.compute(&s, &d).unwrap();
    }
    let cpu = t.elapsed().as_secs_f64() / n_measure as f64;

    let mp = (w as f64 * h as f64) / 1e6;
    println!(
        "{:>5}x{:<5}  cpu {:>7.2} ms  gpu_cwr {:>7.2} ms  gpu_full {:>7.2} ms  ({:.2} MP)",
        w,
        h,
        cpu * 1e3,
        gpu_cwr * 1e3,
        gpu_full * 1e3,
        mp
    );

    // Sanity: scores agree.
    let cpu_score = zcpu.compute(&s, &d).unwrap().score();
    z.set_reference(&img_a).unwrap();
    let gpu_features = z.compute_with_reference(&img_b).unwrap();
    let gpu_score = score_from_features(&gpu_features, &zensim::profile::WEIGHTS_PREVIEW_V0_2);
    let rel = (gpu_score - cpu_score).abs() / cpu_score.abs().max(1.0);
    eprintln!(
        "  cpu_score = {cpu_score:.4}, gpu_score = {gpu_score:.4}, rel = {:.2e}",
        rel
    );
}

fn main() {
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
    println!();
    for (w, h) in [(256, 256), (1024, 1024), (2048, 2048), (4096, 4096)] {
        bench_size_phases(w, h);
    }
}
