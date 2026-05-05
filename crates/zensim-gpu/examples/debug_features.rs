//! Debug helper: print 228 features side-by-side CPU vs GPU on the
//! noisy-gradient case that the parity tests flag.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{Zensim, score_from_features};

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let g = ((x * 255) / w) as u8;
            v.push(g);
            v.push(g);
            v.push(g);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(12345_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

fn main() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    // CPU features.
    let z_cpu = ZensimCpu::new(ZensimProfile::latest());
    let r_pix: Vec<[u8; 3]> = r.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let d_pix: Vec<[u8; 3]> = d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let s = RgbSlice::new(&r_pix, w, h);
    let dd = RgbSlice::new(&d_pix, w, h);
    let cpu_result = z_cpu.compute(&s, &dd).unwrap();
    let cpu_score = cpu_result.score();
    let cpu_features = cpu_result.features().to_vec();
    let cpu_features = cpu_features[..228].to_vec();

    // GPU features.
    let client = Backend::client(&Default::default());
    let mut z = Zensim::<Backend>::new(client, w as u32, h as u32).unwrap();
    let gpu_features = z.compute_features(&r, &d).unwrap();
    let gpu_score = score_from_features(
        &gpu_features,
        &zensim::profile::WEIGHTS_PREVIEW_V0_2,
    );

    println!("cpu score = {cpu_score:.4}, gpu score = {gpu_score:.4e}");
    println!("{:>4} {:>14} {:>14} {:>10}", "i", "cpu", "gpu", "abs");
    let mut max_abs = 0.0f64;
    let mut max_idx = 0usize;
    for i in 0..228 {
        let cpu = cpu_features[i];
        let gpu = gpu_features[i];
        let abs = (gpu - cpu).abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = i;
        }
        if abs > 0.01 {
            println!("{i:>4} {cpu:>14.6} {gpu:>14.6e} {abs:>10.4e}");
        }
    }
    println!("\nworst feature: idx={max_idx}, abs={max_abs:.4e}");
    println!(
        "  cpu = {:.6}, gpu = {:.6e}",
        cpu_features[max_idx], gpu_features[max_idx]
    );
}
