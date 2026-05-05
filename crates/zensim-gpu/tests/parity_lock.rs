//! Integration tests for `zensim-gpu` against the published `zensim`
//! v0.2.8 CPU reference (`ZensimProfile::latest()` =
//! `WEIGHTS_PREVIEW_V0_2`).
//!
//! Backend selection mirrors `dssim-gpu` / `ssim2-gpu`:
//! - `cuda` (default) preferred
//! - `wgpu` fallback when CUDA isn't compiled in
//!
//! cubecl-cpu is build-only here (gotcha G3.3); the per-column
//! partials we write don't use atomics, but the launch geometry would
//! still need cube_count which cubecl-cpu treats inconsistently for
//! Array<f64>.

use cubecl::Runtime;
use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};
use zensim_gpu::{Zensim, score_from_features};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "zensim-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

// ───────────────────────── helpers ─────────────────────────

fn cpu_score(rgb_ref: &[u8], rgb_dis: &[u8], w: usize, h: usize) -> f64 {
    let z = ZensimCpu::new(ZensimProfile::latest());
    let to_pix = |buf: &[u8]| -> Vec<[u8; 3]> {
        buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    };
    let src = to_pix(rgb_ref);
    let dst = to_pix(rgb_dis);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    z.compute(&s, &d).expect("zensim cpu compute").score()
}

fn gpu_score<R: Runtime>(z: &mut Zensim<R>, rgb_ref: &[u8], rgb_dis: &[u8]) -> f64 {
    let features = z.compute_features(rgb_ref, rgb_dis).expect("compute_features");
    score_from_features(&features, &zensim::profile::WEIGHTS_PREVIEW_V0_2)
}

fn solid(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        v.push(r);
        v.push(g);
        v.push(b);
    }
    v
}

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

// ───────────────────────── parity ─────────────────────────

#[test]
fn identical_image_scores_near_100() {
    let w = 64;
    let h = 64;
    let img = gradient(w, h);
    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let s = gpu_score(&mut z, &img, &img);
    eprintln!("identical: gpu = {s:.4} (expect ~100)");
    assert!(
        s > 99.5,
        "GPU score for identical input should be ~100, got {s}"
    );
}

#[test]
fn black_vs_white_is_low() {
    let w = 64;
    let h = 64;
    let black = solid(w, h, 0, 0, 0);
    let white = solid(w, h, 255, 255, 255);
    let cpu = cpu_score(&black, &white, w, h);
    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = gpu_score(&mut z, &black, &white);
    eprintln!("black-vs-white: cpu = {cpu:.4}, gpu = {gpu:.4}");
    let abs = (gpu - cpu).abs();
    assert!(
        abs < 5.0,
        "gpu = {gpu:.4} differs from cpu = {cpu:.4} by {abs:.2} points"
    );
}

#[test]
fn small_distortion_close_to_cpu() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d = add_noise(&r, 8);
    let cpu = cpu_score(&r, &d, w, h);
    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = gpu_score(&mut z, &r, &d);
    eprintln!("noisy-gradient: cpu = {cpu:.4}, gpu = {gpu:.4}");
    let abs = (gpu - cpu).abs();
    assert!(
        abs < 2.0,
        "gpu = {gpu:.4} differs from cpu = {cpu:.4} by {abs:.2} points"
    );
}

#[test]
fn cached_reference_matches_direct() {
    let w = 64_u32;
    let h = 64_u32;
    let r = gradient(w as usize, h as usize);
    let d = add_noise(&r, 12);

    let mut z_one = Zensim::<Backend>::new(make_client!(), w, h).unwrap();
    let direct = gpu_score(&mut z_one, &r, &d);

    let mut z_two = Zensim::<Backend>::new(make_client!(), w, h).unwrap();
    z_two.set_reference(&r).unwrap();
    let cached_features = z_two.compute_with_reference(&d).unwrap();
    let cached =
        score_from_features(&cached_features, &zensim::profile::WEIGHTS_PREVIEW_V0_2);

    eprintln!("direct = {direct:.6}, cached = {cached:.6}");
    let abs = (direct - cached).abs();
    assert!(
        abs < 1e-3,
        "cached path drifted from direct by {abs:.2e} (direct={direct}, cached={cached})"
    );
}

// ───────────────────────── error paths ─────────────────────────

#[test]
fn dimension_mismatch_is_reported() {
    let mut z = Zensim::<Backend>::new(make_client!(), 32, 32).unwrap();
    let small = vec![0_u8; 31 * 32 * 3];
    let r = z.compute_features(&small, &small);
    assert!(r.is_err(), "expected DimensionMismatch error");
}

#[test]
fn invalid_image_size_rejected() {
    let r = Zensim::<Backend>::new(make_client!(), 4, 4);
    assert!(r.is_err(), "expected InvalidImageSize for 4x4");
}

#[test]
fn no_cached_reference_rejected() {
    let mut z = Zensim::<Backend>::new(make_client!(), 32, 32).unwrap();
    let buf = vec![0_u8; 32 * 32 * 3];
    let r = z.compute_with_reference(&buf);
    assert!(r.is_err(), "expected NoCachedReference");
}

// ───────────────────────── corpus parity ─────────────────────────

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

#[test]
fn jpeg_corpus_q70_q90() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIR);
    let src_path = dir.join("source.png");
    if !src_path.exists() {
        eprintln!("skipping: corpus dir absent at {}", dir.display());
        return;
    }
    let img = image::open(&src_path).expect("decode png").to_rgb8();
    let (w, h) = img.dimensions();
    let ref_data = img.into_raw();

    let mut z = Zensim::<Backend>::new(make_client!(), w, h).unwrap();
    for q in &["q70.jpg", "q90.jpg"] {
        let dis = image::open(dir.join(q)).expect("decode jpeg").to_rgb8();
        assert_eq!(dis.dimensions(), (w, h));
        let dis_data = dis.into_raw();
        let cpu = cpu_score(&ref_data, &dis_data, w as usize, h as usize);
        let gpu = gpu_score(&mut z, &ref_data, &dis_data);
        eprintln!("{q}: cpu = {cpu:.4}, gpu = {gpu:.4}");
        let abs = (gpu - cpu).abs();
        assert!(
            abs < 2.0,
            "{q}: gpu = {gpu:.4} differs from cpu = {cpu:.4} by {abs:.2} points"
        );
    }
}
