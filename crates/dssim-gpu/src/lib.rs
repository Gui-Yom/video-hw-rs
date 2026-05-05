//! Multi-vendor GPU implementation of the DSSIM perceptual image
//! quality metric.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl) — single Rust
//! kernel source, dispatchable across:
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** (SIMD) reference path when `cpu` is enabled
//!
//! Algorithmic parity target is the published `dssim-core` v3.4 crate
//! (the canonical Rust DSSIM implementation). At the pyramid level this
//! crate also matches `crates/dssim-cuda/`, which uses the same five
//! pyramid scales, two-pass 3×3 Gaussian, custom-Lab conversion and
//! per-scale MAD score.
//!
//! ## Single-image usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use dssim_gpu::Dssim;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! let dist_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//!
//! let result = d.compute(&ref_srgb, &dist_srgb)?;
//! println!("DSSIM = {:.6}", result.score);
//! # Ok::<(), dssim_gpu::Error>(())
//! ```
//!
//! ## Cached-reference usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use dssim_gpu::Dssim;
//!
//! # fn candidates() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! d.set_reference(&ref_srgb)?;
//!
//! for candidate in candidates() {
//!     let r = d.compute_with_reference(&candidate)?;
//!     // ... use r.score ...
//! }
//! # Ok::<(), dssim_gpu::Error>(())
//! ```
//!
//! ## Score interpretation
//!
//! Output is the standard DSSIM scalar: 0 means identical, larger
//! values mean more distortion. Mirrors the `f64` returned by
//! `dssim_core::Dssim::compare`.
//!
//! ## Status
//!
//! Initial port from `dssim-cuda`. See `PORT_STATUS.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
// NOTE: cubecl's `launch_unchecked` is an `unsafe fn` (the kernel body
// has already been validated by the `#[cube]` macro; the unsafe is for
// the launch-side ArrayArg lifetimes). We mirror ssim2-gpu /
// butteraugli-gpu, which keep unsafe local to the launch-orchestration
// layer. No raw pointer / transmute / get_unchecked is used.

pub mod kernels;
pub mod pipeline;
pub mod pipeline_batch;

pub use pipeline::Dssim;
pub use pipeline_batch::DssimBatch;

/// Number of pyramid scales — matches `dssim-core` and `dssim-cuda`.
pub const NUM_SCALES: usize = 5;

/// Per-scale weights from `dssim-core`. The final score is
/// `Σ (per_scale_score · weight) / Σ weight` where `per_scale_score =
/// 1 - mean(|ssim_i - avg|)` with `avg = mean_ssim ^ (0.5 ^ scale_idx)`.
pub const SCALE_WEIGHTS: [f64; NUM_SCALES] = [0.028, 0.197, 0.322, 0.298, 0.155];

/// Result of a DSSIM comparison. `score = 1/ssim - 1` per dssim-core's
/// `ssim_to_dssim`; 0 = identical, higher = worse.
#[derive(Debug, Clone, Copy)]
pub struct GpuDssimResult {
    pub score: f64,
}

/// Errors that the GPU DSSIM pipeline can return.
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is smaller than 8×8 — the pyramid would collapse before
    /// reaching scale 0.
    InvalidImageSize,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidImageSize => write!(f, "image must be at least 8×8 pixels"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
