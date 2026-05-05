//! Multi-vendor GPU implementation of the zensim perceptual similarity
//! feature extractor.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl) — the same
//! `#[cube]` Rust kernel source dispatches across:
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** (build-only) — `cubecl-cpu` 0.10 doesn't yet implement
//!   `Atomic<f32>::fetch_add` and our reduction relies on it; use the
//!   published `zensim` crate as the CPU reference instead.
//!
//! Algorithmic parity target is the published `zensim` v0.2.8 crate
//! with `ZensimProfile::latest()` (= `WEIGHTS_PREVIEW_V0_2`, 228
//! features = 4 scales × 3 channels × 19 features). At the pyramid
//! level this also matches `crates/zensim-cuda/`, which uses the same
//! SIMD-padded layout, the same `cbrtf_fast` Halley iterations, the
//! same fused H-blur (mu1/mu2/sigma_sq/sigma12) and fused V-blur +
//! per-pixel feature kernels.
//!
//! The CUDA crate stays in the workspace; this one extends reach to
//! AMD / Intel / Apple / WebGPU without giving anything up on NVIDIA.
//!
//! ## Single-image usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use zensim_gpu::Zensim;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//! let dist_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//!
//! let features = z.compute_features(&ref_srgb, &dist_srgb)?;
//! // Apply trained weights from `zensim::profile::WEIGHTS_PREVIEW_V0_2`
//! // (228 entries) and convert to a 0-100 score.
//! # Ok::<(), zensim_gpu::Error>(())
//! ```
//!
//! ## Cached-reference usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use zensim_gpu::Zensim;
//!
//! # fn candidates() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//! z.set_reference(&ref_srgb)?;
//! for candidate in candidates() {
//!     let features = z.compute_with_reference(&candidate)?;
//!     // ... apply weights ...
//! }
//! # Ok::<(), zensim_gpu::Error>(())
//! ```
//!
//! ## Status
//!
//! Initial port from `zensim-cuda`. See `PORT_STATUS.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod kernels;
pub mod pipeline;

pub use pipeline::Zensim;

/// Number of pyramid scales — matches CPU zensim's `WEIGHTS_PREVIEW_V0_2`.
pub const SCALES: usize = 4;

/// Features per channel per scale = 13 basic (mean/L2/L4) + 6 peak
/// (max/L8 pooled) = 19.
pub const FEATURES_PER_CHANNEL: usize = 19;

/// Features per channel per scale, basic ("scored") block — matches
/// CPU `FEATURES_PER_CHANNEL_BASIC`.
pub const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Peak-block features per channel per scale.
pub const FEATURES_PER_CHANNEL_PEAKS: usize = 6;

/// Features per scale (= 19 × 3 channels = 57).
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL * 3;

/// Total features = 4 scales × 57 = 228.
pub const TOTAL_FEATURES: usize = FEATURES_PER_SCALE * SCALES;

/// Blur radius for the V0.1 / V0.2 profiles (`PROFILE_PREVIEW_V0_*`'s
/// `blur_radius`). Diameter = 11. Bumping this changes the HF ratios
/// nonlinearly; the trained weights only pair with this value.
pub const BLUR_RADIUS: u32 = 5;

/// SIMD-alignment padding applied by CPU zensim at scale 0. Every plane
/// is widened to the next multiple of 16, with an extra 16 cols added
/// when the aligned width ≥ 512 and `aligned/16` is even (cache-set
/// avoidance on x86). The mirror-pad fill matches CPU exactly.
///
/// Must match `zensim::blur::simd_padded_width`.
pub fn simd_padded_width(width: usize) -> usize {
    let aligned = (width + 15) & !15;
    if aligned >= 512 && (aligned / 16) % 2 == 0 {
        aligned + 16
    } else {
        aligned
    }
}

/// Apply `100 - 18·d^0.7` to a weighted feature distance, matching CPU
/// zensim's `score_from_features`. Pass the trained weights from
/// `zensim::profile::WEIGHTS_PREVIEW_V0_2`.
pub fn score_from_features(features: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(features.len(), weights.len());
    let raw: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(&f, &w)| w * f)
        .sum();
    let n_scales = features.len() / FEATURES_PER_SCALE;
    let raw_per_scale = raw / n_scales.max(1) as f64;
    if raw_per_scale <= 0.0 {
        100.0
    } else {
        100.0 - 18.0 * raw_per_scale.powf(0.7)
    }
}

#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is too small for the configured pyramid (logical w/h < 8 at scale 0).
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
