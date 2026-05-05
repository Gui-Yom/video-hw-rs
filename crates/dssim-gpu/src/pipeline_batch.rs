//! Sequential-loop batched DSSIM scoring.
//!
//! [`DssimBatch`] is a thin wrapper that scores many distorted candidates
//! against one reference, reusing the cached-reference pyramid + Lab +
//! reference-blur work via [`Dssim::set_reference`] /
//! [`Dssim::compute_with_reference`]. Each call still launches one full
//! distorted-side pipeline per image — there is no kernel-level batching
//! yet.
//!
//! This shape mirrors `ssim2-gpu`'s initial `Ssim2Batch` (sequential loop
//! before the kernel-batched variant landed) and `butteraugli-gpu`'s
//! `ButteraugliBatch` API. Encoder rate-distortion sweeps that scored N
//! candidates against one reference would otherwise pay the full
//! reference-side cost N times.
//!
//! Future work: a kernel-batched variant that packs distorted images into
//! one buffer and broadcasts the cached reference across them. See the
//! `pipeline_batch.rs` files in `butteraugli-gpu` and `ssim2-gpu` for the
//! pattern. Not implemented yet because no in-tree consumer needs it.
//!
//! ## Usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use dssim_gpu::DssimBatch;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut b = DssimBatch::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! # let candidates: Vec<Vec<u8>> = vec![];
//! b.set_reference(&ref_srgb)?;
//! let scores = b.compute_batch(&candidates.iter().map(|v| v.as_slice()).collect::<Vec<_>>())?;
//! # Ok::<(), dssim_gpu::Error>(())
//! ```

use cubecl::prelude::*;

use crate::pipeline::Dssim;
use crate::{Error, GpuDssimResult, Result};

/// Per-instance allocations + cached-reference scoring loop.
///
/// One [`DssimBatch`] holds one [`Dssim`] (so one set of per-resolution
/// buffers) plus the cached-reference state. Construct once per
/// resolution; rotate through reference + many distorted candidates.
pub struct DssimBatch<R: Runtime> {
    inner: Dssim<R>,
}

impl<R: Runtime> DssimBatch<R> {
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            inner: Dssim::new(client, width, height)?,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.inner.dimensions()
    }

    /// Cache the reference. Subsequent [`Self::compute`] /
    /// [`Self::compute_batch`] calls skip the reference-side pyramid
    /// + Lab + reference-blur.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.inner.set_reference(ref_srgb)
    }

    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    pub fn has_cached_reference(&self) -> bool {
        self.inner.has_cached_reference()
    }

    /// Score one distorted candidate against the cached reference.
    /// Returns [`Error::NoCachedReference`] if [`Self::set_reference`]
    /// hasn't been called.
    pub fn compute(&mut self, dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        self.inner.compute_with_reference(dist_srgb)
    }

    /// Score many distorted candidates against the cached reference.
    /// Returns one score per slice in input order. The reference must
    /// already be cached via [`Self::set_reference`].
    ///
    /// Currently a sequential loop over [`Self::compute`]; each call
    /// still launches the full distorted-side pipeline. The savings
    /// come entirely from skipping the reference-side work.
    pub fn compute_batch(&mut self, distorted: &[&[u8]]) -> Result<Vec<GpuDssimResult>> {
        if !self.inner.has_cached_reference() {
            return Err(Error::NoCachedReference);
        }
        let mut out = Vec::with_capacity(distorted.len());
        for dist in distorted {
            out.push(self.inner.compute_with_reference(dist)?);
        }
        Ok(out)
    }
}
