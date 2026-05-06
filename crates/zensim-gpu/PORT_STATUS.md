# zensim-gpu port status

Multi-vendor GPU port of `zensim-cuda` using CubeCL. Algorithmic parity
target is the published `zensim` v0.2.8 crate with
`ZensimProfile::latest()` (= `WEIGHTS_PREVIEW_V0_2`, 228 features = 4
scales × 3 channels × 19 features).

## Module status

| Module | Source | LOC | Status | Notes |
|---|---|---|---|---|
| `kernels::color` | `zensim-cuda-kernel/src/color.rs` | ~110 | ✅ ported | sRGB packed-u8 → planar positive XYB. 256-entry LUT uploaded as `Array<f32>` (cubecl 0.10 can't index host-side `[f32; 256]` constants from `#[cube]`). `cbrt` substituted with `f32::powf(_, 1.0/3.0)` — the magic-constant Newton seed in CPU's `cbrtf_fast` requires `reinterpret_cast<u32>(K_B0)` which cubecl-cuda's codegen rejects for literal-folded constants. Drift vs CPU `cbrtf_fast` is a few ULPs, well below the SSIM normalisation threshold. Same call as dssim-gpu's Lab cbrt. |
| `kernels::pad` | `zensim-cuda-kernel/src/pad.rs` | ~30 | ✅ ported | Mirror-fill SIMD-padded columns; precomputed offset table on device. |
| `kernels::downscale` | `zensim-cuda-kernel/src/downscale.rs` | ~40 | ✅ ported | 2×2 box average with edge clamp on the **padded** plane (CPU zensim does not re-pad after downscaling — pad columns simply downscale along with everything else). |
| `kernels::blur` | `zensim-cuda-kernel/src/blur.rs` | ~70 | ✅ ported | Fused horizontal box-blur producing 4 outputs (`mu1`, `mu2`, `sigma_sq`, `sigma12`) per pixel. Mirror-x logic inlined in pure u32 (cubecl 0.10's `#[cube]` macro fights mixed signed-unsigned arithmetic). |
| `kernels::features` | `zensim-cuda-kernel/src/features.rs` | ~190 | ✅ ported | Fused V-blur + per-pixel feature extraction. **One thread per column** writes 17 f64 sums + 3 f32 maxes to per-column slots — no atomics needed (each column owns a unique slot). Host-side fold across columns produces the per-channel feature scalars. Avoids `Atomic<f64>` (cubecl 0.10 doesn't expose it) and `Atomic<f32>::fetch_max` (broken on Metal per gotcha G3.x). |
| Pipeline (`pipeline::Zensim`) | `zensim-cuda/src/lib.rs` | ~440 | ✅ wired | 4-scale pyramid with cached-reference state. SIMD padding matches `simd_padded_width` exactly so feature footprints stay aligned with CPU's `accum.n = padded_w × h`. |
| `Zensim::set_reference` / `compute_with_reference` | same | (above) | ✅ implemented | Reference pyramid cached after `set_reference`; subsequent `compute_with_reference` reads it without re-running the ref-side sRGB→XYB / pad / downscale chain. Cached-vs-direct drift ≤ 1e-3 on the noisy-gradient lock test. |

## Validated parity (RTX 5070, CUDA 13.2, host-side `zensim` v0.2.8)

`tests/parity_lock.rs` — **8 / 8 pass on CUDA**:

| Case | CPU score | GPU score | rel error |
|---|---|---|---|
| 32×32 identical gradient | 100.0 | 100.0 | ≤ 1e-6 |
| 64×64 black vs white | -208.4879 | -208.4871 | 4.0e-6 |
| 64×64 noisy gradient (±8) | 63.6834 | 63.6860 | 4.1e-5 |
| `dssim-cuda` corpus q70.jpg | 80.9018 | 80.8892 | 1.6e-4 |
| `dssim-cuda` corpus q90.jpg | 91.3509 | 91.3482 | 3.0e-5 |
| Cached-vs-direct drift | (n/a) | (n/a) | ≤ 1e-3 score |

Synthetic edge cases (grayscale, polar opposite, low-magnitude X
channels) sit comfortably under 1e-4 relative error. Real-image
corpus parity at q70 is ≈ 1.6e-4 (0.016 %) — within the cross-arch
FMA contraction floor that bounds CUDA-PTX vs CPU AVX-512 (the
`zensim-cuda` crate documents the same regime as "~ULP of cross-arch
FMA drift"). q90 and the synthetic cases land at ≤ 4e-5.

## The HF feature thresholds

The pipeline's host-side feature extraction matches CPU
`zensim::streaming::compute_features` exactly, including the
**per-pixel-variance threshold** that gates the HF ratios:

```rust
hf_energy_loss = if var_src > 1e-10 { (1.0 - var_dst / var_src).max(0.0) } else { 0.0 };
hf_energy_gain = if var_src > 1e-10 { (var_dst / var_src - 1.0).max(0.0) } else { 0.0 };
hf_mag_loss   = if mad_src > 1e-10 { (1.0 - mad_dst / mad_src).max(0.0) } else { 0.0 };
```

CPU's threshold is `var_src > 1e-10` (per-pixel variance), NOT
`den.abs() > 0.0`. Without this the f32 cancellation residue in
`Σ (s − mu1)²` for constant-colour channels (e.g., the B channel of a
grayscale image, where the XYB transfer collapses to a fixed value)
blows up the ratio and dominates the score. CPU and GPU agree on this
threshold so the feature output is bit-exact across the boundary
where the HF ratios fold to 0.

## FMA fusion match

The kernels use `cubecl::prelude::fma()` explicitly to replicate CPU's
`f32::mul_add` chains in:
- The opsin matrix multiply (`m00*r + (m01*g + (m02*b + K_B0))`)
- The `cbrtf_fast` Halley iterations
- The H-blur sums (`sum_sq = fma(s, s, fma(d, d, sum_sq))`)
- The per-pixel SSIM math (`num_m`, `num_s`, `denom_s`)

`absorbance_bias = -cbrtf_fast(K_B0)` is precomputed on the host using a
direct port of CPU's `cbrtf_fast` (magic-constant Newton seed + 2 Halley
iterations) and passed to the kernel as a runtime scalar — the bit-cast
inside `cbrtf_initial` triggers cubecl-cuda's
`reinterpret_cast<u32 const&>(literal)` codegen failure when applied to
a const-folded `K_B0` literal.

## Performance

Wall-clock measurements on RTX 5070 + CUDA 13.2 (Ryzen 9 7950X CPU
reference). `examples/bench.rs` runs N=8 iterations after 2 warm-ups.
`gpu_cwr` is the cached-reference path (`set_reference` once, then
`compute_with_reference` per call); `gpu_full` includes both phases
each call.

| Size       | CPU      | GPU (cached-ref) | GPU (full)  | GPU vs CPU (cwr) |
|------------|----------|------------------|-------------|------------------|
| 64×64      |  1.49 ms |   0.68 ms        |   0.81 ms   | **2.2× faster**  |
| 256×256    |  4.16 ms |   1.13 ms        |   1.28 ms   | **3.7× faster**  |
| 512×512    |  9.46 ms |   2.42 ms        |   2.26 ms   | **3.9× faster**  |
| 1024×1024  | 16.35 ms |   6.33 ms        |   7.54 ms   | **2.6× faster**  |
| 2048×2048  | 44.59 ms |  15.78 ms        |  24.91 ms   | **2.8× faster**  |
| 4096×4096  | 248.6 ms |  95.57 ms        | 179.49 ms   | **2.6× faster**  |

GPU now beats CPU at every resolution. Per-MP timing (lower is better):

| Size       | gpu_cwr (ms/MP) |
|------------|-----------------|
| 1024²      | 6.0             |
| 2048²      | 3.8             |
| 4096²      | 5.7             |

Best per-MP at 2 K. The 1 ms/MP target needs further work: a tile-fused
H+V kernel that keeps H-blur outputs in shared memory (eliminates the
~50 MB inter-kernel DRAM round-trip per scale) and CUDA-graph capture
to amortise launch overhead in encoder loops. cubecl 0.10 doesn't
expose graph capture, so this is upstream-blocked for the launch-
overhead piece.

Optimisations applied since the initial port:
- **Persistent partials buffers** (`Zensim::new`-time allocation, no
  per-call alloc churn). 12 small allocations / call → 0.
- **Single batched read-back of finals only**. After the on-device
  reduction the host reads ~1.6 KiB instead of ~5.7 MiB per call at
  1 K resolution.
- **3-channel-per-launch H-blur and V-blur+features kernels**.
  Reduces 24 per-call launches → 8.
- **3-channel-per-launch downscale**. Saves 6 launches across the
  pyramid build.
- **Column-strip parallelism in V-blur+features**. Each column is split
  into `n_strips` Y-strips, each processed by its own thread —
  `padded_w × n_strips × 3` parallel threads at the SM-occupancy floor
  (lifts 1 K perf 2× over the old per-column-only kernel).
- **On-device reduction kernel** (`reduce_scale_kernel`). Folds per-
  (col, strip, channel) partials into per-(scale, channel, slot)
  finals on the GPU. One launch per scale (4 total at SCALES = 4)
  with a 60-cube grid (3 channels × 20 slot kinds). Cuts the post-
  compute D2H from a multi-MiB read to a 1.6 KiB read.
- **Fused sRGB → XYB + mirror-pad in one launch**. The kernel covers
  the padded plane and reads from the mirror source column when
  `x ≥ width`. Eliminates the separate per-channel pad pass (was 3
  launches).
- **Packed sRGB-RGB upload**: one u32 per pixel (`R | G<<8 | B<<16`)
  instead of 3 widened u32s. Cuts H2D bandwidth 3× — load-bearing on
  WSL2 where virtualised PCIe is the dominant cost.
- **Persistent host pack scratch**: reused across calls so the u8 →
  u32 packing doesn't re-allocate.
- **No partials zeroing** between calls. Every column thread writes
  all 17 + 3 of its slots in `fused_vblur_features_kernel`, so the
  previous call's contents are fully overwritten before any reduction
  reads them.

## Backend coverage

| Backend | Build | Tests | Notes |
|---|---|---|---|
| CUDA (NVIDIA, native) | ✅ | ✅ 7/8 | Validated on RTX 5070 + CUDA 13.2. |
| WGPU (cross-vendor) | ✅ | ⚠ untested in WSL2 | WSL2 has no Vulkan ICD by default (gotcha G3.2). Validate on native Linux / Mac / Windows. |
| HIP (AMD ROCm) | ✅ (compiles) | ⚠ untested | Same shape as dssim-gpu / ssim2-gpu HIP path. |
| CPU (cubecl-cpu) | ✅ (compiles) | ❌ build-only | cubecl-cpu 0.10 doesn't yet support `Array<f64>` indexing reliably; we use the published `zensim` crate as the CPU reference instead. |

## Known gotchas applied

- **G1.x / `cbrt`** → substituted `f32::powf(_, 1.0/3.0)`. The CPU's
  `cbrtf_fast` magic-constant Newton seed via `reinterpret_cast<u32>` is
  not reachable on cubecl-cuda for compile-time-folded constants.
- **G1.5 / SharedMemory sizing** — n/a (no shared memory; kernels are
  per-pixel / per-column).
- **G2.1 / `CubeCount`/`CubeDim` not Copy** — every launch site
  recomputes the count.
- **G3.2 / WSL2 no Vulkan ICD** — wgpu backend is build-only on the
  reference host; CUDA validates the algorithm.
- **G3.3 / cubecl-cpu no atomics + f64 indexing limitations** —
  cubecl-cpu is build-only.
- **Cancellation in degenerate inputs** — see "HF-ratio noise floor"
  above.
- **Per-column partials, not per-block atomics** — sidesteps the lack
  of `Atomic<f64>` and Metal's broken `Atomic<f32>::fetch_max`. Cost:
  `padded_w × 17 × 8` bytes of scratch per (scale, channel), 557 KiB at
  4 K, well within budget.

## Followups (not blocking)

- **Tighten black-vs-white parity** by adopting the same FMA fusion
  order as the CPU AVX-512 path. cubecl 0.10 has `fma()` (see
  `cubecl::prelude::fma`); using it explicitly may close the 12-point
  gap. Test won't reflect on real images either way.
- **Per-kernel parity examples** modelled on `ssim2-gpu`'s set
  (`color_parity.rs`, `blur_parity.rs`, `features_parity.rs`). The
  integration tests already validate the full pipeline against
  `zensim` v0.2.8; per-kernel diagnostics would only be needed on
  regression.
- **Batched scoring (`ZensimBatch`)** on the dssim-gpu /
  butteraugli-gpu / ssim2-gpu shape. Useful for encoder rate-distortion
  sweeps. Not implemented yet because `zensim-cuda` itself doesn't
  expose a batched API; would be a workspace-level addition rather than
  a port.
- **`zen-metrics-cli` integration** — register `zensim-gpu` alongside
  `dssim-gpu`, `ssim2-gpu`, `butteraugli-gpu` in the metric registry.
- **Tighten `cached_reference_matches_direct` to 1e-5** once the
  fma-ordering match is in place. Currently `< 1e-3`.
