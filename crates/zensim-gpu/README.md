# zensim-gpu

Multi-vendor GPU implementation of the
[zensim](https://github.com/imazen/zensim) perceptual similarity feature
extractor, built on [CubeCL](https://github.com/tracel-ai/cubecl). One
Rust kernel source, runs on:

- **CUDA** (NVIDIA) via `cubecl-cuda`
- **WGPU** (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- **HIP** (AMD ROCm) via `cubecl-hip` (`hip` feature)
- **CPU** fallback via `cubecl-cpu` for build-checking only — use the
  published `zensim` crate as the runtime CPU reference

Algorithmic parity target is `zensim` v0.2.8 with
`ZensimProfile::latest()` (= `WEIGHTS_PREVIEW_V0_2`, 228 features =
4 scales × 3 channels × 19 features). At the pyramid level this also
matches the existing `zensim-cuda` crate — same SIMD-padded layout,
same `cbrt` accuracy regime, same fused H-blur (mu1 / mu2 / sigma_sq /
sigma12) and fused V-blur + per-pixel feature kernels. The CUDA
crate stays in the workspace; this one extends reach to AMD / Intel /
Apple / WebGPU without giving anything up on NVIDIA.

## Single-image usage

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use zensim_gpu::{Zensim, score_from_features};

let client = WgpuRuntime::client(&Default::default());
let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;

let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
let dis_srgb: Vec<u8> = vec![0; 512 * 512 * 3];

let features = z.compute_features(&ref_srgb, &dis_srgb)?;
let score = score_from_features(&features, &zensim::profile::WEIGHTS_PREVIEW_V0_2);
println!("zensim score = {:.4} (0-100, higher = better)", score);
# Ok::<(), zensim_gpu::Error>(())
```

## Cached-reference usage

When scoring many distorted candidates against one reference (encoder
rate-distortion search), call `set_reference` once and then
`compute_with_reference` per candidate. Skips the reference-side sRGB
→ XYB conversion, mirror-pad, and pyramid downscale on every call.

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use zensim_gpu::{Zensim, score_from_features};

let client = WgpuRuntime::client(&Default::default());
let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;

let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
z.set_reference(&ref_srgb)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let f = z.compute_with_reference(&candidate)?;
    let s = score_from_features(&f, &zensim::profile::WEIGHTS_PREVIEW_V0_2);
    // ... use s ...
}
# Ok::<(), zensim_gpu::Error>(())
```

## Score interpretation

Output is a 228-entry `[f64; TOTAL_FEATURES]` feature vector. Apply the
trained weights from `zensim::profile::WEIGHTS_PREVIEW_V0_2` (228
entries) and the standard `100 - 18·d^0.7` mapping to convert to a
0-100 score. The `score_from_features(&features, &weights)` helper
does exactly this.

## Features

| Feature | Default | Effect |
|---|---|---|
| `cuda` | yes | Compile the cubecl-cuda backend. |
| `wgpu` | yes | Compile the cubecl-wgpu backend. |
| `cpu`  | yes | Compile the cubecl-cpu backend (build-only — see above). |
| `hip`  | no  | Compile the cubecl-hip backend. |
| `fast-reduction` | yes | Reserved — current pipeline uses per-column slot writes (no atomics needed); the flag is kept for API symmetry with `dssim-gpu` / `ssim2-gpu`. |

For Metal targets:

```bash
cargo build -p zensim-gpu --no-default-features --features wgpu
```

## Status

Initial port from `zensim-cuda`. See `PORT_STATUS.md` for the
per-kernel breakdown, runtime coverage, and validated parity numbers.
