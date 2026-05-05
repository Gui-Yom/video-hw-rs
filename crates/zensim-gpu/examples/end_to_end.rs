//! Quick end-to-end smoke: extract a feature vector for one synthetic
//! image pair and apply `WEIGHTS_PREVIEW_V0_2` to print a 0-100 score.
//! Defaults to the CUDA backend; pass `--no-default-features --features
//! wgpu` to run on cross-vendor wgpu.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use zensim_gpu::{Zensim, score_from_features};

fn main() {
    let w = 64_u32;
    let h = 64_u32;
    let mut ref_buf = Vec::with_capacity((w * h * 3) as usize);
    let mut dis_buf = Vec::with_capacity((w * h * 3) as usize);
    for _y in 0..h {
        for x in 0..w {
            let g = ((x * 255) / w) as u8;
            ref_buf.extend_from_slice(&[g, g, g]);
            // distortion: shift left by 1 column
            let g2 = (((x.saturating_sub(1)) * 255) / w) as u8;
            dis_buf.extend_from_slice(&[g2, g2, g2]);
        }
    }

    let client = Backend::client(&Default::default());
    let mut z = Zensim::<Backend>::new(client, w, h).unwrap();
    let features = z.compute_features(&ref_buf, &dis_buf).unwrap();
    let weights = zensim::profile::WEIGHTS_PREVIEW_V0_2;
    let score = score_from_features(&features, &weights);
    println!("zensim-gpu score = {:.4}  (0-100, higher = better)", score);

    let self_features = z.compute_features(&ref_buf, &ref_buf).unwrap();
    let self_score = score_from_features(&self_features, &weights);
    println!("self-zensim score = {:.4}  (should be 100.0)", self_score);
}
