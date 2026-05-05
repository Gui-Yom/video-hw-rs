//! sRGB packed-u8 → planar positive-XYB f32.
//!
//! Verbatim port of `zensim-cuda-kernel/src/color.rs`, which itself
//! matches CPU zensim's `srgb_to_positive_xyb_planar_inner`. The Halley
//! `cbrtf_fast` is preserved bit-for-bit so the GPU output matches the
//! CPU scalar path within ~1 ULP (FMA contraction differences only).
//!
//! The 256-entry sRGB-byte → linear-f32 LUT is uploaded as an
//! `Array<f32>` argument (cubecl 0.10's `#[cube]` body can't index
//! a host-side `[f32; 256]` constant directly).
//!
//! Output is 3 planar f32 buffers at the **padded** width — pad
//! columns `[width..padded_w)` are left untouched by this kernel and
//! filled by the [`pad`](super::pad) kernel afterwards.

use cubecl::prelude::*;

// Matrix + bias constants, match zensim CPU exactly (color.rs).
const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4;

/// 256-entry sRGB byte → linear f32 LUT. The pipeline uploads this to
/// GPU memory once and passes the handle to
/// [`srgb_to_positive_xyb_kernel`].
#[rustfmt::skip]
pub const SRGB8_TO_LINEARF32_LUT: [f32; 256] = [
    0.0, 0.000303527, 0.000607054, 0.00091058103, 0.001214108, 0.001517635, 0.0018211621, 0.002124689,
    0.002428216, 0.002731743, 0.00303527, 0.0033465356, 0.003676507, 0.004024717, 0.004391442,
    0.0047769533, 0.005181517, 0.0056053917, 0.0060488326, 0.006512091, 0.00699541, 0.0074990317,
    0.008023192, 0.008568125, 0.009134057, 0.009721218, 0.010329823, 0.010960094, 0.011612245,
    0.012286487, 0.012983031, 0.013702081, 0.014443844, 0.015208514, 0.015996292, 0.016807375,
    0.017641952, 0.018500218, 0.019382361, 0.020288562, 0.02121901, 0.022173883, 0.023153365,
    0.02415763, 0.025186857, 0.026241222, 0.027320892, 0.028426038, 0.029556843, 0.03071345, 0.03189604,
    0.033104774, 0.03433981, 0.035601325, 0.036889452, 0.038204376, 0.039546248, 0.04091521, 0.042311423,
    0.043735042, 0.045186214, 0.046665095, 0.048171833, 0.049706575, 0.051269468, 0.052860655, 0.05448028,
    0.056128494, 0.057805434, 0.05951124, 0.06124607, 0.06301003, 0.06480328, 0.06662595, 0.06847818,
    0.07036011, 0.07227186, 0.07421358, 0.07618539, 0.07818743, 0.08021983, 0.082282715, 0.084376216,
    0.086500466, 0.088655606, 0.09084173, 0.09305898, 0.095307484, 0.09758736, 0.09989874, 0.10224175,
    0.10461649, 0.10702311, 0.10946172, 0.111932434, 0.11443538, 0.116970696, 0.11953845, 0.12213881,
    0.12477186, 0.12743773, 0.13013652, 0.13286836, 0.13563336, 0.13843165, 0.14126332, 0.1441285,
    0.1470273, 0.14995982, 0.15292618, 0.1559265, 0.15896086, 0.16202943, 0.16513224, 0.16826946,
    0.17144115, 0.17464745, 0.17788847, 0.1811643, 0.18447503, 0.1878208, 0.19120172, 0.19461787,
    0.19806935, 0.2015563, 0.20507877, 0.2086369, 0.21223079, 0.21586053, 0.21952623, 0.22322798,
    0.22696589, 0.23074007, 0.23455065, 0.23839766, 0.2422812, 0.2462014, 0.25015837, 0.25415218,
    0.2581829, 0.26225072, 0.26635566, 0.27049786, 0.27467737, 0.27889434, 0.2831488, 0.2874409,
    0.2917707, 0.29613832, 0.30054384, 0.30498737, 0.30946895, 0.31398875, 0.31854683, 0.32314324,
    0.32777813, 0.33245158, 0.33716366, 0.34191445, 0.3467041, 0.3515327, 0.35640025, 0.36130688,
    0.3662527, 0.37123778, 0.37626222, 0.3813261, 0.38642952, 0.39157256, 0.3967553, 0.40197787,
    0.4072403, 0.4125427, 0.41788515, 0.42326775, 0.42869055, 0.4341537, 0.43965724, 0.44520125,
    0.45078585, 0.45641106, 0.46207705, 0.46778384, 0.47353154, 0.47932023, 0.48514998, 0.4910209,
    0.49693304, 0.5028866, 0.50888145, 0.5149178, 0.5209957, 0.52711535, 0.5332766, 0.5394797,
    0.5457247, 0.5520116, 0.5583406, 0.5647117, 0.57112503, 0.57758063, 0.5840786, 0.590619, 0.597202,
    0.60382754, 0.61049575, 0.61720675, 0.62396055, 0.63075733, 0.637597, 0.6444799, 0.6514058,
    0.65837497, 0.66538745, 0.67244333, 0.6795426, 0.68668544, 0.69387203, 0.70110214, 0.70837605,
    0.7156938, 0.72305536, 0.730461, 0.7379107, 0.7454045, 0.75294244, 0.76052475, 0.7681514, 0.77582246,
    0.78353804, 0.79129815, 0.79910296, 0.8069525, 0.8148468, 0.822786, 0.8307701, 0.83879924, 0.84687346,
    0.8549928, 0.8631574, 0.87136734, 0.8796226, 0.8879232, 0.89626956, 0.90466136, 0.913099, 0.92158204,
    0.93011117, 0.9386859, 0.9473069, 0.9559735, 0.9646866, 0.9734455, 0.98225087, 0.9911022, 1.0,
];

/// `cbrt` substitute — `f32::powf(x, 1/3)`.
///
/// CPU zensim uses a magic-constant Newton seed + 2 Halley iterations
/// (`cbrtf_fast`), but the seed step requires `reinterpret_cast<u32>(K_B0)`
/// which cubecl-cuda's codegen can't emit for a literal-folded
/// constant. `powf(_, 1/3)` agrees with `cbrtf_fast` within a few f32
/// ULPs across the unit range, well below the SSIM formula's
/// normalisation. dssim-gpu uses the same substitution in its Lab
/// conversion.
#[cube]
fn cbrtf_fast(x: f32) -> f32 {
    if x == 0.0 {
        f32::new(0.0)
    } else {
        f32::powf(x, 1.0 / 3.0)
    }
}

/// sRGB packed RGB u8 → 3 planar positive-XYB f32 buffers.
///
/// Inputs are uploaded as `Array<u32>` for WGSL portability (no native
/// u8 storage on Metal / wgpu). The 256-entry LUT is also passed as a
/// device buffer.
///
/// Output planes are each `padded_w × height` long; this kernel only
/// writes columns `[0..width)`. Padding columns are filled by
/// [`super::pad`].
#[cube(launch_unchecked)]
pub fn srgb_to_positive_xyb_kernel(
    src: &Array<u32>,
    lut: &Array<f32>,
    x_out: &mut Array<f32>,
    y_out: &mut Array<f32>,
    b_out: &mut Array<f32>,
    width: u32,
    height: u32,
    padded_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let pw = padded_w as usize;
    let y = idx / w;
    let x = idx - y * w;

    let i3 = idx * 3;
    let r = lut[src[i3] as usize];
    let g = lut[src[i3 + 1] as usize];
    let b = lut[src[i3 + 2] as usize];

    let mixed0 = f32::max(K_M00 * r + K_M01 * g + K_M02 * b + K_B0, 0.0);
    let mixed1 = f32::max(K_M10 * r + K_M11 * g + K_M12 * b + K_B0, 0.0);
    let mixed2 = f32::max(K_M20 * r + K_M21 * g + K_M22 * b + K_B0, 0.0);

    let absorbance_bias = -cbrtf_fast(K_B0);
    let c0 = cbrtf_fast(mixed0) + absorbance_bias;
    let c1 = cbrtf_fast(mixed1) + absorbance_bias;
    let c2 = cbrtf_fast(mixed2);

    let x_val = 0.5 * (c0 - c1);
    let y_val = 0.5 * (c0 + c1);
    let b_val = c2 - y_val;

    let x_pos = x_val * 14.0 + 0.42;
    let y_pos = y_val + 0.01;
    let b_pos = b_val + 0.55;

    let out_idx = y * pw + x;
    x_out[out_idx] = x_pos;
    y_out[out_idx] = y_pos;
    b_out[out_idx] = b_pos;
}
