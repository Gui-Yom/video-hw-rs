//! Tile-fused H-blur + V-blur+features kernel.
//!
//! Replaces the separate `fused_blur_h_ssim_3ch_kernel` +
//! `fused_vblur_features_kernel` pipeline with a single launch that
//! keeps the H-blur outputs in shared memory. Eliminates the largest
//! source of inter-kernel DRAM traffic (~50 MiB per scale at 1 MP)
//! and removes the H-blur scratch planes' allocations entirely.
//!
//! ## Block layout
//!
//! Grid: `(ceil(padded_w / TX), n_strips, 3)`. Each block owns a tile
//! of `TX × strip_h` output pixels for one channel. `TX = 64` chosen to
//! be 2 warps — small enough that ~10 blocks fit per SM (with the
//! ~12 KiB shared memory below) and large enough that the
//! `TX + 2·R = 74`-wide cooperative tile load fits in 2 loads/thread.
//!
//! ## Shared memory layout (per block)
//!
//! - `src_row[TX + 2R]` (f32, 296 B)   — one row of `src` for the
//!   block's tile + halo, reloaded for each row entering the V-blur
//!   window.
//! - `dst_row[TX + 2R]` (f32, 296 B)
//! - `buf_mu1, buf_mu2, buf_sq, buf_s12[DIAM × TX]` (f32, 4 × 2816 B)
//!   — circular buffer holding the H-blur outputs for the current
//!   sliding window. Each `buf_*[slot × TX + tx]` is the `slot`-th
//!   row's H-blur output for the column owned by thread `tx`. Slot
//!   indexing is `(y - y_start) mod DIAM`; remarkably, the slot to
//!   subtract (the row leaving the window at step `y`) and the slot
//!   to overwrite (the row entering the window at step `y + R + 1`)
//!   are the SAME slot — `(2R + 1) mod DIAM = 0`.
//!
//! Total shared memory ≈ 12 KiB / block. RTX 5070 (128 KiB shared per
//! SM) → 10 blocks resident, well above the 4-block latency-hiding
//! floor.
//!
//! ## Algorithm
//!
//! Per block, per strip:
//!
//! 1. **Prefix init.** Cooperatively load `src_row` / `dst_row` for
//!    each of the `DIAM` prefix rows `mirror(y_start + k - R)` for
//!    `k ∈ [0, DIAM)`. Each thread sums its column's H-blur from
//!    shared rows, stores into `buf_*[k × TX + tx]`, and accumulates
//!    into the sliding sums `sum_m1 / sum_m2 / sum_sq / sum_s12`.
//!
//! 2. **Walk `y` from `y_start` to `y_end`:**
//!    a. Compute V-blur outputs `mu1 = sum_m1 / DIAM` etc.
//!    b. Read `sv = src[y, col]`, `dv = dst[y, col]` direct from DRAM.
//!    c. Run the SSIM / artifact / detail / HF / MSE math, accumulate
//!       into per-thread `a0..a16` (f64) and `peak0..peak2` (f32 max).
//!    d. **Slide:** read the slot's old H-blur values; cooperatively
//!       load `src_row` / `dst_row` for `mirror(y + R + 1)`; compute
//!       this thread's H-blur for the new row; update sliding sums
//!       (`sum += new − old`); write new H-blur to the same slot;
//!       advance `slot`.
//!
//! 3. **Write partials.** If `col < padded_w`, write the per-thread
//!    accumulator state into the shared `partials_f64` /
//!    `partials_max` buffers at slot
//!    `(channel × n_strips × pw + strip × pw + col)`.
//!
//! ## Mirror handling
//!
//! All mirroring (x and y) is inlined in u32 via the
//! `(idx + period - r) % period` → fold trick. Caller guarantees
//! `width ≥ R + 1` and `height ≥ R + 1`, true for zensim's smallest
//! scale (`min_dim = 8`).

use cubecl::prelude::*;

const TX: u32 = 64;
const TX_US: usize = 64;
const R: u32 = 5;
const DIAM: u32 = 11;
const TILE_COLS: u32 = TX + 2u32 * R;
const TILE_COLS_US: usize = (TX + 2u32 * R) as usize;
const BUF_LEN_US: usize = (DIAM * TX) as usize;

const C2: f32 = 0.0009;
const INV_DIAM: f32 = 1.0 / 11.0;

#[cube(launch_unchecked)]
pub fn fused_features_kernel(
    src_a: &Array<f32>,
    dst_a: &Array<f32>,
    src_b: &Array<f32>,
    dst_b: &Array<f32>,
    src_c: &Array<f32>,
    dst_c: &Array<f32>,
    partials_f64: &mut Array<f64>,
    partials_max: &mut Array<f32>,
    width: u32, // padded_w
    height: u32,
    n_strips: u32,
    slot_off_f64: u32,
    slot_off_max: u32,
) {
    let tx = UNIT_POS_X;
    let col_block = CUBE_POS_X;
    let strip = CUBE_POS_Y;
    let channel = CUBE_POS_Z;
    let col_base = col_block * TX;
    let col = col_base + tx;
    let in_bounds = col < width;

    let w = width as usize;
    let n_strips_us = n_strips as usize;
    let pw = width as usize;
    let period_x = 2u32 * (width - 1u32);
    let period_y = 2u32 * (height - 1u32);

    // Strip range [y_start, y_end). Last strip absorbs any remainder.
    let strip_h_base = height / n_strips;
    let strip_rem = height - strip_h_base * n_strips;
    let y_start = strip * strip_h_base + u32::min(strip, strip_rem);
    let y_end_unclamp = y_start + strip_h_base + (if strip < strip_rem { 1u32 } else { 0u32 });
    let y_end = u32::min(y_end_unclamp, height);

    // Shared memory.
    let mut src_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut dst_row = SharedMemory::<f32>::new(TILE_COLS_US);
    let mut buf_mu1 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_mu2 = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_sq = SharedMemory::<f32>::new(BUF_LEN_US);
    let mut buf_s12 = SharedMemory::<f32>::new(BUF_LEN_US);

    // Per-thread sliding sums + feature accumulators.
    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;

    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut a4 = 0.0_f64;
    let mut a5 = 0.0_f64;
    let mut a6 = 0.0_f64;
    let mut a7 = 0.0_f64;
    let mut a8 = 0.0_f64;
    let mut a9 = 0.0_f64;
    let mut a10 = 0.0_f64;
    let mut a11 = 0.0_f64;
    let mut a12 = 0.0_f64;
    let mut a13 = 0.0_f64;
    let mut a14 = 0.0_f64;
    let mut a15 = 0.0_f64;
    let mut a16 = 0.0_f64;
    let mut peak0 = 0.0_f32;
    let mut peak1 = 0.0_f32;
    let mut peak2 = 0.0_f32;

    // ============================ PREFIX INIT ============================
    // Compute H-blur for the diam prefix rows (window centered at
    // y_start), accumulate sliding sums, store H-blur in buf_*.
    let mut k: u32 = 0u32;
    while k < DIAM {
        // Mirror y_in for prefix row.
        let raw_y = (y_start + k + period_y - R) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        // Cooperative load: each thread loads up to ceil(TILE_COLS / TX) entries.
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            // Mirror x for column col_base - R + load_x.
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off = (y_in as usize) * w + (gx as usize);
            // Channel switch.
            let mut s_val = 0.0_f32;
            let mut d_val = 0.0_f32;
            if channel == 0u32 {
                s_val = src_a[off];
                d_val = dst_a[off];
            } else {
                if channel == 1u32 {
                    s_val = src_b[off];
                    d_val = dst_b[off];
                } else {
                    s_val = src_c[off];
                    d_val = dst_c[off];
                }
            }
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i = i + 1u32;
        }
        sync_cube();

        // Each thread computes H-blur for its column = col_base + tx
        // by summing src_row[tx..tx+DIAM] and dst_row[tx..tx+DIAM].
        let mut m1 = 0.0_f32;
        let mut m2 = 0.0_f32;
        let mut sq = 0.0_f32;
        let mut s12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            m1 += s;
            m2 += d;
            sq = fma(s, s, fma(d, d, sq));
            s12 = fma(s, d, s12);
            j += 1u32;
        }
        m1 = m1 * INV_DIAM;
        m2 = m2 * INV_DIAM;
        sq = sq * INV_DIAM;
        s12 = s12 * INV_DIAM;

        let buf_idx = (k * TX + tx) as usize;
        buf_mu1[buf_idx] = m1;
        buf_mu2[buf_idx] = m2;
        buf_sq[buf_idx] = sq;
        buf_s12[buf_idx] = s12;

        sum_m1 += m1;
        sum_m2 += m2;
        sum_sq += sq;
        sum_s12 += s12;

        k += 1u32;
    }

    // ============================ WALK Y ============================
    let mut slot: u32 = 0u32;
    let mut y: u32 = y_start;
    while y < y_end {
        // V-blur outputs.
        let mu1 = sum_m1 * INV_DIAM;
        let mu2 = sum_m2 * INV_DIAM;
        let ssq = sum_sq * INV_DIAM;
        let s12_v = sum_s12 * INV_DIAM;

        // Read sv, dv from DRAM (one value each).
        let off = (y as usize) * w + (col as usize);
        let mut sv: f32 = 0.0;
        let mut dv: f32 = 0.0;
        if in_bounds {
            if channel == 0u32 {
                sv = src_a[off];
                dv = dst_a[off];
            } else {
                if channel == 1u32 {
                    sv = src_b[off];
                    dv = dst_b[off];
                } else {
                    sv = src_c[off];
                    dv = dst_c[off];
                }
            }
        }

        // SSIMULACRA2-style SSIM (no C1, uses `1 - (mu1-mu2)²`). FMA
        // fusion order matches CPU `zensim::fused::fused_vblur_ssim_inner_v4`.
        let mu_diff = mu1 - mu2;
        let num_m = fma(mu_diff, -mu_diff, 1.0);
        let inner_ns = fma(-mu1, mu2, s12_v);
        let num_s = fma(2.0, inner_ns, C2);
        let inner_ds_inner = fma(-mu1, mu1, ssq);
        let denom_s = fma(-mu2, mu2, inner_ds_inner) + C2;
        let sd_raw = 1.0 - (num_m * num_s) / denom_s;
        let sd = if sd_raw > 0.0 { sd_raw } else { f32::new(0.0) };
        let sd2 = sd * sd;
        let sd4 = sd2 * sd2;
        a0 += sd as f64;
        a1 += sd4 as f64;
        a2 += sd2 as f64;
        a14 += (sd4 * sd4) as f64;
        if sd > peak0 {
            peak0 = sd;
        }

        let diff1 = f32::abs(sv - mu1);
        let diff2 = f32::abs(dv - mu2);
        let ed = (1.0 + diff2) / (1.0 + diff1) - 1.0;
        let artifact = if ed > 0.0 { ed } else { f32::new(0.0) };
        let detail_lost = if ed < 0.0 { -ed } else { f32::new(0.0) };
        let a2_v = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        let a4_v = a2_v * a2_v;
        let dl4 = dl2 * dl2;
        a3 += artifact as f64;
        a4 += a4_v as f64;
        a5 += a2_v as f64;
        a6 += detail_lost as f64;
        a7 += dl4 as f64;
        a8 += dl2 as f64;
        a15 += (a4_v * a4_v) as f64;
        a16 += (dl4 * dl4) as f64;
        if artifact > peak1 {
            peak1 = artifact;
        }
        if detail_lost > peak2 {
            peak2 = detail_lost;
        }

        let vs = sv - mu1;
        let vd = dv - mu2;
        a10 += (vs * vs) as f64;
        a11 += (vd * vd) as f64;
        a12 += diff1 as f64;
        a13 += diff2 as f64;

        let pd = sv - dv;
        a9 += (pd * pd) as f64;

        // Slide: subtract slot's old H-blur from sums, compute new
        // H-blur for row mirror(y + R + 1), add to sums, write to
        // the same slot (because (2R + 1) mod DIAM = 0).
        let buf_idx = (slot * TX + tx) as usize;
        let old_m1 = buf_mu1[buf_idx];
        let old_m2 = buf_mu2[buf_idx];
        let old_sq = buf_sq[buf_idx];
        let old_s12 = buf_s12[buf_idx];

        let raw_y = (y + R + 1u32 + period_y) % period_y;
        let y_in = if raw_y < height {
            raw_y
        } else {
            period_y - raw_y
        };

        sync_cube();
        let mut i: u32 = 0u32;
        while i * TX + tx < TILE_COLS {
            let load_x = i * TX + tx;
            let raw_x = (col_base + load_x + period_x - R) % period_x;
            let gx = if raw_x < width {
                raw_x
            } else {
                period_x - raw_x
            };
            let off2 = (y_in as usize) * w + (gx as usize);
            let mut s_val = 0.0_f32;
            let mut d_val = 0.0_f32;
            if channel == 0u32 {
                s_val = src_a[off2];
                d_val = dst_a[off2];
            } else {
                if channel == 1u32 {
                    s_val = src_b[off2];
                    d_val = dst_b[off2];
                } else {
                    s_val = src_c[off2];
                    d_val = dst_c[off2];
                }
            }
            src_row[load_x as usize] = s_val;
            dst_row[load_x as usize] = d_val;
            i = i + 1u32;
        }
        sync_cube();

        let mut nm1 = 0.0_f32;
        let mut nm2 = 0.0_f32;
        let mut nsq = 0.0_f32;
        let mut ns12 = 0.0_f32;
        let mut j: u32 = 0u32;
        while j < DIAM {
            let s = src_row[(tx + j) as usize];
            let d = dst_row[(tx + j) as usize];
            nm1 += s;
            nm2 += d;
            nsq = fma(s, s, fma(d, d, nsq));
            ns12 = fma(s, d, ns12);
            j += 1u32;
        }
        nm1 = nm1 * INV_DIAM;
        nm2 = nm2 * INV_DIAM;
        nsq = nsq * INV_DIAM;
        ns12 = ns12 * INV_DIAM;

        sum_m1 = sum_m1 + nm1 - old_m1;
        sum_m2 = sum_m2 + nm2 - old_m2;
        sum_sq = sum_sq + nsq - old_sq;
        sum_s12 = sum_s12 + ns12 - old_s12;

        buf_mu1[buf_idx] = nm1;
        buf_mu2[buf_idx] = nm2;
        buf_sq[buf_idx] = nsq;
        buf_s12[buf_idx] = ns12;

        slot = (slot + 1u32) % DIAM;
        y += 1u32;
    }

    // ============================ WRITE PARTIALS ============================
    if !in_bounds {
        terminate!();
    }
    let slot_idx_us =
        (channel as usize) * n_strips_us * pw + (strip as usize) * pw + (col as usize);
    let f64_base = (slot_off_f64 as usize) + slot_idx_us * 17;
    partials_f64[f64_base] = a0;
    partials_f64[f64_base + 1] = a1;
    partials_f64[f64_base + 2] = a2;
    partials_f64[f64_base + 3] = a3;
    partials_f64[f64_base + 4] = a4;
    partials_f64[f64_base + 5] = a5;
    partials_f64[f64_base + 6] = a6;
    partials_f64[f64_base + 7] = a7;
    partials_f64[f64_base + 8] = a8;
    partials_f64[f64_base + 9] = a9;
    partials_f64[f64_base + 10] = a10;
    partials_f64[f64_base + 11] = a11;
    partials_f64[f64_base + 12] = a12;
    partials_f64[f64_base + 13] = a13;
    partials_f64[f64_base + 14] = a14;
    partials_f64[f64_base + 15] = a15;
    partials_f64[f64_base + 16] = a16;
    let max_base = (slot_off_max as usize) + slot_idx_us * 3;
    partials_max[max_base] = peak0;
    partials_max[max_base + 1] = peak1;
    partials_max[max_base + 2] = peak2;
}
