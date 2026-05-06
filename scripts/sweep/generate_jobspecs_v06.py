#!/usr/bin/env python3
"""v06 sweep grid: expanded JXL knob set for picker retraining.

Changes from v05c:
- JXL: adds `butteraugli_iters` ∈ {0, 1, 2} as a third axis. The grid is
  effort × distance × butteraugli_iters = 4 × 22 × 3 = 264 cells per image.
  This decouples the encoder's iterative metric refinement from the
  macro-knob `effort` so the picker can choose "effort=5 + 2 butteraugli
  iters" cells that the v05c grid couldn't represent.

  Rationale: local validation showed cells like (e=5, biters=2) beat
  (e=9, biters=4 implicit) on bytes AND quality AND speed for
  representative content. The picker can't pick them if they're not in
  the data.

- AVIF: unchanged from v05c (already shipped with -6.70% bytes win).

- WebP: unchanged from v05c.

Metrics: zensim + butteraugli (max + pnorm3 from one compute) + ssim2_gpu.
Adding butteraugli is critical for testing the "wrong metric" hypothesis —
training a picker against butteraugli_max may produce a different selection
profile than training against zensim.

Wall-time estimate (per image, JXL only):
  264 cells × ~600ms median encode = ~160s/image
  × ~700 images = ~31 hours single-thread
  At 8 parallel workers per box × 4 boxes = ~1 hour wall-clock.
  At ~$0.30/hr/box that's ~$1.20 for the JXL portion.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Same q grid as v05c — quality is dummy for JXL (distance overrides) but
# kept for AVIF / WebP consistency in the same chunks.
Q_GRID = "5,15,25,35,45,55,65,75,85,95"

KNOB_GRIDS = {
    "zenwebp": json.dumps({
        "method": [4, 6],
    }),
    "zenavif": json.dumps({
        "speed": [3, 5, 7, 9],
        "complex_prediction_modes": [False, True],
    }),
    # JXL with the new butteraugli_iters axis. Distances pick representative
    # tight/mid/loose values from v05c's 22-distance grid; the trained picker
    # can interpolate between them via the scalar distance head.
    "zenjxl": json.dumps({
        "effort": [3, 5, 7, 9],
        "distance": [0.05, 0.1, 0.2, 0.3, 0.5, 0.75,
                     1.0, 1.25, 1.5, 2.0, 2.5,
                     3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
        "butteraugli_iters": [0, 1, 2],
    }),
}

# Metric set: add butteraugli (both columns from one compute)
METRICS = ["zensim", "ssim2_gpu", "butteraugli"]

CHUNK_SIZE = 2  # smaller than v05c (50), v06 (25), v07-v11 (5) — pairs with
                 # onstart_v3.sh's mid-chunk partial-flush sidecar. ~25-30 min
                 # per chunk wall-time so a worker crash loses ≤1 chunk + ≤60s
                 # of in-progress rows. Tradeoff: ~12× more S3 ops than
                 # CHUNK_SIZE=25, still well below R2 op caps (R2 free tier is
                 # 1M class-A ops/mo; even at 200 workers × 12× we're <100k/day).


def main():
    sources_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/lilith/work/zentrain-corpus/mlp-tune-fast")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/chunks_v06.jsonl")

    images = []
    for ext in ("png", "jpg", "jpeg"):
        images.extend(sources_root.rglob(f"*.{ext}"))
    images.sort()
    rels = [str(p.relative_to(sources_root)) for p in images]
    print(f"# {len(rels)} source images", file=sys.stderr)

    with out_path.open("w") as f:
        for codec, knob_grid in KNOB_GRIDS.items():
            for i in range(0, len(rels), CHUNK_SIZE):
                chunk_imgs = rels[i:i+CHUNK_SIZE]
                chunk_id = f"{codec}-{i//CHUNK_SIZE:03d}"
                spec = {
                    "codec": codec,
                    "chunk_id": chunk_id,
                    "q_grid": Q_GRID,
                    "knob_grid": knob_grid,
                    "metrics": METRICS,
                    "images": chunk_imgs,
                }
                f.write(json.dumps(spec))
                f.write("\n")

    n_chunks = sum(
        (len(rels) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for _ in KNOB_GRIDS
    )
    print(f"# wrote {n_chunks} chunks to {out_path}", file=sys.stderr)
    print(f"# Q grid: {Q_GRID}", file=sys.stderr)
    print(f"# Knob grids:", file=sys.stderr)
    for c, k in KNOB_GRIDS.items():
        # Estimate cells per image
        knobs = json.loads(k)
        n = 1
        for v in knobs.values():
            n *= len(v) if isinstance(v, list) else 1
        # JXL has dummy q so the q-grid doesn't multiply
        if c == "zenjxl":
            n_per = n
        else:
            n_per = n * len(Q_GRID.split(","))
        print(f"#   {c}: {n_per} cells/image, grid={k}", file=sys.stderr)
    print(f"# Metrics: {METRICS}", file=sys.stderr)


if __name__ == "__main__":
    main()
