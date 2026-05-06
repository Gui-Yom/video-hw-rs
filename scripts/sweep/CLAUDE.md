# turbo-metrics sweep operations

Captures operational lessons for running zen-metrics sweeps on vast.ai
fleets. See parent CLAUDE.md / README.md for codebase orientation.

## Key files

- `onstart_v3.sh` — main worker entrypoint, static-binary, cgroup-aware,
  invoked by vast.ai via `--onstart-cmd`.
- `generate_jobspecs_v06.py` — chunk generator (image × knob_combo grid
  → `chunks.jsonl` uploaded to `s3://coefficient/jobs/<sweep>/`).
- `vastai_zen_metrics_sweep.sh` — original v04 launcher (see `/tmp/launch_v0X.sh`
  for newer per-sweep variants).
- `Dockerfile.sweep` — historical; v3 onstart bypasses docker, runs
  directly on `ubuntu:24.04`.

## Worker mechanics

1. vast.ai pulls `ubuntu:24.04` (~10 s on warm host).
2. `--onstart-cmd` runs `onstart_v3.sh` which:
   - Imports env from `/proc/1/environ` (filters R2_*, SWEEP_*, WORKER_*, STATS_*).
   - Downloads static `s5cmd` + `jq` to `/usr/local/bin`.
   - Either downloads a `zen-metrics-vX.Y.Z.tar.gz` release tarball OR
     pulls a binary blob via `SWEEP_BIN_OVERRIDE` (s3://path).
   - Writes a heartbeat to `s3://coefficient/heartbeats/<sweep>/<worker>.json`.
   - Reads `chunks.jsonl` from `s3://coefficient/jobs/<sweep>/`.
   - Loops: claim a chunk (atomic via `mc cp` of a `.claim` file),
     download source images, run `zen-metrics --batch`, upload TSV +
     features parquet.

## CRITICAL: cgroup-aware parallelism (2026-05-04 fix)

`nproc` inside a vast.ai container reports the **host** CPU count
(often 56) — NOT the container's effective cgroup allocation (usually
8–16). Setting `xargs -P $(nproc)` oversubscribes and thrashes.

Fix lives in `onstart_v3.sh::cores_from_cgroup()` and `ram_gb_from_cgroup()`.
Both cgroup v1 and v2 are handled. Final formula:

```
PARALLEL = min(cgroup_cores, ram_gb * 2/3) - 2
```

Subtract 2 cores for the heartbeat loop + system overhead. RAM cap
keeps each parallel slot ≤1.5 GB to avoid OOM on lossy encodes (jxl
encoder spikes to ~1 GB on 1024×1024 images).

**Result:** 3-5× throughput improvement on multi-core boxes (2026-05-04
empirical, on v06 sweep). Without this fix, a 56-core/24GB-RAM box
was running 56-way parallel and OOM-thrashing; now runs ~14-way and
finishes 4× faster.

## CRITICAL: vast.ai filter quirks

**`cpu_ram` is in GB, not MB** despite API responses giving MB.
`cpu_ram>16` means >16 GB RAM. The `ram` field doesn't filter at all
on this CLI version (1.0.8); use `cpu_ram`.

**`verified=true` excludes most cheap offers** — drop it for sweep work.
We don't need datacenter-class boxes; cheap consumer GPUs are fine.

**`--raw -o dph_total` conflicts.** `-o` is for the columnar (non-raw)
output; combining with `--raw` produces non-JSON. Use:
```
vastai search offers '...' --order dph_total --raw
```

**Required RAM has trended up** with more knobs sweeping. v06 worked at
12 GB; v07/v08 with patches/gaborish need 24+ GB; v09 with
force_strategy needs 32 GB on multi-MB images. Set `MIN_RAM_GB=24` in
launchers as the default; bump to 32 for v09+.

## Atomic chunk claim mechanism

Each worker, before processing a chunk, attempts:
```
mc cp /tmp/claim-<chunkid>.txt s3://coefficient/claims/<sweep>/<chunkid>.claim
```

If the claim file already exists, `mc cp` fails (S3 If-None-Match
emulation via mc) and the worker moves on. If it succeeds, the worker
processes the chunk.

**FAILURE MODE: orphaned claims.** If a worker crashes mid-chunk, its
claim file remains. No other worker will pick up that chunk. Symptoms:
chunks stop landing on R2 even though instances are "running".

**Recovery:**
```bash
# Cleanest: just delete all claims for the sweep. The done-chunk skip
# logic checks .tsv existence, NOT claim existence — so workers will
# re-iterate the chunk list, skip done chunks (via .tsv check), and
# pick up only the unfinished ones.
aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    s3 rm s3://coefficient/claims/<sweep>/ --recursive
```

We did this 2026-05-06 across v08/v09/v11 (140 stale claims) after
killing 192 idle workers — the surviving 7 then resumed picking up
the last few chunks.

## Per-chunk runtime is HUGE

Each chunk processes 5 images × ~500 knob combos × encode + 4 metrics =
roughly **3-5 hours per chunk**. v08 worker logs showed individual
chunks taking 10,000-18,000 seconds.

Implications:
- A worker crash mid-chunk loses ~3 hours of work.
- "Stalled" sweeps may not be stalled — just slow.
- Don't kill workers preemptively. Check the heartbeat (1-min interval)
  before assuming death.

## Per-sweep instance tracking — IMPORTANT

The original launcher wrote `/tmp/v06_binary_instances.txt` and
overwrote it on every relaunch, regardless of run_id. **Multiple
agents launched in series clobbered each other's tracking and led to
killing the wrong workers.**

Current launcher pattern uses per-sweep file:
```bash
INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"  # truncate per run
echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
```

Always check `/tmp/<sweep>_instances.txt` before bulk-killing. If
unsure, query vast.ai labels and filter by sweep prefix.

## Cost control

Idle "running" instances burn ~$0.04-0.20/hr each. For 200 instances
that's ~$20/hr = $480/day even if no chunks are processing.

Burn-rate audit recipe:
```bash
vastai show instances --raw | python3 -c "
import json,sys
d=json.loads(sys.stdin.read())
insts = d if isinstance(d, list) else d.get('instances', [])
total = sum(i.get('dph_total') or 0 for i in insts)
print(f'instances: {len(insts)}, burn: \${total:.2f}/hr')"
```

After clearing claims, you only need ~1 worker per outstanding chunk
(plus a small buffer). 8 workers for 7 chunks is plenty; the extras
are pure waste once the chunk count drops.

Mass-destroy pattern (parallel via xargs):
```bash
echo "$KILL_IDS" | xargs -P 20 -I {} sh -c 'echo y | vastai destroy instance {} >/dev/null 2>&1; echo killed {}'
```

`vastai destroy instance <id>` requires `y` confirmation; pipe `echo y |`.

## Decoder/encoder bug hunting from sweep data

Sweep workers occasionally fail with encoder hangs or decoder rejects.
Examples surfaced in this fleet:

- **jxl-encoder e9 checker pattern hang** (issue #27, PR #28 in jxl-encoder repo).
- **zenjxl-decoder rejecting files libjxl C accepts** — confirmed via
  `djxl 0.10.3` as the authority. **libjxl is the authority, not
  jxl-oxide.** zenjxl-decoder is a fork of jxl-rs; both inherit the
  same VarDCT-strict-validation lineage and reject some files libjxl C
  accepts. When triaging, always test with `djxl` directly.

## Sweep status, May 2026

| Sweep | Date | Knobs | Chunks (R2) | Status |
|---|---|---|---|---|
| v06 | 2026-05-04 | distance, effort, biters, ziters | ~200 | DONE — 165k cells |
| v07 | 2026-05-05 | + lz77, force_strategy, gaborish, patches, lf_frame, pdl, progressive | 34 | DONE — 32k cells |
| v08 | 2026-05-05 | v06 grid + v07 winners (patches, gaborish, pdl) | 98/100 | 98% complete (2 stragglers) |
| v09 | 2026-05-05 | force_strategy + max_strategy_size + mode | 17/21 | 81% complete |
| v10 | 2026-05-05 | multi-codec (zenjxl/avif/webp) | 60/60 | DONE |
| v11 | 2026-05-05 | per-distance-band specialist | 19/20 | 95% complete |

All sweeps mirrored to R2 at `s3://zentrain/sweep-vXX-YYYY-MM-DD/`.

## Where reports live

- Per-sweep picker reports: `~/work/zen/zenanalyze/benchmarks/picker_v0X_*.md`
- Encoder default decisions: `~/work/zen/zenjxl/benchmarks/INDEX.md`
- Local sweep TSVs + analysis scripts: `~/sweep-data/`
