#!/usr/bin/env bash
# v3 onstart: zero apt-get. Static binaries only.
#
# Replaces awscli + jq + vulkan with:
#   - s5cmd  (single static binary, S3-compatible, ~10 MB)
#   - jq is shipped statically by the GitHub maintainers (~5 MB)
#   - no vulkan (we run --gpu-runtime=cpu by default)
#
# Boot time on a fresh ubuntu:24.04 box:
#   v2 (with apt-get): 3-8 minutes (apt-get update is the bottleneck)
#   v3 (static-only):  10-30 seconds
#
# Required env (passed via vast.ai --env, propagates from PID 1):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID
# Optional:
#   WORKER_ID            default: hostname-pid
#   WORKER_PARALLEL      default: max(2, nproc/4)
#   SWEEP_BIN_VERSION    default: zen-metrics-v0.3.0
#   SWEEP_GPU_RUNTIME    default: cpu  (no vulkan installed)
#   STATS_INTERVAL_SEC   default: 60
#   SWEEP_BIN_OVERRIDE   optional URL or s3:// to a zen-metrics tarball
#                        (skips the GitHub release fetch; use for unreleased builds)

set -euo pipefail

# Pull env from PID 1 (vast.ai --env values land there). The bash that
# invokes onstart sometimes inherits, sometimes not — explicitly
# import to be safe.
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|STATS_*) export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi

SWEEP_BIN_VERSION="${SWEEP_BIN_VERSION:-zen-metrics-v0.3.0}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-v04-2026-05-04}"
WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
WORKDIR="${WORKDIR:-/workspace/sweep}"
GPU_RUNTIME="${SWEEP_GPU_RUNTIME:-cpu}"
STATS_INTERVAL_SEC="${STATS_INTERVAL_SEC:-60}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

PARALLEL="${WORKER_PARALLEL:-}"
if [[ -z "$PARALLEL" ]]; then
    # `nproc` inside a vast.ai container reports the HOST's CPU count
    # (often 56) not the container's actual cgroup allocation (usually
    # 8-16). Read the cgroup limit so we don't oversubscribe and thrash.
    cores_from_cgroup() {
        # cgroup v2: cpu.max is "<quota_us> <period_us>" or "max <period>"
        if [[ -r /sys/fs/cgroup/cpu.max ]]; then
            read q p < /sys/fs/cgroup/cpu.max
            [[ "$q" == "max" || -z "$q" ]] && return 1
            echo $(( (q + p / 2) / p )); return 0
        fi
        # cgroup v1
        if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
            local q p
            q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
            p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
            (( q > 0 && p > 0 )) && { echo $(( (q + p / 2) / p )); return 0; }
        fi
        return 1
    }
    ram_gb_from_cgroup() {
        if [[ -r /sys/fs/cgroup/memory.max ]]; then
            local m; m=$(cat /sys/fs/cgroup/memory.max)
            [[ "$m" == "max" ]] && return 1
            echo $(( m / 1024 / 1024 / 1024 )); return 0
        fi
        if [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
            local m; m=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
            (( m > 0 && m < 1099511627776 )) && { echo $(( m / 1024 / 1024 / 1024 )); return 0; }
        fi
        return 1
    }
    nc=$(cores_from_cgroup) || nc=$(nproc 2>/dev/null || echo 8)
    # RAM-based cap: each parallel slot needs ~1.5 GB peak (encoder + 3 metrics).
    # If RAM is tighter than CPU, RAM wins.
    if rg=$(ram_gb_from_cgroup); then
        ram_slots=$(( rg * 2 / 3 ))  # 1.5 GB / slot
        (( ram_slots < nc )) && nc=$ram_slots
    fi
    PARALLEL=$(( nc > 6 ? nc - 2 : (nc > 2 ? nc - 1 : 2) ))
fi

log() { printf '[onstart-v3 %s %s] %s\n' "$(date -u +%H:%M:%S)" "$WORKER_ID" "$*"; }

trap 'final_state failed "$LINENO" "$BASH_COMMAND"' ERR

final_state() {
    local kind="${1:-done}"; local lineno="${2:-}"; local cmd="${3:-}"
    local stamp; stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"worker":"%s","stamp":"%s","kind":"%s","line":"%s","cmd":"%s"}\n' \
        "$WORKER_ID" "$stamp" "$kind" "$lineno" "$cmd" \
        > /tmp/final.json
    R2 cp /tmp/final.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${kind}" \
        2>/dev/null || true
    log "FINAL: $kind"
    exit 0
}

# ── Required env check (early fail) ────────────────────────────────
: "${R2_ACCOUNT_ID:?must set R2_ACCOUNT_ID via --env}"
: "${R2_ACCESS_KEY_ID:?must set R2_ACCESS_KEY_ID via --env}"
: "${R2_SECRET_ACCESS_KEY:?must set R2_SECRET_ACCESS_KEY via --env}"

# ── Step 1: install s5cmd + jq + minio mc (for atomic claim) statically ──
mkdir -p /usr/local/bin
if [[ ! -x /usr/local/bin/s5cmd ]]; then
    log "installing s5cmd (static)"
    curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" \
        -o /tmp/s5cmd.tgz
    tar xzf /tmp/s5cmd.tgz -C /usr/local/bin s5cmd
    chmod +x /usr/local/bin/s5cmd
    rm /tmp/s5cmd.tgz
fi
if ! command -v jq >/dev/null 2>&1; then
    log "installing jq (static)"
    curl -fsSL -o /usr/local/bin/jq \
        "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64"
    chmod +x /usr/local/bin/jq
fi
# minio mc — single static binary, supports --put with conditional retain
# (and HEAD-then-PUT atomic semantics via lock).
if [[ ! -x /usr/local/bin/mc ]]; then
    log "installing minio mc (static, for atomic claim)"
    curl -fsSL -o /usr/local/bin/mc \
        "https://dl.min.io/client/mc/release/linux-amd64/mc"
    chmod +x /usr/local/bin/mc
fi
# CA certs are in the base ubuntu:24.04 image already.

# R2 wrapper using s5cmd
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() {
    s5cmd --endpoint-url "$R2_ENDPOINT" "$@"
}

heartbeat() {
    local note="$1"
    local stamp; stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"worker":"%s","stamp":"%s","note":"%s","parallel":%d,"runtime":"%s"}\n' \
        "$WORKER_ID" "$stamp" "$note" "$PARALLEL" "$GPU_RUNTIME" \
        > /tmp/heartbeat.json
    R2 cp /tmp/heartbeat.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.json" \
        2>/dev/null || true
}

# ── Step 2: pre-built zen-metrics binary ───────────────────────────
BIN="$WORKDIR/zen-metrics"
if [[ ! -x "$BIN" ]]; then
    arch=$(uname -m)
    case "$arch" in
        x86_64)  pkg="zen-metrics-${SWEEP_BIN_VERSION#zen-metrics-v}-linux-x86_64.tar.gz" ;;
        aarch64) pkg="zen-metrics-${SWEEP_BIN_VERSION#zen-metrics-v}-linux-aarch64.tar.gz" ;;
        *) log "FATAL: unsupported arch $arch"; final_state failed ;;
    esac
    URL="${SWEEP_BIN_OVERRIDE:-https://github.com/imazen/turbo-metrics/releases/download/${SWEEP_BIN_VERSION}/${pkg}}"
    if [[ "$URL" == s3://* ]]; then
        log "fetching $URL via s5cmd"
        R2 cp "$URL" /tmp/zm.bin
        cp /tmp/zm.bin "$BIN"
        chmod +x "$BIN"
    elif [[ "$URL" == *.tar.gz || "$URL" == *.tgz ]]; then
        log "downloading $URL"
        curl -fsSL "$URL" -o /tmp/zm.tgz
        tar xzf /tmp/zm.tgz -C "$WORKDIR" --strip-components=1 \
            "$(tar tzf /tmp/zm.tgz | grep -E '/zen-metrics$' | head -1)"
        chmod +x "$BIN"
    else
        log "downloading raw binary $URL"
        curl -fsSL "$URL" -o "$BIN"
        chmod +x "$BIN"
    fi
fi
"$BIN" --version
heartbeat "starting"

# ── Step 3: stats reporter ─────────────────────────────────────────
stats_loop() {
    local started; started=$(date +%s)
    local stats_file=/tmp/stats.tsv
    printf 'ts_utc\tload1\tcpu_pct\tmem_used_mib\tmem_total_mib\tgpu_util\trows_done\twall_min\n' > "$stats_file"
    while true; do
        local now ts load1 cpu_pct mem_used mem_total gpu_util rows wall
        now=$(date +%s)
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
        cpu_pct=$(top -bn1 -d 0.5 2>/dev/null | awk '/Cpu/ { gsub(/,/, ""); print 100 - $8 }' | head -1 | awk '{printf "%.1f", $1}')
        [[ -z "$cpu_pct" ]] && cpu_pct=0
        mem_used=$(free -m 2>/dev/null | awk '/^Mem:/ {print $3}')
        mem_total=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}')
        gpu_util=0
        if command -v nvidia-smi >/dev/null 2>&1; then
            gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            [[ -n "$gu" ]] && gpu_util="$gu"
        fi
        rows=$(cat /tmp/rows_done 2>/dev/null || echo 0)
        wall=$(awk -v s=$started -v n=$now 'BEGIN{printf "%.1f", (n-s)/60.0}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$ts" "$load1" "$cpu_pct" "$mem_used" "$mem_total" "$gpu_util" "$rows" "$wall" \
            >> "$stats_file"
        R2 cp "$stats_file" "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/stats/${WORKER_ID}.tsv" 2>/dev/null || true
        sleep "$STATS_INTERVAL_SEC"
    done
}
stats_loop > /tmp/stats.log 2>&1 &
log "stats_loop pid=$! interval=${STATS_INTERVAL_SEC}s"

# ── Step 4: source mirror ──────────────────────────────────────────
SOURCES_DIR="$WORKDIR/sources"
if [[ ! -d "$SOURCES_DIR" ]] || [[ -z "$(find "$SOURCES_DIR" -type f -print -quit 2>/dev/null)" ]]; then
    log "syncing sources from R2"
    mkdir -p "$SOURCES_DIR"
    R2 sync "s3://zentrain/${SWEEP_RUN_ID}/sources/*" "$SOURCES_DIR/"
fi
SRC_COUNT=$(find "$SOURCES_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
log "have $SRC_COUNT source images, parallelism=$PARALLEL"

# ── Step 5: chunks list ────────────────────────────────────────────
CHUNK_FILE=/tmp/chunks.jsonl
R2 cp "s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl" "$CHUNK_FILE"
log "loaded $(wc -l < $CHUNK_FILE) chunks"

# ── Step 6: chunk processor with atomic claim ──────────────────────
echo 0 > /tmp/rows_done
process_chunk() {
    local line="$1"
    local codec chunk_id q_grid knob_grid metrics_args
    codec=$(printf '%s' "$line" | jq -r '.codec')
    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    q_grid=$(printf '%s' "$line" | jq -r '.q_grid')
    knob_grid=$(printf '%s' "$line" | jq -r '.knob_grid')
    metrics_args=$(printf '%s' "$line" | jq -r '.metrics | map("--metric " + .) | join(" ")')

    local OUT_KEY="s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv"
    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Quick skip if uploaded
    if R2 ls "$OUT_KEY" 2>/dev/null | grep -q "$chunk_id.tsv"; then
        echo "[skip] $chunk_id already done"
        return 0
    fi

    # Best-effort claim — write to claim key with worker info. Race window
    # exists but shuffle-ordering reduces collisions. s5cmd doesn't support
    # if-none-match natively; we accept some duplicate work.
    local claim_body=/tmp/claim-${chunk_id}.txt
    printf '{"worker":"%s","stamp":"%s"}' "$WORKER_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$claim_body"
    R2 cp "$claim_body" "$CLAIM_KEY" 2>/dev/null || true

    # Re-check post-"claim"
    if R2 ls "$OUT_KEY" 2>/dev/null | grep -q "$chunk_id.tsv"; then
        echo "[skip-after-claim] $chunk_id"
        return 0
    fi

    local STAGE="$WORKDIR/stage-${chunk_id}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    printf '%s' "$line" | jq -r '.images[]' | while IFS= read -r relpath; do
        [[ -z "$relpath" ]] && continue
        flat="${relpath//\//__}"
        ln -sf "$SOURCES_DIR/$relpath" "$STAGE/$flat" 2>/dev/null || true
    done

    local OUT_TSV="$WORKDIR/out-${chunk_id}.tsv"
    # Side-channel parquet of zensim's 300-feature extended vectors per cell.
    # Joins back to the TSV by (image_path, codec, q, knob_tuple_json).
    # Captured per chunk; finalize step concatenates across workers.
    local OUT_PARQUET="$WORKDIR/features-${chunk_id}.parquet"
    local FEATURES_KEY="${OUT_KEY%.tsv}.features.parquet"
    local PARTIAL_KEY="s3://coefficient/partials/${SWEEP_RUN_ID}/${codec}/${chunk_id}.partial.tsv"

    # Mid-chunk flush sidecar: every 60s while zen-metrics is encoding, copy
    # the in-progress TSV to a partial key. On normal completion we delete
    # the partial after the final upload lands; on crash/kill the partial
    # is what survives. Only the TSV is flushed — parquet is written by
    # zen-metrics atomically at end-of-run, no streaming hook available.
    local FLUSH_INTERVAL=60
    local start_t; start_t=$(date +%s)
    (
        while sleep "$FLUSH_INTERVAL"; do
            [[ -f "$OUT_TSV" ]] || continue
            local rows; rows=$(($(wc -l < "$OUT_TSV" 2>/dev/null) - 1))
            (( rows > 0 )) || continue
            R2 cp "$OUT_TSV" "$PARTIAL_KEY" 2>/dev/null || true
            log "[flush] $chunk_id ${rows}rows partial=${PARTIAL_KEY##*/}"
        done
    ) &
    local FLUSH_PID=$!

    if "$BIN" sweep \
        --codec "$codec" \
        --sources "$STAGE" \
        --q-grid "$q_grid" \
        --knob-grid "$knob_grid" \
        $metrics_args \
        --gpu-runtime "$GPU_RUNTIME" \
        --output "$OUT_TSV" \
        --feature-output "$OUT_PARQUET" \
        > "/tmp/sweep-${chunk_id}.log" 2>&1
    then
        # Stop sidecar before final upload to avoid racing with successful path.
        kill "$FLUSH_PID" 2>/dev/null; wait "$FLUSH_PID" 2>/dev/null || true
        local elapsed=$(( $(date +%s) - start_t ))
        local rows
        rows=$(($(wc -l < "$OUT_TSV") - 1))
        R2 cp "$OUT_TSV" "$OUT_KEY"
        if [[ -f "$OUT_PARQUET" ]]; then
            R2 cp "$OUT_PARQUET" "$FEATURES_KEY" || true
        fi
        # Final upload landed — partial is now redundant; remove it.
        R2 rm "$PARTIAL_KEY" 2>/dev/null || true
        ( flock -x 200; echo $(( $(cat /tmp/rows_done) + rows )) > /tmp/rows_done ) 200>/tmp/rows_done.lock
        echo "[done] $chunk_id ${elapsed}s ${rows}rows"
    else
        kill "$FLUSH_PID" 2>/dev/null; wait "$FLUSH_PID" 2>/dev/null || true
        echo "[fail] $chunk_id (see /tmp/sweep-${chunk_id}.log)"
        # Best-effort one last partial flush — captures whatever zen-metrics
        # wrote before exiting, even if the FLUSH_INTERVAL hadn't fired yet.
        if [[ -f "$OUT_TSV" && $(wc -l < "$OUT_TSV") -gt 1 ]]; then
            R2 cp "$OUT_TSV" "$PARTIAL_KEY" 2>/dev/null || true
        fi
        R2 cp "/tmp/sweep-${chunk_id}.log" \
            "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/errors/${chunk_id}.log" 2>/dev/null || true
    fi
    rm -rf "$STAGE" "$OUT_TSV" "$OUT_PARQUET"
}
export -f R2 process_chunk
export BIN R2_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export SWEEP_RUN_ID WORKDIR SOURCES_DIR WORKER_ID GPU_RUNTIME

# Shuffle deterministically per worker
SEED=$(echo -n "$WORKER_ID" | md5sum | cut -c1-16)
shuf < "$CHUNK_FILE" \
    --random-source=<(yes "$SEED" | tr -d '\n' | head -c 32768) 2>/dev/null \
    > "$WORKDIR/chunks_shuf.jsonl" || cp "$CHUNK_FILE" "$WORKDIR/chunks_shuf.jsonl"

heartbeat "running"
log "starting xargs -P $PARALLEL"
< "$WORKDIR/chunks_shuf.jsonl" xargs -d '\n' -I {} -P "$PARALLEL" \
    bash -c 'process_chunk "$@"' _ {}

heartbeat "exhausted-chunks"
log "all chunks done; rows=$(cat /tmp/rows_done 2>/dev/null || echo 0)"
final_state done
