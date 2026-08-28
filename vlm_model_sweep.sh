#!/usr/bin/env bash
# vlm_model_sweep.sh: the same crops judged by several models, one after the
# other. One export, then per model: start a llama-server, judge, evaluate,
# stop the server. The GPU holds one model at a time, so the prod service
# must be stopped first:
#
#   sudo systemctl stop llama-server
#   RES_CSV=$SLADD_VALIDATION/<run>/resultat.csv ./vlm_model_sweep.sh
#   sudo systemctl start llama-server
#
# Scope is EVERY kilde-«yolo» box in the uttrekk (all covering + all BOM),
# the stratum the verifier judges in prod. That is hours of GPU per model;
# the script prints the count and an estimate after the export. HIT_SAMPLE=N
# shrinks the covering side for a cheaper first ranking.
#
# A model that finished is skipped on a re-run, so a sweep killed halfway
# carries on. Delete <out-dir>/.ferdig to force one to run again. A model
# whose GGUF is not on disk is skipped with a note, so a partially
# downloaded MODELS list still runs.
#
# The prompt is the same for every model and is tuned on qwen3.8. A model
# that scores badly may just want its own prompt round; this sweep ranks
# them under OUR prompt, which is the deployment question.
#
# Variables: RES_CSV (required), UTTREKK=4, HIT_SAMPLE=-1 (all), OUT_ROOT,
# PROMPT_FILE, GEOM_FLAGS, PORT=8090, CONCURRENT=4, WORKERS=8, SEED=42,
# LLAMA_BIN, SERVER_FLAGS, VRAM_IDLE_MB=1500.

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run: source activate.sh"
    exit 1
fi

RES_CSV="${RES_CSV:?ERROR: set RES_CSV to the resultat.csv of the model run}"
UTTREKK="${UTTREKK:-4}"
HIT_SAMPLE="${HIT_SAMPLE:--1}"
SEED="${SEED:-42}"
CONCURRENT="${CONCURRENT:-4}"
WORKERS="${WORKERS:-8}"
PORT="${PORT:-8090}"
PYTHON=("${PYTHON:-python}" -u)

FOLDER="${FOLDER:-$SLADD_UTTREKK/uttrekk_$UTTREKK}"
TRUTH_CSV="${TRUTH_CSV:-$SLADD_LABELS/uttrekk_$UTTREKK.csv}"
OCR_CACHE="${OCR_CACHE:-$SLADD_CACHE/uttrekk_$UTTREKK/ocr}"
OUT_ROOT="${OUT_ROOT:-/data2/vlm/uttrekk${UTTREKK}_modeller}"
PROMPT_FILE="${PROMPT_FILE:-$SLADD_REPO/prompts/femsiffer_uten_liste.txt}"
# The winning geometry from the box rounds: up 100, full width, 1600 px cap.
GEOM_FLAGS="${GEOM_FLAGS:---margin-up 100 --full-width --margin-down 60 --max-px 1600}"

LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/bin/llama-server}"
# Same flags as the prod unit. Thinking MUST stay off: a model that spends
# the 150 answer tokens on a think block parses as «ja» and looks fine.
SERVER_FLAGS="${SERVER_FLAGS:---ctx-size 9216 --parallel 3 -ngl 99 --flash-attn auto --reasoning off --reasoning-budget 0 --no-webui}"
VRAM_IDLE_MB="${VRAM_IDLE_MB:-1500}"

# name|gguf|mmproj|extra server flags. The name becomes the output directory
# and the cache fingerprint, so keep it stable across runs. Files that are
# missing are skipped, download the ones to test first (Q4_K_M fits the
# V100S alongside nothing else):
#   nemotron-3-nano-omni  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
#   gemma-3-27b           unsloth/gemma-3-27b-it-GGUF (+ mmproj)
#   mistral-small-3.2     unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF (+ mmproj)
#   minicpm-v-4.5         openbmb/MiniCPM-V-4_5-gguf (+ mmproj)
#   internvl3.5-14b       unsloth/InternVL3_5-14B-GGUF (+ mmproj)
MODELS=(
    "qwen3.8-27b|/data2/llama/qwen3.8-27b.gguf|/data2/llama/qwen3.8-27b-mmproj.gguf|"
    "nemotron-3-nano-omni|/data2/llama/nemotron-3-nano-omni-30b-a3b-q4_k_m.gguf|/data2/llama/nemotron-3-nano-omni-mmproj-f16.gguf|"
    "gemma-3-27b|/data2/llama/gemma-3-27b-it-q4_k_m.gguf|/data2/llama/gemma-3-27b-it-mmproj-f16.gguf|"
    "mistral-small-3.2|/data2/llama/mistral-small-3.2-24b-q4_k_m.gguf|/data2/llama/mistral-small-3.2-24b-mmproj-f16.gguf|"
    "minicpm-v-4.5|/data2/llama/minicpm-v-4.5-q4_k_m.gguf|/data2/llama/minicpm-v-4.5-mmproj-f16.gguf|"
    "internvl3.5-14b|/data2/llama/internvl3.5-14b-q4_k_m.gguf|/data2/llama/internvl3.5-14b-mmproj-f16.gguf|"
)

# ── Preflight ────────────────────────────────────────────────────
for f in "$RES_CSV" "$TRUTH_CSV" "$PROMPT_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: no such file: $f"; exit 1; }
done
[[ -d "$FOLDER" ]] || { echo "ERROR: no PDF directory: $FOLDER"; exit 1; }
[[ -d "$OCR_CACHE" ]] || { echo "ERROR: no OCR cache: $OCR_CACHE"; exit 1; }
[[ -x "$LLAMA_BIN" ]] || { echo "ERROR: no llama-server at $LLAMA_BIN"; exit 1; }

# One model at a time on the GPU. Anything already resident (the prod
# service) makes the first load fail slowly, so refuse up front.
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if (( USED_MB > VRAM_IDLE_MB )); then
    echo "ERROR: the GPU already holds ${USED_MB} MiB. Stop the prod server first:"
    echo "  sudo systemctl stop llama-server"
    exit 1
fi

if curl -s -o /dev/null -m 2 --noproxy '*' "http://127.0.0.1:${PORT}/health"; then
    echo "ERROR: something already answers on port ${PORT}."
    exit 1
fi

mkdir -p "$OUT_ROOT/logg"

SRV_PID=""
cleanup() {
    if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_server() {  # name gguf mmproj extra_flags log
    local name="$1" gguf="$2" mmproj="$3" extra="$4" log="$5"
    # shellcheck disable=SC2086 — the flag strings are argument lists.
    "$LLAMA_BIN" -m "$gguf" --mmproj "$mmproj" --alias "$name" \
        --host 127.0.0.1 --port "$PORT" $SERVER_FLAGS $extra \
        > "$log" 2>&1 &
    SRV_PID=$!
    local waited=0
    until curl -s -o /dev/null -m 2 --noproxy '*' "http://127.0.0.1:${PORT}/health"; do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "!! $name: llama-server died during load, tail of $log:"
            tail -5 "$log"
            SRV_PID=""
            return 1
        fi
        sleep 5; waited=$((waited + 5))
        if (( waited >= 900 )); then
            echo "!! $name: not healthy after ${waited}s, giving up"
            kill "$SRV_PID" 2>/dev/null || true
            wait "$SRV_PID" 2>/dev/null || true
            SRV_PID=""
            return 1
        fi
    done
    echo "   server up after ${waited}s"
}

stop_server() {
    [[ -n "$SRV_PID" ]] || return 0
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    local waited=0
    # The next load OOMs unless this one has actually left the card.
    while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ') > VRAM_IDLE_MB )); do
        sleep 3; waited=$((waited + 3))
        (( waited < 120 )) || { echo "!! VRAM not released after ${waited}s, carrying on"; break; }
    done
}

# ── The crops, exported once ─────────────────────────────────────
CROPS="$OUT_ROOT/crops"
MANIFEST="$CROPS/manifest.csv"
if [[ -f "$CROPS/.ferdig" ]]; then
    echo "== export: already done, skipped"
else
    mkdir -p "$CROPS"
    # shellcheck disable=SC2086
    "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_export.py" \
        --res-csv "$RES_CSV" \
        --truth-csv "$TRUTH_CSV" \
        --folder "$FOLDER" \
        --out-dir "$CROPS" \
        --ocr-cache "$OCR_CACHE" \
        --source yolo \
        --hit-sample "$HIT_SAMPLE" \
        --seed "$SEED" \
        --workers "$WORKERS" \
        $GEOM_FLAGS 2>&1 | tee "$OUT_ROOT/logg/export.log"
    touch "$CROPS/.ferdig"
fi

N_BOX=$(( $(wc -l < "$MANIFEST" | tr -d ' ') - 1 ))
N_PRESENT=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r _ gguf _ _ <<< "$entry"
    [[ -f "$gguf" ]] && N_PRESENT=$((N_PRESENT + 1))
done
echo
echo "Boxes: $N_BOX   models on disk: $N_PRESENT of ${#MODELS[@]}"
echo "At ~3 s/box that is roughly $(( N_BOX * 3 / 3600 )) h per model," \
     "$(( N_BOX * 3 * N_PRESENT / 3600 )) h in total (cache hits are free)."

# ── The sweep ────────────────────────────────────────────────────
declare -a SUMMARY=()
START=$(date +%s)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME GGUF MMPROJ EXTRA <<< "$entry"
    OUT="$OUT_ROOT/$NAME"
    LOG="$OUT_ROOT/logg/$NAME.log"

    if [[ -f "$OUT/.ferdig" ]]; then
        echo "== $NAME: already finished, skipped"
        SUMMARY+=("$NAME  skipped (finished earlier)")
        continue
    fi
    if [[ ! -f "$GGUF" || ! -f "$MMPROJ" ]]; then
        echo "== $NAME: model files not on disk, skipped"
        SUMMARY+=("$NAME  SKIPPED, no files ($GGUF)")
        continue
    fi

    echo
    echo "════════════════════════════════════════════════════════"
    echo "== $NAME   $GGUF"
    echo "   log: $LOG"
    echo "════════════════════════════════════════════════════════"
    mkdir -p "$OUT"

    if ! start_server "$NAME" "$GGUF" "$MMPROJ" "$EXTRA" "$OUT/server.log"; then
        SUMMARY+=("$NAME  FAILED, server never came up")
        continue
    fi

    if ! "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_judge.py" \
            --manifest "$MANIFEST" \
            --out-csv "$OUT/judge_image.csv" \
            --url "http://127.0.0.1:${PORT}/v1" \
            --model "$NAME" \
            --concurrent "$CONCURRENT" \
            --skip-guarded \
            --prompt-file "$PROMPT_FILE" 2>&1 | tee "$LOG"; then
        echo "!! $NAME: the judging failed, moving to the next model"
        SUMMARY+=("$NAME  FAILED in the judging")
        stop_server
        continue
    fi

    "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_evaluate.py" \
        --manifest "$MANIFEST" \
        --judge "$OUT/judge_image.csv" \
        --fnr-override 2>&1 | tee -a "$LOG"

    stop_server
    touch "$OUT/.ferdig"
    SUMMARY+=("$NAME  done")
done

# ── Summary ──────────────────────────────────────────────────────
MINUTES=$(( ($(date +%s) - START) / 60 ))
echo
echo "╭──────────────────────────────────────────╮"
echo "│ vlm_model_sweep.sh: $MINUTES minutes"
echo "├──────────────────────────────────────────┤"
for line in "${SUMMARY[@]}"; do
    printf "│ %s\n" "$line"
done
echo "├──────────────────────────────────────────┤"
printf "│ Results: %s\n" "$OUT_ROOT"
echo "╰──────────────────────────────────────────╯"
echo
for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME _ _ _ <<< "$entry"
    LOG="$OUT_ROOT/logg/$NAME.log"
    [[ -f "$LOG" ]] || continue
    echo "── $NAME"
    sed -n '/^RESULT$/,/per lost fnr/p' "$LOG" | sed '1d;/^==*$/d'
    echo
done
echo "The prod server is still stopped. Bring it back with:"
echo "  sudo systemctl start llama-server"
