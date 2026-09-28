#!/usr/bin/env bash
# vlm_prompt_sweep.sh: the same crops judged under several prompts against
# the RUNNING llama-server, in two phases: first every prompt on a
# deterministic PCT-% slice of the documents (the early ranking), then every
# prompt on the full manifest. The judgement cache makes phase two pay only
# for the boxes phase one did not touch. No export of its own is needed when
# a manifest already exists:
#
#   MANIFEST=/data2/vlm/uttrekk4_modeller/crops/manifest.csv ./vlm_prompt_sweep.sh
#
# Without MANIFEST it exports first (all kilde-«yolo» boxes, the winning
# up100/full/1600 geometry), and then RES_CSV is required:
#
#   RES_CSV=$SLADD_VALIDATION/<run>/resultat.csv ./vlm_prompt_sweep.sh
#
# The slice is drawn per DOCUMENT with a name hash, so it is the same 20 %
# in every arm and every re-run, and needs no seed. Its counts describe the
# slice only; the full-phase numbers are the ones that describe the uttrekk.
# A finished arm is skipped on a re-run; delete <out-dir>/.ferdig to force
# one. To read the ranking before committing to the rest: PHASES=del runs
# the slice only, and a later run with PHASES="del full" (the default)
# skips the finished slice arms. Drop a losing prompt from PROMPTS before
# the full phase and nothing is wasted.
#
# Variables: MANIFEST or RES_CSV (one required), PCT=20, PHASES="del full",
# UTTREKK=4, HIT_SAMPLE=-1, OUT_ROOT, GEOM_FLAGS, MODEL, URL, CONCURRENT=4,
# WORKERS=8, SEED=42.

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run: source activate.sh"
    exit 1
fi

UTTREKK="${UTTREKK:-4}"
HIT_SAMPLE="${HIT_SAMPLE:--1}"
PCT="${PCT:-20}"
PHASES="${PHASES:-del full}"
SEED="${SEED:-42}"
CONCURRENT="${CONCURRENT:-4}"
WORKERS="${WORKERS:-8}"
PYTHON=("${PYTHON:-python}" -u)

FOLDER="${FOLDER:-$SLADD_UTTREKK/uttrekk_$UTTREKK}"
TRUTH_CSV="${TRUTH_CSV:-$SLADD_LABELS/uttrekk_$UTTREKK.csv}"
OCR_CACHE="${OCR_CACHE:-$SLADD_CACHE/uttrekk_$UTTREKK/ocr}"
OUT_ROOT="${OUT_ROOT:-/data2/vlm/uttrekk${UTTREKK}_prompter}"
MODEL="${MODEL:-${SLADD_VLM_MODEL:-}}"
URL="${URL:-${SLADD_VLM_URL:-http://127.0.0.1:8080/v1}}"
GEOM_FLAGS="${GEOM_FLAGS:---margin-up 100 --full-width --margin-down 60 --max-px 1600}"

# name|prompt file. Empty file = the built-in STD_PROMPT (the prod baseline).
# Ordered decision first: the candidate and the baseline settle the adoption
# question even if the night runs short.
PROMPTS=(
    "femsiffer_liste|$SLADD_REPO/prompts/femsiffer_uten_liste.txt"
    "std|"
    "femsiffer|$SLADD_REPO/prompts/femsiffer.txt"
    "uten_liste|$SLADD_REPO/prompts/uten_liste.txt"
)

# ── Preflight ────────────────────────────────────────────────────
[[ -n "$MODEL" ]] || { echo "ERROR: set MODEL (or SLADD_VLM_MODEL)"; exit 1; }
for entry in "${PROMPTS[@]}"; do
    IFS='|' read -r _ file <<< "$entry"
    [[ -z "$file" || -f "$file" ]] || { echo "ERROR: no prompt file: $file"; exit 1; }
done
if ! curl -s -o /dev/null -m 10 --noproxy '*' "${URL%/}/models"; then
    echo "ERROR: no answer from $URL. Start llama-server first:"
    echo "  sudo systemctl start llama-server"
    exit 1
fi

mkdir -p "$OUT_ROOT/logg"

# ── The crops: reuse or export once ──────────────────────────────
if [[ -n "${MANIFEST:-}" ]]; then
    [[ -f "$MANIFEST" ]] || { echo "ERROR: no manifest: $MANIFEST"; exit 1; }
    echo "Crops reused: $MANIFEST"
else
    RES_CSV="${RES_CSV:?ERROR: set MANIFEST to reuse crops, or RES_CSV to export}"
    for f in "$RES_CSV" "$TRUTH_CSV"; do
        [[ -f "$f" ]] || { echo "ERROR: no such file: $f"; exit 1; }
    done
    [[ -d "$FOLDER" ]] || { echo "ERROR: no PDF directory: $FOLDER"; exit 1; }
    [[ -d "$OCR_CACHE" ]] || { echo "ERROR: no OCR cache: $OCR_CACHE"; exit 1; }
    CROPS="$OUT_ROOT/crops"
    MANIFEST="$CROPS/manifest.csv"
    if [[ -f "$CROPS/.ferdig" ]]; then
        echo "== export: already done, skipped"
    else
        mkdir -p "$CROPS"
        # shellcheck disable=SC2086 — GEOM_FLAGS is an argument list.
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
fi

# ── The slice, next to the manifest so crop paths keep resolving ─
DEL_MANIFEST="$(dirname "$MANIFEST")/manifest_del${PCT}.csv"
if [[ ! -f "$DEL_MANIFEST" ]]; then
    "${PYTHON[@]}" - "$MANIFEST" "$DEL_MANIFEST" "$PCT" <<'PYEOF'
import csv, hashlib, sys
src, dst, pct = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    rows = [x for x in r
            if int(hashlib.md5(x["fil"].encode()).hexdigest(), 16) % 100 < pct]
    fields = r.fieldnames
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Slice: {len(rows)} boxes in {dst}")
PYEOF
fi

N_BOX=$(( $(wc -l < "$MANIFEST" | tr -d ' ') - 1 ))
N_DEL=$(( $(wc -l < "$DEL_MANIFEST" | tr -d ' ') - 1 ))
echo
echo "Boxes: $N_BOX full, $N_DEL in the ${PCT}% slice   prompts: ${#PROMPTS[@]}   phases: $PHASES"
echo "At ~3 s/box the slice is at most $(( N_DEL * 3 / 3600 )) h per prompt and the"
echo "full phase $(( (N_BOX - N_DEL) * 3 / 3600 )) h more; guard-skipped boxes cost nothing."

# ── The sweep ────────────────────────────────────────────────────
declare -a SUMMARY=()
START=$(date +%s)

run_arm() {  # arm-name manifest
    local arm="$1" mani="$2"
    local out="$OUT_ROOT/$arm"
    local log="$OUT_ROOT/logg/$arm.log"

    if [[ -f "$out/.ferdig" ]]; then
        echo "== $arm: already finished, skipped"
        SUMMARY+=("$arm  skipped (finished earlier)")
        return 0
    fi
    echo
    echo "════════════════════════════════════════════════════════"
    echo "== $arm   ${PROMPT_FILE_CUR:-built-in STD_PROMPT}"
    echo "   log: $log"
    echo "════════════════════════════════════════════════════════"
    mkdir -p "$out"

    local prompt_args=()
    [[ -n "$PROMPT_FILE_CUR" ]] && prompt_args=(--prompt-file "$PROMPT_FILE_CUR")

    if ! "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_judge.py" \
            --manifest "$mani" \
            --out-csv "$out/judge_image.csv" \
            --url "$URL" \
            --model "$MODEL" \
            --concurrent "$CONCURRENT" \
            --skip-guarded \
            "${prompt_args[@]}" 2>&1 | tee "$log"; then
        echo "!! $arm: the judging failed, moving on"
        SUMMARY+=("$arm  FAILED in the judging")
        return 0
    fi
    "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_evaluate.py" \
        --manifest "$mani" \
        --judge "$out/judge_image.csv" \
        --fnr-override 2>&1 | tee -a "$log"
    touch "$out/.ferdig"
    SUMMARY+=("$arm  done")
}

for PHASE in $PHASES; do
    case "$PHASE" in
        del)  PHASE_MANIFEST="$DEL_MANIFEST"; SUFFIX="_del${PCT}" ;;
        full) PHASE_MANIFEST="$MANIFEST";     SUFFIX="" ;;
        *) echo "ERROR: unknown phase «$PHASE» (use: del full)"; exit 1 ;;
    esac
    echo
    echo "──────────── phase: $PHASE ────────────"
    for entry in "${PROMPTS[@]}"; do
        IFS='|' read -r NAME PROMPT_FILE_CUR <<< "$entry"
        run_arm "${NAME}${SUFFIX}" "$PHASE_MANIFEST"
    done
done

# ── Summary ──────────────────────────────────────────────────────
MINUTES=$(( ($(date +%s) - START) / 60 ))
echo
echo "╭──────────────────────────────────────────╮"
echo "│ vlm_prompt_sweep.sh: $MINUTES minutes"
echo "├──────────────────────────────────────────┤"
for line in "${SUMMARY[@]}"; do
    printf "│ %s\n" "$line"
done
echo "├──────────────────────────────────────────┤"
printf "│ Results: %s\n" "$OUT_ROOT"
echo "╰──────────────────────────────────────────╯"
echo
for PHASE in $PHASES; do
    case "$PHASE" in
        del)  SUFFIX="_del${PCT}" ;;
        full) SUFFIX="" ;;
    esac
    for entry in "${PROMPTS[@]}"; do
        IFS='|' read -r NAME FILE <<< "$entry"
        LOG="$OUT_ROOT/logg/${NAME}${SUFFIX}.log"
        [[ -f "$LOG" ]] || continue
        echo "── ${NAME}${SUFFIX}   ${FILE:-built-in STD_PROMPT}"
        sed -n '/^RESULT$/,/per lost fnr/p' "$LOG" | sed '1d;/^==*$/d'
        echo
    done
done
