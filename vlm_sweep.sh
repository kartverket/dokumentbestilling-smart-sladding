#!/usr/bin/env bash
# vlm_sweep.sh: the same boxes under several crop geometries, one after the
# other. Each geometry gets an export, a judging run and an evaluation.
#
#   RES_CSV=$SLADD_VALIDATION/<run>/resultat.csv ./vlm_sweep.sh
#
# The document list is drawn once and the seed is fixed, so the geometries are
# compared on the same boxes. Only kilde «yolo» is exported, the stratum the
# verifier judges in prod. Requires server.env (source activate.sh) and a
# running llama-server.
#
# A geometry that finished is skipped on a re-run, so a sweep killed halfway
# carries on. Delete <out-dir>/.ferdig to force one to run again.
#
# Each arm also writes hard.txt, the documents the model did badly on, and the
# union lands in harde_dokumenter.txt. Pass that back as DOC_LIST to run the
# next round on those documents alone. A chosen scope is not a random sample,
# so counts from such a run compare against each other, not against the uttrekk.
#
# Variables: RES_CSV (required), UTTREKK=4, N_DOCS=2000, SEED=42,
# HIT_SAMPLE=1000, OUT_ROOT, DOC_LIST, MODEL, URL, CONCURRENT=4, WORKERS=8,
# HARD_KINDS="loss missed", HARD_LIMIT=0, SKIP_GUARDED=1, BOX_LIST (exact
# sladds from vlm_evaluate --hard-boxes, the shortest loop).

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run: source activate.sh"
    exit 1
fi

RES_CSV="${RES_CSV:?ERROR: set RES_CSV to the resultat.csv of the model run}"
UTTREKK="${UTTREKK:-4}"
N_DOCS="${N_DOCS:-2000}"
SEED="${SEED:-42}"
HIT_SAMPLE="${HIT_SAMPLE:-1000}"
CONCURRENT="${CONCURRENT:-4}"
WORKERS="${WORKERS:-8}"
# -u so a tee'd log stays live while the run is going.
PYTHON=("${PYTHON:-python}" -u)

FOLDER="${FOLDER:-$SLADD_UTTREKK/uttrekk_$UTTREKK}"
TRUTH_CSV="${TRUTH_CSV:-$SLADD_LABELS/uttrekk_$UTTREKK.csv}"
OCR_CACHE="${OCR_CACHE:-$SLADD_CACHE/uttrekk_$UTTREKK/ocr}"
OUT_ROOT="${OUT_ROOT:-/data2/vlm/uttrekk${UTTREKK}_margin_sweep}"
MODEL="${MODEL:-${SLADD_VLM_MODEL:-}}"
URL="${URL:-${SLADD_VLM_URL:-http://127.0.0.1:8080/v1}}"
PROMPT_FILE="${PROMPT_FILE:-$SLADD_REPO/prompts/etikettregel.txt}"
# What each arm writes to hard.txt, for the next round's DOC_LIST.
HARD_KINDS="${HARD_KINDS:-loss missed}"
HARD_LIMIT="${HARD_LIMIT:-0}"
# The fnr guard already decides these boxes, so a call cannot change the
# outcome. SKIP_GUARDED=0 judges them anyway, which is what measuring the cost
# of the guard itself needs.
if [[ "${SKIP_GUARDED:-1}" == "0" ]]; then
    GUARD_FLAG="--no-skip-guarded"
else
    GUARD_FLAG="--skip-guarded"
fi

# name|export flags|judge flags|reuse. The name becomes the output directory.
# «reuse» names an EARLIER arm whose crops to judge again instead of exporting
# the same images twice: that is how a prompt is tested against a geometry
# already measured. Ordered so the pairs that answer the most run first: eight
# arms may not fit one night. Arms 2 to 4 are one resolution ladder at the same
# margin, so they are the comparison that survives a night that runs short.
# 2480 px is exactly what A4 portrait renders to at 300 dpi, so --max-px 2480
# leaves those pages alone and caps only what is wider. Uncapped, a landscape
# page comes to 2442 image tokens and overruns the 3072-token slot, which
# would read as a bad geometry rather than a crop that never fit.
CONFIGS=(
    "up100_full_px1024|--margin-up 100 --full-width --margin-down 60 --max-px 1024"
    "up150_full_etikett||--prompt-file $PROMPT_FILE|up100_full_px1024"
    "up100_left250_px1024|--margin-up 100 --margin-left full --margin-right 250 --margin-down 60 --max-px 1024"
    "up150_full|--margin-up 150 --full-width --margin-down 60"
    "up100_full|--margin-up 100 --full-width --margin-down 60"
    "up100_full_native|--margin-up 100 --full-width --margin-down 60 --max-px 2480"
    "up150_full_px1600|--margin-up 150 --full-width --margin-down 60 --max-px 1600"
    "up150_left|--margin-up 150 --margin-left full --margin-right 30 --margin-down 30"
    "up100_left|--margin-up 100 --margin-left full --margin-right 30 --margin-down 30"
)

# ── Preflight ────────────────────────────────────────────────────
# Four exports cost an hour of CPU before the first call is made, so
# everything that can be wrong is checked before any work starts.
for f in "$RES_CSV" "$TRUTH_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: no such file: $f"; exit 1; }
done
[[ -d "$FOLDER" ]] || { echo "ERROR: no PDF directory: $FOLDER"; exit 1; }
[[ -d "$OCR_CACHE" ]] || { echo "ERROR: no OCR cache: $OCR_CACHE"; exit 1; }
[[ -n "$MODEL" ]] || { echo "ERROR: set MODEL (or SLADD_VLM_MODEL)"; exit 1; }

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r _ _ judge_flags _ <<< "$entry"
    [[ "$judge_flags" == *--prompt-file* ]] || continue
    file="${judge_flags##*--prompt-file }"
    file="${file%% *}"
    [[ -f "$file" ]] || { echo "ERROR: no prompt file: $file"; exit 1; }
done

# Liveness only: an endpoint that answers 404 here is still an endpoint.
if ! curl -s -o /dev/null -m 10 --noproxy '*' "${URL%/}/models"; then
    echo "ERROR: no answer from $URL. Start llama-server first:"
    echo "  sudo systemctl start llama-server"
    exit 1
fi

mkdir -p "$OUT_ROOT"

# ── The documents, drawn once ────────────────────────────────────
# Only documents where YOLO actually proposed a box: the rest would sit in the
# scope count without giving the model anything to judge. The draw is sorted
# on a seeded key rather than piped through head, which would close the pipe
# early and trip pipefail.
#
# A DOC_LIST from outside is the short feedback loop: point it at the hard
# list from an earlier round. It is never redrawn, and a missing one is an
# error rather than a silent fall back to a fresh random 2000. BOX_LIST is the
# shortest loop of all — exact sladds from vlm_evaluate --hard-boxes — and
# replaces the document machinery entirely.
if [[ -n "${BOX_LIST:-}" ]]; then
    [[ -s "$BOX_LIST" ]] || { echo "ERROR: no box list: $BOX_LIST"; exit 1; }
    N_BOX=$(( $(wc -l < "$BOX_LIST" | tr -d ' ') - 1 ))
    echo "Boxes given: $N_BOX from $BOX_LIST"
    echo "NB: a picked set of boxes. Counts compare arms against each other"
    echo "    and nothing scales back to the uttrekk."
    SCOPE_ARGS=(--box-list "$BOX_LIST")
elif [[ -n "${DOC_LIST:-}" ]]; then
    [[ -s "$DOC_LIST" ]] || { echo "ERROR: no document list: $DOC_LIST"; exit 1; }
    echo "Documents given: $(wc -l < "$DOC_LIST" | tr -d ' ') from $DOC_LIST"
    echo "NB: a chosen scope is not a random sample. Gain and loss RATES from"
    echo "    this run do not carry back to the uttrekk, only the counts do."
    SCOPE_ARGS=(--processed-list "$DOC_LIST")
elif [[ ! -s "$OUT_ROOT/dokumenter.txt" ]]; then
    DOC_LIST="$OUT_ROOT/dokumenter.txt"
    awk -F, -v seed="$SEED" '
        NR==1 { for (i=1; i<=NF; i++) {
                    if ($i == "navn")  name_col = i
                    if ($i == "kilde") source_col = i }
                if (!name_col || !source_col) {
                    print "ERROR: resultat.csv has no navn/kilde column" > "/dev/stderr"
                    exit 1 }
                srand(seed)
                next }
        $source_col == "yolo" && !seen[$name_col]++ { print rand() "\t" $name_col }
    ' "$RES_CSV" | sort -n | awk -v n="$N_DOCS" 'NR <= n { print $2 }' \
        > "$DOC_LIST"
    echo "Documents drawn: $(wc -l < "$DOC_LIST" | tr -d ' ') to $DOC_LIST"
    FOUND=$(wc -l < "$DOC_LIST" | tr -d ' ')
    if (( FOUND < N_DOCS )); then
        echo "NB: asked for $N_DOCS, the result CSV holds $FOUND with a yolo box."
    fi
    SCOPE_ARGS=(--processed-list "$DOC_LIST")
else
    DOC_LIST="$OUT_ROOT/dokumenter.txt"
    echo "Documents reused: $(wc -l < "$DOC_LIST" | tr -d ' ') from $DOC_LIST"
    SCOPE_ARGS=(--processed-list "$DOC_LIST")
fi

# ── The sweep ────────────────────────────────────────────────────
mkdir -p "$OUT_ROOT/logg"
declare -a SUMMARY=()
START=$(date +%s)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME FLAGS JUDGE_FLAGS REUSE <<< "$entry"
    OUT="$OUT_ROOT/$NAME"
    LOG="$OUT_ROOT/logg/$NAME.log"

    if [[ -f "$OUT/.ferdig" ]]; then
        echo "== $NAME: already finished, skipped"
        SUMMARY+=("$NAME  skipped (finished earlier)")
        continue
    fi

    echo
    echo "════════════════════════════════════════════════════════"
    echo "== $NAME   ${FLAGS:-crops from $REUSE}  $JUDGE_FLAGS"
    echo "   log: $LOG"
    echo "════════════════════════════════════════════════════════"
    mkdir -p "$OUT"

    # The crops, and with them utvalg.json and the sampling factors, stay with
    # the arm that exported them. vlm_judge and vlm_evaluate find both from the
    # manifest path, so a reusing arm only writes its own judgements.
    if [[ -n "$REUSE" ]]; then
        MANIFEST="$OUT_ROOT/$REUSE/manifest.csv"
        if [[ ! -f "$MANIFEST" ]]; then
            echo "!! $NAME: $REUSE has not exported, nothing to judge"
            SUMMARY+=("$NAME  SKIPPED, $REUSE has no manifest")
            continue
        fi
        echo "   crops from $REUSE"
    else
        MANIFEST="$OUT/manifest.csv"
        # shellcheck disable=SC2086 — FLAGS is a list of arguments on purpose.
        if ! "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_export.py" \
                --res-csv "$RES_CSV" \
                --truth-csv "$TRUTH_CSV" \
                --folder "$FOLDER" \
                --out-dir "$OUT" \
                "${SCOPE_ARGS[@]}" \
                --ocr-cache "$OCR_CACHE" \
                --source yolo \
                --hit-sample "$HIT_SAMPLE" \
                --seed "$SEED" \
                --workers "$WORKERS" \
                $FLAGS 2>&1 | tee "$LOG"; then
            echo "!! $NAME: the export failed, moving to the next arm"
            SUMMARY+=("$NAME  FAILED in the export")
            continue
        fi
    fi

    # shellcheck disable=SC2086 — JUDGE_FLAGS is a list of arguments too.
    if ! "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_judge.py" \
            --manifest "$MANIFEST" \
            --out-csv "$OUT/judge_image.csv" \
            --url "$URL" \
            --model "$MODEL" \
            --concurrent "$CONCURRENT" \
            "$GUARD_FLAG" \
            $JUDGE_FLAGS 2>&1 | tee -a "$LOG"; then
        echo "!! $NAME: the judging failed, moving to the next arm"
        SUMMARY+=("$NAME  FAILED in the judging")
        continue
    fi

    # The fnr guard runs in prod, so the number that describes prod is the
    # one measured with it on.
    "${PYTHON[@]}" "$SLADD_REPO/utils/vlm_evaluate.py" \
        --manifest "$MANIFEST" \
        --judge "$OUT/judge_image.csv" \
        --hard-list "$OUT/hard.txt" \
        --hard-kinds $HARD_KINDS \
        --hard-limit "$HARD_LIMIT" \
        --fnr-override 2>&1 | tee -a "$LOG"

    touch "$OUT/.ferdig"
    SUMMARY+=("$NAME  done")
done

# ── Summary ──────────────────────────────────────────────────────
MINUTES=$(( ($(date +%s) - START) / 60 ))
echo
echo "╭──────────────────────────────────────────╮"
echo "│ vlm_sweep.sh: $MINUTES minutes"
echo "├──────────────────────────────────────────┤"
for line in "${SUMMARY[@]}"; do
    printf "│ %s\n" "$line"
done
echo "├──────────────────────────────────────────┤"
printf "│ Results: %s\n" "$OUT_ROOT"
echo "╰──────────────────────────────────────────╯"

# Hard in ANY arm. A geometry that only one arm fails is exactly the case the
# next round has to keep, so the union is the safe join, not the intersection.
HARD_ALL="$OUT_ROOT/harde_dokumenter.txt"
if compgen -G "$OUT_ROOT/*/hard.txt" > /dev/null; then
    sort -u "$OUT_ROOT"/*/hard.txt > "$HARD_ALL"
    echo
    echo "$(wc -l < "$HARD_ALL" | tr -d ' ') documents were hard in at least one arm:"
    echo "  $HARD_ALL"
fi
echo
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME FLAGS JUDGE_FLAGS REUSE <<< "$entry"
    LOG="$OUT_ROOT/logg/$NAME.log"
    [[ -f "$LOG" ]] || continue
    echo "── $NAME   ${FLAGS:-crops from $REUSE}  $JUDGE_FLAGS"
    sed -n '/^RESULT$/,/per lost fnr/p' "$LOG" | sed '1d;/^==*$/d'
    echo
done
