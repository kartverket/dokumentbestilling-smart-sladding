#!/usr/bin/env bash
# valider_yolo.sh: validation with YOLO only (no OCR). See valider_full.sh for OCR + YOLO.
#
#   ./valider_yolo.sh model=PATH uttrekk=N [list=NAME] [name=ALIAS] [precache=no]
#
#   model     path to the YOLO weights file (required)
#   uttrekk   uttrekk number to validate against (required)
#   list      name of the ID list (default: all documents)
#   name      custom name for the output directory
#   precache  'no' skips filling the cache (default: yes)
#
# Requires server.env to be sourced (source activate.sh).

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run first:"
    echo "  source activate.sh"
    exit 1
fi

# ── Parse named parameters ────────────────────────────────────────
MODEL=""
UTTREKK_NR=""
LIST=""
NAME=""
PRECACHE="yes"
EXTRA_FLAGS=()

for arg in "$@"; do
    case "$arg" in
        model=*)  MODEL="${arg#model=}" ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}" ;;
        list=*)   LIST="${arg#list=}" ;;
        name=*)    NAME="${arg#name=}" ;;
        precache=*) PRECACHE="${arg#precache=}" ;;
        -*)        EXTRA_FLAGS+=("$arg") ;;
        *)
            echo "ERROR: Unknown parameter: $arg"
            echo "Valid: model=PATH uttrekk=N [list=NAME] [name=ALIAS] [precache=no]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: model= is required"
    echo "Example: $0 model=\$SLADD_PRODWEIGHTS uttrekk=5 list=jou"
    exit 1
fi

if [[ -z "$UTTREKK_NR" ]]; then
    echo "ERROR: uttrekk= is required"
    echo "Example: $0 model=\$SLADD_PRODWEIGHTS uttrekk=5 list=jou"
    exit 1
fi

# Published models are <name>/<name>.pt; raw runs are <run>/weights/best.pt, where only the run dir names it.
derive_model_name() {
    local name
    name=$(basename "$1"); name=${name%.pt}
    if [[ "$name" == best || "$name" == last ]]; then
        name=$(basename "$(dirname "$(dirname "$1")")")
    fi
    echo "$name"
}

if [[ -n "$NAME" ]]; then
    MODEL_NAME="$NAME"
else
    MODEL_NAME=$(derive_model_name "$MODEL")
fi

# ── Build paths ──────────────────────────────────────────────────
UTTREKK_DIR="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
TRUTH="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

LIST_FILE=""
if [[ -n "$LIST" ]]; then
    LIST_FILE="$SLADD_LISTS/uttrekk_${UTTREKK_NR}_${LIST}.txt"
    OUT_NAME="${MODEL_NAME}_validated_on_uttrekk_${UTTREKK_NR}_${LIST}"
else
    OUT_NAME="${MODEL_NAME}_validated_on_uttrekk_${UTTREKK_NR}_all"
fi
OUT_DIR="$SLADD_VALIDATION/$OUT_NAME"

# ── Check that the files exist ───────────────────────────────────
CHECK_FILES=("$MODEL" "$TRUTH")
if [[ -n "$LIST_FILE" ]]; then
    CHECK_FILES+=("$LIST_FILE")
fi
for file in "${CHECK_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Cannot find: $file"
        exit 1
    fi
done

if [[ ! -d "$UTTREKK_DIR" ]]; then
    echo "ERROR: uttrekk directory does not exist: $UTTREKK_DIR"
    exit 1
fi

# Without list= all documents run: labels cover the whole uttrekk, so a document with no rows holds zero fnr.

# ── Show what is being run ───────────────────────────────────────
echo "╭─────────────────────────────────────────────╮"
echo "│ Validation: $OUT_NAME"
echo "├─────────────────────────────────────────────┤"
printf "│ modell:   %s\n" "$MODEL"
printf "│ uttrekk:  %s\n" "$UTTREKK_DIR"
if [[ -n "$LIST_FILE" ]]; then
    printf "│ liste:    %s\n" "$LIST_FILE"
else
    printf "│ liste:    (all documents)\n"
fi
printf "│ truth:    %s\n" "$TRUTH"
printf "│ out dir:  %s\n" "$OUT_DIR"
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/yolo"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Fill the caches first ────────────────────────────────────────
# "--only both" is deliberate: --only-yolo needs the rotations, and they live in the OCR cache.
if [[ "$PRECACHE" == "yes" ]]; then
    echo "── Filling cache (precache.py) ──"
    PRECACHE_CMD=(python -u "${SLADD_PRECACHE:-$SLADD_REPO/utils/precache.py}"
        --folder "$UTTREKK_DIR"
        --only both
        --yolo-weights "$MODEL"
    )
    if [[ -n "$LIST_FILE" ]]; then
        PRECACHE_CMD+=(--select-from-file "$LIST_FILE")
    fi
    "${PRECACHE_CMD[@]}"
    echo ""
fi

# ── Build the command ────────────────────────────────────────────
CMD=(python -u "$SLADD_RUN"
    --folder "$UTTREKK_DIR"
    --yolo-weights "$MODEL"
    --only-yolo
    --csv --truth --only-error
    --truth-csv "$TRUTH"
    --csv-out "$OUT_DIR/resultat.csv"
    --png-dir "$OUT_DIR/error_images"
    --result-dir "$OUT_DIR"
)

if [[ -n "$LIST_FILE" ]]; then
    CMD+=(--select-from-file "$LIST_FILE")
else
    CMD+=(--count all)
fi

"${CMD[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
