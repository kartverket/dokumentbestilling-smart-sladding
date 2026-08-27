#!/usr/bin/env bash
# valider_full.sh: validation with full production logic (OCR + YOLO + matching).
#
#   ./valider_full.sh model=PATH uttrekk=N [list=NAME] [name=ALIAS] [precache=no]
#                     [rules=no] [metadata=yes|PATH] [images=N] [processes=N]
#
#   model      path to the YOLO weights file (required)
#   uttrekk    uttrekk number to validate against (required)
#   list       name of the ID list (default: all documents)
#   name       custom name for the output directory
#   precache   'no' skips filling the cache (default: yes)
#   rules      'no' skips ALL postfilters, giving raw detection for baselining the rules
#   metadata   'yes' sends rettsstiftelse types from $SLADD_METADATA/uttrekk_N.csv
#              (rule profiles as in prod), or an explicit path. Default: global behaviour
#   images     'no'/0 skips the error images, or N draws at most N documents
#              (default: all). Summary and resultat.csv are unaffected
#   processes  worker processes for cache hits (default: auto = min(8, cores))
#
# Uses the OCR and YOLO caches ($SLADD_CACHE), so rerunning the same model is nearly free.
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
IMAGES="all"
METADATA=""
RULES="yes"
PROCESSES=""
EXTRA_FLAGS=()

# The script has no positional arguments, so a bare word can only be the value
# of the run.py flag in front of it. Without this «--vlm-model qwen» fails on
# «qwen» instead of reaching run.py.
AFTER_FLAG=0

for arg in "$@"; do
    case "$arg" in
        model=*)  MODEL="${arg#model=}"; AFTER_FLAG=0 ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}"; AFTER_FLAG=0 ;;
        list=*)   LIST="${arg#list=}"; AFTER_FLAG=0 ;;
        name=*)    NAME="${arg#name=}"; AFTER_FLAG=0 ;;
        precache=*) PRECACHE="${arg#precache=}"; AFTER_FLAG=0 ;;
        metadata=*) METADATA="${arg#metadata=}"; AFTER_FLAG=0 ;;
        rules=*)   RULES="${arg#rules=}"; AFTER_FLAG=0 ;;
        images=*)  IMAGES="${arg#images=}"; AFTER_FLAG=0 ;;
        processes=*) PROCESSES="${arg#processes=}"; AFTER_FLAG=0 ;;
        -*)        EXTRA_FLAGS+=("$arg"); AFTER_FLAG=1 ;;
        *)
            if [[ "$AFTER_FLAG" == 1 ]]; then
                EXTRA_FLAGS+=("$arg")
                AFTER_FLAG=0
            else
                echo "ERROR: Unknown parameter: $arg"
                echo "Valid: model=PATH uttrekk=N [list=NAME] [name=ALIAS] [precache=no] [rules=no] [metadata=yes] [images=N] [processes=N]"
                echo "run.py flags are passed on as they are, e.g. --vlm --vlm-concurrent 1"
                exit 1
            fi
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

MODEL_NAME=$(derive_model_name "$MODEL")

# ── Build paths ──────────────────────────────────────────────────
UTTREKK_DIR="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
TRUTH="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

LIST_FILE=""
if [[ -n "$LIST" ]]; then
    LIST_FILE="$SLADD_LISTS/uttrekk_${UTTREKK_NR}_${LIST}.txt"
fi

if [[ -n "$NAME" ]]; then
    OUT_NAME="$NAME"
elif [[ -n "$LIST" ]]; then
    OUT_NAME="full_${MODEL_NAME}_validated_on_uttrekk_${UTTREKK_NR}_${LIST}"
else
    OUT_NAME="full_${MODEL_NAME}_validated_on_uttrekk_${UTTREKK_NR}_all"
fi
OUT_DIR="$SLADD_VALIDATION/$OUT_NAME"

# ── Check that the files exist ───────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Cannot find model: $MODEL"
    exit 1
fi

if [[ -n "$LIST_FILE" && ! -f "$LIST_FILE" ]]; then
    echo "ERROR: Cannot find: $LIST_FILE"
    exit 1
fi

if [[ ! -f "$TRUTH" ]]; then
    echo "ERROR: Cannot find: $TRUTH"
    exit 1
fi

if [[ ! -d "$UTTREKK_DIR" ]]; then
    echo "ERROR: uttrekk directory does not exist: $UTTREKK_DIR"
    exit 1
fi

# Without list= all documents run: labels cover the whole uttrekk, so a document with no rows holds zero fnr.

# ── Show what is being run ───────────────────────────────────────
echo "╭─────────────────────────────────────────────╮"
echo "│ Full validation (OCR+YOLO): $OUT_NAME"
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
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/{ocr,yolo}"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Fill the caches first ────────────────────────────────────────
# precache.py does run.py's work in parallel processes against the same GPU, measured 3.3x on V100S.
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
    --csv --truth --only-error
    --truth-csv "$TRUTH"
    --csv-out "$OUT_DIR/resultat.csv"
    --png-dir "$OUT_DIR/error_images"
    --result-dir "$OUT_DIR"
    --time
)

if [[ -n "$LIST_FILE" ]]; then
    CMD+=(--select-from-file "$LIST_FILE")
else
    CMD+=(--count all)
fi

if [[ -n "$METADATA" ]]; then
    if [[ "$METADATA" == "yes" || "$METADATA" == "auto" ]]; then
        METADATA="$SLADD_METADATA/uttrekk_${UTTREKK_NR}.csv"
    fi
    if [[ ! -f "$METADATA" ]]; then
        echo "ERROR: Cannot find metadata CSV: $METADATA"
        exit 1
    fi
    CMD+=(--metadata-csv "$METADATA")
fi

if [[ "$RULES" == "no" || "$RULES" == "0" ]]; then
    CMD+=(--without-postfilter)
fi

if [[ "$IMAGES" != "all" ]]; then
    if [[ "$IMAGES" == "no" ]]; then
        IMAGES=0
    fi
    if ! [[ "$IMAGES" =~ ^[0-9]+$ ]]; then
        echo "ERROR: images= must be 'all', 'no' or a number (got: $IMAGES)"
        exit 1
    fi
    CMD+=(--max-error-images "$IMAGES")
fi

if [[ -n "$PROCESSES" ]]; then
    if ! [[ "$PROCESSES" =~ ^[0-9]+$ ]]; then
        echo "ERROR: processes= must be a number (got: $PROCESSES)"
        exit 1
    fi
    CMD+=(--processes "$PROCESSES")
fi

"${CMD[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
