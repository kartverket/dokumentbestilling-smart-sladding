#!/usr/bin/env bash
# valider_full.sh: validation with full production logic (OCR + YOLO + matching).
#
#   ./valider_full.sh model=PATH uttrekk=N [list=NAME] [name=ALIAS] [precache=no]
#                     [rules=no] [metadata=yes|PATH] [images=N] [processes=N]
#   ./valider_full.sh deploy=test|prod|URL uttrekk=N [list=NAME] [name=ALIAS]
#                     [metadata=yes|PATH] [images=N]
#
#   model      path to the YOLO weights file (required without deploy=)
#   deploy     validate a running container instead of the local model: 'test'
#              (port from .env, default 5072), 'prod' (5071) or a full URL.
#              Every PDF goes over HTTP to /model, so the image decides
#              weights and rules. The caches, rules= and processes= belong to
#              the local path and are not available
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
#   live       'no' turns the live summary off, or N sets its interval in
#              seconds (default 60). One line per tick from resultat.csv.
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
DEPLOY=""
UTTREKK_NR=""
LIST=""
NAME=""
PRECACHE="yes"
IMAGES="all"
METADATA=""
RULES="yes"
PROCESSES=""
LIVE=""
EXTRA_FLAGS=()

# The script has no positional arguments, so a bare word can only be the value
# of the run.py flag in front of it. Without this «--vlm-model qwen» fails on
# «qwen» instead of reaching run.py.
AFTER_FLAG=0

for arg in "$@"; do
    case "$arg" in
        model=*)  MODEL="${arg#model=}"; AFTER_FLAG=0 ;;
        deploy=*) DEPLOY="${arg#deploy=}"; AFTER_FLAG=0 ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}"; AFTER_FLAG=0 ;;
        list=*)   LIST="${arg#list=}"; AFTER_FLAG=0 ;;
        name=*)    NAME="${arg#name=}"; AFTER_FLAG=0 ;;
        precache=*) PRECACHE="${arg#precache=}"; AFTER_FLAG=0 ;;
        metadata=*) METADATA="${arg#metadata=}"; AFTER_FLAG=0 ;;
        rules=*)   RULES="${arg#rules=}"; AFTER_FLAG=0 ;;
        images=*)  IMAGES="${arg#images=}"; AFTER_FLAG=0 ;;
        processes=*) PROCESSES="${arg#processes=}"; AFTER_FLAG=0 ;;
        live=*)    LIVE="${arg#live=}"; AFTER_FLAG=0 ;;
        -*)        EXTRA_FLAGS+=("$arg"); AFTER_FLAG=1 ;;
        *)
            if [[ "$AFTER_FLAG" == 1 ]]; then
                EXTRA_FLAGS+=("$arg")
                AFTER_FLAG=0
            else
                echo "ERROR: Unknown parameter: $arg"
                echo "Valid: model=PATH|deploy=test uttrekk=N [list=NAME] [name=ALIAS] [precache=no] [rules=no] [metadata=yes] [images=N] [processes=N] [live=no|N]"
                echo "run.py flags are passed on as they are, e.g. --vlm --vlm-concurrent 1"
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$MODEL" && -z "$DEPLOY" ]]; then
    echo "ERROR: model= or deploy= is required"
    echo "Example: $0 model=\$SLADD_PRODWEIGHTS uttrekk=5 list=jou"
    echo "         $0 deploy=test uttrekk=5 list=jou"
    exit 1
fi

if [[ -n "$MODEL" && -n "$DEPLOY" ]]; then
    echo "ERROR: model= and deploy= cannot be combined. The image holds its own"
    echo "       weights: see which with ./deploy.sh status"
    exit 1
fi

if [[ -n "$DEPLOY" && ( "$RULES" == "no" || "$RULES" == "0" ) ]]; then
    echo "ERROR: rules=no is not available with deploy=: the postfilters run inside"
    echo "       the container. Measure that baseline locally instead."
    exit 1
fi

if [[ -n "$DEPLOY" && -n "$PROCESSES" ]]; then
    echo "ERROR: processes= is not available with deploy=: the container answers one"
    echo "       request at a time (gunicorn workers=1)."
    exit 1
fi

LIVE_INTERVAL=60
case "$LIVE" in
    ""|yes)   ;;
    no|0)     LIVE_INTERVAL=0 ;;
    *[!0-9]*) echo "ERROR: live= must be 'no' or a number of seconds (got: $LIVE)"
              exit 1 ;;
    *)        LIVE_INTERVAL="$LIVE" ;;
esac

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

# Docker tags and URLs both go into a directory name, so anything else is stripped.
slug() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/-/g; s/--*/-/g; s/^-*//; s/-*$//'
}

# deploy.sh writes .env next to itself, which is this script's directory.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
env_value() {
    [[ -f "$SCRIPT_DIR/.env" ]] || return 0
    { grep "^${1}=" "$SCRIPT_DIR/.env" 2>/dev/null || true; } | tail -1 | cut -d= -f2-
}

MODEL_NAME=""
API_URL=""
TAG=""
if [[ -z "$DEPLOY" ]]; then
    MODEL_NAME=$(derive_model_name "$MODEL")
else
    case "$DEPLOY" in
        prod) API_URL="http://localhost:5071/model"; TAG=$(env_value PROD_TAG) ;;
        test) PORT=$(env_value TEST_PORT)
              API_URL="http://localhost:${PORT:-5072}/model"
              TAG=$(env_value TEST_TAG) ;;
        http://*|https://*)
              API_URL="$DEPLOY"
              [[ "$API_URL" == */model ]] || API_URL="${API_URL%/}/model" ;;
        *)    echo "ERROR: deploy= must be 'test', 'prod' or a URL (got: $DEPLOY)"
              exit 1 ;;
    esac

    # The tag names the code; the label names the model inside it.
    if [[ -n "$TAG" ]] && command -v docker >/dev/null 2>&1; then
        MODEL_NAME=$(docker image inspect "${IMAGE:-smart-sladding}:${TAG}" \
            --format '{{index .Config.Labels "no.kartverket.smsl.modell"}}' 2>/dev/null || true)
    fi
fi

# ── Build paths ──────────────────────────────────────────────────
UTTREKK_DIR="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
TRUTH="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

LIST_FILE=""
if [[ -n "$LIST" ]]; then
    LIST_FILE="$SLADD_LISTS/uttrekk_${UTTREKK_NR}_${LIST}.txt"
fi

if [[ -n "$DEPLOY" ]]; then
    # The tag identifies the deployment; without it, whatever was asked for.
    PREFIX="deploy_$(slug "${TAG:-$DEPLOY}")"
else
    PREFIX="full_${MODEL_NAME}"
fi

if [[ -n "$NAME" ]]; then
    OUT_NAME="$NAME"
elif [[ -n "$LIST" ]]; then
    OUT_NAME="${PREFIX}_validated_on_uttrekk_${UTTREKK_NR}_${LIST}"
else
    OUT_NAME="${PREFIX}_validated_on_uttrekk_${UTTREKK_NR}_all"
fi
OUT_DIR="$SLADD_VALIDATION/$OUT_NAME"

# ── Check that the files exist ───────────────────────────────────
if [[ -z "$DEPLOY" && ! -f "$MODEL" ]]; then
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

if [[ -n "$DEPLOY" ]]; then
    HEALTH_URL="${API_URL%/model}/health"
    if ! curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
        echo "ERROR: no answer from $HEALTH_URL"
        case "$DEPLOY" in
            prod|test) echo "       Start the deployment first: ./deploy.sh start $DEPLOY" ;;
            *)         echo "       Check that something is listening there." ;;
        esac
        exit 1
    fi
fi

# Without list= all documents run: labels cover the whole uttrekk, so a document with no rows holds zero fnr.

# ── Everything below goes to the run directory as well ───────────
# Appended, not truncated: a rerun lands under a new date line in the same
# file, so the log tells the whole story of the directory.
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"
echo "════════ $(date '+%Y-%m-%d %H:%M:%S') ════════" >> "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Show what is being run ───────────────────────────────────────
echo "╭─────────────────────────────────────────────╮"
if [[ -n "$DEPLOY" ]]; then
    echo "│ Deployment validation (over HTTP): $OUT_NAME"
else
    echo "│ Full validation (OCR+YOLO): $OUT_NAME"
fi
echo "├─────────────────────────────────────────────┤"
if [[ -n "$DEPLOY" ]]; then
    printf "│ api:      %s\n" "$API_URL"
    printf "│ tag:      %s\n" "${TAG:-(unknown: no .env in $SCRIPT_DIR)}"
    printf "│ modell:   %s\n" "${MODEL_NAME:-(unknown: read it with ./deploy.sh status)}"
else
    printf "│ modell:   %s\n" "$MODEL"
fi
printf "│ uttrekk:  %s\n" "$UTTREKK_DIR"
if [[ -n "$LIST_FILE" ]]; then
    printf "│ liste:    %s\n" "$LIST_FILE"
else
    printf "│ liste:    (all documents)\n"
fi
printf "│ truth:    %s\n" "$TRUTH"
printf "│ out dir:  %s\n" "$OUT_DIR"
printf "│ logg:     %s\n" "$LOG_FILE"
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    printf "│ flagg:    %s\n" "${EXTRA_FLAGS[*]}"
fi
if [[ -n "$DEPLOY" ]]; then
    printf "│ cache:    (none: the container renders and reads every document itself)\n"
else
    printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/{ocr,yolo}"
fi
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Fill the caches first ────────────────────────────────────────
# precache.py does run.py's work in parallel processes against the same GPU, measured 3.3x on V100S.
# With deploy= there is nothing to fill: the container keeps its own caches.
START_TS=$(date +%s)
if [[ "$PRECACHE" == "yes" && -z "$DEPLOY" ]]; then
    echo "── $(date '+%H:%M:%S') Filling cache (precache.py) ──"
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
    --csv --truth --only-error
    --truth-csv "$TRUTH"
    --csv-out "$OUT_DIR/resultat.csv"
    --png-dir "$OUT_DIR/error_images"
    --result-dir "$OUT_DIR"
    --time
)

if [[ -n "$DEPLOY" ]]; then
    CMD+=(--api-url "$API_URL")
else
    CMD+=(--yolo-weights "$MODEL")
fi

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

# ── Live summary while the run goes ─────────────────────────────
# resultat.csv grows one row per surviving box, so documents, boxes and the
# kilde split can be counted live. Boxes the verifier drops never reach the
# CSV; the judgement-cache growth stands in as "fresh VLM calls" instead,
# counted from the first tick that sees the cache directory in the log.
live_summary() {
    local csv="$OUT_DIR/resultat.csv" cache="" base=""
    while sleep "$LIVE_INTERVAL"; do
        [[ -f "$csv" ]] || continue
        if [[ -z "$cache" ]]; then
            cache=$({ grep -m1 "Judgement cache: " "$LOG_FILE" 2>/dev/null || true; } | sed 's/.*Judgement cache: //')
        fi
        local extra=""
        if [[ -n "$cache" && -d "$cache" ]]; then
            local n_cache
            n_cache=$(find "$cache" -type f 2>/dev/null | wc -l | tr -d ' ')
            [[ -z "$base" ]] && base="$n_cache"
            extra="  vlm-nydømt $((n_cache - base))"
        fi
        awk -F, -v extra="$extra" -v ts="$(date '+%H:%M:%S')" '
            NR > 1 { n++; k[$9]++; if (!seen[$1]++) d++ }
            END { printf "── live %s  dok %d  bokser %d", ts, d, n
                  for (s in k) printf "  %s %d", s, k[s]
                  printf "%s ──\n", extra }' "$csv"
    done
}

echo "── $(date '+%H:%M:%S') Validation (run.py) ──"
printf 'kommando:'
printf ' %q' "${CMD[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
echo ""

LIVE_PID=""
if (( LIVE_INTERVAL > 0 )); then
    live_summary &
    LIVE_PID=$!
fi

RC=0
"${CMD[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} || RC=$?
if [[ -n "$LIVE_PID" ]]; then
    kill "$LIVE_PID" 2>/dev/null || true
fi
MINUTES=$(( ($(date +%s) - START_TS) / 60 ))
echo ""
echo "── $(date '+%H:%M:%S') Done: exit $RC after $MINUTES min ──"
echo "   Log: $LOG_FILE"
exit $RC
