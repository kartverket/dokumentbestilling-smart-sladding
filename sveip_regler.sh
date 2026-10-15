#!/usr/bin/env bash
# sveip_regler.sh: re-sweeps the filter rules after a model swap.
#
#   ./sveip_regler.sh res=PATH [uttrekk=N] [list=NAME] [cost=N] [seeds=A,B,C]
#                     [name=ALIAS]
#
#   res      resultat.csv from a rules=no run of the new model. A run WITH
#            rules cannot be swept: the boxes the rules drop are already gone
#            from the CSV, so every rule measures as free
#   uttrekk  uttrekk the run was made against (default: 6)
#   list     ID list the run used. Becomes --processed-list, so a document
#            where the model found nothing counts as run and its fasit stays
#            in scope (default: holdout48)
#   cost     removed oversladdinger one lost fasit box is worth (default: 50)
#   seeds    seeds for the internal holdout split, comma separated
#            (default: 42,7,2026). One seed decides nothing: oversladdinger
#            cluster in coordinate-heavy documents and land unevenly
#   name     output directory name (default: derived from the res path)
#
# Three passes per run: the whole set unbounded, the whole set cut to the
# configurations that reach the exchange rate, and one per seed with an
# internal holdout. Output lands in $SLADD_VALIDATION/sveip_<name>/.
#
# Requires server.env (source activate.sh).

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run first:"
    echo "  source activate.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RES=""
UTTREKK_NR="6"
LIST="holdout48"
COST="50"
SEEDS="42,7,2026"
NAME=""

for arg in "$@"; do
    case "$arg" in
        res=*)     RES="${arg#res=}" ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}" ;;
        list=*)    LIST="${arg#list=}" ;;
        cost=*)    COST="${arg#cost=}" ;;
        seeds=*)   SEEDS="${arg#seeds=}" ;;
        name=*)    NAME="${arg#name=}" ;;
        *) echo "ERROR: unknown parameter: $arg"
           echo "Valid: res=PATH [uttrekk=N] [list=NAME] [cost=N] [seeds=A,B,C] [name=ALIAS]"
           exit 1 ;;
    esac
done

if [[ -z "$RES" ]]; then
    echo "ERROR: res= is required (resultat.csv from a rules=no run)."
    echo ""
    echo "Produce one with:"
    echo "  ./valider_full.sh model=\$SLADD_WEIGHTS/<model>/<model>.pt \\"
    echo "      uttrekk=$UTTREKK_NR list=$LIST rules=no name=<model>_raa"
    exit 1
fi

TRUTH="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"
LIST_FILE=""
if [[ -n "$LIST" ]]; then
    LIST_FILE="$SLADD_LISTS/uttrekk_${UTTREKK_NR}_${LIST}.txt"
fi

for f in "$RES" "$TRUTH" ${LIST_FILE:+"$LIST_FILE"}; do
    [[ -f "$f" ]] || { echo "ERROR: cannot find $f"; exit 1; }
done

if [[ -z "$NAME" ]]; then
    NAME="$(basename "$(dirname "$RES")")"
fi
OUT_DIR="$SLADD_VALIDATION/sveip_${NAME}"
mkdir -p "$OUT_DIR"

echo "╭─────────────────────────────────────────────╮"
printf "│ res:      %s\n" "$RES"
printf "│ truth:    %s\n" "$TRUTH"
printf "│ liste:    %s\n" "${LIST_FILE:-(all documents in the result CSV)}"
printf "│ cost:     %s removed oversladdinger per lost fasit box\n" "$COST"
printf "│ seeds:    %s\n" "$SEEDS"
printf "│ out dir:  %s\n" "$OUT_DIR"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Preflight: is this really a rules=no run? ─────────────────────
# A CSV that already went through the postfilters holds no box the rules
# would drop, so every rule measures as free and the whole sweep is a lie.
echo "── Preflight ──"
python3 - "$RES" "$SCRIPT_DIR" <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(sys.argv[2], "utils"))
sys.path.insert(0, os.path.join(sys.argv[2], "app"))
from collections import Counter

from config import (MIN_SHORT_SIDE_YOLO_PT, MIN_SHORT_SIDE_PADDLE_PT,
                    MIN_LONG_SIDE_PADDLE_PT, MAX_ELONGATION_PADDLE)
from filter_common import read_predictions

pred = read_predictions(sys.argv[1])
if not pred:
    sys.exit("  EMPTY result CSV")

per_source = Counter(p["kilde"] for p in pred)
print(f"  {len(pred)} predictions   " +
      "  ".join(f"{k}:{v}" for k, v in sorted(per_source.items())))

n_features = sum(1 for p in pred if p.get("har_tokens"))
n_window = sum(1 for p in pred if p.get("maks_luke") is not None)
print(f"  {n_features} boxes with OCR features, {n_window} with window features")
if not n_features:
    print("  WARNING: no OCR features, the whole OCR half of the sweep is skipped")

# Boxes prod's dimension filters would have dropped. None left means the
# postfilters already ran.
raw = sum(1 for p in pred
          if (p["kilde"] == "yolo" and p["short_side"] < MIN_SHORT_SIDE_YOLO_PT)
          or (p["kilde"] == "paddle"
              and (p["short_side"] < MIN_SHORT_SIDE_PADDLE_PT
                   or p["long_side"] < MIN_LONG_SIDE_PADDLE_PT
                   or p["elongation"] > MAX_ELONGATION_PADDLE)))
print(f"  {raw} boxes below prod's dimension filters")
if not raw:
    sys.exit("  ERROR: not a single box the dimension filters would drop.\n"
             "  This looks like a run WITH rules. Rerun with rules=no.")
PY
echo ""

SWEEP=(python3 -u "$SCRIPT_DIR/utils/filter_sweep.py"
    --truth-csv "$TRUTH"
    --res-csv "$RES"
    --cost "$COST")
if [[ -n "$LIST_FILE" ]]; then
    SWEEP+=(--processed-list "$LIST_FILE")
fi

# ── Pass 1: whole set, nothing hidden ────────────────────────────
# The reference report. Today's operating point is in here, and so is every
# axis, including the ones that do not pay.
echo "── $(date '+%H:%M:%S') Pass 1: whole set ──"
"${SWEEP[@]}" --out "$OUT_DIR/01_alle.txt" --out-csv "$OUT_DIR/sveip.csv"

# ── Pass 2: whole set, cut to the exchange rate ──────────────────
# Same numbers, but only configurations that clear cost:1 and stay under
# three lost boxes. This is the shortlist to read.
echo ""
echo "── $(date '+%H:%M:%S') Pass 2 shortlist (ov/lost > $COST, lost <= 2) ──"
"${SWEEP[@]}" --max-lost 2 --min-ov-lost "$COST" \
    --out "$OUT_DIR/02_kortliste.txt"

# ── Pass 3: one per seed ─────────────────────────────────────────
# The internal holdout answers a different question than uttrekk 6's own:
# whether a configuration picked on one set of documents still pays on
# documents it was not picked on.
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
    seed="$(tr -d '[:space:]' <<< "$seed")"
    if [[ -z "$seed" ]]; then
        continue
    fi
    echo ""
    echo "── $(date '+%H:%M:%S') Pass 3 holdout, seed $seed ──"
    "${SWEEP[@]}" --holdout 0.3 --seed "$seed" --max-lost 2 \
        --min-ov-lost "$COST" --out "$OUT_DIR/03_holdout_seed${seed}.txt"
done

echo ""
echo "╭─────────────────────────────────────────────╮"
printf "│ Reports:  %s\n" "$OUT_DIR"
echo "│"
echo "│ Read in this order:"
echo "│   02_kortliste.txt  what pays at $COST:1, and today's operating point"
echo "│   03_holdout_*.txt  whether the same rows survive on other documents"
echo "│   01_alle.txt       every axis, including the ones that do not pay"
echo "│"
echo "│ Then hand-check the losses of the chosen configuration:"
echo "│   python3 utils/filter_review.py --truth-csv <truth> --res-csv <res> \\"
echo "│       --folder \$SLADD_UTTREKK/uttrekk_$UTTREKK_NR --only-lost <the flags>"
echo "╰─────────────────────────────────────────────╯"
