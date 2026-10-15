#!/usr/bin/env bash
# lag_liste.sh: generate a document ID list from a metadata filter.
#
#   ./lag_liste.sh uttrekk=N name=ALIAS [docs=TYPE[,TYPE]] [years=RANGE]
#
#   uttrekk  uttrekk number (required)
#   name     filename alias for the output file (required)
#   docs     comma-separated document types to filter on
#   years    year filter: range (2020-2026) or comma-separated (2024,2025)
#
# At least one of docs/years must be given. Writes $SLADD_LISTS/uttrekk_<nr>_<name>.txt.
# Requires server.env to be sourced (source activate.sh).

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "ERROR: the SLADD_ variables are not set. Run: source activate.sh"
    exit 1
fi

# ── Parse named parameters ────────────────────────────────────────
UTTREKK=""
NAME=""
DOCS=""
YEARS=""

for arg in "$@"; do
    case "$arg" in
        uttrekk=*) UTTREKK="${arg#uttrekk=}" ;;
        name=*)    NAME="${arg#name=}" ;;
        docs=*)    DOCS="${arg#docs=}" ;;
        years=*)   YEARS="${arg#years=}" ;;
        *)
            echo "ERROR: Unknown parameter: $arg"
            echo "Valid: uttrekk=N docs=TYPE[,TYPE] years=RANGE name=ALIAS"
            exit 1
            ;;
    esac
done

if [[ -z "$UTTREKK" ]]; then
    echo "ERROR: uttrekk= is required"
    echo "Example: $0 uttrekk=5 docs=SR_JOU name=jou"
    exit 1
fi

if [[ -z "$NAME" ]]; then
    echo "ERROR: name= is required"
    echo "Example: $0 uttrekk=5 docs=SR_JOU name=jou"
    exit 1
fi

if [[ -z "$DOCS" && -z "$YEARS" ]]; then
    echo "ERROR: at least one of docs= or years= must be given"
    echo "Examples:"
    echo "  $0 uttrekk=5 docs=SR_JOU name=jou"
    echo "  $0 uttrekk=5 years=2020-2026 name=nyere"
    echo "  $0 uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou_reg"
    exit 1
fi

METADATA="$SLADD_METADATA/uttrekk_${UTTREKK}.csv"
OUT_FILE="$SLADD_LISTS/uttrekk_${UTTREKK}_${NAME}.txt"

if [[ ! -f "$METADATA" ]]; then
    echo "ERROR: metadata file does not exist: $METADATA"
    exit 1
fi

mkdir -p "$SLADD_LISTS"

# ── Build the awk filter ─────────────────────────────────────────
# Column positions vary, so the header is used to locate them.
AWK_SCRIPT='
BEGIN { FS=","; doc_col=0; year_col=0 }

NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "rettsstiftelsestyper") doc_col = i
        if ($i == "dokument_aar") year_col = i
    }
    next
}

{
    # Document type filter
    if (docs != "") {
        n_docs = split(docs, doc_arr, ",")
        doc_match = 0
        for (d=1; d<=n_docs; d++) {
            if ($doc_col ~ doc_arr[d]) { doc_match = 1; break }
        }
        if (!doc_match) next
    }

    # Year filter
    if (years != "") {
        year_val = $year_col + 0
        if (year_from > 0 && year_to > 0) {
            if (year_val < year_from || year_val > year_to) next
        } else {
            n_years = split(years, year_arr, ",")
            year_match = 0
            for (y=1; y<=n_years; y++) {
                if (year_val == year_arr[y] + 0) { year_match = 1; break }
            }
            if (!year_match) next
        }
    }

    print $1
}
'

# years= is either a range (2020-2026) or a comma-separated list (2024,2025).
YEAR_FROM=0
YEAR_TO=0
YEARS_LIST=""

if [[ -n "$YEARS" ]]; then
    if [[ "$YEARS" == *-* ]]; then
        YEAR_FROM="${YEARS%-*}"
        YEAR_TO="${YEARS#*-}"
    else
        YEARS_LIST="$YEARS"
    fi
fi

awk -v docs="$DOCS" \
    -v years="$YEARS_LIST" \
    -v year_from="$YEAR_FROM" \
    -v year_to="$YEAR_TO" \
    "$AWK_SCRIPT" "$METADATA" > "$OUT_FILE"

COUNT=$(wc -l < "$OUT_FILE")

# ── Show a summary ───────────────────────────────────────────────
echo "╭──────────────────────────────────────────╮"
echo "│ lag_liste.sh"
echo "├──────────────────────────────────────────┤"
printf "│ uttrekk:  %s\n" "$UTTREKK"
[[ -n "$DOCS" ]]  && printf "│ docs:     %s\n" "$DOCS"
[[ -n "$YEARS" ]] && printf "│ years:    %s\n" "$YEARS"
printf "│ name:     %s\n" "$NAME"
echo "├──────────────────────────────────────────┤"
printf "│ → %s IDs written to:\n" "$COUNT"
printf "│   %s\n" "$OUT_FILE"
echo "╰──────────────────────────────────────────╯"
