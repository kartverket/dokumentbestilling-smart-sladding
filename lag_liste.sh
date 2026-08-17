#!/usr/bin/env bash
# lag_liste.sh — generer en dokument-ID-liste basert på metadata-filter
#
# Bruk (eksplisitte navngitte parametere):
#   ./lag_liste.sh uttrekk=5 docs=SR_JOU name=jou
#   ./lag_liste.sh uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou
#   ./lag_liste.sh uttrekk=5 years=2024,2025 name=samlet
#
# Parametere:
#   uttrekk  — uttrekk-nummer (påkrevd)
#   name     — filnavn-alias for utfilen (påkrevd)
#   docs     — kommaseparert liste med dokumenttyper å filtrere på (valgfri)
#   years    — årsfilter: range (2020-2026) eller kommaseparert (2024,2025) (valgfri)
#
# Minst én av docs/years må angis.
# Lagrer listen til $SLADD_LISTER/uttrekk_<nr>_<name>.txt
# Krever at server.env er sourcet.

set -euo pipefail

if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "FEIL: SLADD_-variablene er ikke satt. Kjør: source activate.sh"
    exit 1
fi

# ── Parse navngitte parametere ────────────────────────────────────
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
            echo "FEIL: Ukjent parameter: $arg"
            echo "Gyldige: uttrekk=N docs=TYPE[,TYPE] years=RANGE name=ALIAS"
            exit 1
            ;;
    esac
done

# ── Validering ────────────────────────────────────────────────────
if [[ -z "$UTTREKK" ]]; then
    echo "FEIL: uttrekk= er påkrevd"
    echo "Eksempel: $0 uttrekk=5 docs=SR_JOU name=jou"
    exit 1
fi

if [[ -z "$NAME" ]]; then
    echo "FEIL: name= er påkrevd"
    echo "Eksempel: $0 uttrekk=5 docs=SR_JOU name=jou"
    exit 1
fi

if [[ -z "$DOCS" && -z "$YEARS" ]]; then
    echo "FEIL: Minst én av docs= eller years= må angis"
    echo "Eksempler:"
    echo "  $0 uttrekk=5 docs=SR_JOU name=jou"
    echo "  $0 uttrekk=5 years=2020-2026 name=nyere"
    echo "  $0 uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou_reg"
    exit 1
fi

METADATA="$SLADD_METADATA/uttrekk_${UTTREKK}.csv"
UT_FIL="$SLADD_LISTER/uttrekk_${UTTREKK}_${NAME}.txt"

if [[ ! -f "$METADATA" ]]; then
    echo "FEIL: Metadata-fil finnes ikke: $METADATA"
    exit 1
fi

mkdir -p "$SLADD_LISTER"

# ── Bygg awk-filter ──────────────────────────────────────────────
# Kolonne 6 = rettsstiftelsestyper, kolonne for dokument_aar finnes i headeren.
# Vi bruker headeren for å finne riktig kolonne dynamisk.

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
    # Dokumenttype-filter
    if (docs != "") {
        n_docs = split(docs, doc_arr, ",")
        doc_match = 0
        for (d=1; d<=n_docs; d++) {
            if ($doc_col ~ doc_arr[d]) { doc_match = 1; break }
        }
        if (!doc_match) next
    }

    # Årsfilter
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

# Parse years-parameter: range (2020-2026) eller kommaseparert (2024,2025)
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
    "$AWK_SCRIPT" "$METADATA" > "$UT_FIL"

ANTALL=$(wc -l < "$UT_FIL")

# ── Vis sammendrag ───────────────────────────────────────────────
echo "╭──────────────────────────────────────────╮"
echo "│ lag_liste.sh"
echo "├──────────────────────────────────────────┤"
printf "│ uttrekk:  %s\n" "$UTTREKK"
[[ -n "$DOCS" ]]  && printf "│ docs:     %s\n" "$DOCS"
[[ -n "$YEARS" ]] && printf "│ years:    %s\n" "$YEARS"
printf "│ name:     %s\n" "$NAME"
echo "├──────────────────────────────────────────┤"
printf "│ → %s IDer lagret til:\n" "$ANTALL"
printf "│   %s\n" "$UT_FIL"
echo "╰──────────────────────────────────────────╯"
