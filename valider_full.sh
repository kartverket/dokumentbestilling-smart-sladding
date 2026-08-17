#!/usr/bin/env bash
# valider_full.sh — validering med full produksjonslogikk (OCR + YOLO + matching)
#
# Bruk (eksplisitte navngitte parametere):
#   ./valider_full.sh uttrekk=5 liste=jou
#   ./valider_full.sh uttrekk=5 liste=jou navn=mitt-eksperiment
#
# Parametere:
#   uttrekk  — uttrekk-nummer å validere på (påkrevd)
#   liste    — navn på ID-listen (filnavn uten uttrekk_N_-prefix) (påkrevd)
#   navn     — egendefinert navn på utmappen (valgfri)
#
# Bruker OCR-cache ($SLADD_CACHE) for å unngå å kjøre OCR på nytt.
# Se også: valider_yolo.sh — kun YOLO (raskere, men uten OCR-matching)
# Krever at server.env er sourcet (SLADD_-variablene må finnes).

set -euo pipefail

# ── Sjekk at miljøvariabler er satt ──────────────────────────────
if [[ -z "${SLADD_REPO:-}" ]]; then
    echo "FEIL: SLADD_-variablene er ikke satt. Kjør først:"
    echo "  source activate.sh"
    exit 1
fi

# ── Parse navngitte parametere ────────────────────────────────────
UTTREKK_NR=""
LISTE=""
NAVN=""
EKSTRA_FLAGG=()

for arg in "$@"; do
    case "$arg" in
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}" ;;
        liste=*)   LISTE="${arg#liste=}" ;;
        navn=*)    NAVN="${arg#navn=}" ;;
        -*)        EKSTRA_FLAGG+=("$arg") ;;
        *)
            echo "FEIL: Ukjent parameter: $arg"
            echo "Gyldige: uttrekk=N liste=NAVN [navn=ALIAS]"
            exit 1
            ;;
    esac
done

# ── Validering ────────────────────────────────────────────────────
if [[ -z "$UTTREKK_NR" ]]; then
    echo "FEIL: uttrekk= er påkrevd"
    echo "Eksempel: $0 uttrekk=5 liste=jou"
    exit 1
fi

if [[ -z "$LISTE" ]]; then
    echo "FEIL: liste= er påkrevd"
    echo "Eksempel: $0 uttrekk=5 liste=jou"
    exit 1
fi

# ── Bygg stier ───────────────────────────────────────────────────
LISTE_FIL="$SLADD_LISTER/uttrekk_${UTTREKK_NR}_${LISTE}.txt"
UTTREKK_MAPPE="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
FASIT="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

if [[ -n "$NAVN" ]]; then
    UT_NAVN="$NAVN"
else
    UT_NAVN="full_validert_pa_uttrekk_${UTTREKK_NR}_${LISTE}"
fi
UT_MAPPE="$SLADD_VALIDERING/$UT_NAVN"

# ── Sjekk at filer finnes ────────────────────────────────────────
for fil in "$LISTE_FIL" "$FASIT"; do
    if [[ ! -f "$fil" ]]; then
        echo "FEIL: Finner ikke: $fil"
        exit 1
    fi
done

if [[ ! -d "$UTTREKK_MAPPE" ]]; then
    echo "FEIL: Uttrekk-mappe finnes ikke: $UTTREKK_MAPPE"
    exit 1
fi

# ── Vis hva som kjøres ──────────────────────────────────────────
echo "╭─────────────────────────────────────────────╮"
echo "│ Full validering (OCR+YOLO): $UT_NAVN"
echo "├─────────────────────────────────────────────┤"
printf "│ uttrekk:  %s\n" "$UTTREKK_MAPPE"
printf "│ liste:    %s\n" "$LISTE_FIL"
printf "│ fasit:    %s\n" "$FASIT"
printf "│ utmappe:  %s\n" "$UT_MAPPE"
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/ocr"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Kjør full validering (produksjonslogikk) ─────────────────────
python -u "$SLADD_RUN" \
    --mappe "$UTTREKK_MAPPE" \
    --velg-fra-fil "$LISTE_FIL" \
    --csv --fasit --kun-feil \
    --fasit-csv "$FASIT" \
    --csv-ut "$UT_MAPPE/resultat.csv" \
    --png-mappe "$UT_MAPPE/feilbilder" \
    --resultat-mappe "$UT_MAPPE" \
    --tid \
    ${EKSTRA_FLAGG[@]+"${EKSTRA_FLAGG[@]}"}


