#!/usr/bin/env bash
# valider_full.sh — validering med full produksjonslogikk (OCR + YOLO + matching)
#
# Bruk (eksplisitte navngitte parametere):
#   ./valider_full.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou
#   ./valider_full.sh modell=$SLADD_PRODVEKTER uttrekk=5           # alle dokumenter
#   ./valider_full.sh modell=$SLADD_RUNS/mitt-run/weights/best.pt uttrekk=5 liste=jou navn=mitt-eksperiment
#
# Parametere:
#   modell   — sti til YOLO-vektfil (påkrevd)
#   uttrekk  — uttrekk-nummer å validere på (påkrevd)
#   liste    — navn på ID-listen (valgfri; uten = kjører alle dokumenter)
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
MODELL=""
UTTREKK_NR=""
LISTE=""
NAVN=""
EKSTRA_FLAGG=()

for arg in "$@"; do
    case "$arg" in
        modell=*)  MODELL="${arg#modell=}" ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}" ;;
        liste=*)   LISTE="${arg#liste=}" ;;
        navn=*)    NAVN="${arg#navn=}" ;;
        -*)        EKSTRA_FLAGG+=("$arg") ;;
        *)
            echo "FEIL: Ukjent parameter: $arg"
            echo "Gyldige: modell=STI uttrekk=N [liste=NAVN] [navn=ALIAS]"
            exit 1
            ;;
    esac
done

# ── Validering ────────────────────────────────────────────────────
if [[ -z "$MODELL" ]]; then
    echo "FEIL: modell= er påkrevd"
    echo "Eksempel: $0 modell=\$SLADD_PRODVEKTER uttrekk=5 liste=jou"
    exit 1
fi

if [[ -z "$UTTREKK_NR" ]]; then
    echo "FEIL: uttrekk= er påkrevd"
    echo "Eksempel: $0 modell=\$SLADD_PRODVEKTER uttrekk=5 liste=jou"
    exit 1
fi

# ── Utled modellnavn fra stien ───────────────────────────────────
if [[ "$MODELL" == "$SLADD_PRODVEKTER" ]]; then
    MODELL_NAVN="yolo-yearly-10000"
else
    MODELL_NAVN=$(basename "$(dirname "$(dirname "$MODELL")")")
fi

# ── Bygg stier ───────────────────────────────────────────────────
UTTREKK_MAPPE="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
FASIT="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

LISTE_FIL=""
if [[ -n "$LISTE" ]]; then
    LISTE_FIL="$SLADD_LISTER/uttrekk_${UTTREKK_NR}_${LISTE}.txt"
fi

if [[ -n "$NAVN" ]]; then
    UT_NAVN="$NAVN"
elif [[ -n "$LISTE" ]]; then
    UT_NAVN="full_${MODELL_NAVN}_validert_pa_uttrekk_${UTTREKK_NR}_${LISTE}"
else
    UT_NAVN="full_${MODELL_NAVN}_validert_pa_uttrekk_${UTTREKK_NR}_alle"
fi
UT_MAPPE="$SLADD_VALIDERING/$UT_NAVN"

# ── Sjekk at filer finnes ────────────────────────────────────────
if [[ ! -f "$MODELL" ]]; then
    echo "FEIL: Finner ikke modell: $MODELL"
    exit 1
fi

if [[ -n "$LISTE_FIL" && ! -f "$LISTE_FIL" ]]; then
    echo "FEIL: Finner ikke: $LISTE_FIL"
    exit 1
fi

if [[ ! -f "$FASIT" ]]; then
    echo "FEIL: Finner ikke: $FASIT"
    exit 1
fi

if [[ ! -d "$UTTREKK_MAPPE" ]]; then
    echo "FEIL: Uttrekk-mappe finnes ikke: $UTTREKK_MAPPE"
    exit 1
fi

# ── Vis hva som kjøres ──────────────────────────────────────────
echo "╭─────────────────────────────────────────────╮"
echo "│ Full validering (OCR+YOLO): $UT_NAVN"
echo "├─────────────────────────────────────────────┤"
printf "│ modell:   %s\n" "$MODELL"
printf "│ uttrekk:  %s\n" "$UTTREKK_MAPPE"
if [[ -n "$LISTE_FIL" ]]; then
    printf "│ liste:    %s\n" "$LISTE_FIL"
else
    printf "│ liste:    (alle dokumenter)\n"
fi
printf "│ fasit:    %s\n" "$FASIT"
printf "│ utmappe:  %s\n" "$UT_MAPPE"
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/ocr"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Bygg kommando ────────────────────────────────────────────────
CMD=(python -u "$SLADD_RUN"
    --mappe "$UTTREKK_MAPPE"
    --yolo-vekter "$MODELL"
    --csv --fasit --kun-feil
    --fasit-csv "$FASIT"
    --csv-ut "$UT_MAPPE/resultat.csv"
    --png-mappe "$UT_MAPPE/feilbilder"
    --resultat-mappe "$UT_MAPPE"
    --tid
)

if [[ -n "$LISTE_FIL" ]]; then
    CMD+=(--velg-fra-fil "$LISTE_FIL")
else
    CMD+=(--antall alle)
fi

# ── Kjør full validering (produksjonslogikk) ─────────────────────
"${CMD[@]}" ${EKSTRA_FLAGG[@]+"${EKSTRA_FLAGG[@]}"}
