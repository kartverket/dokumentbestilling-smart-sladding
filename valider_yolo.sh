#!/usr/bin/env bash
# valider_yolo.sh — validering kun med YOLO (uten OCR)
#
# Bruk (eksplisitte navngitte parametere):
#   ./valider_yolo.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou
#   ./valider_yolo.sh modell=$SLADD_PRODVEKTER uttrekk=5           # alle dokumenter
#   ./valider_yolo.sh modell=$SLADD_RUNS/uttrekk_4_jou_med_negative/weights/best.pt uttrekk=5 liste=jou
#   ./valider_yolo.sh modell=$SLADD_RUNS/uttrekk_4_jou_based_pat20/weights/best.pt uttrekk=4 liste=mob
#
# Se også: valider_full.sh — full produksjonslogikk (OCR + YOLO)
#
# Parametere:
#   modell   — sti til YOLO-vektfil (påkrevd)
#   uttrekk  — uttrekk-nummer å validere på (påkrevd)
#   liste    — navn på ID-listen (valgfri; uten = kjører alle dokumenter)
#   navn     — egendefinert navn på utmappen (valgfri, utledes fra modell ellers)
#
# Modellnavnet utledes automatisk fra stien for å navngi utmappen.
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
if [[ -n "$NAVN" ]]; then
    MODELL_NAVN="$NAVN"
elif [[ "$MODELL" == "$SLADD_PRODVEKTER" ]]; then
    MODELL_NAVN="yolo-yearly-10000"
else
    # Bruk mappenavnet som inneholder weights/best.pt
    MODELL_NAVN=$(basename "$(dirname "$(dirname "$MODELL")")")
fi

# ── Bygg stier ───────────────────────────────────────────────────
UTTREKK_MAPPE="$SLADD_UTTREKK/uttrekk_${UTTREKK_NR}"
FASIT="$SLADD_LABELS/uttrekk_${UTTREKK_NR}.csv"

LISTE_FIL=""
if [[ -n "$LISTE" ]]; then
    LISTE_FIL="$SLADD_LISTER/uttrekk_${UTTREKK_NR}_${LISTE}.txt"
    UT_NAVN="${MODELL_NAVN}_validert_pa_uttrekk_${UTTREKK_NR}_${LISTE}"
else
    UT_NAVN="${MODELL_NAVN}_validert_pa_uttrekk_${UTTREKK_NR}_alle"
fi
UT_MAPPE="$SLADD_VALIDERING/$UT_NAVN"

# ── Sjekk at filer finnes ────────────────────────────────────────
SJEKK_FILER=("$MODELL" "$FASIT")
if [[ -n "$LISTE_FIL" ]]; then
    SJEKK_FILER+=("$LISTE_FIL")
fi
for fil in "${SJEKK_FILER[@]}"; do
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
echo "│ Validering: $UT_NAVN"
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
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/yolo"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Bygg kommando ────────────────────────────────────────────────
CMD=(python -u "$SLADD_RUN"
    --mappe "$UTTREKK_MAPPE"
    --yolo-vekter "$MODELL"
    --kun-yolo
    --csv --fasit --kun-feil
    --fasit-csv "$FASIT"
    --csv-ut "$UT_MAPPE/resultat.csv"
    --png-mappe "$UT_MAPPE/feilbilder"
    --resultat-mappe "$UT_MAPPE"
)

if [[ -n "$LISTE_FIL" ]]; then
    CMD+=(--velg-fra-fil "$LISTE_FIL")
else
    CMD+=(--antall alle)
fi

# ── Kjør validering ──────────────────────────────────────────────
"${CMD[@]}" ${EKSTRA_FLAGG[@]+"${EKSTRA_FLAGG[@]}"}
