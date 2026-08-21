#!/usr/bin/env bash
# valider_full.sh — validering med full produksjonslogikk (OCR + YOLO + matching)
#
# Bruk (eksplisitte navngitte parametere):
#   ./valider_full.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou
#   ./valider_full.sh modell=$SLADD_PRODVEKTER uttrekk=5           # alle dokumenter
#   ./valider_full.sh modell=$SLADD_VEKTER/mitt-run/mitt-run.pt uttrekk=5 liste=jou navn=mitt-eksperiment
#
# Parametere:
#   modell   — sti til YOLO-vektfil (påkrevd)
#   uttrekk  — uttrekk-nummer å validere på (påkrevd)
#   liste    — navn på ID-listen (valgfri; uten = kjører alle dokumenter)
#   navn     — egendefinert navn på utmappen (valgfri)
#   precache — 'nei' for å hoppe over cache-fyllingen (default: ja)
#   regler   — 'nei' for å hoppe over ALLE etterfiltrene (rå deteksjon;
#              basislinjemåling av regelverkets totalbidrag)
#   metadata — 'ja' for å sende rettsstiftelsestyper fra
#              $SLADD_METADATA/uttrekk_N.csv (regelprofiler som i prod),
#              eller en eksplisitt sti. Uten = global oppførsel.
#   bilder   — 'nei'/0 for å hoppe over feilbildene, eller et tall N for å
#              tegne maks N dokumenter (default: alle). Sammendrag og
#              resultat.csv påvirkes ikke — de beregnes fra boksene alene.
#   prosesser— antall arbeiderprosesser for cache-treff i valideringen
#              (default: auto = min(8, kjerner); 1 = sekvensielt)
#
# Bruker OCR- og YOLO-cache ($SLADD_CACHE) for å unngå å kjøre OCR og YOLO på nytt.
# YOLO-cachen er per vektfil. Treffer begge, hoppes også PDF-renderingen over,
# så en ny kjøring av samme modell koster nesten ingenting.
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
PRECACHE="ja"
BILDER="alle"
METADATA=""
REGLER="ja"
PROSESSER=""
EKSTRA_FLAGG=()

for arg in "$@"; do
    case "$arg" in
        modell=*)  MODELL="${arg#modell=}" ;;
        uttrekk=*) UTTREKK_NR="${arg#uttrekk=}" ;;
        liste=*)   LISTE="${arg#liste=}" ;;
        navn=*)    NAVN="${arg#navn=}" ;;
        precache=*) PRECACHE="${arg#precache=}" ;;
        metadata=*) METADATA="${arg#metadata=}" ;;
        regler=*)   REGLER="${arg#regler=}" ;;
        bilder=*)  BILDER="${arg#bilder=}" ;;
        prosesser=*) PROSESSER="${arg#prosesser=}" ;;
        -*)        EKSTRA_FLAGG+=("$arg") ;;
        *)
            echo "FEIL: Ukjent parameter: $arg"
            echo "Gyldige: modell=STI uttrekk=N [liste=NAVN] [navn=ALIAS] [precache=nei] [bilder=N] [prosesser=N]"
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
# Publiserte modeller heter <navn>/<navn>.pt og bærer navnet sitt selv.
# Rå treningskjøringer heter <run>/weights/best.pt — da er navnet på
# run-mappen det eneste navnet som finnes.
utled_modellnavn() {
    local navn
    navn=$(basename "$1"); navn=${navn%.pt}
    if [[ "$navn" == best || "$navn" == last ]]; then
        navn=$(basename "$(dirname "$(dirname "$1")")")
    fi
    echo "$navn"
}

MODELL_NAVN=$(utled_modellnavn "$MODELL")

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

# Kjøringen omfatter ALLE dokumentene i mappa (uten liste=): labels-filen
# dekker hele uttrekket, så et dokument uten rader der er gjennomgått med
# null fnr — prediksjoner på det er ekte oversladdinger som skal telles.
# Analyseverktøyene regner dem med (inkluder-ulabelte er default på).

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
printf "│ cache:    %s\n" "$SLADD_CACHE/uttrekk_${UTTREKK_NR}/{ocr,yolo}"
echo "╰─────────────────────────────────────────────╯"
echo ""

# ── Fyll cachene først ───────────────────────────────────────────
# run.py kjører ett dokument om gangen i én prosess. precache.py gjør
# det samme arbeidet i parallelle prosesser mot samme GPU — målt 3,3×
# på V100S — og legger det i cachen run.py leser. Etterpå er
# valideringen nesten gratis. precache=nei hopper over steget.
if [[ "$PRECACHE" == "ja" ]]; then
    echo "── Fyller cache (precache.py) ──"
    PRECACHE_CMD=(python -u "${SLADD_PRECACHE:-$SLADD_REPO/utils/precache.py}"
        --mappe "$UTTREKK_MAPPE"
        --kun begge
        --yolo-vekter "$MODELL"
    )
    if [[ -n "$LISTE_FIL" ]]; then
        PRECACHE_CMD+=(--velg-fra-fil "$LISTE_FIL")
    fi
    "${PRECACHE_CMD[@]}"
    echo ""
fi

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

if [[ -n "$METADATA" ]]; then
    if [[ "$METADATA" == "ja" || "$METADATA" == "auto" ]]; then
        METADATA="$SLADD_METADATA/uttrekk_${UTTREKK_NR}.csv"
    fi
    if [[ ! -f "$METADATA" ]]; then
        echo "FEIL: Finner ikke metadata-CSV: $METADATA"
        exit 1
    fi
    CMD+=(--metadata-csv "$METADATA")
fi

if [[ "$REGLER" == "nei" || "$REGLER" == "0" ]]; then
    CMD+=(--uten-etterfilter)
fi

if [[ "$BILDER" != "alle" ]]; then
    if [[ "$BILDER" == "nei" ]]; then
        BILDER=0
    fi
    if ! [[ "$BILDER" =~ ^[0-9]+$ ]]; then
        echo "FEIL: bilder= må være 'alle', 'nei' eller et tall (fikk: $BILDER)"
        exit 1
    fi
    CMD+=(--maks-feilbilder "$BILDER")
fi

if [[ -n "$PROSESSER" ]]; then
    if ! [[ "$PROSESSER" =~ ^[0-9]+$ ]]; then
        echo "FEIL: prosesser= må være et tall (fikk: $PROSESSER)"
        exit 1
    fi
    CMD+=(--prosesser "$PROSESSER")
fi

# ── Kjør full validering (produksjonslogikk) ─────────────────────
"${CMD[@]}" ${EKSTRA_FLAGG[@]+"${EKSTRA_FLAGG[@]}"}
