#!/usr/bin/env bash
# kjor_48t.sh: unattended 48-hour training window on the uttrekk 4 + 6 mix.
# Usage: preflight | start | status. Documented in train/KJOR_48T.md (untracked).

set -uo pipefail

HOURS="${HOURS:-48}"
COST="${COST:-20}"
IMGSZ="${IMGSZ:-1280}"
PATIENCE="${PATIENCE:-12}"
RESERVE_H="${RESERVE_H:-5}"        # hours held back for eval + ranking
B_MIN_H="${B_MIN_H:-6}"           # run B is skipped below this budget
MIN_FREE_GB="${MIN_FREE_GB:-80}"
SEED="${SEED:-48}"
HOLDOUT_SHARE="${HOLDOUT_SHARE:-0.4}"
BASE_L="${BASE_L:-yolo26l.pt}"
PROXY="${PROXY:-http://159.162.48.7:3128}"
DATASET="${DATASET:-/data2/smartsladding-trening/dataset_48t}"
OLD_DATASET="${OLD_DATASET:-/data2/smartsladding-trening/dataset_2026-07-15T18-34-22}"

TAG=48t
LOGDIR="${LOGDIR:-$SLADD_RUNS/${TAG}_styring}"
LOG="$LOGDIR/kjor.log"
RUNLIST="$LOGDIR/runs.list"
DEADLINE_FILE="$LOGDIR/deadline"
SCRIPTS="$SLADD_REPO/train/scripts"
LIST_TRAIN="$SLADD_LISTS/uttrekk_6_tren48.txt"
LIST_HOLDOUT="$SLADD_LISTS/uttrekk_6_holdout48.txt"

log() { printf '%s  %s\n' "$(date '+%d.%m %H:%M:%S')" "$*"; }

left_s() { echo $(( $(cat "$DEADLINE_FILE") - $(date +%s) )); }

left_h() { echo $(( $(left_s) / 3600 )); }

free_gb() { df -BG --output=avail /data2 | tail -1 | tr -dc 0-9; }

gpu_procs() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | sed '/^$/d'; }

krev() { [[ -e "$1" ]] || { log "MANGLER: $1"; return 1; }; }

# ── preflight ────────────────────────────────────────────────────

preflight() {
    local ok=0
    [[ -n "${SLADD_REPO:-}" ]] || { echo "Kjør først: source activate.sh"; exit 1; }
    for f in "$SLADD_LABELS/uttrekk_4.csv" "$SLADD_LABELS/uttrekk_6.csv" \
             "$SLADD_METADATA/uttrekk_4.csv" "$SLADD_METADATA/uttrekk_6.csv" \
             "$SLADD_UTTREKK/uttrekk_4" "$SLADD_UTTREKK/uttrekk_6" \
             "$OLD_DATASET/images_all" "$SLADD_PRODWEIGHTS" \
             "$SCRIPTS/rank_models.py" "$SCRIPTS/make_holdout_lists.py"; do
        krev "$f" || ok=1
    done
    local free; free=$(free_gb)
    if (( free < MIN_FREE_GB + 120 )); then
        log "ADVARSEL: ${free}G ledig på /data2; renders + splittkopier vil trolig trenge ~150G"
        ok=1
    else
        log "Disk: ${free}G ledig på /data2"
    fi
    if [[ -n "$(gpu_procs)" ]]; then
        log "GPU er IKKE ledig:"; gpu_procs
        log "Stopp llama-server og vent på kjørende valideringer før start."
        ok=1
    else
        log "GPU er ledig."
    fi
    if systemctl is-active --quiet llama-server 2>/dev/null; then
        log "ADVARSEL: llama-server er aktiv (sudo systemctl stop llama-server)"
        ok=1
    fi
    if [[ ! -f "$SLADD_TRAIN/$BASE_L" ]]; then
        log "Henter $BASE_L via proxy ..."
        ( cd "$SLADD_TRAIN" &&
          https_proxy="$PROXY" HTTPS_PROXY="$PROXY" \
          python -c "from ultralytics import YOLO; YOLO('$BASE_L')" ) || ok=1
    fi
    python -c "import torch, ultralytics, fitz, pandas" || ok=1
    if (( ok == 0 )); then
        log "Preflight OK. Start med:  tmux new -s 48t '$SLADD_REPO/train/kjor_48t.sh start'"
    else
        log "Preflight fant problemer, se over."
    fi
    exit $ok
}

# ── phases ───────────────────────────────────────────────────────

fase_lister() {
    [[ -f "$LIST_HOLDOUT" && -f "$LIST_TRAIN" ]] && { log "Lister finnes, gjenbrukes."; return 0; }
    comm -12 <(ls "$SLADD_UTTREKK/uttrekk_4" | sed 's/\.pdf$//' | sort) \
             <(ls "$SLADD_UTTREKK/uttrekk_6" | sed 's/\.pdf$//' | sort) \
             > "$LOGDIR/overlapp_4_6.txt"
    log "Overlapp uttrekk 4/6: $(wc -l < "$LOGDIR/overlapp_4_6.txt") dokumenter"
    python "$SCRIPTS/make_holdout_lists.py" \
        --labels "$SLADD_LABELS/uttrekk_6.csv" \
        --metadata "$SLADD_METADATA/uttrekk_6.csv" \
        --force-train "$LOGDIR/overlapp_4_6.txt" \
        --holdout-share "$HOLDOUT_SHARE" --seed "$SEED" \
        --out-train "$LIST_TRAIN" --out-holdout "$LIST_HOLDOUT"
}

fase_datasett() {
    mkdir -p "$DATASET/images_all" "$DATASET/labels_all"
    if [[ ! -f "$DATASET/.48t_linket" ]]; then
        log "Hardlenker gamle renders fra $OLD_DATASET ..."
        cp -al "$OLD_DATASET/images_all/." "$DATASET/images_all/"
        cp -a  "$OLD_DATASET/labels_all/." "$DATASET/labels_all/"
        touch "$DATASET/.48t_linket"
    fi
    if [[ ! -f "$DATASET/.48t_convert_u4" ]]; then
        log "Convert uttrekk 4 (hele fasiten) ..."
        python "$SCRIPTS/convert_csv_to_yolo.py" "$SLADD_LABELS/uttrekk_4.csv" \
            "$SLADD_UTTREKK/uttrekk_4" --output "$DATASET" && touch "$DATASET/.48t_convert_u4"
    fi
    if [[ ! -f "$DATASET/.48t_convert_u6" ]]; then
        log "Convert uttrekk 6 (treningsandelen) ..."
        python "$SCRIPTS/convert_csv_to_yolo.py" "$SLADD_LABELS/uttrekk_6.csv" \
            "$SLADD_UTTREKK/uttrekk_6" --output "$DATASET" --ids "$LIST_TRAIN" \
            && touch "$DATASET/.48t_convert_u6"
    fi
    if [[ ! -f "$DATASET/.48t_splittet" ]]; then
        rm -rf "$DATASET/images" "$DATASET/labels"
        python "$SCRIPTS/split_train_val.py" --dataset "$DATASET" \
            --train-ratio 0.8 --val-ratio 0.1 --seed 42 \
            && touch "$DATASET/.48t_splittet"
    fi
    { echo "path: $DATASET"; echo "train: images/train"; echo "val: images/val";
      echo "test: images/test"; echo ""; echo "names:"; echo "  0: fnr"; } > "$DATASET/data.yaml"
    log "Datasett: $(ls "$DATASET/images_all" | wc -l) sider totalt, $(ls "$DATASET/images/train" | wc -l) i train."
}

sekunder_per_epoke() {
    python - "$1" <<'EOF'
import sys
import pandas as pd
r = pd.read_csv(sys.argv[1] + "/results.csv")
r.columns = [c.strip() for c in r.columns]
t = r["time"]
print(int(t.iloc[-1] - t.iloc[-2]) if len(t) > 1 else int(t.iloc[-1]))
EOF
}

# Stdout is the measured number alone; everything else must go to stderr.
fase_pilot() {
    local navn=$1 base=$2
    local run="$LOGDIR/pilot/$navn"
    if [[ ! -f "$run/results.csv" ]]; then
        log "Pilot $navn (2 epoker, autobatch) ..." >&2
        yolo detect train data="$DATASET/data.yaml" model="$base" epochs=2 \
            imgsz="$IMGSZ" batch=-1 device=0 workers=16 plots=False \
            project="$LOGDIR/pilot" name="$navn" exist_ok=True >&2 || return 1
    fi
    sekunder_per_epoke "$run"
}

tren() {
    local navn=$1 base=$2 epoker=$3 datayaml=$4
    log "Trener $navn: $epoker epoker fra $base ..."
    if ! yolo detect train data="$datayaml" model="$base" epochs="$epoker" \
            imgsz="$IMGSZ" batch=-1 device=0 workers=16 patience="$PATIENCE" \
            project="$SLADD_RUNS" name="$navn" exist_ok=True; then
        log "$navn feilet, prøver igjen med batch=8 ..."
        yolo detect train data="$datayaml" model="$base" epochs="$epoker" \
            imgsz="$IMGSZ" batch=8 device=0 workers=16 patience="$PATIENCE" \
            project="$SLADD_RUNS" name="$navn" exist_ok=True || return 1
    fi
}

publiser_og_valider() {
    local navn=$1 datasett=$2
    python "$SCRIPTS/publish_model.py" --run "$SLADD_RUNS/$navn" --name "$navn" \
        --out "$SLADD_WEIGHTS" --weights best --dataset "$datasett" \
        --info base_model="$3" --info strategy=48t --overwrite || return 1
    "$SLADD_REPO/valider_yolo.sh" model="$SLADD_WEIGHTS/$navn/$navn.pt" \
        uttrekk=6 list=holdout48 || return 1
    echo "$navn=$SLADD_VALIDATION/${navn}_validated_on_uttrekk_6_holdout48/resultat.csv" >> "$RUNLIST"
    ranger
}

kandidat() {
    local navn=$1 base=$2 epoker=$3
    if [[ -f "$SLADD_WEIGHTS/$navn/$navn.pt" ]]; then
        log "$navn er allerede publisert, hopper over treningen."
        grep -q "^$navn=" "$RUNLIST" || publiser_og_valider "$navn" "$DATASET" "$base" \
            || log "Eval av $navn feilet."
        return 0
    fi
    if tren "$navn" "$base" "$epoker" "$DATASET/data.yaml"; then
        publiser_og_valider "$navn" "$DATASET" "$base" || log "Eval av $navn feilet."
    else
        log "$navn feilet."
    fi
}

ranger() {
    local specs=()
    while IFS= read -r line; do
        [[ -f "${line#*=}" ]] && specs+=(--run "$line")
    done < <(sort -u "$RUNLIST")
    (( ${#specs[@]} )) || return 0
    python "$SCRIPTS/rank_models.py" --truth-csv "$SLADD_LABELS/uttrekk_6.csv" \
        --processed-list "$LIST_HOLDOUT" \
        --metadata "$SLADD_METADATA/uttrekk_6.csv" "$SLADD_METADATA/uttrekk_4.csv" \
        --cost "$COST" --out "$LOGDIR/rangering.csv" "${specs[@]}" \
        | tee "$LOGDIR/rangering.txt"
}

fase_probe() {
    local kode
    kode=$(awk '/^WORST_CODE/ {print $3}' "$LOGDIR/rangering.txt" | tail -1)
    [[ "$kode" =~ ^[A-Z0-9_]+$ ]] || { log "Ingen entydig verste-kode, dropper proben."; return 0; }
    local navn="${TAG}_probe_$kode"
    if [[ -f "$SLADD_WEIGHTS/$navn/$navn.pt" ]]; then
        log "$navn er allerede publisert, hopper over."
        return 0
    fi
    local pd="$DATASET-probe"
    log "Probe: eget modell-forsøk for rettsstiftelsen $kode ..."
    rm -rf "$pd"; mkdir -p "$pd/images_all" "$pd/labels_all"
    python - "$kode" "$DATASET" "$pd" <<'EOF'
import os, sys
import pandas as pd
kode, kilde, maal = sys.argv[1], sys.argv[2], sys.argv[3]
har_kode = set()
for f in (os.environ["SLADD_METADATA"] + "/uttrekk_4.csv",
          os.environ["SLADD_METADATA"] + "/uttrekk_6.csv"):
    m = pd.read_csv(f)
    rst = m.get("rettsstiftelsestyper")
    if rst is None:
        continue
    for d, koder in zip(m["fil_revisjon_id"], rst.fillna("")):
        if any(p.strip().partition(" - ")[0].strip() == kode
               for p in str(koder).split(",")):
            har_kode.add(str(d))
n = 0
for fn in os.listdir(kilde + "/images_all"):
    if fn.rsplit("_p", 1)[0] in har_kode:
        os.link(f"{kilde}/images_all/{fn}", f"{maal}/images_all/{fn}")
        lbl = fn.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(f"{kilde}/labels_all/{lbl}"):
            os.link(f"{kilde}/labels_all/{lbl}", f"{maal}/labels_all/{lbl}")
        n += 1
print(f"{n} sider i probedatasettet for {kode}")
EOF
    python "$SCRIPTS/split_train_val.py" --dataset "$pd" \
        --train-ratio 0.8 --val-ratio 0.1 --seed 42 || return 1
    { echo "path: $pd"; echo "train: images/train"; echo "val: images/val";
      echo "test: images/test"; echo ""; echo "names:"; echo "  0: fnr"; } > "$pd/data.yaml"
    local basis="$SLADD_WEIGHTS/${TAG}_l/${TAG}_l.pt"
    [[ -f "$basis" ]] || basis="$SLADD_TRAIN/$BASE_L"
    local andel epoker
    andel=$(( $(ls "$pd/images/train" | wc -l) * 100 / ($(ls "$DATASET/images/train" | wc -l) + 1) ))
    epoker=$(( (($(left_s) - RESERVE_H * 1800)) / ((SEPOCH_L * andel / 100) + 60) ))
    (( epoker > 80 )) && epoker=80
    (( epoker < 15 )) && { log "For lite tid til proben ($epoker epoker), dropper."; return 0; }
    tren "$navn" "$basis" "$epoker" "$pd/data.yaml" || return 1
    publiser_og_valider "$navn" "$pd" "$basis"
}

fase_rapport() {
    ranger
    {
        echo "# 48-timers kjøring, sluttrapport"
        echo
        echo "Avsluttet $(date '+%d.%m.%Y %H:%M'), $(left_h) timer igjen av vinduet."
        echo "Kostnadsmodell: kostnad = $COST * tapte fnr + oversladdede bokser."
        echo "Dommer: uttrekk 6 holdout ($LIST_HOLDOUT), aldri sett under trening."
        echo "NB: tapte_manuell er tallet som unnslipper sirkulariteten i ML-godkjente labels."
        echo "Probe-modellen skal kun leses på sin egen rettsstiftelse i kodelinjen."
        echo
        echo '```'
        cat "$LOGDIR/rangering.txt" 2>/dev/null || echo "(ingen rangering)"
        echo '```'
        echo
        echo "Neste steg: filter_sweep for vinneren (conf-reglene er kalibrert mot"
        echo "den gamle modellens fordeling), deretter valider_full og deploy test."
    } > "$LOGDIR/SLUTTRAPPORT.md"
    log "Rapport: $LOGDIR/SLUTTRAPPORT.md"
}

# ── start ────────────────────────────────────────────────────────

start() {
    [[ -n "${SLADD_REPO:-}" ]] || { echo "Kjør først: source activate.sh"; exit 1; }
    mkdir -p "$LOGDIR/pilot"
    touch "$RUNLIST"
    [[ -f "$DEADLINE_FILE" ]] || echo $(( $(date +%s) + HOURS * 3600 )) > "$DEADLINE_FILE"
    exec > >(tee -a "$LOG") 2>&1
    log "=== Start, frist om $(left_h) t. Logg: $LOG ==="
    if [[ -n "$(gpu_procs)" ]]; then
        log "AVBRYTER: GPU-en er ikke ledig:"; gpu_procs; exit 1
    fi

    fase_lister    || { log "AVBRYTER: listene feilet."; exit 1; }
    fase_datasett  || { log "AVBRYTER: datasettbyggingen feilet."; exit 1; }

    log "Baseline: prod-modellen på holdouten ..."
    if ! grep -q '^baseline=' "$RUNLIST"; then
        if "$SLADD_REPO/valider_yolo.sh" model="$SLADD_PRODWEIGHTS" uttrekk=6 list=holdout48; then
            echo "baseline=$SLADD_VALIDATION/yolo-yearly-10000-docs_validated_on_uttrekk_6_holdout48/resultat.csv" >> "$RUNLIST"
        else
            log "Baseline-valideringen feilet, fortsetter uten."
        fi
    fi

    SEPOCH_L=$(fase_pilot pilot_l "$SLADD_TRAIN/$BASE_L")
    [[ "$SEPOCH_L" =~ ^[0-9]+$ ]] || SEPOCH_L=1200
    SEPOCH_XFT=$(fase_pilot pilot_xft "$SLADD_PRODWEIGHTS")
    [[ "$SEPOCH_XFT" =~ ^[0-9]+$ ]] || SEPOCH_XFT=2400
    log "Målt: l=$SEPOCH_L s/epoke, x-finetune=$SEPOCH_XFT s/epoke."

    local budsjett epoker
    budsjett=$(( $(left_s) - RESERVE_H * 3600 - B_MIN_H * 3600 ))
    epoker=$(( budsjett / SEPOCH_L ))
    (( epoker > 120 )) && epoker=120
    if (( epoker >= 25 )); then
        kandidat "${TAG}_l" "$SLADD_TRAIN/$BASE_L" "$epoker"
    else
        log "Hopper over ${TAG}_l: bare $epoker epoker mulig."
    fi

    budsjett=$(( $(left_s) - RESERVE_H * 3600 ))
    epoker=$(( budsjett / SEPOCH_XFT ))
    (( epoker > 60 )) && epoker=60
    if (( epoker >= 12 )); then
        kandidat "${TAG}_x_ft" "$SLADD_PRODWEIGHTS" "$epoker"
    else
        log "Hopper over ${TAG}_x_ft: bare $epoker epoker mulig."
    fi

    if (( $(left_s) > RESERVE_H * 1800 + 7200 )); then
        fase_probe || log "Proben feilet."
    else
        log "Hopper over proben: for lite tid igjen."
    fi

    fase_rapport
    log "=== Ferdig. ==="
}

case "${1:-}" in
    preflight) preflight ;;
    start)     start ;;
    status)    tail -n 40 -f "$LOG" ;;
    *)         echo "Bruk: $0 preflight|start|status  (se train/KJOR_48T.md)"; exit 1 ;;
esac
