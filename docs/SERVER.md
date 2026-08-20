# Treningsserver: Oppsett og bruk

## Første gang: Sett opp miljøet

```bash
tmux new -s trening
source /home/smartsladding/dokumentbestilling-smart-sladding_test/dokumentbestilling-smart-sladding/activate.sh
```

Én linje aktiverer både venv og alle `SLADD_`-variabler.

> **Tips:** Legg gjerne `source`-linjen i `~/.bashrc` så slipper du å gjenta den.

## Tilgjengelige variabler

| Variabel            | Verdi                                          | Beskrivelse                     |
|--------------------|--------------------------------------------------|---------------------------------|
| `SLADD_REPO`       | `.../dokumentbestilling-smart-sladding`         | Repo-rot                        |
| `SLADD_UTTREKK`    | `/data2/smartsladding-uttrekk`                  | PDFer per uttrekk               |
| `SLADD_LABELS`     | `.../smartsladding-uttrekk-labels`              | Labels-CSVer                    |
| `SLADD_METADATA`   | `.../smartsladding-uttrekk-metadata`            | Metadata-CSVer                  |
| `SLADD_RUNS`       | `/data2/runs`                                   | Rå treningskjøringer            |
| `SLADD_VEKTER`     | `/data2/vekter`                                 | Publiserte modeller             |
| `SLADD_VALIDERING` | `/data2/validering`                             | Valideringsresultater           |
| `SLADD_LISTER`     | `/data2/validering/lister`                      | Fil-ID-lister                   |
| `SLADD_PRODVEKTER` | `$SLADD_VEKTER/<modell>/<modell>.pt`            | Standardmodell (prod)           |
| `SLADD_CACHE`      | `/data2/cache`                                  | Alt derivert, per uttrekk       |
| `SLADD_RUN`        | `.../utils/run.py`                              | Validerings-script              |
| `SLADD_TRAIN`      | `.../train`                                     | Trenings-mappe                  |

---

## Trening

### Standard trening (fra scratch)

```bash
make -C $SLADD_TRAIN \
  PDFS=$SLADD_UTTREKK/uttrekk_4/ \
  CSV=$SLADD_LABELS/uttrekk_4.csv \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset \
  STRATEGY=doc_type \
  METADATA=$SLADD_METADATA/uttrekk_4.csv \
  DOC_TYPE=SR_JOU \
  DEVICE=cuda \
  NAME=uttrekk_4_jou_med_negative
```

Kjøringen havner i `$SLADD_RUNS/<NAME>/`, og publiseres til slutt som en ferdig
modell i `$SLADD_VEKTER/<NAME>/` med `<NAME>.pt` og `modell.json`. Det er den
publiserte modellen validering og `deploy.sh` peker på — ikke `best.pt` inne i
kjøringen.

### Trening basert på en eksisterende modell

```bash
make -C $SLADD_TRAIN \
  PDFS=$SLADD_UTTREKK/uttrekk_4/ \
  CSV=$SLADD_LABELS/uttrekk_4.csv \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset \
  STRATEGY=doc_type \
  METADATA=$SLADD_METADATA/uttrekk_4.csv \
  DOC_TYPE=SR_JOU \
  DEVICE=cuda \
  NAME=uttrekk_4_jou_based_pat20 \
  PATIENCE=20 \
  MODEL=$SLADD_PRODVEKTER
```

### Publisere en kjøring som ble trent uten publisering

```bash
make -C $SLADD_TRAIN publiser \
  NAME=uttrekk_4_jou_med_negative \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset
```

`modell.json` blir bare så fullstendig som variablene du oppgir: kjør
`publiser` med de samme `PDFS`/`CSV`/`STRATEGY`-verdiene som treningen brukte,
ellers står de tomme i metadataen.

---

## Validering

### Steg 1: Lag en dokumentliste

```bash
./lag_liste.sh uttrekk=5 docs=SR_JOU name=jou
```

Flere dokumenttyper og årsfilter:

```bash
./lag_liste.sh uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou_reg
./lag_liste.sh uttrekk=5 years=2024,2025 name=nyere
./lag_liste.sh uttrekk=4 docs=OB_MOB name=mob
```

### Steg 2: Kjør validering

Det finnes to wrapper-script:

| Script | Hva det kjører | Bruksområde |
|--------|---------------|-------------|
| `valider_yolo.sh` | Kun YOLO (uten OCR) | Rask testing av YOLO-vekter |
| `valider_full.sh` | OCR + YOLO + matching (produksjonslogikk) | Validere full pipeline |

#### Kun YOLO (`valider_yolo.sh`)

```bash
# Valider prod-modellen på uttrekk 5, JOU-dokumenter
./valider_yolo.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou

# Valider en trent modell
./valider_yolo.sh modell=$SLADD_VEKTER/uttrekk_4_jou_med_negative/uttrekk_4_jou_med_negative.pt uttrekk=5 liste=jou

# Valider en annen modell på et annet uttrekk/doctype
./valider_yolo.sh modell=$SLADD_RUNS/uttrekk_4_jou_based_pat20/weights/best.pt uttrekk=4 liste=mob   # upublisert kjøring

# Egendefinert navn på utmappen
./valider_yolo.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou navn=prod_test
```

#### Full pipeline (`valider_full.sh`)

```bash
# Valider full produksjonslogikk (OCR + YOLO) på uttrekk 5
./valider_full.sh uttrekk=5 liste=jou

# Med egendefinert navn
./valider_full.sh uttrekk=5 liste=jou navn=ocr-test
```

OCR-cachen (`$SLADD_CACHE`) brukes automatisk. Første kjøring prosesserer alle dokumenter; påfølgende kjøringer med samme uttrekk hopper over PaddleOCR.

#### Felles egenskaper

Begge script:
- Bygger alle stier automatisk fra uttrekk-nr og liste-navn
- Sjekker at alle filer/mapper finnes før det starter
- Viser en oppsummering av hva som kjøres
- Kjører `--csv --fasit --kun-feil` (CSV-resultat, evaluering mot fasit, feilbilder)

---

## OCR-cache

PaddleOCR er den tyngste operasjonen per dokument. Resultatet er deterministisk for en gitt PDF, så det caches per dokument og gjenbrukes på tvers av kjøringer.

### Cache-struktur (alt per uttrekk)

```
/data2/cache/
  uttrekk_4/
    ocr/                  ← PaddleOCR-tokens + orientering
      123456789.json
      234567890.json
    dataset/              ← Konverterte bilder + YOLO-labels (trening)
      images_all/
      labels_all/
  uttrekk_5/
    ocr/
      345678901.json
    dataset/
      ...
```

Alt derivert for ett uttrekk samlet i én mappe. Slett alt cachet for ett uttrekk med:

```bash
rm -rf $SLADD_CACHE/uttrekk_5
```

### Hvordan det aktiveres

Cachen er **på som standard** når `SLADD_CACHE` er satt (via `server.env`). OCR-cache-stien utledes automatisk:

```
--mappe $SLADD_UTTREKK/uttrekk_5/  →  cache: $SLADD_CACHE/uttrekk_5/ocr/
--mappe $SLADD_UTTREKK/uttrekk_4/  →  cache: $SLADD_CACHE/uttrekk_4/ocr/
```

Eksplisitt overstyring:

```bash
# Spesifiser cache-sti manuelt
python -u $SLADD_RUN --ocr-cache /data2/cache/uttrekk_5/ocr ...

# Deaktiver cache helt
python -u $SLADD_RUN --no-ocr-cache ...
```

### Flyt

```
Første kjøring (cache-miss):
  render PDF → orientering → PaddleOCR → lagre til cache → YOLO → resultat

Neste kjøring, samme uttrekk (cache-treff):
  render PDF → last fra cache → YOLO → resultat
```

Ved `--kun-yolo`-kjøringer brukes ikke cachen (ingen OCR å cache).

### Invalidering

Hver cache-fil inneholder forutsetningene (OCR-modellversjon, DPI). Ved oppslag sjekkes disse automatisk. Hvis du bytter OCR-modell (f.eks. v6 → v7), vil alle oppslag misse og dokumentene prosesseres på nytt.

For å tvinge full reprosessering av ett uttrekk:

```bash
rm -rf $SLADD_CACHE/uttrekk_5/ocr
```

### Hva caches?

| Operasjon | Caches? | Begrunnelse |
|-----------|---------|-------------|
| PaddleOCR-tokens | ✅ | Tung GPU, deterministisk per dokument |
| Orientering | ✅ | Lagres sammen med tokens |
| YOLO-inferens | ❌ | Avhenger av modellvekter som endres ofte |
| PDF-rendering | ❌ | For store filer (~25 MB/side) |

---

## Modellageret

`$SLADD_VEKTER` er stedet alle ferdige modeller bor. Én mappe per modell:

```
/data2/vekter/
  yolo-yearly-10000-docs/
    yolo-yearly-10000-docs.pt    ← vektene
    modell.json                  ← datasett, hyperparametre, mål, git-sha
    trening/                     ← results.csv, args.yaml, data.yaml, split_log.txt
  uttrekk_4_jou/
    …
```

`$SLADD_RUNS` er arbeidsmapper: checkpoints, plott og `weights/best.pt` fra hver
kjøring. Der heter alle modeller det samme, så ingenting utenfor treningen skal
peke dit. `make publiser` er broen mellom de to.

### Engangsjobb: flytte gamle vekter inn i lageret

Vektene lå tidligere i repoet (`app/weights/weights/`, levert via et git-submodul).
Submodulet er borte, og vekter skal aldri i git igjen. Modeller derfra flyttes inn
i lageret med `--vektfil`, som gir en tynn `modell.json` — treningskjøringen finnes
ikke lenger, og det står da eksplisitt i metadataen:

```bash
python $SLADD_TRAIN/scripts/publiser_modell.py \
  --vektfil $SLADD_REPO/app/weights/weights/yolo-yearly-10000-docs.pt \
  --navn yolo-yearly-10000-docs \
  --ut $SLADD_VEKTER
```

Når alle modellene er flyttet, kan `app/weights/` slettes fra utsjekken.

## Filstruktur

```
activate.sh        ← Source denne: aktiverer venv + laster variabler
server.env         ← Globale stier (lastes av activate.sh)
valider_yolo.sh    ← Validering kun med YOLO
valider_full.sh    ← Validering med full produksjonslogikk (OCR + YOLO)
lag_liste.sh       ← Generer dokument-ID-lister fra metadata
app/ocr_cache.py   ← Per-dokument OCR-cache (les/skriv)
train/Makefile     ← Treningspipeline
train/scripts/publiser_modell.py  ← Kjøring → ferdig modell i $SLADD_VEKTER
```




