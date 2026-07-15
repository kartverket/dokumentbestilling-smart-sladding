# dokumentbestilling-smart-sladding

## VIKTIG: For bruk i Kartverket
Hvis du jobber med dette i Kartverket-sammenheng, så skal du være inneforstått med de relevante rutinene før du begynner med å arbeide. [Les rutinene på confluence her](https://kartverket.atlassian.net/wiki/x/F4Dwn)

## Beskrivelse
Prosjektet genererer automatiske sladdinger av personnummer og d-nummer i dokumentbestillinger.

Se [docs/TEKNISK.md](docs/TEKNISK.md) for teknisk beskrivelse av arkitektur og deteksjonspipeline med figurer.

## Repo-struktur

```
app/       API og modell-kode (Flask + PaddleOCR + YOLO)
utils/     Analyse- og testverktøy (run, tegn, statistikk)
```

## Forutsetninger
- Python 3.12

## Installasjon

```sh
git clone https://github.com/kartverket/dokumentbestilling-smart-sladding.git
cd dokumentbestilling-smart-sladding
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PaddleOCR — separat installasjon

Paddle-pakken installeres for seg fordi CPU- og GPU-bygget er forskjellige pakker:

```sh
# CPU (Mac / maskin uten GPU):
pip install paddlepaddle paddleocr

# GPU (Linux-server med CUDA):
pip install paddlepaddle-gpu paddleocr
```

## Modeller

`best.pt` (vektene til den trente YOLO-modellen) leveres separat og legges i `app/`.

PaddleOCR-modellene er ferdig trente vekter fra PaddlePaddle sitt modellbibliotek og lastes ned manuelt (kjøres fra `app/`):

```sh
# v6 (standard)
curl -L -O https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_infer.tar
curl -L -O https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_infer.tar
tar -xvf PP-OCRv6_medium_det_infer.tar
tar -xvf PP-OCRv6_medium_rec_infer.tar

# v5 (sett MODELL_SETT = "v5" i app/config.py)
curl -L -O https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
curl -L -O https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar
tar -xvf PP-OCRv5_server_det_infer.tar
tar -xvf PP-OCRv5_server_rec_infer.tar

# Orienteringsmodell (felles for v5 og v6)
curl -L -O https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar
tar -xvf PP-LCNet_x1_0_doc_ori_infer.tar
```

## Kjøre API lokalt

```sh
mkdir -p app/logs   # må opprettes første gang
cd app
python app.py
```

Test i ny terminal:
```sh
curl http://localhost:5070/health

curl -X POST http://localhost:5070/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@/sti/til/dokument.pdf"
```

## Valideringskjøring med testdokument

Et testdokument (`utils/testdokument.pdf`) følger med repoet. Bruk det for å verifisere at oppsettet fungerer:

```sh
cd utils
python run.py --mappe . --velg testdokument.pdf --csv --csv-ut test_ut.csv --png --png-mappe visning_test --fasit --tid
```

PNG-resultat lagres i `utils/visning_test/`.

## Produksjon

```sh
cd app
chmod +x start_production.sh
./start_production.sh
```

---

## run.py for testing — kjøre modellen mot fasit

Kjøres fra `utils/`-mappa. Pek på PDF-mappen din med `--mappe` (standard er `../uttrekk_3/`).

```sh
cd utils
python run.py [flagg]
```

| Flagg                    | Standard                                      | Beskrivelse                                      |
|--------------------------|-----------------------------------------------|--------------------------------------------------|
| `--mappe STI`            | `../uttrekk_3`                                | Mappe med PDF-er                                 |
| `--velg FIL [FIL ...]`   | —                                             | Kjør bare disse filene (filnavn/delstreng)       |
| `--antall N`             | `20`                                          | Antall filer når `--velg` er tom (`alle` = alle) |
| `--csv`                  | av                                            | Skriv funne bokser til CSV                       |
| `--csv-ut FIL`           | `sladd_koordinater.csv`                       | Filnavn for boks-CSV                             |
| `--fasit`                | av                                            | Mål recall mot fasit-CSV                         |
| `--fasit-csv FIL`        | `smartsladding_uttrekk_labels_3_07_07_26.csv` | Fasit-CSV                                        |
| `--terskel FLOAT`        | `0.15`                                        | Andel fasit-areal som kreves for TRUFFET         |
| `--y-origin topp\|bunn`  | `topp`                                        | Y-origo i fasit-CSV                              |
| `--png`                  | av                                            | Tegn funne + fasit-bokser til PNG                |
| `--png-mappe STI`        | `visning`                                     | Mappe for PNG-ene                                |
| `--sladd`                | av                                            | Lag faktisk sladdede PDF-er                      |
| `--sladd-mappe STI`      | `sladdet`                                     | Mappe for sladdede PDF-er                        |
| `--ocr-logg`             | av                                            | Skriv OCR-tekst linje for linje til fil          |
| `--ocr-logg-fil FIL`     | `ocr_linjer.txt`                              | Filnavn for OCR-loggen                           |
| `--tid`                  | av                                            | Skriv timing per dokument                        |
| `--beskrivelse TEKST`    | —                                             | Suffiks i result-mappenavnet                     |

```sh
python run.py --velg 10000676.pdf --csv --fasit --tid
python run.py --antall alle --fasit --csv --png
```

---

## tegn.py — visualisere sladde-bokser som PNG

Brukes for manuell gjennomgang av hva modellen har funnet. Tegner sladde-boksene oppå PDF-sidene og lagrer som PNG-bilder — slik kan man raskt bla gjennom og se om boksene treffer riktig. Bruker da sladd_koordinater laget av en run.py kjøring.

```sh
cd utils
python tegn.py --csv sladd_koordinater.csv --png-mappe visning
```

Åpne bildene i `utils/visning/` for å se resultatet. Med `--fasit` tegnes også fasit-boksene i grønt for sammenligning.

```sh
python tegn.py [flagg]
```

| Flagg                   | Standard                                      | Beskrivelse                                      |
|-------------------------|-----------------------------------------------|--------------------------------------------------|
| `--csv FIL`             | `res.csv`                                     | Koordinat-CSV (fra `run.py --csv`)               |
| `--mappe STI`           | `../uttrekk_3`                                | Mappe med original-PDF-ene                       |
| `--png-mappe STI`       | `visning`                                     | Mappe for PNG-ene                                |
| `--fasit-csv FIL`       | `smartsladding_uttrekk_labels_3_07_07_26.csv` | Fasit-CSV (tegnes som grønne rammer)             |
| `--velg FIL [FIL ...]`  | —                                             | Begrens til disse filene                         |
| `--fasit`               | av                                            | Mål recall mot fasit og skriv ut i terminal      |
| `--terskel FLOAT`       | `0.15`                                        | Overlapp-terskel for TRUFFET                     |
| `--yolo`                | av                                            | Kjør YOLO og vis treff som røde rammer           |
| `--kun-oversladd`       | av                                            | Tegn kun sider med over-sladding                 |
| `--kun-bom`             | av                                            | Tegn kun sider med minst én MANGLER (+ `--fasit`)|
| `--y-origin topp\|bunn` | `topp`                                        | Y-origo i fasit-CSV                              |

```sh
python tegn.py --csv sladd_koordinater.csv --fasit --velg 10000676.pdf
python tegn.py --csv t.csv --fasit --kun-bom
python tegn.py --csv t.csv --kun-oversladd
```

---

## statistikk.py — samlet rapport for en kjøring

```sh
python statistikk.py [mappe] [flagg]
```

| Argument/flagg  | Standard | Beskrivelse                                            |
|-----------------|----------|--------------------------------------------------------|
| `mappe`         | —        | Result-mappe fra `run.py --fasit` (valgfri)            |
| `--labels FIL`  | —        | Labels-CSV for sammenligning med nåværende løsning     |
| `--ingen-graf`  | av       | Dropp statistikk.png                                   |

```sh
python statistikk.py result-2026-07-14T08-15-20 --labels smartsladding_uttrekk_labels_3_07_07_26.csv

python statistikk.py --labels smartsladding_uttrekk_labels_3_07_07_26.csv
```

Lager `statistikk.txt` og `statistikk.png` i result-mappa.

---

## Konfigurasjon

| Fil                     | Innhold                                              |
|-------------------------|------------------------------------------------------|
| `app/config.py`         | PDF-DPI, YOLO-terskel, OCR-parametere, orientering   |
| `utils/utils_config.py` | Stier, evalueringsterskler, visualiseringsfarger      |

---

## CSV-formater

### Boks-CSV (`run.py --csv`, leses av `tegn.py`)

| Kolonne        | Beskrivelse                                      |
|----------------|--------------------------------------------------|
| `navn`         | Filnavn (PDF)                                    |
| `side`         | Sidenummer (1-basert)                            |
| `bilde_bredde` | Bildebredde i piksler                            |
| `bilde_hoyde`  | Bildehøyde i piksler                             |
| `x0, y0`       | Øvre venstre hjørne av sladde-boks (piksler)     |
| `x1, y1`       | Nedre høyre hjørne av sladde-boks (piksler)      |
| `kilde`        | `paddle`, `yolo`, `begge` eller `yolo_vertikal`  |
| `conf`         | YOLO-konfidensverdi (tom for rene Paddle-treff)  |

### Fasit-CSV (labels fra eksisterende løsning)

| Kolonne           | Beskrivelse                                        |
|-------------------|----------------------------------------------------|
| `fil_revisjon_id` | Dokument-ID (tilsvarer tall i PDF-filnavnet)       |
| `sidetall`        | Sidenummer (1-basert)                              |
| `x, y`            | Øvre venstre hjørne i PDF-punkter                  |
| `width, height`   | Størrelse i PDF-punkter                            |
| `type`            | Kategori (f.eks. `PERSONNUMMER`)                   |
| `ml_generated`    | `true` hvis modellen fant den, `false` = manuell   |
| `ml_status`       | `ACCEPTED` / `REJECTED`                            |

### Detaljer-CSV (`run.py --fasit`, leses av `statistikk.py`)

| Kolonne                      | Beskrivelse                              |
|------------------------------|------------------------------------------|
| `fil`                        | Filnavn (PDF)                            |
| `side`                       | Sidenummer                               |
| `fasit_nr`                   | Løpenummer for fasit-boks på siden       |
| `type`                       | Kategori fra fasit                       |
| `dekning_pst`                | Andel av fasit-boks dekket (%)           |
| `resultat`                   | `TRUFFET` eller `MANGLER`                |
| `kilde`                      | Hvilken modell som traff                 |
| `conf`                       | Konfidensverdi                           |
| `fasit_x0/y0/x1/y1`          | Fasit-boks i normaliserte koordinater    |

---

## Lisens
[MIT](LICENSE)

## Bidrag
Se [CONTRIBUTING.md](CONTRIBUTING.md).

## Sikkerhet
Se [SECURITY.md](.github/SECURITY.md) for rapportering av sårbarheter.


