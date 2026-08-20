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

### PaddleOCR — CPU vs GPU (må velges konsistent i hele systemet)

CPU- og GPU-bygget av Paddle er to forskjellige pakker. Du må velge **samme variant alle stedene under** 

| Sted | CPU (Mac / uten GPU) | GPU (Linux-server med CUDA) |
|------|----------------------|------------------------------|
| Manuell install | `pip install paddlepaddle paddleocr` | `pip install paddlepaddle-gpu paddleocr` |
| `requirements.txt` | `paddlepaddle==3.3.1` | `paddlepaddle-gpu==3.3.1` |
| `Dockerfile` (`--extra-index-url`) | `.../packages/stable/cpu/` | `.../packages/stable/cu126/` |


## Modeller

YOLO-vektene ligger **ikke** i repoet. De bor i modellageret på serveren, `$SLADD_VEKTER` (se `server.env`), med én mappe per publisert modell:

```
$SLADD_VEKTER/yolo-yearly-10000-docs/
  yolo-yearly-10000-docs.pt    ← vektene, navngitt etter modellen
  modell.json                  ← hva den er trent på, med hvilke parametere
  trening/                     ← results.csv, args.yaml, data.yaml, split_log.txt
```

Mappene lages av `make -C $SLADD_TRAIN publiser` etter en treningskjøring — se [train/README.md](train/README.md). Vektfilen heter det samme som modellen, så navnet følger med uansett hvor filen kopieres, og `modell.json` gjør at man i ettertid kan se hva en modell faktisk er trent på.

`$SLADD_PRODVEKTER` peker på modellen som er standardvalget: den `./deploy.sh build` bygger inn, og den `run.py` bruker uten `--yolo-vekter`. Andre modeller velges per kjøring med `--yolo-vekter` (se run.py-flaggene under) eller per bygg med `vekter=` (se under).

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

## Produksjon (Docker)

Produksjon kjøres som Docker-container på GPU-serveren. **Det som kjører på port 5071 er produksjon** — ingenting annet.

Et image bygges én gang og får en uforanderlig tag (`<dato>-<commit>-<modell>`, f.eks. `20260820-6d7e6820-yolo-yearly-10000-docs`). Etter det flyttes bare *hvilken* tag som kjører hvor. Prod bygger aldri selv, så det som står på 5071 endrer seg ikke av at noen bygger noe nytt, og rollback er å peke tilbake på en tag som allerede har kjørt.

Imagene lagres **bare lokalt på serveren** — ingen registry foreløpig. Det betyr at `docker image prune -a` sletter muligheten til å rulle tilbake, og at en ny maskin må bygge alt på nytt.

| Rolle | Port | Container   | Tag styres av |
|-------|------|-------------|---------------|
| Prod  | 5071 | `smsl-prod` | `PROD_TAG`    |
| Test  | 5072 | `smsl-test` | `TEST_TAG`    |

`deploy.sh` gjør alt arbeidet. Første gang på en ny maskin:

```sh
cp .env.example .env
source activate.sh                        # laster server.env, som peker på modellageret
ls $SLADD_VEKTER                          # modellene som kan bygges inn
```

### Normal flyt: build → test → promote

```sh
./deploy.sh build                         # modell fra $SLADD_PRODVEKTER
./deploy.sh test 20260820-6d7e6820-yolo-yearly-10000-docs
```

En annen modell inn i imaget — samme kode, nye vekter:

```sh
./deploy.sh build vekter=$SLADD_VEKTER/uttrekk_4_jou/uttrekk_4_jou.pt
```

Verifiser mot testporten før du promoterer:

```sh
curl http://localhost:5072/health
curl -X POST http://localhost:5072/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@/sti/til/dokument.pdf"
```

Når den ser bra ut, settes *samme tag* i prod — ingen ny bygging, altså samme bits som ble testet:

```sh
./deploy.sh promote 20260820-6d7e6820-yolo-yearly-10000-docs
./deploy.sh stop test                   # frigi GPU-minnet testcontaineren holder
```

`promote` krever at du oppgir taggen eksplisitt, ber om bekreftelse, og **ruller automatisk tilbake** til forrige tag hvis `/health` ikke svarer.

### Øvrige kommandoer

```sh
./deploy.sh status            # hva kjører hvor, og er det friskt
./deploy.sh versions          # lokalt bygde tagger med modell, nyest først
./deploy.sh rollback          # tilbake til forrige tag i prod
./deploy.sh stop prod|test    # ta ned en container
./deploy.sh start prod|test   # opp igjen med taggen som står i .env
./deploy.sh logs prod|test    # følg loggen
./deploy.sh prune             # slett gamle images (kan ikke angres)
```

`start` og `stop` bytter aldri versjon — de tar bare ned og opp den taggen som allerede står i `.env`. Skal du bytte versjon, er det `promote` (prod) eller `test <tag>` (test). `stop prod` ber om bekreftelse, siden det tar ned produksjon; `stop test` gjør ikke.

`stop` stopper containeren uten å slette den, så `start` henter opp nøyaktig samme oppsett. GPU-minnet frigis uansett, siden prosessen dør, og `restart: unless-stopped` tar den ikke opp av seg selv etter en eksplisitt stopp.

Deploy-historikken ligger i `.deploy-historikk` (tid, ny tag, forrige tag) og er det `rollback` leser.

`prune` er den ene kommandoen som ikke kan angres, siden imagene bare finnes her. Den verner de 5 nyeste, taggen i prod, taggen i test og alt som står i deploy-historikken — men sletter du noe eldre, er eneste vei tilbake å bygge på nytt fra commiten taggen navngir.

### Imaget

| | |
|---|---|
| Navn | `smart-sladding:<dato>-<commit>` |
| Lagring | bare lokalt på serveren |
| Størrelse | ~10–15 GB |
| Ny versjon koster | ~64 kB ved en ren kodeendring (lagene deles) |

Imaget inneholder alle modellene det trenger:

```
/app/
  PP-OCRv6_medium_det_infer/     59 MB   ← curl i Dockerfile
  PP-OCRv6_medium_rec_infer/     73 MB   ← curl i Dockerfile
  PP-LCNet_x1_0_doc_ori_infer/  6,6 MB   ← curl i Dockerfile
  weights/modell.pt              51 MB   ← modellen ./deploy.sh build valgte
  weights/modell.json                    ← hva den modellen er trent på
  *.py                           64 kB   ← COPY app/*.py
```

Merk at `Dockerfile` hardkoder nedlasting av **v6**-modellene, mens `app/config.py` velger settet med `MODELL_SETT`. Bytter du til `"v5"` må Dockerfile endres tilsvarende, ellers bygges et image uten de modellene koden ber om.

Vektene ligger i et eget lag fra koden. Det gjør at ti versjoner av samme modell koster 51 MB vekter til sammen og ikke 51 MB hver, og at et rebuild etter en kodeendring gjenbruker alt det tunge.

Docker kan bare kopiere fra byggekonteksten, og vektene ligger utenfor repoet. `./deploy.sh build` legger derfor den valgte modellen i `.byggvekter/` rett før bygget og fjerner mappen etterpå. Derfor bygger heller ikke `docker build .` direkte lenger — bygg går gjennom `deploy.sh`.

### Logger

Tre strømmer, alle gjennom samme roterende handler: de zippes ved døgnskiftet og eldste zip slettes når historikken er full.

| Logg | Fil i loggmappa | Kilde |
|------|-----------------|-------|
| Applikasjon | `app.log` | `app/app.py` |
| Access | `gunicorn_access_prod.log` | `config/gunicorn_config_prod.py` |
| Gunicorn-feil | `gunicorn_error_prod.log` | samme |

Etter rotasjon: `app.log.2026-08-19.zip` — datoen er **døgnet zipen dekker**. Skjer det et ekstra rollover i samme døgn (nedetid over midnatt, eller flere workers), får den et løpenummer (`.2.zip`) i stedet for å overskrive.

Hvor de havner defineres i `server.env`:

```sh
export SLADD_LOGS=/data/docker       # loggrot på verten
export SLADD_LOGG_DAGER=30           # døgn historikk per fil
```

`deploy.sh` leser dem og sender dem videre til compose som `LOG_ROOT` og `LOG_BACKUP_DAYS`. Under loggroten lages:

```
$SLADD_LOGS/
  gunicorn_logs/        ← prod: access + error
  ml_logs/              ← prod: app.log
  gunicorn_logs_test/   ← test, holdt for seg
  ml_logs_test/
```

Test skriver til egne mapper med vilje. Delte de prods, ville to rotasjons-handlere kappet om samme fil ved midnatt, og testtrafikk havnet i prods access-logg. Mappene opprettes av Docker ved første kjøring.

Inne i containeren ligger stiene på `/data/gunicorn_logs` og `/data/ml_logs` uansett, satt med `GUNICORN_LOG_DIR` og `ML_LOG_DIR`. `./deploy.sh status` viser hvilken loggrot og retensjon som gjelder.

`_prod`-suffikset i filnavnene er historisk — det finnes bare én gunicorn-config nå. Navnene er beholdt så eksisterende logghistorikk på serveren ikke splittes i to serier. Test skriver samme filnavn, men i sine egne mapper.

Gunicorns error-logg går også til stdout, så `./deploy.sh logs prod` fortsatt viser oppstart og feil. Stray stdout/stderr havner i dockers egen loggfil, som compose kapper på 50 MB × 5.

### Verdt å vite

- Prod og test deler GPU-en. Kjører du en testcontainer ved siden av prod, konkurrerer de om samme kort — kjør `./deploy.sh stop test` når du er ferdig.
- Hvilken modell som bygges inn er et eksplisitt valg. `./deploy.sh build` uten argumenter tar `$SLADD_PRODVEKTER`; `./deploy.sh build vekter=$SLADD_VEKTER/<modell>/<modell>.pt` tar en annen. Uten en modell nekter den å bygge, siden et image uten vekter starter fint og feiler først ved første `/model`-kall.
- Modellnavnet står i taggen (`20260820-eb6f64dd-yolo-yearly-10000-docs`), og modellnavn + sha256 av vektfilen ligger som merkelapper på imaget. Vektene ligger utenfor repoet, så commiten alene sier ikke hva imaget inneholder — uten navnet i taggen ville samme kode med to modeller fått samme tag, og `rollback` ville rullet tilbake koden uten modellen. `./deploy.sh status` og `versions` viser modellen per tag.
- En tag bygget på ucommittede endringer får suffikset `-dirty` og blir avvist av `./deploy.sh promote`. Commit først.
- Modellene lastes først ved første `/model`-kall, så `/health` svarer lenge før containeren er varm.
- Bekreft at GPU-en faktisk brukes:

```sh
./deploy.sh logs prod | grep -i "GPU tilgjengelig"   # -> True hvis du kjører på gpu
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
| `--yolo-vekter FIL`      | `$SLADD_PRODVEKTER`                           | Path til YOLO-vektfil (for å teste andre vekter) |
| `--csv`                  | av                                            | Skriv funne bokser til CSV                       |
| `--csv-ut FIL`           | `sladd_koordinater.csv`                       | Filnavn for boks-CSV                             |
| `--fasit`                | av                                            | Mål recall mot fasit-CSV                         |
| `--fasit-csv FIL`        | `smartsladding_uttrekk_labels_3_07_07_26.csv` | Fasit-CSV                                        |
| `--terskel FLOAT`        | `0.32`                                        | Andel fasit-areal som kreves for TRUFFET         |
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
python run.py --velg 10000676.pdf --yolo-vekter $SLADD_VEKTER/yolo-yearly-10000-docs/yolo-yearly-10000-docs.pt --fasit --tid
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
| `--terskel FLOAT`       | `0.32`                                        | Overlapp-terskel for TRUFFET                     |
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

## API-kontrakt

### `GET /health`

Returnerer `{"health": "healthy"}` med status 200 når tjenesten kjører.
Merk: modellene lastes først ved første kall til `/model`, så første
forespørsel etter oppstart tar vesentlig lengre tid enn de påfølgende.

### `POST /model`

Tar imot en PDF som rå bytes i request-body (`Content-Type: application/pdf`)
og returnerer funne sladde-bokser som JSON.

200        | OK, respons som beskrevet under            
400        | Tom request-body                           
500        | Intern feil, `{"error": "<beskrivelse>"}` 

#### Responsformat

```json
{
  "sider": [
    {
      "side": 1,
      "bilde_bredde": 2480,
      "bilde_hoyde": 3510,
      "bokser": [
        { "x0": 856, "y0": 1203, "x1": 998, "y1": 1240, "kilde": "begge", "conf": 0.871 }
      ]
    }
  ]
}
```

| Felt                        | Beskrivelse                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `sider`                     | Én oppføring per side i PDF-en, også sider uten funn (`bokser` er da tom)  |
| `side`                      | Sidenummer, 1-basert                                                        |
| `bilde_bredde`/`bilde_hoyde`| Sidens størrelse i piksler                                                  |
| `x0, y0`                    | Øvre venstre hjørne av sladde-boksen (origo øverst til venstre)             |
| `x1, y1`                    | Nedre høyre hjørne av sladde-boksen                                         |
| `kilde`                     | `paddle`, `yolo`, `begge` eller `yolo_vertikal`                             |
| `conf`                      | YOLO-konfidens (0–1). Finnes bare når YOLO var involvert, rene Paddle-treff har ikke feltet |

Koordinatene refererer til sidens opprinnelige orientering, eventuell
rotasjon under analysen er allerede regnet tilbake.

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