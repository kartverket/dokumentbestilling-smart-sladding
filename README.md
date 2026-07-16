# dokumentbestilling-smart-sladding

## VIKTIG: For bruk i Kartverket
Hvis du jobber med dette i Kartverket-sammenheng, så skal du være inneforstått med de relevante rutinene før du begynner med å arbeide. [Les rutinene på confluence her](https://kartverket.atlassian.net/wiki/x/F4Dwn)

## Beskrivelse
Prosjektet genererer automatiske sladdinger av personnummer og d-nummer i dokumentbestillinger. Løsningen benytter Tesseract OCR for å gjenkjenne tekst i dokumenter, og klassifiserer områder med sensitive opplysninger gjennom regex- og nøkkelordssøk.

## Forutsetninger


### Docker (anbefalt)
- Docker
- NVIDIA GPU med CUDA 12.1-støtte
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Lokal utvikling
- Python 3.10+
- [Poppler](https://poppler.freedesktop.org/) (`pdftoppm -v` for å verifisere)
- [Tesseract](https://tesseract-ocr.github.io/) (`tesseract -v` for å verifisere)

På macOS:
```sh
brew install poppler tesseract
```

## Installasjon

### 1. Docker-oppsett (anbefalt)

Prosjektet bruker `nvidia/cuda:12.1.1-runtime-ubuntu22.04` som base-image med Python 3.10. Docker og NVIDIA Container Toolkit må være installert og konfigurert på forhånd.

#### 1.1 Docker build

```sh
# Bygg image
docker build --tag smart_sladding_app:latest .

# Bak proxy:
docker build \
  --build-arg http_proxy=http://159.162.48.7:3128 \
  --build-arg https_proxy=http://159.162.48.7:3128 \
  --build-arg no_proxy=localhost,127.0.0.1 \
  --tag smart_sladding_app:latest .
```

#### 1.2 Start containere med compose

```sh
docker compose up -d
```

Dette starter to tjenester:
- **prod** på port 5071 (`MODE=prod`)
- **dev** på port 5072 (`MODE=dev`)

#### 1.3 Start container manuelt

```sh
docker run -it --gpus all \
  -v /data/docker/ml_logs:/data/ml_logs \
  -p 5071:8080 \
  -e MODE=prod \
  -e HTTP_PROXY=http://159.162.48.7:3128 \
  -e HTTPS_PROXY=http://159.162.48.7:3128 \
  --name smsl-server-prod \
  smart_sladding_app:latest
```

#### 1.4 Test med curl

```sh
curl -X POST http://localhost:5071/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@app/testdokument-2.pdf"
```

### 2. Lokalt oppsett (uten Docker)

#### 2.1. Klon repository
```sh
git clone https://github.com/kartverket/dokumentbestilling-smart-sladding.git
cd dokumentbestilling-smart-sladding
```

#### 2.2. Opprett virtuelt miljø og installer avhengigheter
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For onprem maskin, finn de nødvendige installasjonspakkene for Tesseract og Poppler, og installer disse.
* poppler versjon kan sjekkes med `pdftoppm -v`
* tesseract versjon kan sjekkes med `tesseract -v`

#### 2.3. Kjør opp appen
```sh
cd app
python3 app.py
```

#### 2.4. Test appen via curl i ny terminal
```sh
mkdir -p app/logs
curl -X POST http://localhost:5070/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@app/testdokument-2.pdf"
```

#### 2.5. For production/test kjører vi appen med gunicorn
```sh
cd app
chmod +x start_production.sh
./start_production.sh
```

## EasyOCR-modeller bak brannmur

Dersom maskinen ikke har direkte internettilgang, last ned modellene manuelt og plasser dem i `tmp/.EasyOCR/model`:

```sh
curl -L -x http://<proxyip>:<proxyport> -o "latin_g2.zip" https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip
curl -L -x http://<proxyip>:<proxyport> -o "craft_mlt_25k.zip" https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip

mkdir -p tmp/.EasyOCR/model
mv latin_g2.zip craft_mlt_25k.zip tmp/.EasyOCR/model/
cd tmp/.EasyOCR/model && unzip latin_g2.zip && unzip craft_mlt_25k.zip
```

## Testing

### Enkelt dokument
Kjør modellen på ett dokument med `doc_id` på formen `<aar>_<nr>_<embete>`:
```python
predicted_boxes = get_predicted_boxes_on_doc(doc_id, base_url)
```
Funksjonen returnerer predikerte avgrensingsboksene og lagrer et bilde.

### Evaluering på flere dokumenter
Kjør `evaluation_main.py`. Minor-versjonsøkning i `current_model_version_number` kjører reglene på nytt med cachet OCR-lesing. Major-versjonsøkning kjører OCR på nytt.

Første gang tar OCR-lesingen 8+ timer for 1400 dokumenter.

## SSL-problemer på macOS
Se [denne Stack Overflow-responsen](https://stackoverflow.com/a/57795811).

## Lisens
[MIT](LICENSE)

## Bidrag
Se [CONTRIBUTING.md](CONTRIBUTING.md).

## Sikkerhet
Se [SECURITY.md](.github/SECURITY.md) for rapportering av sårbarheter.
