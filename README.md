# dokumentbestilling-smart-sladding

## Beskrivelse
Prosjektet genererer automatiske sladdinger av personnummer og d-nummer i dokumentbestillinger. Løsningen benytter Tesseract OCR for å gjenkjenne tekst i dokumenter, og klassifiserer områder med sensitive opplysninger gjennom regex- og nøkkelordssøk.

## Forutsetninger
- Python 3.11+
- [Poppler](https://poppler.freedesktop.org/) (`pdftoppm -v` for å verifisere)
- [Tesseract](https://tesseract-ocr.github.io/) (`tesseract -v` for å verifisere)

På macOS:
```sh
brew install poppler tesseract
```

## Installasjon
1. Hvordan sette opp prosjektet til å kjøre via docker
Det forutsettes at docker er installert og konfigurert på forhånd.

1.1 Hent python manuelt
#Fikk feil "ERROR [internal] load metadata for docker.io/library/python:3.14-slim" ved forsøk via requirements.txt
docker pull python:3.14-slim

1.2 Docker build image
docker build  \
  --build-arg http_proxy=http://<proxyip>:<proxyport> \
  --build-arg https_proxy=http://<proxyip>:<proxyport> \
  --build-arg no_proxy=localhost,<localhost_ip>\
  --tag smart_sladding_app:latest .
  
1.3 Start containers manuelt
docker run -it -v data:/data/ml_logs -p <containerport>:<exposeport> -m 32g -e http_proxy=http://<proxyip>:<proxyport> -e https_proxy=http://<proxyip>:<proxyport> --name smsl-server-prod smart_sladding_app
docker run -it -v data:/data/ml_logs -p <containerport>:<exposeport> -m 32g -e http_proxy=http://<proxyip>:<proxyport> -e https_proxy=http://<proxyip>:<proxyport> --name smsl-server-dev smart_sladding_app

1.4 Start containere med compose
docker compose up -d

1.5 Test med curl
curl -X POST http://localhost:<containerport>/model -H "Content-Type: application/pdf" --data-binary "@testdokument-2.pdf"
curl -X POST http://localhost:<containerport>/model -H "Content-Type: application/pdf" --data-binary "@testdokument-2.pdf"

2. Hvordan sette opp prosjektet og kjøre lokalt:

2.1. Klon repository
```sh
git clone https://github.com/kartverket/dokumentbestilling-smart-sladding.git
cd dokumentbestilling-smart-sladding
```

2.2. Opprett virtuelt miljø og installer avhengigheter
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For de som bruker mac: 
* Last ned Poppler `brew install poppler`
* Last ned Tesseract `brew install tesseract`

For onprem maskin, finn de nødvendige installasjonspakkene for Tesseract og Poppler, og installer disse.
* poppler versjon kan sjekkes med `pdftoppm -v` 
* tesseract versjon kan sjekkes med `tesseract -v` 


2.3. Kjør opp appen
```sh
cd app
python3 app.py
```

2.4. test appen via Curl i ny terminal:
```sh
mkdir -p app/logs
curl -X POST http://localhost:5070/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@app/testdokument-2.pdf"
```

2.5. Manuelt laste ned easyOCR modeller bak brannmuren (kun nødvendig for onprem maskin):
```sh
curl -L -x http://<proxyip>:<proxyport> -o "latin_g2.zip" https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip
curl -L -x http://<proxyip>:<proxyport> -o "craft_mlt_25k.zip" https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip

mv latin_g2.zip dokumentbestilling-smart-sladding/tmp/.EasyOCR/model
mv craft_mlt_25k.zip dokumentbestilling-smart-sladding/tmp/.EasyOCR/model

unzip latin_g2.zip
unzip craft_mlt_25k.zip
```

2.6. For production / test så kjører vi appen med gunicorn:
```sh
cd app
chmod +x start_production.sh
./start_production.sh
```

## EasyOCR-modeller bak brannmur
Dersom utviklingsmaskinen ikke har direkte internettilgang, last ned modellene manuelt og plasser dem i `tmp/.EasyOCR/model`:

```sh
curl -L -o latin_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip
curl -L -o craft_mlt_25k.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip

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
