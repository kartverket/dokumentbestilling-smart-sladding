# dokumentbestilling-smart-sladding

## VIKTIG: For bruk i Kartverket
Hvis du jobber med dette i Kartverket-sammenheng, så skal du være inneforstått med de relevante rutinene før du begynner med å arbeide. [Les rutinene på confluence her](https://kartverket.atlassian.net/wiki/x/F4Dwn)

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
1. Klon repository
```sh
git clone https://github.com/kartverket/dokumentbestilling-smart-sladding.git
cd dokumentbestilling-smart-sladding
```

2. Opprett virtuelt miljø og installer avhengigheter
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Kjør lokalt
```sh
cd app
python3 app.py
```

Test endepunktet i ny terminal:
```sh
mkdir -p app/logs
curl -X POST http://localhost:5070/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@app/testdokument-2.pdf"
```

## Produksjon
Appen kjøres med Gunicorn i produksjon:
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
