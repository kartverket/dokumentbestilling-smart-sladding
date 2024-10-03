# smart_sladding_ml

## Beskrivelse
Prosjektet er utviklet for å generere automatiske sladdinger av personnummer og d-nummer i dokumentbestillinger. Dette gjøres ved å bruke en Tesseract OCR modell til å gjenkjenne og detektere innholdet i dokumentene. Deretter gjennomføres et regex og/eller keyword søk på innholdet for å klassifisere områder med personnummer og d-nummer. 

## Installasjon
Hvordan sette opp prosjektet og kjøre lokalt:

1. Klon repository
```sh
git clone https://github.com/kartverket/smart_sladding_ml.git
cd smart_sladding_ml
```

2. Installer nødvendige pakker
```sh
pip install -r requirements.txt
```

Last ned Poppler `brew install poppler`
Last ned Tesseract `brew install tesseract`

NB: Hvis du får SSL-problematikk på Mac, så kan du sjekke ut denne Stack Overflow-responsen https://stackoverflow.com/a/57795811

## How to test

### Teste modellen for ett dokument
For å kjøre ett dokument med dokument id (doc_id) på formatet '<dokument_aar>_<dokument_nr>_<embete>' kjører man:
```sh
predicted_boxes = get_predicted_boxes_on_doc(doc_id, "https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok")
```
Denne funksjonen returnerer de predikerte avgrensingsboksene og lagrer et bilde med disse.

### Evaluere på flere dokumenter
For å evaluere modellen på flere dokumenter, kan man kjøre `evaluation_main.py`. Denne filen tar inn en liste med dokumenter og kjører modellen på disse.  
Hvis du oppdaterer `current_model_version_number`, så vil en ny minor version gjøre at den kjører reglene på nytt, men med cachet OCR-lesing. Ved ny major-versjon, så vil den kjøre OCR-lesing på nytt.  

Første gang du kjører evaluering, vil du måtte kjøre OCR på alle dokumenter. Dette tar ofte mange(8+) timer for 1400 dokumenter.
