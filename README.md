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

### Evaluere et sett med flere dokumenter
For å evaluere flere dokumenter kjører man:
```sh
total_results, total_tp, total_fp, total_fn, df_results = evaluate_model(document_folder, labels_csv, docids_csv, savefolder_name)
```
Denne funksjonen returnerer total_results (en liste med metrics per side per dokument), total_tp (totalt antall true positives), total_fp (totalt antall false positives), total_fn (totalt antall false negatives) og df_results (dataframe med antall true positives, false positives og false negatives per dokument). Funksjonen trenger document_folder (path til mappen med de nedlastede dokumentene), labels_cv (path til en .csv fil med de labella boksene per dokument), docids_csv (path til en .csv fil med dokument id til de ulike dokumentene som skal bli evaluert) og savefolder_name (path til en folder der bilder av de evaluerte dokumentene blir lagret).
