# smart_sladding_ml

## Beskrivelse
Prosjektet er utviklet for å generere automatiske sladdinger av personnummer og d-nummer i dokumentbestillinger. Dette gjøres ved å bruke en Tesseract OCR modell til å gjenkjenne og detektere innholdet i dokumentene. Deretter gjennomføres et regex og/eller keyword søk på innholdet for å klassifisere personnummer og d-nummer. 

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


