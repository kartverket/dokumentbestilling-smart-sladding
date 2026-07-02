# Smartsladding — OCR-sladding av fødselsnumre

Finner norske fødselsnumre i PDF-er (skannede pantebok-dokumenter) med OCR og
gir ut koordinatene til sladde-bokser. To bruksmåter:

- **Produksjon (API):** PDF som bytes inn → koordinater ut (`model_main.py`).
- **Test:** kjør mot en fasit-CSV, få recall-måling + bilder, og evt. faktisk
  sladdede PDF-er (`run.py`).

## Oppsett

Krever Python 3.12.

```bash
python3.12 -m venv venv
source venv/bin/activate
python --version          # skal si Python 3.12.x
pip install --upgrade pip
```

### 1) torch + torchvision — egen installasjon, FØR resten

torch installeres for seg fordi GPU-bygget ikke ligger på vanlig PyPI. Gjør det
*før* `requirements.txt`, ellers drar easyocr inn CPU-versjonen av torch som
transitiv avhengighet. Installer begge i samme kommando så pip matcher
kompatible versjoner.

```bash
# Linux-server med GPU — CUDA-bygg.
# Bytt cu121 til å matche "CUDA Version" øverst i nvidia-smi (cu118 / cu121 / cu124).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Mac / maskin uten GPU — vanlig CPU-bygg.
pip install torch torchvision
```

Sjekk at GPU-en faktisk brukes:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- `True NVIDIA ...` → GPU brukes. Klar.
- `False CPU` → kjører på CPU. Da matcher som regel ikke `cuXXX`-tallet driveren,
  eller CPU-torch ligger der fra før:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2) Resten av avhengighetene

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # presidio laster denne under panseret
```

## Kjøre (test)

Kjøres fra undermappa `testing av sladding sommer 2026`. Last opp PDF-ene i en
`panteboksdokumenter`-mappe der (lurt å se på oppsettet lokalt før du gjør noe
på server).

```bash
python run.py                      # 20 første filer i standardmappa
python run.py --antall alle        # alle filene (eller et tall, f.eks. 40)
python run.py --velg 1020868081    # bare filer som matcher (filnavn/delstreng)
python run.py --ingen-visning      # raskere: hopp over PNG-tegningen
python run.py --sladd              # lag også faktisk sladdede PDF-er
python run.py --antall 2 --png.    
python run.py --help               # alle flagg
```

Resultat (standardmapper, alle gitignoret):

- `sladd_koordinater.csv` — alle funne bokser (`--csv-ut`)
- `visning/` — én PNG per side, rødt = funnet, grønt = fasit (`--ut-mappe`)
- recall mot fasit skrives i terminalen
- `sladdet/` — faktisk sladdede PDF-er, kun med `--sladd` (`--sladd-ut`)

Inn-mappe og fasit-CSV styres med `--mappe` og `--csv`.

## Produksjon (API)

Flask-ruten kaller `model_main.run_model_on_pdf_bytes(pdf_bytes)` og får
koordinatene tilbake som JSON. I produksjon trengs bare `model_main.py`,
`loading.py` og `recognition.py` — resten er testverktøy.

## Filene

| Fil                 | Ansvar                                            |
|---------------------|---------------------------------------------------|
| `model_main.py`     | PDF-bytes → koordinater (API-et kaller denne)     |
| `loading.py`        | Leser PDF (filsti + bytes) → bilder               |
| `recognition.py`    | OCR + fnr-gjenkjenning + boks-logikk (kjernen)    |
| `csv_export.py`     | Skriver alle bokser til CSV / leser tilbake       |
| `ground_truth.py`   | Leser fasit-CSV                                    |
| `evaluation.py`     | Måler recall (treff/bom) mot fasit                |
| `visualization.py`  | Tegner funne + fasit-bokser, lagrer PNG           |
| `redaction.py`      | Faktisk sladding av PDF                           |
| `file_selection.py` | Velger hvilke filer som kjøres                    |
| `run.py`            | Test-CLI (den du starter)                         |