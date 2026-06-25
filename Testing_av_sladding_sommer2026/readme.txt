
## Oppsett

```bash
# 1) lag og aktiver et venv på Python 3.12
python3.12 -m venv venv
source venv/bin/activate
python --version          # skal si Python 3.12.x

# 2) oppgrader pip og installer alt UNNTATT torch
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` inneholder alt programmet trenger **bortsett fra**
`torch` og `torchvision`. De installeres for seg (se neste avsnitt), fordi
GPU-versjonen ikke ligger på vanlig PyPI.

## torch / torchvision (egen installasjon)

Velg den ene som passer maskinen:

```bash
# Linux-server med GPU — CUDA-bygg.
# Bytt cu121 til å matche "CUDA Version" øverst i `nvidia-smi` (cu118 / cu121 / cu124).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Mac / maskin uten GPU — vanlig CPU-bygg.
pip install torch torchvision
```

Installer torch og torchvision i **samme** kommando, så pip matcher
kompatible versjoner.

## Sjekk at GPU-en faktisk brukes

Etter installasjon på serveren:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- `True NVIDIA ...`  → GPU brukes. Klar.
- `False CPU`        → kjører på CPU. Da matcher som regel ikke `cuXXX`-tallet
  driveren, eller CPU-torch ligger der fra før. Fiks:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Kjøre gjøres ifra egen undermappe testing av sladding sommer 2026

```bash
python kjor.py                       # 20 første filer i standardmappa
python kjor.py --antall alle         # alle filene eller et antall f.eks 40 
python kjor.py --velg 1020868081     # bare filer som matcher (filnavn/delstreng)
python kjor.py --antall 2 --ingen-logg   # rask test uten den trege per-linje-loggen
python kjor.py --help                # alle flagg
```

Input-mappe, output-mappe og fasit-CSV styres med `--mappe`, `--ut-mappe` og
`--csv` (har standardverdier i `kjor.py`). Output (PNG-er, `bokser.csv`,
`ocr_linjer.txt`) havner i ut-mappa. standar mapper er gitignoret 

Det som må gjøres er å laste opp dokumentene i testing av sladding sommer2026 i en egen panteboksdokumenter mapper (lurt å se på oppsettet lokalt før man gjør noe på server)
## Filene

| Fil             | Ansvar                                            |
|-----------------|---------------------------------------------------|
| `kjor.py`       | Kjører hele pipelinen (den du starter)            |
| `sladd_lib.py`  | OCR, fnr-gjenkjenning, boks-logikk (kjernen)      |
| `filvalg.py`    | Velger hvilke filer som kjøres                    |
| `fasit.py`      | Leser fasit-CSV                                   |
| `visning.py`    | Tegner bokser, lagrer PNG + CSV + logg            |
| `evaluering.py` | Måler recall mot fasit                            |