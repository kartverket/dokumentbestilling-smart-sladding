"""Publiser en treningskjøring som en ferdig modell i vektlageret.

Ultralytics skriver til <project>/<name>/weights/best.pt sammen med
checkpoints, plott og alt annet som hører til selve kjøringen. Det er en
arbeidsmappe, ikke et modellager: filen heter best.pt uansett hvilken
modell det er, og navnet finnes bare i mappen over.

Dette skriptet lager en modell som kan flyttes uten å miste hvem den er:

    $SLADD_VEKTER/<navn>/
        <navn>.pt        vektene, navngitt etter modellen
        modell.json      hva den er trent på, med hvilke parametere
        trening/         results.csv, args.yaml, data.yaml, split_log.txt

Metadataen leses ut av kjøringen selv (args.yaml, results.csv, data.yaml)
og av git, slik at den beskriver det som faktisk ble kjørt. Det som bare
Makefilen vet — hvilke PDF-er og hvilken fasit datasettet kom fra — sendes
inn med --info.

Bruk:
    python publiser_modell.py --run $SLADD_RUNS/uttrekk_4_jou --ut $SLADD_VEKTER
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Filer fra kjøringen som er verdt å ta vare på ved siden av vektene.
# Resten (checkpoints, batch-bilder) blir liggende i $SLADD_RUNS.
TRENINGSFILER = ["args.yaml", "results.csv", "results.png",
                 "confusion_matrix_normalized.png", "PR_curve.png"]

# Kolonnen som avgjør hvilken epoke som var best. Samme mål ultralytics
# selv bruker når den velger best.pt.
BESTE_KOLONNE = "metrics/mAP50-95(B)"


def sha256(sti):
    h = hashlib.sha256()
    with open(sti, "rb") as f:
        for blokk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blokk)
    return h.hexdigest()


def les_yaml(sti):
    """args.yaml/data.yaml som dict.

    pyyaml følger med ultralytics, så på treningsserveren er den der. Uten
    den faller vi tilbake på en enkel parser — args.yaml og data.yaml er
    flate «nokkel: verdi»-filer med ett unntak (names), og det er bedre å
    få med hyperparameterne enn å skrive «ukjent» i metadataen.
    """
    try:
        tekst = Path(sti).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import yaml
        return yaml.safe_load(tekst) or {}
    except ImportError:
        pass
    except Exception:
        return {}
    return _enkel_yaml(tekst)


def _enkel_yaml(tekst):
    def verdi(v):
        v = v.strip().strip("'\"")
        if v in ("", "null", "None"):
            return None
        if v in ("true", "True"):
            return True
        if v in ("false", "False"):
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    ut, forelder = {}, None
    for linje in tekst.splitlines():
        if not linje.strip() or linje.lstrip().startswith("#") or ":" not in linje:
            continue
        nokkel, _, rest = linje.partition(":")
        if linje.startswith((" ", "\t")) and isinstance(ut.get(forelder), dict):
            ut[forelder][verdi(nokkel)] = verdi(rest)
            continue
        nokkel = nokkel.strip()
        if rest.strip():
            ut[nokkel] = verdi(rest)
        else:
            ut[nokkel] = {}
            forelder = nokkel
    return ut


def les_resultater(sti):
    """Beste epoke og målene den ga, hentet fra results.csv."""
    try:
        with open(sti, newline="", encoding="utf-8") as f:
            rader = [{k.strip(): v.strip() for k, v in rad.items() if k}
                     for rad in csv.DictReader(f)]
    except (OSError, ValueError):
        return {}
    if not rader:
        return {}

    def score(rad):
        try:
            return float(rad.get(BESTE_KOLONNE, "nan"))
        except ValueError:
            return float("-inf")

    beste = max(rader, key=score)
    ut = {"epoker_kjort": len(rader)}
    for nokkel, verdi in beste.items():
        if nokkel == "epoch" or nokkel.startswith("metrics/"):
            try:
                ut[nokkel] = float(verdi)
            except ValueError:
                ut[nokkel] = verdi
    return ut


def tell_bilder(datasett):
    """Antall bilder per split — hva modellen faktisk så."""
    ut = {}
    for split in ("train", "val", "test"):
        mappe = datasett / "images" / split
        if mappe.is_dir():
            ut[split] = sum(1 for _ in mappe.iterdir())
    return ut


def git_status(repo):
    def kjor(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = kjor("rev-parse", "HEAD")
    if not sha:
        return {}
    status = kjor("status", "--porcelain")
    return {"git_sha": sha, "git_rent_tre": status == ""}


def miljo():
    ut = {"python": sys.version.split()[0]}
    for modul, navn in (("ultralytics", "ultralytics"), ("torch", "torch")):
        try:
            ut[navn] = __import__(modul).__version__
        except Exception:
            pass
    return ut


def finn_run(run):
    """Advar hvis ultralytics har lagt resultatet i en søskenmappe.

    Kjører man samme NAME to ganger, skriver ultralytics til <navn>2 og
    lar den gamle mappen stå. Da ville vi publisert forrige kjøring uten
    å merke det.
    """
    if not run.is_dir():
        sys.exit(f"FEIL: finner ikke treningskjøringen: {run}")
    sosken = sorted((d for d in run.parent.glob(run.name + "*") if d.is_dir()),
                    key=lambda d: d.stat().st_mtime)
    if sosken and sosken[-1] != run:
        sys.exit(f"FEIL: {sosken[-1]} er nyere enn {run}. Ultralytics nummererer\n"
                 f"      mapper ved gjentatt NAME. Velg kjøring eksplisitt:\n"
                 f"      --run {sosken[-1]}")
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None,
                   help="treningskjøringen (mappen med weights/best.pt)")
    p.add_argument("--vektfil", default=None,
                   help="publiser en løs .pt-fil uten treningskjøring (for å flytte "
                        "gamle modeller inn i lageret). Metadataen blir da tynn.")
    p.add_argument("--ut", default=os.environ.get("SLADD_VEKTER"),
                   help="vektlageret; default $SLADD_VEKTER")
    p.add_argument("--navn", default=None,
                   help="navn på modellen; default er navnet på treningskjøringen")
    p.add_argument("--vekter", default="best", choices=["best", "last"],
                   help="hvilken checkpoint som publiseres (default: best)")
    p.add_argument("--dataset", default=None, help="datasettmappen som ble trent på")
    p.add_argument("--info", action="append", default=[], metavar="NOKKEL=VERDI",
                   help="ekstra opplysninger om datasettet (kan gjentas)")
    p.add_argument("--overskriv", action="store_true",
                   help="overskriv en modell som allerede er publisert med dette navnet")
    args = p.parse_args()

    if not args.ut:
        sys.exit("FEIL: --ut mangler og SLADD_VEKTER er ikke satt. Kjør «source activate.sh».")
    if bool(args.run) == bool(args.vektfil):
        sys.exit("FEIL: oppgi enten --run (en treningskjøring) eller --vektfil (en løs .pt).")

    if args.vektfil:
        run = None
        kilde = Path(args.vektfil).resolve()
        if not kilde.is_file():
            sys.exit(f"FEIL: finner ikke {kilde}")
        if not args.navn:
            sys.exit("FEIL: --navn er påkrevd sammen med --vektfil. Filnavnet alene "
                     "sier ikke hva modellen heter (best.pt heter alle).")
    else:
        run = finn_run(Path(args.run).resolve())
        kilde = run / "weights" / f"{args.vekter}.pt"
        if not kilde.is_file():
            sys.exit(f"FEIL: finner ikke {kilde}. Ble treningen fullført?")

    navn = args.navn or run.name
    mal = Path(args.ut).resolve() / navn
    vektfil = mal / f"{navn}.pt"

    if vektfil.exists() and not args.overskriv:
        if sha256(vektfil) == sha256(kilde):
            print(f"{navn} er allerede publisert med samme vekter: {vektfil}")
            return
        sys.exit(f"FEIL: {vektfil} finnes med andre vekter.\n"
                 f"      Bruk et annet --navn, eller --overskriv hvis den skal erstattes.")

    datasett = Path(args.dataset).resolve() if args.dataset else None
    argumenter = les_yaml(run / "args.yaml") if run else {}
    ekstra = dict(kv.split("=", 1) for kv in args.info if "=" in kv and kv.split("=", 1)[1])

    mal.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kilde, vektfil)
    for mappe, filer in ((run, TRENINGSFILER), (datasett, ("data.yaml", "split_log.txt"))):
        for fil in filer if mappe else ():
            if (mappe / fil).is_file():
                (mal / "trening").mkdir(exist_ok=True)
                shutil.copy2(mappe / fil, mal / "trening" / fil)

    metadata = {
        "navn": navn,
        "publisert": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "vekter": {
            "fil": vektfil.name,
            "sha256": sha256(vektfil),
            "bytes": vektfil.stat().st_size,
            "checkpoint": args.vekter,
            "kilde": str(kilde),
        },
        "trent": {
            "dato": datetime.fromtimestamp(kilde.stat().st_mtime).astimezone()
                    .isoformat(timespec="seconds"),
            "run": str(run) if run else None,
            "basismodell": ekstra.pop("basismodell", None) or argumenter.get("model"),
            "epochs": argumenter.get("epochs"),
            "imgsz": argumenter.get("imgsz"),
            "batch": argumenter.get("batch"),
            "patience": argumenter.get("patience"),
            "device": argumenter.get("device"),
            "argumenter": argumenter,
        },
        "datasett": {
            "sti": str(datasett) if datasett else None,
            "antall_bilder": tell_bilder(datasett) if datasett else {},
            "klasser": les_yaml(datasett / "data.yaml").get("names") if datasett else None,
            **ekstra,
        },
        "resultater": les_resultater(run / "results.csv") if run else {},
        "kode": git_status(Path(__file__).resolve().parents[2]),
        "miljo": miljo(),
    }
    # En løs .pt uten treningskjøring: si det rett ut i metadataen, i stedet
    # for å la tomme felter se ut som om noe gikk galt under lesingen.
    if run is None:
        metadata["trent"]["ukjent_opphav"] = ("publisert fra en løs vektfil — "
                                              "treningskjøringen finnes ikke lenger")
    (mal / "modell.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Publisert modell «{navn}»")
    print(f"  vekter:   {vektfil}")
    print(f"  metadata: {mal / 'modell.json'}")
    print(f"  sha256:   {metadata['vekter']['sha256'][:16]}…")
    print()
    print("Valider den:")
    print(f"  ./valider_yolo.sh modell={vektfil} uttrekk=N")
    print("Bygg et image med den:")
    print(f"  ./deploy.sh build vekter={vektfil}")


if __name__ == "__main__":
    main()
