"""Publish a training run as a finished model in the weights store.

An ultralytics run directory is a working directory, not a model store: the
file is called best.pt whichever model it is, and the name only exists in the
folder above. This copies the run out to $SLADD_WEIGHTS/<name>/ as <name>.pt +
modell.json + trening/, so the model can be moved without losing its identity.

The metadata is read from the run itself and from git, so it describes what
was actually run. What only the Makefile knows, which PDFs and which truth
the dataset came from, is passed in with --info.

Run:
    python publish_model.py --run $SLADD_RUNS/uttrekk_4_jou --out $SLADD_WEIGHTS
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

# Worth keeping next to the weights. The rest stays behind in $SLADD_RUNS.
TRAINING_FILES = ["args.yaml", "results.csv", "results.png",
                 "confusion_matrix_normalized.png", "PR_curve.png"]

# Same measure ultralytics itself uses when it picks best.pt.
BEST_COLUMN = "metrics/mAP50-95(B)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_yaml(path):
    """args.yaml/data.yaml as a dict.

    pyyaml ships with ultralytics, so it is there on the training server.
    Without it a crude parser takes over: both files are flat "key: value"
    with one exception (names), and having the hyperparameters beats writing
    "unknown" into the metadata.
    """
    try:
        line_text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import yaml
        return yaml.safe_load(line_text) or {}
    except ImportError:
        pass
    except Exception:
        return {}
    return _simple_yaml(line_text)


def _simple_yaml(line_text):
    def value(v):
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

    out, parent_key = {}, None
    for line in line_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        sort_key, _, rest = line.partition(":")
        if line.startswith((" ", "\t")) and isinstance(out.get(parent_key), dict):
            out[parent_key][value(sort_key)] = value(rest)
            continue
        sort_key = sort_key.strip()
        if rest.strip():
            out[sort_key] = value(rest)
        else:
            out[sort_key] = {}
            parent_key = sort_key
    return out


def read_results(path):
    """Best epoch and the metrics it reached, from results.csv."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in row.items() if k}
                     for row in csv.DictReader(f)]
    except (OSError, ValueError):
        return {}
    if not rows:
        return {}

    def score(row):
        try:
            return float(row.get(BEST_COLUMN, "nan"))
        except ValueError:
            return float("-inf")

    best = max(rows, key=score)
    out = {"epoker_kjort": len(rows)}
    for sort_key, value in best.items():
        if sort_key == "epoch" or sort_key.startswith("metrics/"):
            try:
                out[sort_key] = float(value)
            except ValueError:
                out[sort_key] = value
    return out


def count_images(dataset_dir):
    """Images per split: what the model actually saw."""
    out = {}
    for split in ("train", "val", "test"):
        folder = dataset_dir / "images" / split
        if folder.is_dir():
            out[split] = sum(1 for _ in folder.iterdir())
    return out


def git_status(repo):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = git("rev-parse", "HEAD")
    if not sha:
        return {}
    status = git("status", "--porcelain")
    return {"git_sha": sha, "git_rent_tre": status == ""}


def env():
    out = {"python": sys.version.split()[0]}
    for module, name in (("ultralytics", "ultralytics"), ("torch", "torch")):
        try:
            out[name] = __import__(module).__version__
        except Exception:
            pass
    return out


def find_run(run):
    """Refuses if ultralytics has put the result in a sibling directory.

    Running the same NAME twice writes to <name>2 and leaves the old folder
    alone, so we would publish the previous run without noticing.
    """
    if not run.is_dir():
        sys.exit(f"ERROR: no such training run: {run}")
    sibling = sorted((d for d in run.parent.glob(run.name + "*") if d.is_dir()),
                    key=lambda d: d.stat().st_mtime)
    if sibling and sibling[-1] != run:
        sys.exit(f"ERROR: {sibling[-1]} is newer than {run}. Ultralytics numbers\n"
                 f"      directories on a repeated NAME. Pick the run explicitly:\n"
                 f"      --run {sibling[-1]}")
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None,
                   help="the training run (the directory holding weights/best.pt)")
    p.add_argument("--weight-file", default=None,
                   help="publish a loose .pt with no training run, to move old models "
                        "into the store. The metadata is thin then.")
    p.add_argument("--out", default=os.environ.get("SLADD_WEIGHTS"),
                   help="the weights store; default $SLADD_WEIGHTS")
    p.add_argument("--name", default=None,
                   help="model name; defaults to the name of the training run")
    p.add_argument("--weights", default="best", choices=["best", "last"],
                   help="which checkpoint to publish (default: best)")
    p.add_argument("--dataset", default=None, help="the dataset directory that was trained on")
    p.add_argument("--info", action="append", default=[], metavar="KEY=VALUE",
                   help="extra facts about the dataset (repeatable)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite a model already published under this name")
    args = p.parse_args()

    if not args.out:
        sys.exit("ERROR: --out is missing and SLADD_WEIGHTS is not set. Run `source activate.sh`.")
    if bool(args.run) == bool(args.weight_file):
        sys.exit("ERROR: give either --run (a training run) or --weight-file (a loose .pt).")

    if args.weight_file:
        run = None
        source = Path(args.weight_file).resolve()
        if not source.is_file():
            sys.exit(f"ERROR: no such file: {source}")
        if not args.name:
            sys.exit("ERROR: --name is required with --weight-file. The filename alone does "
                     "not say what the model is called. Every one of them is best.pt.")
    else:
        run = find_run(Path(args.run).resolve())
        source = run / "weights" / f"{args.weights}.pt"
        if not source.is_file():
            sys.exit(f"ERROR: no such file: {source}. Did the training finish?")

    name = args.name or run.name
    mal = Path(args.out).resolve() / name
    weight_file = mal / f"{name}.pt"

    if weight_file.exists() and not args.overwrite:
        if sha256(weight_file) == sha256(source):
            print(f"{name} is already published with the same weights: {weight_file}")
            return
        sys.exit(f"ERROR: {weight_file} exists with different weights.\n"
                 f"      Use another --name, or --overwrite to replace it.")

    dataset_dir = Path(args.dataset).resolve() if args.dataset else None
    arguments = read_yaml(run / "args.yaml") if run else {}
    extra = dict(kv.split("=", 1) for kv in args.info if "=" in kv and kv.split("=", 1)[1])

    mal.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, weight_file)
    for folder, files in ((run, TRAINING_FILES), (dataset_dir, ("data.yaml", "split_log.txt"))):
        for file in files if folder else ():
            if (folder / file).is_file():
                (mal / "training").mkdir(exist_ok=True)
                shutil.copy2(folder / file, mal / "training" / file)

    metadata = {
        "name": name,
        "published": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "weights": {
            "file": weight_file.name,
            "sha256": sha256(weight_file),
            "bytes": weight_file.stat().st_size,
            "checkpoint": args.weights,
            "source": str(source),
        },
        "trained": {
            "date": datetime.fromtimestamp(source.stat().st_mtime).astimezone()
                    .isoformat(timespec="seconds"),
            "run": str(run) if run else None,
            "base_model": extra.pop("base_model", None) or arguments.get("model"),
            "epochs": arguments.get("epochs"),
            "imgsz": arguments.get("imgsz"),
            "batch": arguments.get("batch"),
            "patience": arguments.get("patience"),
            "device": arguments.get("device"),
            "arguments": arguments,
        },
        "dataset": {
            "path": str(dataset_dir) if dataset_dir else None,
            "n_images": count_images(dataset_dir) if dataset_dir else {},
            "classes": read_yaml(dataset_dir / "data.yaml").get("names") if dataset_dir else None,
            **extra,
        },
        "results": read_results(run / "results.csv") if run else {},
        "code": git_status(Path(__file__).resolve().parents[2]),
        "env": env(),
    }
    # Say it outright, so the empty fields do not read as a failed metadata read.
    if run is None:
        metadata["trained"]["unknown_origin"] = ("published from a loose weights file. "
                                              "The training run no longer exists")
    (mal / "modell.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Published model «{name}»")
    print(f"  weights:  {weight_file}")
    print(f"  metadata: {mal / 'modell.json'}")
    print(f"  sha256:   {metadata['weights']['sha256'][:16]}…")
    print()
    print("Validate it:")
    print(f"  ./valider_yolo.sh modell={weight_file} uttrekk=N")
    print("Build an image with it:")
    print(f"  ./deploy.sh build vekter={weight_file}")


if __name__ == "__main__":
    main()
