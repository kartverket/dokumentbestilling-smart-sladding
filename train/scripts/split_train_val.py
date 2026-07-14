"""
Split images_all/labels_all into train/val/test sets.

Strategies:
  - random (default): shuffle all images and split by ratio
  - yearly: use a metadata CSV with dokument_aar to pick X images per year
"""

import argparse
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def _copy_split(dataset: Path, splits: list, labels_all: Path, log_lines: list):
    """Copy images/labels into train/val/test folders and write log."""
    for subset, group in splits:
        img_dest = dataset / "images" / subset
        label_dest = dataset / "labels" / subset
        img_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)

        log_lines.append(f"--- {subset}: {len(group)} images ---")

        doc_ids = sorted(set(img.stem.rsplit("_p", 1)[0] for img in group))
        log_lines.append(f"Documents ({len(doc_ids)}):")
        for doc_id in doc_ids:
            log_lines.append(f" {doc_id}")

        for img in group:
            label = labels_all / (img.stem + ".txt")
            shutil.copy(img, img_dest / img.name)
            if label.exists():
                shutil.copy(label, label_dest / label.name)

        log_lines.append("")
        print(f"  {subset}: {len(group)} images")


def split_random(dataset_dir: str, train_ratio: float, val_ratio: float, seed: int):
    """Randomly shuffle all images and split by ratio."""
    dataset = Path(dataset_dir)
    images_all = dataset / "images_all"
    labels_all = dataset / "labels_all"

    imgs = sorted(images_all.glob("*.png"))
    random.seed(seed)
    random.shuffle(imgs)

    n = len(imgs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = [
        ("train", imgs[:train_end]),
        ("val", imgs[train_end:val_end]),
        ("test", imgs[val_end:]),
    ]

    log_lines = []
    log_lines.append(f"Split log - {datetime.now().isoformat()}")
    log_lines.append(f"Strategy: random")
    log_lines.append(f"Seed: {seed}")
    log_lines.append(f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}")
    log_lines.append(f"Total images: {n}")
    log_lines.append("")

    _copy_split(dataset, splits, labels_all, log_lines)

    log_path = dataset / "split_log.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"  Log written to {log_path}")
    print(f"\nSplit complete: {n} total -> {dataset}/images/{{train,val,test}}")


def split_yearly(dataset_dir: str, metadata_csv: str, per_year: int,
                 train_ratio: float, val_ratio: float, seed: int):
    """Lager dataset splits for hvert år ved å bruke tingslysningdatoen fra metadata CSV.

    Tar per_year antall dokumenter per år, og splitter disse random til train/val/test basert på gitt ratio.
    per_year variabel er en maks grense, så hvis det finnes færre dokumenter for et år, vil alle bli brukt i splittet.
    """

    dataset = Path(dataset_dir)
    images_all = dataset / "images_all"
    labels_all = dataset / "labels_all"

    meta = pd.read_csv(metadata_csv)
    if "dokument_aar" not in meta.columns or "fil_revisjon_id" not in meta.columns:
        print("ERROR: Metadata CSV must have 'fil_revisjon_id' and 'dokument_aar' columns")
        raise SystemExit(1)

    meta["year"] = pd.to_numeric(meta["dokument_aar"], errors="coerce")
    id_to_year = dict(zip(meta["fil_revisjon_id"].astype(str), meta["year"]))

    # Grupperer images per år
    imgs = sorted(images_all.glob("*.png"))
    by_year = defaultdict(list)
    no_year = []
    for img in imgs:
        doc_id = img.stem.rsplit("_p", 1)[0]
        year = id_to_year.get(doc_id)
        if year and not pd.isna(year):
            by_year[int(year)].append(img)
        else:
            no_year.append(img)

    if no_year:
        print(f"  Warning: {len(no_year)} images had no matching year in metadata")

    random.seed(seed)

    # Samler alle bilder per år for train,val,test split
    train_imgs, val_imgs, test_imgs = [], [], []

    for year in sorted(by_year.keys()):
        year_imgs = by_year[year]
        random.shuffle(year_imgs)
        selected = year_imgs[:per_year]

        n = len(selected)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_imgs.extend(selected[:train_end])
        val_imgs.extend(selected[train_end:val_end])
        test_imgs.extend(selected[val_end:])

        print(f"  Year {year}: {len(year_imgs)} available, {n} selected")

    splits = [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]

    total = len(train_imgs) + len(val_imgs) + len(test_imgs)
    log_lines = []
    log_lines.append(f"Split log - {datetime.now().isoformat()}")
    log_lines.append(f"Strategy: yearly")
    log_lines.append(f"Metadata: {metadata_csv}")
    log_lines.append(f"Per year: {per_year}")
    log_lines.append(f"Seed: {seed}")
    log_lines.append(f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}")
    log_lines.append(f"Total images selected: {total}")
    log_lines.append("")
    log_lines.append("Per year:")
    for year in sorted(by_year.keys()):
        available = len(by_year[year])
        used = min(available, per_year)
        log_lines.append(f"  {year}: {used} used of {available} available")
    if no_year:
        log_lines.append(f"  (no year): {len(no_year)} images skipped")
    if no_year:
        log_lines.append(f"Images without year: {len(no_year)}")
    log_lines.append("")

    _copy_split(dataset, splits, labels_all, log_lines)

    log_path = dataset / "split_log.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"  Log written to {log_path}")
    print(f"\nSplit complete: {total} total -> {dataset}/images/{{train,val,test}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--strategy", choices=["random", "yearly"], default="random",
                        help="Split strategy (default: random)")
    parser.add_argument("--metadata", default="", help="Path to metadata CSV with dokument_aar (required for yearly)")
    parser.add_argument("--per-year", type=int, default=100, help="Max images per year for yearly strategy (default: 100)")
    args = parser.parse_args()

    if args.strategy == "yearly":
        if not args.metadata:
            print("ERROR: --metadata is required for yearly strategy")
            raise SystemExit(1)
        if not Path(args.metadata).exists():
            print(f"ERROR: Metadata file not found: {args.metadata}")
            raise SystemExit(1)
        split_yearly(args.dataset, args.metadata, args.per_year,
                     args.train_ratio, args.val_ratio, args.seed)
    else:
        split_random(args.dataset, args.train_ratio, args.val_ratio, args.seed)
