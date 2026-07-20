"""
Split images_all/labels_all into train/val/test sets.

Strategies:
  - random: shuffle all images and split by ratio
  - yearly: group by dokument_aar, select N per year, split per group
  - doc_type: filter by rettsstiftelsestyper, split by ratio
  - year_and_doc_type: filter by rettsstiftelsestyper + group by year
"""

import argparse
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# ─── Core split engine ────────────────────────────────────────────────────────

NEGATIVE_RATIO = 0.10


def _is_negative(img: Path, labels_all: Path) -> bool:
    """An image is a negative sample if its label file is empty or missing."""
    label = labels_all / (img.stem + ".txt")
    return not label.exists() or label.stat().st_size == 0


def _separate_positives_negatives(imgs: list, labels_all: Path) -> tuple[list, list]:
    """Split image list into positives (have annotations) and negatives (empty labels)."""
    positives = []
    negatives = []
    for img in imgs:
        if _is_negative(img, labels_all):
            negatives.append(img)
        else:
            positives.append(img)
    return positives, negatives


def _add_negatives(splits: list, negatives: list, ratio: float = NEGATIVE_RATIO):
    """Add negatives to each split subset proportional to its positive count."""
    random.shuffle(negatives)
    offset = 0
    for i, (subset, positives) in enumerate(splits):
        n_neg = int(len(positives) * ratio)
        selected_neg = negatives[offset:offset + n_neg]
        offset += n_neg
        splits[i] = (subset, positives + selected_neg)
        if selected_neg:
            print(f"  {subset}: +{len(selected_neg)} negatives")
    return splits


def _do_split(imgs: list, train_ratio: float, val_ratio: float):
    """Compute train/val/test split boundaries on an already-shuffled list."""
    n = len(imgs)
    train = int(n * train_ratio)
    val= int(n * (train_ratio + val_ratio))
    return [("train", imgs[:train]), ("val", imgs[train:val]), ("test", imgs[val:])]


def _copy_and_log(dataset: Path, splits: list, log_lines: list):
    """Copy images+labels into split folders and write log."""
    labels_all = dataset / "labels_all"
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

    total = sum(len(g) for _, g in splits)
    log_path = dataset / "split_log.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"  Log written to {log_path}")
    print(f"\nSplit complete: {total} total -> {dataset}/images/{{train,val,test}}")


def _shuffle_and_split(dataset: Path, imgs: list, train_ratio: float, val_ratio: float,
                       seed: int, log_header: list):
    """Shuffle positives, split by ratio, add 10% negatives per subset, copy and log."""
    labels_all = dataset / "labels_all"
    positives, negatives = _separate_positives_negatives(imgs, labels_all)

    random.seed(seed)
    random.shuffle(positives)
    splits = _do_split(positives, train_ratio, val_ratio)
    splits = _add_negatives(splits, negatives)

    log_header.append(f"Negatives: {sum(len(s) for _, s in splits) - len(positives)} of {len(negatives)} available")
    log_header.append("")
    _copy_and_log(dataset, splits, log_header)


def _split_per_group(dataset: Path, groups: dict, per_group: int,
                     train_ratio: float, val_ratio: float, seed: int, log_header: list):
    """Split per group: shuffle each, select up to per_group, split, merge. Add 10% negatives."""
    labels_all = dataset / "labels_all"
    random.seed(seed)
    train, val, test = [], [], []
    all_negatives = []

    for key in sorted(groups):
        g = groups[key]
        positives, negatives = _separate_positives_negatives(g, labels_all)
        random.shuffle(positives)
        selected = positives[:per_group]
        s = _do_split(selected, train_ratio, val_ratio)
        train.extend(s[0][1])
        val.extend(s[1][1])
        test.extend(s[2][1])
        all_negatives.extend(negatives)
        print(f"  {key}: {len(positives)} positives available, {len(selected)} selected, {len(negatives)} negatives")

    for key in sorted(groups):
        pos_in_group = [img for img in groups[key] if not _is_negative(img, labels_all)]
        log_header.append(f"  {key}: {min(len(pos_in_group), per_group)} used of {len(pos_in_group)} available")
    log_header.append("")

    total_positives = len(train) + len(val) + len(test)
    splits = [("train", train), ("val", val), ("test", test)]
    splits = _add_negatives(splits, all_negatives)
    total_negatives = sum(len(s) for _, s in splits) - total_positives

    log_header.append(f"Negatives: {total_negatives} added from {len(all_negatives)} available")
    _copy_and_log(dataset, splits, log_header)


# ─── Metadata helpers ─────────────────────────────────────────────────────────


def _load_metadata(metadata_csv: str, required_columns: list) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    missing = [c for c in required_columns if c not in meta.columns]
    if missing:
        print(f"ERROR: Metadata CSV missing columns: {missing}")
        raise SystemExit(1)
    return meta


def _group_images_by(images_all: Path, id_to_key: dict) -> tuple[dict, list]:
    """Group images by a key from id_to_key. Returns (groups, unmatched)."""
    imgs = sorted(images_all.glob("*.png"))
    groups = defaultdict(list)
    unmatched = []
    for img in imgs:
        doc_id = img.stem.rsplit("_p", 1)[0]
        key = id_to_key.get(doc_id)
        if key is not None:
            groups[key].append(img)
        else:
            unmatched.append(img)
    return dict(groups), unmatched


def _year_map(meta: pd.DataFrame) -> dict:
    """Build {fil_revisjon_id(str) -> year(int)} from a DataFrame with dokument_aar."""
    years = pd.to_numeric(meta["dokument_aar"], errors="coerce")
    ids = meta["fil_revisjon_id"].astype(str)
    return {id_: int(y) for id_, y in zip(ids, years) if not pd.isna(y)}


# ─── Strategies ───────────────────────────────────────────────────────────────


def split_random(dataset_dir: str, train_ratio: float, val_ratio: float, seed: int):
    dataset = Path(dataset_dir)
    imgs = sorted((dataset / "images_all").glob("*.png"))

    log_header = [
        f"Split log - {datetime.now().isoformat()}",
        f"Strategy: random | Seed: {seed}",
        f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}",
        f"Total images: {len(imgs)}",
        "",
    ]
    _shuffle_and_split(dataset, imgs, train_ratio, val_ratio, seed, log_header)


def split_yearly(dataset_dir: str, metadata_csv: str, per_year: int,
                 train_ratio: float, val_ratio: float, seed: int):
    dataset = Path(dataset_dir)
    meta = _load_metadata(metadata_csv, ["fil_revisjon_id", "dokument_aar"])
    groups, unmatched = _group_images_by(dataset / "images_all", _year_map(meta))

    if unmatched:
        print(f"  Warning: {len(unmatched)} images had no matching year")

    total_available = sum(min(len(g), per_year) for g in groups.values())
    log_header = [
        f"Split log - {datetime.now().isoformat()}",
        f"Strategy: yearly | Per year: {per_year} | Seed: {seed}",
        f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}",
        f"Total images available: {total_available}",
        "",
    ]
    _split_per_group(dataset, groups, per_year, train_ratio, val_ratio, seed, log_header)


def split_doc_type(dataset_dir: str, metadata_csv: str, doc_type: str,
                   train_ratio: float, val_ratio: float, seed: int):
    dataset = Path(dataset_dir)
    meta = _load_metadata(metadata_csv, ["fil_revisjon_id", "rettsstiftelsestyper"])
    valid_ids = set(meta.loc[meta["rettsstiftelsestyper"].str.split(" ", n=1).str[0] == doc_type, "fil_revisjon_id"].astype(str))

    imgs = sorted((dataset / "images_all").glob("*.png"))
    selected = [img for img in imgs if img.stem.rsplit("_p", 1)[0] in valid_ids]

    print(f"  Doc type '{doc_type}': {len(selected)} images from {len(valid_ids)} documents")

    log_header = [
        f"Split log - {datetime.now().isoformat()}",
        f"Strategy: doc_type | Doc type: {doc_type} | Seed: {seed}",
        f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}",
        f"Total images: {len(selected)}",
        "",
    ]
    _shuffle_and_split(dataset, selected, train_ratio, val_ratio, seed, log_header)


def split_year_and_doc_type(dataset_dir: str, metadata_csv: str, doc_type: str,
                            year_from: int, year_to: int,
                            train_ratio: float, val_ratio: float, seed: int):
    dataset = Path(dataset_dir)
    meta = _load_metadata(metadata_csv, ["fil_revisjon_id", "dokument_aar", "rettsstiftelsestyper"])
    filtered = meta[meta["rettsstiftelsestyper"].str.split(" ", n=1).str[0] == doc_type]
    id_to_year = _year_map(filtered)

    # Keep only IDs within the year range
    valid_ids = {id_ for id_, year in id_to_year.items() if year_from <= year <= year_to}

    imgs = sorted((dataset / "images_all").glob("*.png"))
    selected = [img for img in imgs if img.stem.rsplit("_p", 1)[0] in valid_ids]

    print(f"  Doc type '{doc_type}', years {year_from}-{year_to}: {len(selected)} images from {len(valid_ids)} documents")

    log_header = [
        f"Split log - {datetime.now().isoformat()}",
        f"Strategy: year_and_doc_type | Doc type: {doc_type} | Years: {year_from}-{year_to} | Seed: {seed}",
        f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}",
        f"Total images: {len(selected)}",
        "",
    ]
    _shuffle_and_split(dataset, selected, train_ratio, val_ratio, seed, log_header)


# ─── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["random", "yearly", "doc_type", "year_and_doc_type"], default="random")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--per-year", type=int, default=100)
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    args = parser.parse_args()

    if args.strategy != "random":
        if not args.metadata:
            print(f"ERROR: --metadata is required for {args.strategy} strategy")
            raise SystemExit(1)
        if not Path(args.metadata).exists():
            print(f"ERROR: Metadata file not found: {args.metadata}")
            raise SystemExit(1)

    if args.strategy in ("doc_type", "year_and_doc_type") and not args.doc_type:
        print(f"ERROR: --doc-type is required for {args.strategy} strategy")
        raise SystemExit(1)

    if args.strategy == "year_and_doc_type" and (args.year_from is None or args.year_to is None):
        print("ERROR: --year-from and --year-to are required for year_and_doc_type strategy")
        raise SystemExit(1)

    match args.strategy:
        case "yearly":
            split_yearly(args.dataset, args.metadata, args.per_year,
                         args.train_ratio, args.val_ratio, args.seed)
        case "doc_type":
            split_doc_type(args.dataset, args.metadata, args.doc_type,
                           args.train_ratio, args.val_ratio, args.seed)
        case "year_and_doc_type":
            split_year_and_doc_type(args.dataset, args.metadata, args.doc_type,
                                    args.year_from, args.year_to,
                                    args.train_ratio, args.val_ratio, args.seed)
        case _:
            split_random(args.dataset, args.train_ratio, args.val_ratio, args.seed)
