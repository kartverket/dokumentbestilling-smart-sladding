"""Split images_all/labels_all into train/val/test sets.

Whole documents go to one subset: pages of the same document share handwriting,
stamps and layout, so splitting per page leaks train into val/test and inflates
both early stopping and selection. Ratios are filled by image count, negatives
are whole zero-fnr documents.

Strategies: random, yearly (group by dokument_aar), doc_type (filter by
rettsstiftelsestyper) and year_and_doc_type (both).

Run:
    python train/scripts/split_train_val.py --dataset dataset --strategy yearly \
        --metadata meta.csv --per-year 100
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


def _group_by_doc(imgs: list) -> dict:
    """{document id -> [page images]} from image stems like <id>_p<N>."""
    docs = defaultdict(list)
    for img in imgs:
        docs[img.stem.rsplit("_p", 1)[0]].append(img)
    return dict(docs)


def _separate_positives_negatives(imgs: list, labels_all: Path) -> tuple[list, list]:
    """Doc granularity: (doc, [imgs]) lists; a doc is positive if any page has boxes."""
    positives = []
    negatives = []
    for doc, group in sorted(_group_by_doc(imgs).items()):
        if all(_is_negative(img, labels_all) for img in group):
            negatives.append((doc, group))
        else:
            positives.append((doc, group))
    return positives, negatives


def _n_imgs(doc_groups: list) -> int:
    return sum(len(g) for _, g in doc_groups)


def _add_negatives(splits: list, negatives: list, ratio: float = NEGATIVE_RATIO):
    """Adds whole negative documents until each subset holds about `ratio`
    negative images per positive image."""
    random.shuffle(negatives)
    it = iter(negatives)
    for i, (subset, positives) in enumerate(splits):
        target = int(_n_imgs(positives) * ratio)
        selected, n_neg = [], 0
        while n_neg < target:
            try:
                doc, group = next(it)
            except StopIteration:
                break
            selected.append((doc, group))
            n_neg += len(group)
        splits[i] = (subset, positives + selected)
        if selected:
            print(f"  {subset}: +{n_neg} negative images from {len(selected)} documents")
    return splits


def _do_split(doc_groups: list, train_ratio: float, val_ratio: float):
    """Allocate already-shuffled documents to train/val/test by image quota."""
    total = _n_imgs(doc_groups)
    bounds = [("train", total * train_ratio),
              ("val", total * (train_ratio + val_ratio)),
              ("test", float("inf"))]
    out = {"train": [], "val": [], "test": []}
    filled = 0
    for doc, group in doc_groups:
        subset = next(name for name, limit in bounds if filled < limit)
        out[subset].append((doc, group))
        filled += len(group)
    return [(name, out[name]) for name in ("train", "val", "test")]


def _copy_and_log(dataset: Path, splits: list, log_lines: list):
    labels_all = dataset / "labels_all"
    for subset, doc_groups in splits:
        img_dest = dataset / "images" / subset
        label_dest = dataset / "labels" / subset
        img_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)

        group = [img for _, imgs in doc_groups for img in imgs]
        log_lines.append(f"--- {subset}: {len(group)} images ---")
        doc_ids = sorted(doc for doc, _ in doc_groups)
        log_lines.append(f"Documents ({len(doc_ids)}):")
        for doc_id in doc_ids:
            log_lines.append(f" {doc_id}")

        for img in group:
            label = labels_all / (img.stem + ".txt")
            shutil.copy(img, img_dest / img.name)
            if label.exists():
                shutil.copy(label, label_dest / label.name)

        log_lines.append("")
        print(f"  {subset}: {len(group)} images in {len(doc_ids)} documents")

    total = sum(_n_imgs(g) for _, g in splits)
    log_path = dataset / "split_log.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"  Log written to {log_path}")
    print(f"\nSplit complete: {total} total -> {dataset}/images/{{train,val,test}}")


def _shuffle_and_split(dataset: Path, imgs: list, train_ratio: float, val_ratio: float,
                       seed: int, log_header: list):
    labels_all = dataset / "labels_all"
    positives, negatives = _separate_positives_negatives(imgs, labels_all)

    random.seed(seed)
    random.shuffle(positives)
    splits = _do_split(positives, train_ratio, val_ratio)
    splits = _add_negatives(splits, negatives)

    log_header.append(f"Negatives: {sum(_n_imgs(s) for _, s in splits) - _n_imgs(positives)} images"
                      f" of {_n_imgs(negatives)} available")
    log_header.append("")
    _copy_and_log(dataset, splits, log_header)


def _split_per_group(dataset: Path, groups: dict, per_group: int,
                     train_ratio: float, val_ratio: float, seed: int, log_header: list):
    """Splits each group on its own, capped at per_group images, then merges the subsets."""
    labels_all = dataset / "labels_all"
    random.seed(seed)
    train, val, test = [], [], []
    all_negatives = []

    for key in sorted(groups):
        positives, negatives = _separate_positives_negatives(groups[key], labels_all)
        random.shuffle(positives)
        selected = []
        for doc, group in positives:
            if _n_imgs(selected) >= per_group:
                break
            selected.append((doc, group))
        s = _do_split(selected, train_ratio, val_ratio)
        train.extend(s[0][1])
        val.extend(s[1][1])
        test.extend(s[2][1])
        all_negatives.extend(negatives)
        print(f"  {key}: {_n_imgs(positives)} positive images available, "
              f"{_n_imgs(selected)} selected, {_n_imgs(negatives)} negative")
        log_header.append(f"  {key}: {_n_imgs(selected)} used of {_n_imgs(positives)} available")
    log_header.append("")

    total_positives = _n_imgs(train) + _n_imgs(val) + _n_imgs(test)
    splits = [("train", train), ("val", val), ("test", test)]
    splits = _add_negatives(splits, all_negatives)
    total_negatives = sum(_n_imgs(s) for _, s in splits) - total_positives

    log_header.append(f"Negatives: {total_negatives} images added from "
                      f"{_n_imgs(all_negatives)} available")
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
    """Returns (groups, images whose document id was not in id_to_key)."""
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
    """{fil_revisjon_id (str) -> year (int)}, skipping rows without dokument_aar."""
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


def _has_doc_type(rettsstiftelsestyper: str, doc_type: str) -> bool:
    """rettsstiftelsestyper is a comma-separated list; match the code before the first space."""
    return any(
        part.strip().split(" ", 1)[0] == doc_type
        for part in str(rettsstiftelsestyper).split(",")
    )


def split_doc_type(dataset_dir: str, metadata_csv: str, doc_type: str,
                   train_ratio: float, val_ratio: float, seed: int):
    dataset = Path(dataset_dir)
    meta = _load_metadata(metadata_csv, ["fil_revisjon_id", "rettsstiftelsestyper"])
    mask = meta["rettsstiftelsestyper"].apply(lambda s: _has_doc_type(s, doc_type))
    valid_ids = set(meta.loc[mask, "fil_revisjon_id"].astype(str))

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
    mask = meta["rettsstiftelsestyper"].apply(lambda s: _has_doc_type(s, doc_type))
    filtered = meta[mask]
    id_to_year = _year_map(filtered)

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

    if args.strategy == "yearly":
        split_yearly(args.dataset, args.metadata, args.per_year,
                     args.train_ratio, args.val_ratio, args.seed)
    elif args.strategy == "doc_type":
        split_doc_type(args.dataset, args.metadata, args.doc_type,
                       args.train_ratio, args.val_ratio, args.seed)
    elif args.strategy == "year_and_doc_type":
        split_year_and_doc_type(args.dataset, args.metadata, args.doc_type,
                                args.year_from, args.year_to,
                                args.train_ratio, args.val_ratio, args.seed)
    else:
        split_random(args.dataset, args.train_ratio, args.val_ratio, args.seed)
