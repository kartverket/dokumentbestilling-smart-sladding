"""
Split images_all/labels_all into train/val/test sets (70/15/15 by default).
"""

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path


def split(dataset_dir: str, train_ratio: float, val_ratio: float, seed: int):
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
    log_lines.append(f"Seed: {seed}")
    log_lines.append(f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.2f}")
    log_lines.append(f"Total images: {n}")
    log_lines.append("")

    for subset, group in splits:
        img_dst = dataset / "images" / subset
        lbl_dst = dataset / "labels" / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        log_lines.append(f"--- {subset}: {len(group)} images ---")

        doc_ids = sorted(set(img.stem.rsplit("_p", 1)[0] for img in group))
        log_lines.append(f"Documents ({len(doc_ids)}):")
        for doc_id in doc_ids:
            log_lines.append(f" {doc_id}")

        for img in group:
            label = labels_all / (img.stem + ".txt")
            shutil.copy(img, img_dst / img.name)
            has_label = label.exists()
            if has_label:
                shutil.copy(label, lbl_dst / label.name)

        log_lines.append("")
        print(f"  {subset}: {len(group)} images")

    log_path = dataset / "split_log.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"  Log written to {log_path}")
    print(f"\nSplit complete: {n} total -> {dataset}/images/{{train,val,test}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    split(args.dataset, args.train_ratio, args.val_ratio, args.seed)
