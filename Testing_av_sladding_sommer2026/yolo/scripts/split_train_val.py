"""
Split images_all/labels_all into train/val/test sets (70/15/15 by default).
"""

import argparse
import random
import shutil
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

    for subset, group in splits:
        img_dst = dataset / "images" / subset
        lbl_dst = dataset / "labels" / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in group:
            lbl = labels_all / (img.stem + ".txt")
            shutil.copy(img, img_dst / img.name)
            if lbl.exists():
                shutil.copy(lbl, lbl_dst / lbl.name)

        print(f"  {subset}: {len(group)} images")

    print(f"\nSplit complete: {n} total -> {dataset}/images/{{train,val,test}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    split(args.dataset, args.train_ratio, args.val_ratio, args.seed)
