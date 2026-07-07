"""
Split images_all/labels_all into train/val sets (80/20 by default).
"""

import argparse
import random
import shutil
from pathlib import Path


def split(dataset_dir: str, ratio: float, seed: int):
    dataset = Path(dataset_dir)
    images_all = dataset / "images_all"
    labels_all = dataset / "labels_all"

    imgs = sorted(images_all.glob("*.png"))
    random.seed(seed)
    random.shuffle(imgs)

    cut = int(len(imgs) * ratio)
    splits = [("train", imgs[:cut]), ("val", imgs[cut:])]

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

    print(f"\nSplit complete: {len(imgs)} total -> {dataset}/images/{{train,val}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    split(args.dataset, args.ratio, args.seed)
