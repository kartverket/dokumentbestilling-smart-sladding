"""Flag rendered training pages that carry suspiciously few labels.

An unlabeled fnr on a training page teaches the model to ignore real ones.

Run:
    python train/scripts/check_coverage.py --images dataset/images_all --labels dataset/labels_all
"""

import argparse
from pathlib import Path


def check(images_dir: str, labels_dir: str, min_boxes: int):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    suspicious = []
    total = 0

    for img_path in sorted(images_dir.glob("*.png")):
        total += 1
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            suspicious.append((img_path.name, 0, "NO LABEL FILE"))
            continue

        content = label_path.read_text().strip()
        box_count = len(content.split("\n")) if content else 0

        if box_count < min_boxes:
            suspicious.append((img_path.name, box_count, "few boxes"))

    print(f"Checked {total} images")
    if suspicious:
        print(f"\nSuspicious pages ({len(suspicious)}):")
        for name, count, reason in suspicious:
            print(f"  {name}: {count} boxes ({reason})")
        print(f"\nReview these pages manually to ensure all FNRs are labeled.")
        print("Unlabeled FNRs on training pages teach the model to ignore real FNRs.")
    else:
        print(f"All pages have at least {min_boxes} labeled box(es).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for incomplete label coverage")
    parser.add_argument("--images", default="dataset/images_all", help="Images directory")
    parser.add_argument("--labels", default="dataset/labels_all", help="Labels directory")
    parser.add_argument("--min-boxes", type=int, default=1, help="Minimum expected boxes per page")
    args = parser.parse_args()
    check(args.images, args.labels, args.min_boxes)
