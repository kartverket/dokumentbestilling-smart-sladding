"""Draw YOLO labels back onto the rendered images, to check before training
that the boxes actually land on the fnr.

Run:
    python train/scripts/verify_boxes.py --images dataset/images_all --labels dataset/labels_all
"""

import argparse
import cv2
from pathlib import Path


def verify(images_dir: str, labels_dir: str, output_dir: str, max_files: int):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(images_dir.glob("*.png"))
    if max_files > 0:
        images = images[:max_files]

    verified = 0
    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        box_count = 0
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            _, xc, yc, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "FNR", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            box_count += 1

        cv2.imwrite(str(output_dir / img_path.name), img)
        verified += 1
        print(f"  {img_path.name}: {box_count} boxes")

    print(f"\nVerified {verified} images -> {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify YOLO labels by drawing boxes on images")
    parser.add_argument("--images", default="dataset/images_all", help="Images directory")
    parser.add_argument("--labels", default="dataset/labels_all", help="Labels directory")
    parser.add_argument("--output", default="verification", help="Output directory for annotated images")
    parser.add_argument("--max", type=int, default=10, help="Max files to verify (0 = all)")
    args = parser.parse_args()
    verify(args.images, args.labels, args.output, args.max)
