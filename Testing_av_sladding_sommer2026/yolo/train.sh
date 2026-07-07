#!/bin/bash
set -e

# Usage:
#   bash train.sh <csv_path> <pdfs_path>              # Steps 1-3 (generate + verify)
#   bash train.sh <csv_path> <pdfs_path> --continue   # Steps 1-5 (full pipeline)

CSV_PATH="${1:?Usage: bash train.sh <csv_path> <pdfs_path> [--continue]}"
PDFS_PATH="${2:?Usage: bash train.sh <csv_path> <pdfs_path> [--continue]}"
CONTINUE="$3"

echo "=== Step 1: Convert CSV to YOLO format ==="
python scripts/convert_csv_to_yolo.py --csv "$CSV_PATH" --pdfs "$PDFS_PATH" --output dataset

echo ""
echo "=== Step 2: Verify boxes on sample images ==="
python scripts/verify_boxes.py --images dataset/images_all --labels dataset/labels_all --output verification --max 10

echo ""
echo "=== Step 3: Check label coverage ==="
python scripts/check_coverage.py --images dataset/images_all --labels dataset/labels_all

echo ""
echo ">>> STOP HERE. Open verification/ and check that red boxes land on FNRs."
echo ">>> If boxes are correct, run: bash train.sh $CSV_PATH $PDFS_PATH --continue"
echo ""

if [ "$CONTINUE" = "--continue" ]; then
    echo "=== Step 4: Split train/val/test ==="
    python scripts/split_train_val.py --dataset dataset --train-ratio 0.7 --val-ratio 0.15

    echo ""
    echo "=== Step 5: Smoke test (3 epochs) ==="
    yolo detect train data=data.yaml model=yolo11n.pt epochs=3 imgsz=640 batch=2 device=cuda

    echo ""
    echo "=== Smoke test passed. For full training run: ==="
    echo "yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=1280 batch=4 device=cuda"
fi
