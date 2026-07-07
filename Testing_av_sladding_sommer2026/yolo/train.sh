#!/bin/bash
set -e

echo "=== Step 1: Convert CSV to YOLO format ==="
python scripts/convert_csv_to_yolo.py --csv coordinates.csv --pdfs pdfs --output dataset

echo ""
echo "=== Step 2: Verify boxes on sample images ==="
python scripts/verify_boxes.py --images dataset/images_all --labels dataset/labels_all --output verification --max 10

echo ""
echo "=== Step 3: Check label coverage ==="
python scripts/check_coverage.py --images dataset/images_all --labels dataset/labels_all

echo ""
echo ">>> STOP HERE. Open verification/ and check that red boxes land on FNRs."
echo ">>> If boxes are correct, run: bash train.sh --continue"
echo ""

if [ "$1" = "--continue" ]; then
    echo "=== Step 4: Split train/val ==="
    python scripts/split_train_val.py --dataset dataset --ratio 0.8

    echo ""
    echo "=== Step 5: Smoke test (3 epochs) ==="
    yolo detect train data=data.yaml model=yolo11n.pt epochs=3 imgsz=640 batch=2 device=mps

    echo ""
    echo "=== Smoke test passed. For full training run: ==="
    echo "yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=1280 batch=4 device=mps"
fi
