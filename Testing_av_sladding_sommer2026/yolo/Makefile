# ============================================================
#  YOLO FNR Detection Pipeline
#
#  Usage:
#    make                 # split + train (default)
#    make split           # CSV + PDFs -> dataset/{train,val,test}
#    make train           # train YOLO on the split dataset
#
#  Eksempel bruk:
#    make PDFS=data/pdfs CSV=data/fnr.csv TRAIN_RATIO=0.8 VAL_RATIO=0.1
# ============================================================

# ---- Inputs -------------------------------------------------
# folder with source PDFs
PDFS ?= pdfs
# ground-truth annotations
CSV ?= labels.csv
# where generated data lands (timestamped by default)
DATASET ?= dataset_$(shell date +%Y-%m-%dT%H-%M-%S)
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO
TRAIN_RATIO ?= 0.8
VAL_RATIO ?= 0.2

# ---- Training -----------------------------------------------
MODEL ?= yolo26x.pt
EPOCHS ?= 100
IMGSZ ?= 1280
BATCH ?= 4
DEVICE ?= cuda
PATIENCE ?= 20

PYTHON ?= python
YOLO := $(shell dirname $(shell which $(PYTHON)))/yolo
SCRIPTS := scripts

.DEFAULT_GOAL := all
.PHONY: all split train convert verify coverage smoke clean help

all: split train

# ---- Step 1: CSV -> YOLO format -----------------------------
convert: $(DATASET)/.converted

$(DATASET)/.converted:
	@if [ -d "$(DATASET)/images_all" ]; then \
		echo "Dataset already converted: $(DATASET)"; \
		mkdir -p $(DATASET) && touch $@; \
	else \
		test -f "$(SCRIPTS)/convert_csv_to_yolo.py" || { \
			echo "ERROR: $(SCRIPTS)/convert_csv_to_yolo.py not found."; \
			echo "       Run make from the yolo/ directory (where data.yaml lives)."; \
			exit 1; }; \
		test -f "$(CSV)"  || { echo "ERROR: CSV not found: $(CSV)"; exit 1; }; \
		test -d "$(PDFS)" || { echo "ERROR: PDF folder not found: $(PDFS)"; exit 1; }; \
		$(PYTHON) $(SCRIPTS)/convert_csv_to_yolo.py "$(CSV)" "$(PDFS)" --output "$(DATASET)"; \
		touch $@; \
	fi

# ---- Step 4: train/val/test split ---------------------------
split: $(DATASET)/.split

$(DATASET)/.split: $(DATASET)/.converted
	$(PYTHON) $(SCRIPTS)/split_train_val.py \
		--dataset $(DATASET) \
		--train-ratio $(TRAIN_RATIO) \
		--val-ratio $(VAL_RATIO)
	@touch $@

# ---- Step 5: training ---------------------------------------
train: split $(DATASET)/data.yaml
	$(YOLO) detect train \
		data=$(DATASET)/data.yaml \
		model=$(MODEL) \
		epochs=$(EPOCHS) \
		imgsz=$(IMGSZ) \
		batch=$(BATCH) \
		device=$(DEVICE) \
		patience=$(PATIENCE)

$(DATASET)/data.yaml: $(DATASET)/.split
	@echo "path: $(abspath $(DATASET))" > $@
	@echo "train: images/train" >> $@
	@echo "val: images/val" >> $@
	@echo "test: images/test" >> $@
	@echo "" >> $@
	@echo "names:" >> $@
	@echo "  0: fnr" >> $@
	@echo "Generated $@"

# ---- Optional checks (steps 2-3) ----------------------------
verify: convert
	$(PYTHON) $(SCRIPTS)/verify_boxes.py \
		--images $(DATASET)/images_all \
		--labels $(DATASET)/labels_all \
		--output verification --max 10

coverage: convert
	$(PYTHON) $(SCRIPTS)/check_coverage.py \
		--images $(DATASET)/images_all \
		--labels $(DATASET)/labels_all

smoke: split $(DATASET)/data.yaml
	$(YOLO) detect train data=$(DATASET)/data.yaml model=$(MODEL) \
		epochs=3 imgsz=640 batch=2 device=$(DEVICE)


help:
	@echo "make               split + train"
	@echo "make split         convert CSV/PDFs and split into train/val/test"
	@echo "make train         train YOLO (runs split first if needed)"
	@echo "make verify        draw labels on images for visual check"
	@echo "make coverage      find pages without labels"
	@echo "make smoke         3-epoch smoke test on $(MODEL)"
	@echo "make clean         remove $(DATASET)/"
	@echo ""
	@echo "Vars: PDFS CSV DATASET TRAIN_RATIO VAL_RATIO MODEL EPOCHS IMGSZ BATCH DEVICE PATIENCE"