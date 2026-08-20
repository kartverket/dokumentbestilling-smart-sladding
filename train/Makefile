
# YOLO FNR Detection Pipeline

# ===Variabler===

# ---- Inputs -------------------------------------------------
# folder with source PDFs
PDFS ?= pdfs
# ground-truth annotations
CSV ?= labels.csv
# where generated data lands (base dir + timestamped subfolder)
OUTPUT_DIR ?= .
# Set DATASET explicitly to reuse a previously converted dataset
DATASET ?= $(OUTPUT_DIR)/dataset_$(shell date +%Y-%m-%dT%H-%M-%S)
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO
TRAIN_RATIO ?= 0.7
VAL_RATIO ?= 0.15
SEED ?= 42

# ---- Split strategy -----------------------------------------
# "random" (default), "yearly", "doc_type", or "year_and_doc_type"
STRATEGY ?= random
# metadata CSV with tinglysingsdato (required for non-random strategies)
METADATA ?=
# max images per year for yearly strategy
PER_YEAR ?= 100
# rettsstiftelsestyper value (required for doc_type / year_and_doc_type)
DOC_TYPE ?=
# year range for year_and_doc_type strategy
YEAR_FROM ?=
YEAR_TO ?=

# ---- Training -----------------------------------------------
MODEL ?= yolo26x.pt
EPOCHS ?= 200
IMGSZ ?= 1280
BATCH ?= 4
# Can either be cpu or cuda (if you have a GPU)
DEVICE ?= cuda
PATIENCE ?= 20
# Name for the training run (output in <PROJECT>/<NAME>/) and, unless
# PUBLISER_NAVN says otherwise, the name of the published model.
NAME ?= trening_$(shell date +%Y-%m-%dT%H-%M-%S)
# Project directory for training output. $SLADD_RUNS on the server.
PROJECT ?= $(if $(SLADD_RUNS),$(SLADD_RUNS),runs/detect)
# Set RESUME=1 to resume from last.pt of a previous run
RESUME ?=

# ---- Publisering --------------------------------------------
# Vektlageret. Trening skriver rått til $(PROJECT); «publiser» lager en
# navngitt modell med metadata her, som er det deploy.sh og valideringen
# peker på. Se server.env.
VEKTER ?= $(if $(SLADD_VEKTER),$(SLADD_VEKTER),vekter)
# Navn på den publiserte modellen (default: navnet på treningskjøringen)
PUBLISER_NAVN ?= $(NAME)
# best (default) eller last
PUBLISER_VEKTER ?= best
# Sett OVERSKRIV=1 for å erstatte en modell som alt er publisert
OVERSKRIV ?=

PYTHON ?= python
YOLO := $(shell dirname $(shell which $(PYTHON)))/yolo
SCRIPTS := scripts

.DEFAULT_GOAL := all
.PHONY: all split train publiser convert verify coverage smoke count clean help

# Hele veien: datasett -> trening -> ferdig modell i vektlageret. En
# treningskjøring som ikke er publisert er ikke til å bruke — den heter
# best.pt som alle andre og sier ikke hva den er trent på.
all: split train publiser

# ===Treningssteg===

# ---- Step 1: CSV -> YOLO format -----------------------------
convert: $(DATASET)/.converted

$(DATASET)/.converted:
	@test -f "$(SCRIPTS)/convert_csv_to_yolo.py" || { \
		echo "ERROR: $(SCRIPTS)/convert_csv_to_yolo.py not found."; \
		echo "       Run make from the train/ directory."; \
		exit 1; }
	@test -f "$(CSV)"  || { echo "ERROR: CSV not found: $(CSV)"; exit 1; }
	@test -d "$(PDFS)" || { echo "ERROR: PDF folder not found: $(PDFS)"; exit 1; }
	@if [ -n "$(METADATA)" ] && [ "$(STRATEGY)" != "random" ]; then \
		echo "Generating ID list from metadata filter..."; \
		$(PYTHON) $(SCRIPTS)/count_documents.py \
			--metadata $(METADATA) \
			--strategy $(STRATEGY) \
			$(if $(DOC_TYPE),--doc-type $(DOC_TYPE)) \
			$(if $(YEAR_FROM),--year-from $(YEAR_FROM)) \
			$(if $(YEAR_TO),--year-to $(YEAR_TO)) \
			--output-ids > $(DATASET)/.convert_ids.txt; \
		$(PYTHON) $(SCRIPTS)/convert_csv_to_yolo.py "$(CSV)" "$(PDFS)" --output "$(DATASET)" --ids $(DATASET)/.convert_ids.txt; \
	else \
		$(PYTHON) $(SCRIPTS)/convert_csv_to_yolo.py "$(CSV)" "$(PDFS)" --output "$(DATASET)"; \
	fi
	@touch $@

# ---- Step 4: train/val/test split ---------------------------
split: $(DATASET)/.split

$(DATASET)/.split: $(DATASET)/.converted
	$(PYTHON) $(SCRIPTS)/split_train_val.py \
		--dataset $(DATASET) \
		--train-ratio $(TRAIN_RATIO) \
		--val-ratio $(VAL_RATIO) \
		--seed $(SEED) \
		--strategy $(STRATEGY) \
		$(if $(METADATA),--metadata $(METADATA)) \
		--per-year $(PER_YEAR) \
		$(if $(DOC_TYPE),--doc-type $(DOC_TYPE)) \
		$(if $(YEAR_FROM),--year-from $(YEAR_FROM)) \
		$(if $(YEAR_TO),--year-to $(YEAR_TO))
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
		patience=$(PATIENCE) \
		$(if $(NAME),name=$(NAME)) \
		$(if $(PROJECT),project=$(PROJECT)) \
		$(if $(RESUME),resume=True)

# ---- Step 6: publisering ------------------------------------
publiser:
	$(PYTHON) $(SCRIPTS)/publiser_modell.py \
		--run "$(PROJECT)/$(NAME)" \
		--navn "$(PUBLISER_NAVN)" \
		--ut "$(VEKTER)" \
		--vekter $(PUBLISER_VEKTER) \
		--dataset "$(DATASET)" \
		--info basismodell=$(MODEL) \
		--info strategi=$(STRATEGY) \
		--info pdfs=$(PDFS) \
		--info labels_csv=$(CSV) \
		--info metadata_csv=$(METADATA) \
		--info doc_type=$(DOC_TYPE) \
		--info per_year=$(PER_YEAR) \
		--info year_from=$(YEAR_FROM) \
		--info year_to=$(YEAR_TO) \
		--info train_ratio=$(TRAIN_RATIO) \
		--info val_ratio=$(VAL_RATIO) \
		--info seed=$(SEED) \
		$(if $(OVERSKRIV),--overskriv)

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

# list PDF filenames per year (set LIST_FILES=1 to enable)
LIST_FILES ?=

# ---- Count matching documents (pre-check) --------------------
count:
	@test -f "$(METADATA)" || { echo "ERROR: METADATA not set or not found: $(METADATA)"; exit 1; }
	$(PYTHON) $(SCRIPTS)/count_documents.py \
		--metadata $(METADATA) \
		$(if $(CSV),--labels $(CSV)) \
		--strategy $(STRATEGY) \
		$(if $(DOC_TYPE),--doc-type $(DOC_TYPE)) \
		$(if $(YEAR_FROM),--year-from $(YEAR_FROM)) \
		$(if $(YEAR_TO),--year-to $(YEAR_TO)) \
		$(if $(LIST_FILES),--list-files)


help:
	@echo "make               split + train + publiser"
	@echo "make split         convert CSV/PDFs and split into train/val/test"
	@echo "make train         train YOLO (runs split first if needed)"
	@echo 'make publiser      copy <PROJECT>/<NAME>/weights/best.pt into $$SLADD_VEKTER'
	@echo "                   as <navn>/<navn>.pt + modell.json (training metadata)"
	@echo "make verify        draw labels on images for visual check"
	@echo "make coverage      find pages without labels"
	@echo "make smoke         3-epoch smoke test on $(MODEL)"
	@echo "make count         count matching documents before training"
	@echo "make clean         remove $(DATASET)/"
	@echo ""
	@echo "Vars: PDFS CSV DATASET TRAIN_RATIO VAL_RATIO STRATEGY METADATA PER_YEAR DOC_TYPE YEAR_FROM YEAR_TO MODEL EPOCHS IMGSZ BATCH DEVICE PATIENCE"
	@echo "      NAME PROJECT VEKTER PUBLISER_NAVN PUBLISER_VEKTER OVERSKRIV"