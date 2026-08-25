# --- Paths / run defaults (run.py) ---
DOC_DIR        = "../uttrekk_3"
DEFAULT_FILE_COUNT       = "20"
TRUTH_CSV    = "smartsladding_uttrekk_labels_3_07_07_26.csv"
CSV_OUT       = "sladd_koordinater.csv"
OCR_LOG_FILE = "ocr_linjer.txt"
PNG_DIR    = "visning"
SLADD_DIR  = "sladdet"
Y_ORIGIN     = "top"         

# --- Evaluation ---
# Share of the truth area required for a HIT. Manually validated, see
# STD_THRESHOLD in filter_common.py.
HIT_THRESHOLD      = 0.32

# --- Colors (visualization.py) ---
# RGBA at 0.8 opacity (alpha=204). Outline only, never filled.
MISSED_TRUTH_COLOR              = (220, 30, 30, 204)     # red = truth box that was missed
CORRECT_PADDLE_COLOR   = (30, 180, 30, 204)     # green = correct sladd from Paddle
CORRECT_YOLO_COLOR     = (30, 80, 220, 204)     # blue = correct sladd from YOLO
CORRECT_BOTH_COLOR    = (0, 180, 180, 204)     # teal = correct sladd from both
OVERSLADD_PADDLE_COLOR = (255, 140, 0, 204)     # orange = over-sladding from Paddle
OVERSLADD_YOLO_COLOR   = (255, 180, 0, 204)     # amber = over-sladding from YOLO
OVERSLADD_BOTH_COLOR  = (180, 180, 0, 204)     # dark yellow-green = over-sladding from both
UNKNOWN_COLOR           = (110, 110, 110, 204)   # gray = source unknown (prod CSV)
