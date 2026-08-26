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
# Share of the truth area a prediction must cover to count as a HIT. Defined
# once here; filter_common, evaluation, draw_from_csv and send_to_prod all read
# it from this line, so run.py and the sweep measure the same thing.
# Set by MANUAL review of borderline crops (filter_review.py --band areal LO HI),
# not by geometry: false hits on the neighbouring line are rare (12 of 20019 on
# uttrekk 4), so discarding real hits costs more. Do not change without a review.
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
# Half opacity so the covered labels sit behind the errors they explain.
COVERED_TRUTH_COLOR    = (190, 60, 230, 128)     # magenta at 0.5 = truth box that WAS covered
