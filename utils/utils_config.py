# --- Stier / kjøre-standarder (run.py) ---
MAPPE        = "../uttrekk_3"
ANTALL       = "20"
FASIT_CSV    = "smartsladding_uttrekk_labels_3_07_07_26.csv"
CSV_UT       = "sladd_koordinater.csv"
OCR_LOGG_FIL = "ocr_linjer.txt"
PNG_MAPPE    = "visning"
SLADD_MAPPE  = "sladdet"
Y_ORIGIN     = "topp"         

# --- Evaluering ---
TERSKEL      = 0.15            # andel fasit-areal som kreves for TRUFFET

# --- Farger (visualization.py) ---
FUNNET_FARGE    = (0, 0, 0)        # svart fyll = faktisk sladding
PADDLE_FARGE    = (30, 80, 200)    # blå outline = Paddle-treff
YOLO_FARGE      = (200, 0, 0)      # rød outline = YOLO-treff
OVERSLADD_FARGE = (255, 140, 0)    # oransje outline = over-sladding
FASIT_FARGE     = (30, 160, 30)    # grønt = ground_truth
