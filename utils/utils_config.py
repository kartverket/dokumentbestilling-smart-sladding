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
TERSKEL      = 0.32            # andel fasit-areal som kreves for TRUFFET
                               # (manuelt validert — se STD_TERSKEL i filter_felles.py)

# --- Farger (visualization.py) ---
# Alle farger RGBA med 0.8 opacity (alpha=204). Aldri fyll, kun outline.
BOM_FARGE              = (220, 30, 30, 204)     # rød = fasit-boks som ble bommet (MANGLER)
KORREKT_PADDLE_FARGE   = (30, 180, 30, 204)     # grønn = korrekt sladding fra Paddle
KORREKT_YOLO_FARGE     = (30, 80, 220, 204)     # blå = korrekt sladding fra YOLO
KORREKT_BEGGE_FARGE    = (0, 180, 180, 204)     # teal = korrekt sladding fra begge
OVERSLADD_PADDLE_FARGE = (255, 140, 0, 204)     # oransje = over-sladding fra Paddle
OVERSLADD_YOLO_FARGE   = (255, 180, 0, 204)     # mørk gul/amber = over-sladding fra YOLO
OVERSLADD_BEGGE_FARGE  = (180, 180, 0, 204)     # mørk gul-grønn = over-sladding fra begge
UKJENT_FARGE           = (110, 110, 110, 204)   # grå = kilde ukjent (prod-CSV)
