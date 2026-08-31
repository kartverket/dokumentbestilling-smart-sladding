# ---- PDF rendering -----------------------------------------
PDF_DPI = 300                  # higher = slower, more accurate

# ---- Dimension filters -----------------------------------------------------
# Universal across kilde; only high YOLO confidence exempts. Short/long side,
# not width/height, so vertical sladdinger survive.
MIN_BOX_AREA     = 965       # px²
MIN_ELONGATION     = 1.44
MAX_ELONGATION    = 9         # 3-4x wider than the field itself
MIN_SHORT_SIDE_PT    = 6

# Kilde "yolo" only ("begge", paddle and "yolo_vertikal" are spared).
# Short side applies at any conf; long side sits behind the exemption.
MIN_SHORT_SIDE_YOLO_PT = 7
MIN_LONG_SIDE_YOLO_PT = 20      # too short to hold 5 digits

# Kilde "paddle" only, never exempted by conf: OCR conf is read quality, not
# detection certainty. Paddle noise is thin strokes.
MIN_SHORT_SIDE_PADDLE_PT = 7
MIN_LONG_SIDE_PADDLE_PT = 20
MAX_ELONGATION_PADDLE = 6

# conf at which a YOLO box skips the geometry filters.
YOLO_CONF_GEOMETRY_THRESHOLD = 0.5

# Only the default for --max-width in utils/draw.py; not in the filter path.
MAX_WIDTH_ELECTRONIC_PT = 50

# ---- YOLO weights ------------------------------------------
# The weights do not live in the repo. In the container ./deploy.sh bakes the
# chosen model in as weights/modell.pt; outside it SLADD_PRODWEIGHTS
# (server.env) points at the weight store. YOLO_WEIGHTS overrides both,
# --yolo-weights overrides everything (utils/run.py).
import os as _os


def default_weights():
    for var in ("YOLO_WEIGHTS", "SLADD_PRODWEIGHTS"):
        path = _os.environ.get(var)
        if path:
            return path
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "weights", "modell.pt")


# ---- YOLO --------------------------------------------------
YOLO_CONF          = 0.05      # predict threshold
YOLO_CONF_NO_TEXT = 0.40   # required when Paddle read nothing in the box
YOLO_CONF_VERTICAL = 0.90
VERTICAL_FACTOR    = 1.3       # height > 1.3 x width counts as vertical
YOLO_IMGSZ         = 1280
MIN_DIGITS         = 1         # lenient_check floor
MAX_LETTERS     = 1         # 2+ letters is not an fnr, whatever YOLO says

# An fnr never contains a decimal separator; confident detections are exempt.
RULE_DECIMAL = {"reject_decimal": 1, "rec_veto": 0.98, "ocr_conf_exempt": 0.6}
# A weaker read suffices when the detection itself is weak.
RULE_DECIMAL_LOW_TIER = {"reject_decimal": 1, "rec_veto": 0.95,
                         "ocr_conf_exempt": 0.4}

# A confidently read digit run of 6-8 or a valid orgnr rules out an fnr.
# Stops at 8: 9-runs are often real fnr with OCR-dropped characters.
RULE_LINE_EVIDENCE = {"reject_run_6_10": 8, "reject_orgnr": 1,
                      "line_veto": 0.99, "ocr_conf_exempt": 0.5}

# An 11-digit window stitched across a decimal or a gap outside the legal
# positions (after digit 2/4/6) is a coordinate column, not an fnr.
# max_gap is in digit widths; handwritten fnr reach ~6.
RULE_WINDOW = {"max_gap": 6.5, "reject_decimal_gap": 1}

# ── Rule profile per rettsstiftelsestype ──────────────────────────
# The coordinate family is maps, measurement tables and coordinate lists.
# Codes come from the skip job; without them the global behaviour applies.
KOORDFAM_CODES = frozenset((
    "SR_JOU",   # Jordskifte
    "AH_JOU",   # Jordskifte (annen hjemmel)
    "KA_MOB",   # Målebrev
    "KA_GRE",   # Grensejustering
    "TR_MAS",   # Massetransport
    "SR_SKN",   # Skjønn
    "JS_JSA",   # Opprettelse av jordsameie
    "FR_REG",   # Registrering av grunn
    "SR_UTS",   # Utskifting, historical jordskifte
))

# A number without fnr evidence in a koordfam document is a coordinate.
# Token-less boxes are map graphics with no text evidence, so they need
# detection conf instead. Loses real fnr outside koordfam.
RULE_KOORDFAM = {"require_fnr_candidate": 1, "reject_decimal": 1,
                 "without_text_conf": 0.7}

# SE_SEK documents are table-heavy and the false positives are table cells.
# The koordfam rules lose real fnr here, hence a separate profile. Only
# SE_SEK is measured; the reseksjonering relatives stay out until measured.
SEKSJONERING_CODES = frozenset(("SE_SEK",))
# Side limits in pt, both kilder. Geometry applies at any conf; the digit
# requirement (fewer digits = fraction or snr) only on a certain read.
RULE_SEKSJONERING_YOLO = {"max_short_side": 40.0, "max_long_side": 80.0,
                          "min_digits": 6, "rec_veto": 0.98,
                          "ocr_conf_exempt": 0.5}
# Same side limits; square table cells rejected on elongation.
RULE_SEKSJONERING_PADDLE = {"min_elongation": 3.0, "max_short_side": 40.0,
                            "max_long_side": 80.0}

# Features box_features computes per YOLO box and writes to the result CSV.
# The list lives here so utils/csv_export.py and utils/filter_common.py get it
# without pulling in PaddleOCR. The last two are paddle/begge only, see
# _window_features in paddle_ocr_model_fnr.py.
FEATURE_FIELDS = ("har_tokens", "n_siffer", "n_bokstaver", "rec_min", "rec_median",
              "rec_min_linje", "n_siffer_linje", "siffer_run",
              "har_fnr_kandidat", "har_desimal_naer",
              "har_00_run", "har_orgnr", "har_org_ord", "lang_run",
              "maks_luke", "har_desimal_luke")

# Must stay below every threshold one might change without invalidating the
# YOLO cache: boxes are stored down to the floor and filtered against
# YOLO_CONF on read. A lower floor feeds more candidates into NMS, but the
# survivors are the same. NMS always suppresses with a higher-scoring box.
YOLO_CACHE_CONF_FLOOR = 0.05

# ---- Paddle OCR --------------------------------------------
PADDLE_MODEL_SET        = "v6"      # "v5" or "v6"
DET_PAGE_LEN       = 2048      # detection: max page side in pixels
REC_BATCH          = 64        # text lines per recognition batch
PAGES_PER_OCR_BATCH = 8        # pages per predict call (GPU utilisation)

# ---- Sladd box ---------------------------------------------
SLADD_DIGITS      = 5
PAD_X_FACTOR             = 0.35      # share of digit width
PAD_Y_FACTOR             = 0.0
MAX_HEIGHT_FACTOR  = 3.0       # max N x median digit width

# ---- Pipeline ----------------------------------------------
DEDUP_OVERLAP     = 0.5       # coverage at which a YOLO box is "the same" as a Paddle box

# ---- Orientation -------------------------------------------
ORIENTATION_DOWNSCALE       = 4
ORIENTATION_MIN_CONFIDENCE      = 0.7       # below this: do not trust the guess, do not rotate

# ---- VLM verifier ------------------------------------------
# A vision model that re-reads each proposed sladdeboks and may drop it. Off
# unless switched on: SLADD_VLM=1 with a URL and a model name in prod,
# --vlm in utils/run.py. See app/vlm_verifier.py for what it may and may not
# do, and docs/VLM-ISOLATION.md for what the model server is allowed to reach.
# The endpoint is OpenAI-compatible /v1, so llama-server, vLLM and LM Studio
# all work.
VLM_ENABLED = _os.environ.get("SLADD_VLM", "").strip().lower() in (
    "1", "true", "yes", "on")
VLM_URL = _os.environ.get("SLADD_VLM_URL", "")        # comma-separated = several backends
VLM_MODEL = _os.environ.get("SLADD_VLM_MODEL", "")
VLM_API_KEY = _os.environ.get("SLADD_VLM_API_KEY") or None
VLM_TIMEOUT = float(_os.environ.get("SLADD_VLM_TIMEOUT", "20"))   # seconds per box
VLM_CONCURRENT = int(_os.environ.get("SLADD_VLM_CONCURRENT", "4"))  # boxes in flight per page
# Empty = no cache, which is what the containers run with: compose sets nothing.
VLM_CACHE = _os.environ.get("SLADD_VLM_CACHE", "").strip()
VLM_MAX_TOKENS = 150

# Circuit breaker. A hung endpoint costs one full VLM_TIMEOUT per box, so a
# document with ten boxes holds a prod request for VLM_TIMEOUT x 10. After
# this many failures in a row the verifier stops calling for the cooldown and
# keeps every box. 0 turns it off.
VLM_BREAKER_FAILURES = int(_os.environ.get("SLADD_VLM_BREAKER_FAILURES", "5"))
VLM_BREAKER_COOLDOWN = float(_os.environ.get("SLADD_VLM_BREAKER_COOLDOWN", "30"))

# The stratum. Measured on uttrekk4: after the fnr guard the gain is 806 of
# 1027 removable boxes on kilde «yolo», 0 of 203 on «begge» and 1 of 129 on
# «paddle» — judging «begge» costs GPU and buys nothing.
VLM_SOURCES = ("yolo", "paddle")

# Crop geometry, read by prod and by utils/vlm_export.py alike. The crop is
# cut in vlm_client.crop_with_marker; change either and prod shows the model
# other images than the runs it was measured on.
VLM_MARGIN_UP_PT = 100.0       # pt, or "full" for the page edge
VLM_MARGIN_DOWN_PT = 60.0
VLM_MARGIN_LEFT_PT = "full"
VLM_MARGIN_RIGHT_PT = "full"
VLM_MAX_PX = 1600              # px crop width cap; lower is faster, coarser
