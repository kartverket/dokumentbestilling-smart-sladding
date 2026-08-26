# ---- PDF rendering -----------------------------------------
PDF_DPI = 300                  # higher = slower, more accurate

# ---- Dimension filters -----------------------------------------------------
# Universal across kilde; only high YOLO confidence exempts (paddle has none
# and is always filtered). Short/long side, not width/height: width/height
# discarded vertical sladdinger, 30 of 34 wrong removals in uttrekk 4.
# Tuned on 41 reviewed losses in uttrekk 4: 64 oversladdinger removed, 7 fasit
# boxes lost, all confirmed as boxes that should not be sladdet.
MIN_BOX_AREA     = 965       # px²
MIN_ELONGATION     = 1.44      # 1.5 cost 3 real sladdinger at 1.47-1.49
MAX_ELONGATION    = 9         # 3-4x wider than the field itself
MIN_SHORT_SIDE_PT    = 6

# Kilde "yolo" only ("begge", paddle and "yolo_vertikal" are spared). Uttrekk
# 6: ~53 oversladdinger removed, 4 fasit boxes lost, all confirmed reference
# numbers. Short side applies at any conf; long side sits behind the exemption.
MIN_SHORT_SIDE_YOLO_PT = 7
MIN_LONG_SIDE_YOLO_PT = 20      # too short to hold 5 digits

# Kilde "paddle" only, never exempted by conf: OCR conf is read quality, not
# detection certainty. Uttrekk 6: ~20 oversladdinger removed, 0 fasit lost.
# Paddle noise is thin strokes, short side p99.9 = 14.6pt against 8.6pt for
# the smallest real paddle hit.
MIN_SHORT_SIDE_PADDLE_PT = 7
MIN_LONG_SIDE_PADDLE_PT = 20
MAX_ELONGATION_PADDLE = 6

# Skips the geometry filters. "begge" used to be exempt at any confidence;
# removed, see _hopp_over_geometrifilter.
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
YOLO_CONF          = 0.12      # predict threshold
YOLO_CONF_NO_TEXT = 0.40   # required when Paddle read nothing in the box
YOLO_CONF_VERTICAL = 0.90
VERTICAL_FACTOR    = 1.3       # height > 1.3 x width counts as vertical
YOLO_IMGSZ         = 1280
MIN_DIGITS         = 1         # lenient_check floor
MAX_LETTERS     = 1         # 2+ letters is not an fnr, whatever YOLO says

# A decimal separator in confidently read text means coordinate or amount; an
# fnr never has one. High detection conf exempts: real fnr caught by the rule
# sit at conf >= 0.5, coordinates at <= 0.37. Final kilde after dedup, so pure
# "yolo" only. Uttrekk 6: ~1250 oversladdinger removed / 3 real fnr lost.
# Mirrored by _decimal_rule_discards (model_main) and _ocr_reason
# (utils/filter_common.py).
REJECT_DECIMAL_REC_VETO    = 0.98
REJECT_DECIMAL_CONF_EXEMPT = 0.6
# Lower tier: a weaker read suffices when the detection itself is weak.
# Uttrekk 6: +38 oversladdinger removed / 1 lost.
REJECT_DECIMAL_LOW_TIER_REC_VETO  = 0.95
REJECT_DECIMAL_LOW_TIER_CONF_MAX  = 0.4

# A confidently read line can prove the number is no fnr: a digit run of 6-8
# (too long for a personnummer, too short for an fnr, so dagboknummer, amount
# or coordinate) or a valid orgnr mod11. 9-runs are excluded on purpose, they
# are often real fnr with two characters dropped by OCR. Uttrekk 6: 79
# oversladdinger removed / 1 real fnr. Mirrored by _ocr_reason.
LINE_EVIDENCE_REC_VETO  = 0.99
LINE_EVIDENCE_CONF_EXEMPT = 0.5
LINE_EVIDENCE_RUN_MAX    = 8

# Reject paddle boxes whose 11-digit window was stitched across a decimal or a
# large gap outside the legal positions (after digit 2/4/6). Coordinate
# columns are the source; a real fnr has no such gaps. Final kilde "paddle"
# only. Position-blind on "begge" it cost 974 fnr. 8, not 3: handwritten fnr
# reach ~6 digit widths (5.94 measured), coordinate gaps >= 10. Uttrekk 6:
# 72 oversladdinger removed / 0 lost (gap 3 gave 91/1). Mirrored by _ocr_reason.
WINDOW_MAX_GAP          = 8.0   # digit widths
WINDOW_REJECT_DECIMAL_IN_GAP = True

# ── Rule profile per rettsstiftelsestype ──────────────────────────
# The coordinate family is maps, measurement tables and coordinate lists:
# 21 % precision against 88.5 % elsewhere. Inside it _koordfam_discards
# removed 576 oversladdinger / 0 real fnr; globally it costs hundreds of fnr.
# Codes come from the skip job; without them the global behaviour applies.
KOORDFAM_CODES = frozenset((
    "SR_JOU",   # Jordskifte
    "AH_JOU",   # Jordskifte (annen hjemmel)
    "KA_MOB",   # Målebrev
    "KA_GRE",   # Grensejustering
    "TR_MAS",   # Massetransport
    "SR_SKN",   # Skjønn
    "JS_JSA",   # Opprettelse av jordsameie
    "FR_REG",   # Registrering av grunn, marginal: 183 removed / 1 real fnr
    "SR_UTS",   # Utskifting, historical jordskifte
))

# Token-less boxes in koordfam documents are map graphics the text rules never
# see, so they need detection conf instead: 26 oversladdinger removed / 0
# lost. Globally the same rule loses fnr, so it is safe ONLY in koordfam.
KOORDFAM_NO_TEXT_CONF = 0.7

# SE_SEK documents are table-heavy: 72 % precision against 92 % elsewhere, and
# the false positives are table cells: oversized boxes or short digit runs.
# Uttrekk 6: 143 oversladdinger removed / 2 real fnr lost (smin 5 would save
# them but costs ~80 oversladdinger). NB: the koordfam rules are lethal here.
# fnr sit in the same table lines as the decimal areas, 130 lost globally,
# hence separate profiles per document type. Only SE_SEK is measured; the
# reseksjonering relatives (RS_RES, SB_SEB, …) stay out until measured.
SEKSJONERING_CODES = frozenset(("SE_SEK",))
SEKSJONERING_MAX_SHORT_SIDE_PT = 40.0  # yolo+paddle: fnr-sladd p99.9 = 29.5
SEKSJONERING_MAX_LONG_SIDE_PT = 80.0  # yolo+paddle: fnr-sladd p99.9 = 78.9
SEKSJONERING_PADDLE_MIN_ELONG = 3.0   # paddle: square cells rejected
SEKSJONERING_MIN_DIGITS  = 6          # yolo: fewer digits = fraction/snr
SEKSJONERING_REC_VETO    = 0.98       # digit rule only on a certain read
SEKSJONERING_CONF_EXEMPT = 0.5

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
