import time
from contextlib import contextmanager

import fitz
import numpy as np

from config import (DEDUP_OVERLAP, PDF_DPI, YOLO_CACHE_CONF_FLOOR, YOLO_CONF,
                    YOLO_CONF_NO_TEXT, YOLO_CONF_VERTICAL, YOLO_CONF_GEOMETRY_THRESHOLD,
                    REJECT_DECIMAL_REC_VETO, REJECT_DECIMAL_CONF_EXEMPT,
                    REJECT_DECIMAL_LOW_TIER_REC_VETO, REJECT_DECIMAL_LOW_TIER_CONF_MAX,
                    LINE_EVIDENCE_REC_VETO, LINE_EVIDENCE_CONF_EXEMPT,
                    LINE_EVIDENCE_RUN_MAX,
                    WINDOW_MAX_GAP, WINDOW_REJECT_DECIMAL_IN_GAP,
                    KOORDFAM_CODES, KOORDFAM_NO_TEXT_CONF,
                    SEKSJONERING_CODES,
                    SEKSJONERING_MAX_SHORT_SIDE_PT, SEKSJONERING_MAX_LONG_SIDE_PT,
                    SEKSJONERING_PADDLE_MIN_ELONG, SEKSJONERING_MIN_DIGITS,
                    SEKSJONERING_REC_VETO, SEKSJONERING_CONF_EXEMPT)
from load_pdf import read_pages_from_bytes
from paddle_ocr_model_fnr import (read_tokens_batched, sladd_boxes_from_tokens,
                                  lines_with_fnr_marks, build_lines)
from orientation import find_rotations_batch, unrotate_box
from yolo_fnr import (find_yolo_boxes, lenient_check, tokens_in_box, overlap_share_box,
                      is_vertical, is_too_small, has_wrong_ratio, is_too_thin,
                      is_too_narrow_yolo, is_too_short_yolo, has_paddle_noise_shape)
from box_features import features_for_box
from ocr_cache import read_cache as read_ocr_cache, write_cache as write_ocr_cache
from yolo_cache import read_cache as read_yolo_cache, write_cache as write_yolo_cache


@contextmanager
def _take_time(t, post):
    start = time.perf_counter()
    yield
    t[post] = t.get(post, 0.0) + (time.perf_counter() - start)


def _skip_over_geometry_filter(conf, source):
    """High confidence -> trust the model, skip the geometry filters.

    "begge" used to be exempt at any confidence, but low-confidence "begge"
    caused real oversladding (4 of 7 losses at conf 0.17-0.31). Paddle is
    never exempt: OCR confidence is read quality, not detection certainty.
    """
    if source == "paddle":
        return False
    return conf is not None and conf >= YOLO_CONF_GEOMETRY_THRESHOLD


def _decimal_rule_discards(features, conf):
    """Decimal separator in confidently read text -> coordinate, not fnr.

    Mirrors _ocr_grunn in utils/filter_common.py (des=1, rveto, cfritak).
    Called on the FINAL kilde after dedup, so a box that became "begge" is
    spared. See config.
    """
    if not features or not features.get("har_tokens"):
        return False
    rec = features.get("rec_min")
    if rec is None or not features.get("har_desimal_naer"):
        return False
    if rec >= REJECT_DECIMAL_REC_VETO \
            and (conf is None or conf < REJECT_DECIMAL_CONF_EXEMPT):
        return True
    # Low tier: a weaker read suffices when the detection itself is weak.
    return (rec >= REJECT_DECIMAL_LOW_TIER_REC_VETO
            and (conf is None or conf < REJECT_DECIMAL_LOW_TIER_CONF_MAX))


def _line_evidence_discards(features, conf):
    """A confidently read line proves the number cannot be an fnr.

    Mirrors _ocr_grunn in utils/filter_common.py (avvis_run_6_10, avvis_orgnr,
    linje_veto, ocr_conf_fritak). Final kilde after dedup. See config.
    """
    if not features or not features.get("har_tokens"):
        return False
    if conf is not None and conf >= LINE_EVIDENCE_CONF_EXEMPT:
        return False
    line = features.get("rec_min_linje")
    if line is None or line < LINE_EVIDENCE_REC_VETO:
        return False
    long = features.get("lang_run")
    if long is not None and 6 <= long <= LINE_EVIDENCE_RUN_MAX:
        return True
    return bool(features.get("har_orgnr"))


_SEKSJONERING_MAX_SHORT_SIDE_PX = SEKSJONERING_MAX_SHORT_SIDE_PT * PDF_DPI / 72.0
_SEKSJONERING_MAX_LONG_SIDE_PX = SEKSJONERING_MAX_LONG_SIDE_PT * PDF_DPI / 72.0


def _seksjonering_geometry_discards(box):
    """Orientation-free shape: too large to be a 5-digit sladd."""
    x0, y0, x1, y1 = box[:4]
    short, long = sorted((x1 - x0, y1 - y0))
    return (short > _SEKSJONERING_MAX_SHORT_SIDE_PX
            or long > _SEKSJONERING_MAX_LONG_SIDE_PX)


def _seksjonering_yolo_discards(box, features, conf):
    """Seksjonering document: table cell, not an fnr sladd.

    Mirrors er_filtrert in utils/filter_common.py, per-kilde spec
    "yolo:kmaks=40,lmaks=80,smin=6,rveto=0.98,cfritak=0.5". Geometry always
    applies; the digit requirement only on confidently read text.
    """
    if _seksjonering_geometry_discards(box):
        return True
    if not features or not features.get("har_tokens"):
        return False
    if conf is not None and conf >= SEKSJONERING_CONF_EXEMPT:
        return False
    rec = features.get("rec_min")
    if rec is None or rec < SEKSJONERING_REC_VETO:
        return False
    return (features.get("n_siffer") or 0) < SEKSJONERING_MIN_DIGITS


def _seksjonering_paddle_discards(box):
    """Mirrors "paddle:e=3,kmaks=40,lmaks=80": square or oversized cells."""
    x0, y0, x1, y1 = box[:4]
    short, long = sorted((x1 - x0, y1 - y0))
    if short <= 0:
        return True
    if long / short < SEKSJONERING_PADDLE_MIN_ELONG:
        return True
    return _seksjonering_geometry_discards(box)


def _koordfam_discards(features, conf):
    """Coordinate document: a number without fnr evidence is a coordinate.

    Mirrors _ocr_grunn in utils/filter_common.py (krev_fnr_kandidat,
    avvis_desimal, uten_tekst_conf). Only when the document's
    rettsstiftelsestyper hit KOORDFAM_CODES. Globally the same rules cost
    hundreds of real fnr. Token-less boxes are map graphics with no text
    evidence, so they need detection conf instead.
    """
    if not features:
        return False
    har_tokens = features.get("har_tokens")
    if har_tokens is None:
        return False
    if not har_tokens:
        return conf is None or conf < KOORDFAM_NO_TEXT_CONF
    if not features.get("har_fnr_kandidat"):
        return True
    return bool(features.get("har_desimal_naer"))


def _paddle_window_discards(features):
    """The 11-digit window the box was built from is a seam, not an fnr.

    Mirrors _ocr_grunn in utils/filter_common.py (avvis_desimal_luke,
    maks_luke). Final kilde after dedup: boxes that became "begge" are
    YOLO-confirmed and spared. The features are already position-aware from
    _window_features. Gaps after digit 2/4/6 do not count. See config.
    """
    if not features:
        return False
    if WINDOW_REJECT_DECIMAL_IN_GAP and features.get("har_desimal_luke"):
        return True
    gap = features.get("maks_luke")
    return gap is not None and gap >= WINDOW_MAX_GAP


def _find_boxes_only_yolo(yolo_boxes):
    boxes = []
    for (x0, y0, x1, y1, conf) in yolo_boxes:
        yb = (x0, y0, x1, y1)
        if not is_too_small(yb) and not is_too_narrow_yolo(yb) and (
                _skip_over_geometry_filter(round(conf, 3), "yolo")
                or (not has_wrong_ratio(yb) and not is_too_thin(yb)
                    and not is_too_short_yolo(yb))):
            boxes.append([(x0, y0, x1, y1), "yolo", round(conf, 3), None, None])
    return [tuple(pair) for pair in boxes]


def _find_boxes_with_source(tokens, yolo_boxes, koordfam=False,
                           seksjonering=False, postfilter=True):
    """Merge Paddle and YOLO boxes.

    An empty `yolo_boxes` gives pure Paddle detection. That is how
    elektronisk tinglyste documents are handled.

    Internal per-box layout: [box, kilde, yolo_conf, paddle_rec_score, trekk]
    """
    # Line grouping is the expensive part: done once per page and shared by
    # the fnr search and the per-box feature computation.
    lines = build_lines(tokens) if tokens else []

    boxes = [[box, "paddle", None, rec_score, window_features]
              for (box, _mod11, rec_score, window_features)
              in sladd_boxes_from_tokens(tokens, lines)]

    for (x0, y0, x1, y1, conf) in yolo_boxes:
        yb = (x0, y0, x1, y1)
        covered = [pair for pair in boxes if overlap_share_box(yb, pair[0]) > DEDUP_OVERLAP]
        # Only Paddle hits may be promoted to "begge". Renaming earlier YOLO
        # boxes too contaminated the "begge" bucket with pure YOLO hits and
        # hid them from the OCR rules, which only apply to kilde "yolo".
        paddle_covered = [pair for pair in covered if pair[1] in ("paddle", "begge")]
        if paddle_covered:
            for pair in paddle_covered:
                pair[1] = "begge"
                pair[2] = round(conf, 3)           # yolo_conf
        elif covered:
            # Overlaps only an earlier YOLO box: duplicate, so keep the first.
            pass
        elif source := _accept_yolo_box(tokens, yb, conf):
            # Features describe what lenient_check had to go on; they go to
            # the result CSV so stricter variants can be swept without a
            # rerun. Only "yolo" gets them: "yolo_vertikal" reads no tokens,
            # and paddle/begge carry the window features instead.
            features = features_for_box(tokens, lines, yb) if source == "yolo" else None
            boxes.append([yb, source, round(conf, 3), None, features])

    # ── Decimal and line-evidence rules ────────────────────────────────────
    # Deliberately after the dedup loop: the kilde is final, so boxes that
    # became "begge" keep their sladding.
    # postfilter=False skips BOTH blocks below (rules + geometry) to measure
    # raw detection+mod11+dedup. YOLO_CONF and lenient_check still apply,
    # they are the model's operating point, not postfilters.
    if postfilter:
        boxes = [pair for pair in boxes
                  if not (pair[1] == "yolo"
                          and (_decimal_rule_discards(pair[4], pair[2])
                               or _line_evidence_discards(pair[4], pair[2])
                               or (koordfam
                                   and _koordfam_discards(pair[4], pair[2]))
                               or (seksjonering
                                   and _seksjonering_yolo_discards(
                                       pair[0], pair[4], pair[2]))))
                  and not (pair[1] == "paddle"
                           and (_paddle_window_discards(pair[4])
                                or (seksjonering
                                    and _seksjonering_paddle_discards(
                                        pair[0]))))]

        # ── Dimension filters ──────────────────────────────────
        # Universal limits; only high yolo confidence exempts. On top: the
        # short-side limit for "yolo" and the full noise shape for "paddle"
        # apply at any confidence; the long-side limit for "yolo" sits behind
        # the conf gate.
        boxes = [pair for pair in boxes
                  if not is_too_small(pair[0])
                  and not (pair[1] == "yolo" and is_too_narrow_yolo(pair[0]))
                  and not (pair[1] == "paddle" and has_paddle_noise_shape(pair[0]))
                  and (
                      _skip_over_geometry_filter(pair[2], pair[1])
                      or (not has_wrong_ratio(pair[0]) and not is_too_thin(pair[0])
                          and not (pair[1] == "yolo"
                                   and is_too_short_yolo(pair[0])))
                  )]

    return [tuple(pair) for pair in boxes]


def _accept_yolo_box(tokens, box, conf):
    if is_vertical(box):
        return "yolo_vertikal" if conf >= YOLO_CONF_VERTICAL else None
    if tokens_in_box(tokens, box):
        return "yolo" if lenient_check(tokens, box) else None
    return "yolo" if conf >= YOLO_CONF_NO_TEXT else None


def _build_page(si, size, tokens, boxes_with_source, k, with_lines):
    w, h = size
    boxes = []
    for box, source, yolo_conf, paddle_rec_score, features in boxes_with_source:
        x0, y0, x1, y1 = unrotate_box(box, k, w, h)
        b = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "kilde": source}
        if yolo_conf is not None:
            b["yolo_conf"] = yolo_conf
        if paddle_rec_score is not None:
            b["paddle_rec_score"] = paddle_rec_score
        if features is not None:
            b["trekk"] = features
        boxes.append(b)
    page = {"side": si, "bilde_bredde": w, "bilde_hoyde": h, "boxes": boxes}
    if with_lines:
        page["linjer"] = lines_with_fnr_marks(tokens)
    return page


def _page_geometry(pdf_bytes):
    """Page sizes without rasterising: (pixels at PDF_DPI, points).

    The pixel sizes must match get_pixmap(dpi=PDF_DPI), since they are used as
    the image size when rendering is skipped: hence irect of page * zoom.
    """
    zoom = PDF_DPI / 72.0
    m = fitz.Matrix(zoom, zoom)
    pixels, point = [], []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            r = page.rect
            ir = (r * m).irect
            pixels.append((ir.width, ir.height))
            point.append((r.width, r.height))
    return pixels, point


def _to_flat(pages, page_field):
    ut = []
    for page in pages:
        n = page["side"]
        bb, bh = page.get("bilde_bredde"), page.get("bilde_hoyde")
        if not bb or not bh or not (1 <= n <= len(page_field)):
            continue
        pt_width, pt_height = page_field[n - 1]
        sx = pt_width / bb
        sy = pt_height / bh
        for b in page.get("boxes", []):
            x0, x1 = sorted((b["x0"] * sx, b["x1"] * sx))
            y0, y1 = sorted((b["y0"] * sy, b["y1"] * sy))
            d = {"page": n, "x": x0, "y": y0,
                 "width": x1 - x0, "height": y1 - y0,
                 "kilde": b.get("kilde")}
            if "yolo_conf" in b:
                d["yolo_conf"] = b["yolo_conf"]
            if "paddle_rec_score" in b:
                d["paddle_rec_score"] = b["paddle_rec_score"]
            if "trekk" in b:
                d["trekk"] = b["trekk"]
            ut.append(d)
    return ut


def run_model_on_pdf_bytes(pdf_bytes, write_time=False, with_lines=False, name=None,
                           elektronisk_tinglyst=False, only_yolo=False,
                           cache_dir=None, yolo_cache_dir=None,
                           only_cache=False, rettsstiftelsestyper=None,
                           postfilter=True):
    """only_cache=True: return None instead of running the models on a cache
    miss. Lets GPU-less worker processes handle cache hits and send the misses
    back to a process that has the models.

    rettsstiftelsestyper: the document's XX_YYY codes from the grunnbok.
    Enables per-document-type rule profiles (KOORDFAM_CODES in config).
    None/empty = global behaviour, so missing metadata can never cost recall.
    """
    t = {}
    codes = {k.strip().upper() for k in (rettsstiftelsestyper or ()) if k}
    koordfam = bool(KOORDFAM_CODES & codes)
    seksjonering = bool(SEKSJONERING_CODES & codes)
    uses_yolo = only_yolo or not elektronisk_tinglyst
    yolo_cache = bool(yolo_cache_dir and name and uses_yolo)

    with _take_time(t, "render"):
        page_target, page_field = _page_geometry(pdf_bytes)
    n_pages = len(page_target)

    # The rotations are worth fetching even with --only-yolo: YOLO runs on the
    # orientation-corrected image, so without them --only-yolo would re-render
    # and re-orient every document even with a full YOLO cache.
    rotations, tokens_per_page = None, None
    ocr_hit = root_hit = False
    if cache_dir and name:
        cached = read_ocr_cache(cache_dir, name)
        if cached is not None and len(cached[0]) == n_pages:
            rotations = cached[0]
            root_hit = True
            if not only_yolo:
                tokens_per_page = cached[1]
                ocr_hit = True

    # The lookup needs the rotations. Having them before render lets us skip
    # rasterising entirely; otherwise orientation must run first.
    yolo_boxes_per_page = None
    if yolo_cache and root_hit:
        yolo_boxes_per_page = read_yolo_cache(yolo_cache_dir, name, rotations)

    # Hits in both caches mean nothing reads the images: the page sizes above
    # cover what _build_page needs. With --only-yolo, rotations + YOLO is
    # enough.
    missing_ocr = not ocr_hit and not only_yolo
    needs_pixels = (missing_ocr or not root_hit
                       or (uses_yolo and yolo_boxes_per_page is None))
    images = images_ocr = None

    if needs_pixels:
        if only_cache:
            return None
        with _take_time(t, "render"):
            images = list(read_pages_from_bytes(pdf_bytes))

        if not root_hit:
            with _take_time(t, "orientation"):
                # One model call for the whole document; per page the GPU was
                # spun up and down once per page.
                rotations = find_rotations_batch(images)

        images_ocr = [np.rot90(b, k) if k else b for b, k in zip(images, rotations)]

        if not ocr_hit and not only_yolo:
            with _take_time(t, "ocr"):
                tokens_per_page = read_tokens_batched(images_ocr)
            # Cache for future runs
            if cache_dir and name:
                write_ocr_cache(cache_dir, name, rotations, tokens_per_page)

    if tokens_per_page is None:
        tokens_per_page = [[] for _ in range(n_pages)]

    # On a miss the rotations only became known now, after orientation
    if yolo_cache and not root_hit:
        yolo_boxes_per_page = read_yolo_cache(yolo_cache_dir, name, rotations)

    yolo_hit = yolo_boxes_per_page is not None
    # Raw boxes to cache afterwards (only when we actually ran the model)
    new_yolo_boxes = [] if (yolo_cache and not yolo_hit) else None

    pages = []
    for si in range(n_pages):
        k = rotations[si]
        tokens = tokens_per_page[si]

        with _take_time(t, "yolo+match"):
            if not uses_yolo:
                yolo_boxes = []
            elif yolo_hit:
                yolo_boxes = yolo_boxes_per_page[si]
            else:
                image_yolo = images_ocr[si]
                # With a cache we predict down to the floor and filter here, so
                # the cache survives later YOLO_CONF changes. The survivors are
                # the same boxes a predict at YOLO_CONF would give.
                raw = find_yolo_boxes(
                    image_yolo, conf=YOLO_CACHE_CONF_FLOOR if yolo_cache else None)
                if new_yolo_boxes is not None:
                    new_yolo_boxes.append(raw)
                yolo_boxes = [b for b in raw if b[4] >= YOLO_CONF] if yolo_cache else raw

            if only_yolo:
                boxes_with_source = _find_boxes_only_yolo(yolo_boxes)
            else:
                boxes_with_source = _find_boxes_with_source(
                    tokens, yolo_boxes, koordfam=koordfam,
                    seksjonering=seksjonering, postfilter=postfilter)

        with _take_time(t, "postprocessing"):
            pages.append(_build_page(si + 1, page_target[si], tokens, boxes_with_source,
                                    k, with_lines))

    if new_yolo_boxes is not None and len(new_yolo_boxes) == n_pages:
        write_yolo_cache(yolo_cache_dir, name, rotations, new_yolo_boxes)

    if write_time:
        _write_time(t, len(pages), name, ocr_hit, yolo_hit)

    return _to_flat(pages, page_field)


def _write_time(t, n_pages, name=None, ocr_hit=False, yolo_hit=False):
    entries = ["render", "orientation", "ocr", "yolo+match", "postprocessing"]
    total = sum(t.get(p, 0.0) for p in entries)

    label = f"Timing [{name}]:" if name else "Timing:"
    from_cache = [n for n, hit in (("OCR", ocr_hit), ("YOLO", yolo_hit)) if hit]
    if from_cache:
        label += f" ({' + '.join(from_cache)} from cache)"
    print(label)
    for post in entries:
        sec = t.get(post, 0.0)
        pct = (sec / total * 100) if total else 0.0
        print(f"  {post:<18}{sec:9.3f} s{pct:7.1f}%")
    print(f"  {'Total':<18}{total:9.3f} s")
    print(f"  {'Pages total':<18}{n_pages:9d}")
    if n_pages:
        print(f"  {'Per page':<18}{total / n_pages:9.3f} s")
