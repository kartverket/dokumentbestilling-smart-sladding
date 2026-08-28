import time
from contextlib import contextmanager

import fitz
import numpy as np

from config import (DEDUP_OVERLAP, PDF_DPI, YOLO_CACHE_CONF_FLOOR, YOLO_CONF,
                    YOLO_CONF_NO_TEXT, YOLO_CONF_VERTICAL, YOLO_CONF_GEOMETRY_THRESHOLD,
                    KOORDFAM_CODES, SEKSJONERING_CODES,
                    RULE_DECIMAL, RULE_DECIMAL_LOW_TIER, RULE_LINE_EVIDENCE,
                    RULE_WINDOW, RULE_KOORDFAM,
                    RULE_SEKSJONERING_YOLO, RULE_SEKSJONERING_PADDLE)
from filter_rules import make_filter, rule_input
from geometry import smallest_share
from load_pdf import read_pages_from_bytes
from paddle_ocr_model_fnr import (read_tokens_batched, sladd_boxes_from_tokens,
                                  lines_with_fnr_marks, build_lines)
from orientation import find_rotations_batch, unrotate_box
from yolo_fnr import (find_yolo_boxes, lenient_check, tokens_in_box,
                      is_vertical, is_too_small, has_wrong_ratio, is_too_thin,
                      is_too_narrow_yolo, is_too_short_yolo, has_paddle_noise_shape)
from box_features import features_for_box
from ocr_cache import read_cache as read_ocr_cache, write_cache as write_ocr_cache
from yolo_cache import read_cache as read_yolo_cache, write_cache as write_yolo_cache
import vlm_verifier


@contextmanager
def _take_time(t, post):
    start = time.perf_counter()
    yield
    t[post] = t.get(post, 0.0) + (time.perf_counter() - start)


def _skip_over_geometry_filter(conf, source):
    """High confidence -> trust the model, skip the geometry filters.

    Paddle is never exempt: OCR confidence is read quality, not detection
    certainty.
    """
    if source == "paddle":
        return False
    return conf is not None and conf >= YOLO_CONF_GEOMETRY_THRESHOLD


# ── OCR and profile rules ────────────────────────────────────
# Shared with the sweep tools: predicates in filter_rules.py, operating
# point in config.py.

_decimal_discards = make_filter(**RULE_DECIMAL)
_decimal_low_tier_discards = make_filter(**RULE_DECIMAL_LOW_TIER)
_line_evidence_discards = make_filter(**RULE_LINE_EVIDENCE)
_window_discards = make_filter(**RULE_WINDOW)
_koordfam_discards = make_filter(**RULE_KOORDFAM)
_seksjonering_yolo_discards = make_filter(**RULE_SEKSJONERING_YOLO)
_seksjonering_paddle_discards = make_filter(**RULE_SEKSJONERING_PADDLE)


def _rules_discard(pair, koordfam, seksjonering):
    """Whether the OCR and profile rules drop the box.

    Runs on the FINAL kilde after dedup: "begge" and "yolo_vertikal" are
    spared, and the koordfam/seksjonering profiles apply only when the
    document's rettsstiftelsestyper say so. Paddle boxes carry the window
    features from _window_features instead of the box features.
    """
    box, kilde, conf, _rec, features = pair
    if kilde == "yolo":
        p = rule_input(box, features, conf)
        return (_decimal_discards(p) or _decimal_low_tier_discards(p)
                or _line_evidence_discards(p)
                or (koordfam and _koordfam_discards(p))
                or (seksjonering and _seksjonering_yolo_discards(p)))
    if kilde == "paddle":
        p = rule_input(box, features, conf)
        return (_window_discards(p)
                or (seksjonering and _seksjonering_paddle_discards(p)))
    return False


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


def _find_boxes_with_source(tokens, lines, yolo_boxes, koordfam=False,
                           seksjonering=False, postfilter=True):
    """Merge Paddle and YOLO boxes.

    An empty `yolo_boxes` gives pure Paddle detection. That is how
    elektronisk tinglyste documents are handled.

    Internal per-box layout: [box, kilde, yolo_conf, paddle_rec_score, trekk]
    """
    boxes = [[box, "paddle", None, rec_score, window_features]
              for (box, _mod11, rec_score, window_features)
              in sladd_boxes_from_tokens(tokens, lines)]

    for (x0, y0, x1, y1, conf) in yolo_boxes:
        yb = (x0, y0, x1, y1)
        covered = [pair for pair in boxes if smallest_share(yb, pair[0]) > DEDUP_OVERLAP]
        # Only Paddle hits may be promoted to "begge": renaming YOLO boxes
        # too would hide them from the OCR rules, which only see kilde "yolo".
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

    # ── OCR and profile rules ──────────────────────────────────────────────
    # Deliberately after the dedup loop: the kilde is final, so boxes that
    # became "begge" keep their sladding.
    # postfilter=False skips BOTH blocks below (rules + geometry) to measure
    # raw detection+mod11+dedup. YOLO_CONF and lenient_check still apply,
    # they are the model's operating point, not postfilters.
    if postfilter:
        boxes = [pair for pair in boxes
                  if not _rules_discard(pair, koordfam, seksjonering)]

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
                           postfilter=True, vlm=None, stats=None):
    """only_cache=True: return None instead of running the models on a cache
    miss. Lets GPU-less worker processes handle cache hits and send the misses
    back to a process that has the models.

    rettsstiftelsestyper: the document's XX_YYY codes from the grunnbok.
    Enables per-document-type rule profiles (KOORDFAM_CODES in config).
    None/empty = global behaviour, so missing metadata can never cost recall.

    vlm: a vlm_verifier.VlmConfig turns the VLM verifier on. It can only
    remove boxes. The crops come from the page image, so a document served
    entirely from the caches is rendered the first time a page has a box in
    the stratum, and not at all if no page does.

    stats: an optional dict this document's phase timings, cache hits and VLM
    counters are written to. Left untouched on a cache miss under only_cache.
    """
    t = {}
    n_judged = n_dropped = 0
    vlm_stats = {}
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
    needs_models = (missing_ocr or not root_hit
                    or (uses_yolo and yolo_boxes_per_page is None))
    if needs_models and only_cache:
        return None
    images_ocr = None

    if needs_models:
        with _take_time(t, "render"):
            images = list(read_pages_from_bytes(pdf_bytes))

        if not root_hit:
            with _take_time(t, "orientation"):
                # One model call for the whole document, not one per page.
                rotations = find_rotations_batch(images)

        images_ocr = [np.rot90(b, k) if k else b for b, k in zip(images, rotations)]

        if not ocr_hit and not only_yolo:
            with _take_time(t, "ocr"):
                tokens_per_page = read_tokens_batched(images_ocr)
            if cache_dir and name:
                write_ocr_cache(cache_dir, name, rotations, tokens_per_page)

    def _pages_as_ocr_saw_them():
        """The pages the verifier crops from, rasterised on first use.

        A document served from the caches renders only if some page has a box
        to judge, and never if none has. Rendering is pure CPU, so a
        cache-only worker gets here without the models.
        """
        nonlocal images_ocr
        if images_ocr is None:
            with _take_time(t, "render"):
                raw = list(read_pages_from_bytes(pdf_bytes))
            images_ocr = [np.rot90(b, k) if k else b
                          for b, k in zip(raw, rotations)]
        return images_ocr

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

            # Line grouping is the expensive part: done once per page and
            # shared by the fnr search, the per-box features and the verifier.
            lines = build_lines(tokens) if tokens else []
            if only_yolo:
                boxes_with_source = _find_boxes_only_yolo(yolo_boxes)
            else:
                boxes_with_source = _find_boxes_with_source(
                    tokens, lines, yolo_boxes, koordfam=koordfam,
                    seksjonering=seksjonering, postfilter=postfilter)

        if vlm is not None and vlm_verifier.needs_image(
                boxes_with_source, vlm):
            with _take_time(t, "vlm"):
                boxes_with_source, judged, dropped = vlm_verifier.verify_page(
                    boxes_with_source, _pages_as_ocr_saw_them()[si], lines, vlm,
                    stats=vlm_stats)
                n_judged += judged
                n_dropped += dropped

        with _take_time(t, "postprocessing"):
            pages.append(_build_page(si + 1, page_target[si], tokens, boxes_with_source,
                                    k, with_lines))

    if new_yolo_boxes is not None and len(new_yolo_boxes) == n_pages:
        write_yolo_cache(yolo_cache_dir, name, rotations, new_yolo_boxes)

    if write_time:
        _write_time(t, len(pages), name, ocr_hit, yolo_hit,
                    n_judged, n_dropped)

    if stats is not None:
        stats["timings"] = dict(t)
        stats["pages"] = n_pages
        stats["ocr_cache_hit"] = ocr_hit
        stats["yolo_cache_hit"] = yolo_hit
        if vlm is not None:
            stats["vlm"] = vlm_stats

    return _to_flat(pages, page_field)


def _write_time(t, n_pages, name=None, ocr_hit=False, yolo_hit=False,
                n_judged=0, n_dropped=0):
    entries = ["render", "orientation", "ocr", "yolo+match", "vlm",
               "postprocessing"]
    total = sum(t.get(p, 0.0) for p in entries)

    label = f"Timing [{name}]:" if name else "Timing:"
    from_cache = [n for n, hit in (("OCR", ocr_hit), ("YOLO", yolo_hit)) if hit]
    if from_cache:
        label += f" ({' + '.join(from_cache)} from cache)"
    print(label)
    for post in entries:
        if post == "vlm" and post not in t:
            continue
        sec = t.get(post, 0.0)
        pct = (sec / total * 100) if total else 0.0
        print(f"  {post:<18}{sec:9.3f} s{pct:7.1f}%")
    print(f"  {'Total':<18}{total:9.3f} s")
    print(f"  {'Pages total':<18}{n_pages:9d}")
    if n_judged:
        print(f"  {'VLM judged':<18}{n_judged:9d}  ({n_dropped} dropped)")
    if n_pages:
        print(f"  {'Per page':<18}{total / n_pages:9.3f} s")
