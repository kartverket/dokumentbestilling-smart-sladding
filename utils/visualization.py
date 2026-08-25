import glob
import os
import re

import fitz
from PIL import Image, ImageDraw, ImageFont

from load_pdf import PDF_DPI
from utils_config import (
    MISSED_TRUTH_COLOR, COVERED_TRUTH_COLOR,
    CORRECT_PADDLE_COLOR, CORRECT_YOLO_COLOR, CORRECT_BOTH_COLOR,
    OVERSLADD_PADDLE_COLOR, OVERSLADD_YOLO_COLOR, OVERSLADD_BOTH_COLOR, UNKNOWN_COLOR
)

SCALE = PDF_DPI / 72.0           # PDF points -> pixels

_CONF_FONT = None


def _conf_font():
    global _CONF_FONT
    if _CONF_FONT is None:
        _CONF_FONT = ImageFont.load_default(size=28)
    return _CONF_FONT


def _draw_conf(drawer, r, conf, color):
    if conf is None:
        return
    drawer.text((r[0] + 2, max(r[1] + 2, 2)), f"{conf:.2f}",
                fill=color, font=_conf_font())


def _doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def _fasit_pixels(box, ph_px, y_origin):
    x, y, w, h = box
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bottom":
        ph_pt = ph_px / SCALE
        y0, y1 = ph_pt - max(y, y + h), ph_pt - min(y, y + h)
    return [x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE]


def _render_page(page):
    pix = page.get_pixmap(dpi=PDF_DPI)
    mode = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(mode, (pix.w, pix.h), pix.samples).convert("RGB")


def _is_oversladd(box, oversladd_list):
    """Whether the box appears in the oversladding list (by coordinates)."""
    x0, y0, x1, y1 = box[:4]
    for ob in oversladd_list:
        ox0, oy0, ox1, oy1 = ob[:4]
        if abs(x0 - ox0) < 0.5 and abs(y0 - oy0) < 0.5 and \
           abs(x1 - ox1) < 0.5 and abs(y1 - oy1) < 0.5:
            return True
    return False


def _select_color(source, er_over):
    """Colour by kilde and by whether the box is oversladding."""
    if source not in ("paddle", "yolo", "begge"):
        return UNKNOWN_COLOR
    if er_over:
        if source == "paddle":
            return OVERSLADD_PADDLE_COLOR
        if source == "yolo":
            return OVERSLADD_YOLO_COLOR
        return OVERSLADD_BOTH_COLOR
    else:
        if source == "paddle":
            return CORRECT_PADDLE_COLOR
        if source == "yolo":
            return CORRECT_YOLO_COLOR
        return CORRECT_BOTH_COLOR


def _pages_to_draw(sladd_boxes, ground_truth, folder):
    per_file = {}
    for (name, si) in sladd_boxes:
        per_file.setdefault(name, set()).add(si)

    if ground_truth:
        no_to_name = {}
        for name in per_file:
            no_to_name.setdefault(_doc_no(name), name)
        for (nr, si) in ground_truth:
            name = no_to_name.get(nr)
            if name:
                per_file.setdefault(name, set()).add(si)

    return {name: sorted(pages) for name, pages in per_file.items()}


def draw_and_save(sladd_boxes, ground_truth, folder, out_dir, y_origin="top",
                  write_log=True, clean=True, yolo_boxes=None, sources=None,
                  oversladd_boxes=None, miss_indices=None):
    os.makedirs(out_dir, exist_ok=True)
    if clean:
        for png in glob.glob(os.path.join(out_dir, "*.png")):
            os.remove(png)

    per_file = _pages_to_draw(sladd_boxes, ground_truth, folder)

    for name in sorted(per_file):
        nr = _doc_no(name)
        try:
            d = fitz.open(os.path.join(folder, name))
        except Exception as e:
            print(f"   {name}: could not be opened ({e!r})")
            continue
        n_pages_drawn = 0
        n_with_funn = 0
        for si in per_file[name]:
            if not 1 <= si <= len(d):
                continue
            image = _render_page(d[si - 1])
            base = image.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            drawer = ImageDraw.Draw(overlay)

            bw, bh, found = sladd_boxes.get((name, si), (image.width, image.height, []))
            sx, sy = image.width / bw, image.height / bh

            over_names = []
            if oversladd_boxes and (name, si) in oversladd_boxes:
                _, _, over_names = oversladd_boxes[(name, si)]

            # 1) Prediction boxes, outline only
            if sources and (name, si) in sources:
                _, _, with_source = sources[(name, si)]
                for box in with_source:
                    x0, y0, x1, y1 = box[:4]
                    source_choice = box[4] if len(box) > 4 else "paddle"
                    conf = box[5] if len(box) > 5 else None
                    r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

                    er_over = _is_oversladd(box, over_names) if over_names else False
                    color = _select_color(source_choice, er_over)
                    drawer.rectangle(r, outline=color, width=3)
                    if source_choice in ("yolo", "begge"):
                        _draw_conf(drawer, r, conf, color)

            # 2) YOLO run live (--yolo): its own frames, with conf
            if yolo_boxes and (name, si) in yolo_boxes:
                yw, yh, yolo_f = yolo_boxes[(name, si)]
                ysx, ysy = image.width / yw, image.height / yh
                for box in yolo_f:
                    x0, y0, x1, y1 = box[:4]
                    r = [x0 * ysx, y0 * ysy, x1 * ysx, y1 * ysy]
                    drawer.rectangle(r, outline=CORRECT_YOLO_COLOR, width=3)
                    _draw_conf(drawer, r, box[4] if len(box) > 4 else None,
                               CORRECT_YOLO_COLOR)

            # 3) Truth: every label, the missed ones in red and the rest
            # faint, so an oversladd page still shows what the fasit holds.
            if ground_truth:
                for fi, (x, y, w, h, _t) in enumerate(ground_truth.get((nr, si), [])):
                    missed = miss_indices is None or (nr, si, fi) in miss_indices
                    drawer.rectangle(
                        _fasit_pixels((x, y, w, h), image.height, y_origin),
                        outline=MISSED_TRUTH_COLOR if missed else COVERED_TRUTH_COLOR,
                        width=3)

            image = Image.alpha_composite(base, overlay).convert("RGB")

            ut = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_side{si}.png")
            image.save(ut)
            n_pages_drawn += 1
            if found:
                n_with_funn += 1

        if write_log:
            print(f"  PNG: {name}, {n_pages_drawn} pages, {n_with_funn} with detections")
        d.close()