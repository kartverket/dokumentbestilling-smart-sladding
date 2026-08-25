"""Draw labels onto the PDF pages, colored by error category.

Green is ACCEPTED, orange REJECTED (false positive) and red hand-placed
(true negative). Documents where everything is ACCEPTED are dropped, and the
rest are written to false_positive/ and true_negative/. A page with both
lands in both directories.

Run:
    python utils/render_error_categories.py --folder ../uttrekk_3 \
        --labels-csv labels.csv --out-dir error_categories
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import fitz
from PIL import Image, ImageDraw

PDF_DPI = 300
SCALE = PDF_DPI / 72.0  # PDF points -> pixels

COLOR_ACCEPTED = (30, 160, 30)      # green
COLOR_REJECTED = (255, 140, 0)      # orange (false positive)
COLOR_MANUAL = (200, 0, 0)         # red (true negative)


def _doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def read_labels_csv(path):
    """Returns {(doc_no, page): [label, ...]}."""
    per_page = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                doc_no = int(r["fil_revisjon_id"])
                page = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                continue

            ml = (r.get("ml_generated") or "").strip().lower() == "true"
            status = (r.get("ml_status") or "").strip().upper()

            if ml and status == "ACCEPTED":
                category = "accepted"
            elif ml and status == "REJECTED":
                category = "rejected"
            else:
                category = "manual"

            per_page[(doc_no, page)].append({
                "x": x, "y": y, "w": w, "h": h,
                "type": (r.get("type") or "").strip(),
                "category": category,
            })
    return per_page


def read_document_csv(path):
    """Returns the set of fil_revisjon_id to process."""
    ids = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                ids.add(int(r["fil_revisjon_id"]))
            except (TypeError, ValueError, KeyError):
                continue
    return ids


def _render_page(page):
    pix = page.get_pixmap(dpi=PDF_DPI)
    mode = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(mode, (pix.w, pix.h), pix.samples).convert("RGB")


def _color_for(category):
    if category == "accepted":
        return COLOR_ACCEPTED
    elif category == "rejected":
        return COLOR_REJECTED
    return COLOR_MANUAL


def draw_page(image, labels, page_width_pt, page_height_pt):
    drawer = ImageDraw.Draw(image)
    bw, bh = image.width, image.height

    for lab in labels:
        x0 = lab["x"] * SCALE
        y0 = lab["y"] * SCALE
        x1 = (lab["x"] + lab["w"]) * SCALE
        y1 = (lab["y"] + lab["h"]) * SCALE

        # Some stored boxes have negative width or height
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        color = _color_for(lab["category"])
        drawer.rectangle([x0, y0, x1, y1], outline=color, width=4)

    return image


def find_pdf_for_doc(folder, doc_no):
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".pdf"):
            continue
        if _doc_no(fn) == doc_no:
            return fn
    return None


def main():
    p = argparse.ArgumentParser(
        description="Draw labels by error category: green=accepted, "
                    "orange=rejected (FP), red=hand-placed (TN).")
    p.add_argument("--folder", required=True,
                   help="directory holding the PDF documents")
    p.add_argument("--labels-csv", required=True,
                   help="labels CSV with columns fil_revisjon_id, sidetall, x, y, width, height, "
                        "ml_generated, ml_status")
    p.add_argument("--document-csv", default=None,
                   help="optional document CSV with fil_revisjon_id, to limit the documents")
    p.add_argument("--out-dir", default="error_categories",
                   help="output root directory (default: error_categories/)")
    p.add_argument("--proceed", action="store_true",
                   help="skip pages already drawn (default: overwrite everything)")
    args = p.parse_args()

    per_page = read_labels_csv(args.labels_csv)
    n_labels = sum(len(v) for v in per_page.values())
    n_doc = len({k[0] for k in per_page})
    print(f"Read {n_labels} labels for {n_doc} documents from {args.labels_csv}")

    if args.document_csv:
        doc_ids = read_document_csv(args.document_csv)
        print(f"Document CSV: {len(doc_ids)} documents from {args.document_csv}")
        per_page = {k: v for k, v in per_page.items() if k[0] in doc_ids}
        n_labels = sum(len(v) for v in per_page.values())
        n_doc = len({k[0] for k in per_page})
        print(f"After filtering: {n_labels} labels for {n_doc} documents")

    doc_categories = defaultdict(set)
    for (doc_no, page), labels in per_page.items():
        for lab in labels:
            doc_categories[doc_no].add(lab["category"])

    interesting_doc = {
        doc_no for doc_no, kats in doc_categories.items()
        if kats != {"accepted"}
    }
    per_page = {k: v for k, v in per_page.items() if k[0] in interesting_doc}

    n_filtered = n_doc - len(interesting_doc)
    print(f"Dropped {n_filtered} documents where everything was ACCEPTED")
    print(f"Processing {len(interesting_doc)} documents with errors")

    if not interesting_doc:
        print("No documents with errors, done.")
        return

    fp_dir = os.path.join(args.out_dir, "false_positive")
    tn_dir = os.path.join(args.out_dir, "true_negative")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(tn_dir, exist_ok=True)

    n_drawn = 0
    n_fp = 0
    n_tn = 0
    processed_doc = set()

    for doc_no in sorted(interesting_doc):
        pdf_name = find_pdf_for_doc(args.folder, doc_no)
        if not pdf_name:
            print(f"  doc {doc_no}: no PDF found in {args.folder}")
            continue

        pdf_path = os.path.join(args.folder, pdf_name)
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  {pdf_name}: could not be opened ({e!r})")
            continue

        processed_doc.add(doc_no)

        doc_pages = {page: labels for (d, page), labels in per_page.items() if d == doc_no}

        for si in sorted(doc_pages):
            if not 1 <= si <= len(doc):
                print(f"  {pdf_name} page {si}: does not exist ({len(doc)} pages)")
                continue

            labels = doc_pages[si]

            categories_on_side = {lab["category"] for lab in labels}
            filename = f"{os.path.splitext(pdf_name)[0]}_side{si}.png"

            has_fp = "rejected" in categories_on_side
            has_tn = "manual" in categories_on_side

            if args.proceed:
                fp_finnes = (not has_fp) or os.path.exists(os.path.join(fp_dir, filename))
                tn_finnes = (not has_tn) or os.path.exists(os.path.join(tn_dir, filename))
                if fp_finnes and tn_finnes:
                    n_drawn += 1
                    if has_fp:
                        n_fp += 1
                    if has_tn:
                        n_tn += 1
                    continue

            page_obj = doc[si - 1]
            image = _render_page(page_obj)
            pw, ph = page_obj.rect.width, page_obj.rect.height
            draw_page(image, labels, pw, ph)

            if has_fp:
                image.save(os.path.join(fp_dir, filename))
                n_fp += 1
            if has_tn:
                image.save(os.path.join(tn_dir, filename))
                n_tn += 1

            n_drawn += 1

        doc.close()

    print(f"\nDone!")
    print(f"  Documents processed:  {len(processed_doc)}")
    print(f"  Pages drawn:          {n_drawn}")
    print(f"  False positive (FP):  {n_fp} pages -> {fp_dir}/")
    print(f"  True negative (TN):   {n_tn} pages -> {tn_dir}/")

    legend_path = os.path.join(args.out_dir, "LEGENDE.txt")
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write("COLOR CODES\n")
        f.write("===========\n")
        f.write("Green frame:  ACCEPTED, ML found it, the case worker accepted it\n")
        f.write("Orange frame: REJECTED, ML found it, the case worker rejected it (false positive)\n")
        f.write("Red frame:    BY HAND . The case worker added it (true negative / model miss)\n")
        f.write("\n")
        f.write("DIRECTORIES\n")
        f.write("===========\n")
        f.write("false_positive/ , pages with at least one REJECTED box (orange)\n")
        f.write("true_negative/  , pages with at least one hand-placed box (red)\n")
        f.write("\nNote: a page with both FP and TN appears in both directories.\n")
    print(f"  Legend:               {legend_path}")


if __name__ == "__main__":
    main()

