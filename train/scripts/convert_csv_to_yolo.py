"""Convert a labels CSV into rendered pages plus YOLO label files.

Rendering is the whole cost, pure CPU and disk, so documents run in parallel
processes: each one is independent, with its own PDF handle and output files.

Rows come through filter_common.iter_label_rows, so training sees the same
fasit as evaluation: REJECTED and ugyldige_labels.txt rows are out,
manglende_labels.csv rows are in. Labels for already-rendered documents in
this run's scope are rewritten from the current CSV, so a re-run refreshes
stale label files without re-rendering.

CSV coordinates are PDF points with (x, y) at the TOP-LEFT corner and the
y-axis growing downwards, the same convention as the image, so no flip.

Run:
    python train/scripts/convert_csv_to_yolo.py labels.csv pdfs/ --output dataset
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np

# ugyldige_labels.txt lives at the repo root; reuse the reader in utils/.
_UTILS = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from filter_common import iter_label_rows


DPI = 300
SCALE = DPI / 72.0


def _default_workers():
    """One process per physical core, roughly, capped to keep RAM sane."""
    return min(max((os.cpu_count() or 4) // 2, 1), 32)


def _write_label(labels_dir, stem, boxes_pt, img_w, img_h):
    """Writes one YOLO label file from boxes in PDF points; empty file if none."""
    if not boxes_pt:
        (Path(labels_dir) / f"{stem}.txt").write_text("")
        return 0
    arr = np.asarray(boxes_pt, dtype=float) * SCALE
    x_px, y_px, w_px, h_px = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    x_center = np.clip((x_px + w_px / 2) / img_w, 0.0, 1.0)
    y_center = np.clip((y_px + h_px / 2) / img_h, 0.0, 1.0)
    bw = np.clip(w_px / img_w, 0.0, 1.0)
    bh = np.clip(h_px / img_h, 0.0, 1.0)
    lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
             for xc, yc, w, h in zip(x_center, y_center, bw, bh)]
    (Path(labels_dir) / f"{stem}.txt").write_text("\n".join(lines))
    return len(lines)


def _png_size(path):
    """(width, height) from the PNG header, without decoding the image."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return (int.from_bytes(head[16:20], "big"),
            int.from_bytes(head[20:24], "big"))


def _render_document(job):
    """Render one PDF to page images + YOLO labels, in a worker process.

    `job["pages"]` maps a 1-based page number to (x, y, w, h) boxes in PDF
    points. Pages not listed are rendered as negatives (empty labels).
    """
    file_id = job["fil_id"]
    images_dir = Path(job["images_dir"])
    labels_dir = Path(job["labels_dir"])
    out = {"fil_id": file_id, "pages": 0, "boxes": 0, "negatives": 0,
           "skipped_pages": 0, "error": None}

    try:
        doc = fitz.open(job["pdf_path"])
    except Exception as e:
        out["error"] = repr(e)
        return out

    try:
        annotated = set()
        for page_no, boxes in sorted(job["pages"].items()):
            idx = int(page_no) - 1
            if idx < 0 or idx >= len(doc):
                out["skipped_pages"] += 1
                continue

            annotated.add(int(page_no))
            pix = doc[idx].get_pixmap(dpi=DPI)
            img_w, img_h = pix.width, pix.height
            stem = f"{file_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))

            out["boxes"] += _write_label(labels_dir, stem, boxes, img_w, img_h)
            out["pages"] += 1

        # Unannotated pages become negatives (empty label files)
        for idx in range(len(doc)):
            page_no = idx + 1
            if page_no in annotated:
                continue
            pix = doc[idx].get_pixmap(dpi=DPI)
            stem = f"{file_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))
            (labels_dir / f"{stem}.txt").write_text("")
            out["negatives"] += 1
    except Exception as e:
        out["error"] = repr(e)
    finally:
        doc.close()
    return out


def convert(csv_path: str, pdf_dir: str, output_dir: str, only_ids: set = None,
            workers: int = 0):
    csv_path = Path(csv_path)
    pdf_dir = Path(pdf_dir)
    output = Path(output_dir)
    images_dir = output / "images_all"
    labels_dir = output / "labels_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # One fasit policy for training and evaluation: iter_label_rows drops
    # REJECTED and ugyldige_labels.txt rows and appends manglende_labels.csv.
    info = {}
    doc_pages = {}
    n_boxes = 0
    for r in iter_label_rows(str(csv_path), info=info):
        try:
            doc = str(int(float(r["fil_revisjon_id"])))
            page = int(float(r["sidetall"]))
            box = (float(r["x"]), float(r["y"]),
                   float(r["width"]), float(r["height"]))
        except (TypeError, ValueError, KeyError):
            continue
        if only_ids is not None and doc not in only_ids:
            continue
        doc_pages.setdefault(doc, {}).setdefault(page, []).append(box)
        n_boxes += 1

    discarded = dict(info.get("discarded", {}))
    added = dict(info.get("added", {}))
    print(f"Kept {n_boxes} boxes in {len(doc_pages)} documents"
          + (f" (matching --ids)" if only_ids is not None else ""))
    if discarded:
        print(f"Discarded: {discarded}")
    if added:
        print(f"Added from manglende_labels.csv: {added}")

    missing = set()
    skipped_docs = 0
    jobs = []

    def _add_to(file_id, pages):
        """Queue one document unless the PDF is gone or it is already done."""
        nonlocal skipped_docs
        pdf_path = pdf_dir / f"{file_id}.pdf"
        if not pdf_path.exists():
            missing.add(str(file_id))
            return
        if (images_dir / f"{file_id}_p1.png").exists():   # cache
            skipped_docs += 1
            return
        jobs.append({"fil_id": file_id, "pdf_path": str(pdf_path), "pages": pages,
                     "images_dir": str(images_dir), "labels_dir": str(labels_dir)})

    for file_id, pages in sorted(doc_pages.items()):
        _add_to(file_id, pages)

    # Documents in the ID list but not in the CSV are pure negatives
    negative_docs = 0
    if only_ids:
        unlabeled_ids = only_ids - set(doc_pages)
        if unlabeled_ids:
            print(f"Rendering {len(unlabeled_ids)} unlabeled documents as negatives...")
        for file_id in sorted(unlabeled_ids):
            before = len(jobs)
            _add_to(file_id, {})
            negative_docs += len(jobs) - before

    done = total_boxes = negatives = skipped_pages = 0
    failed = []
    n_workers = min(workers or _default_workers(), max(len(jobs), 1))
    print(f"Converting {len(jobs)} documents with {n_workers} processes...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_document, j): j["fil_id"] for j in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r["error"]:
                failed.append((r["fil_id"], r["error"]))
                print(f"  [{n}/{len(jobs)}] FAILED {r['fil_id']}: {r['error']}")
                continue
            done += r["pages"]
            total_boxes += r["boxes"]
            negatives += r["negatives"]
            skipped_pages += r["skipped_pages"]
            print(f"  [{n}/{len(jobs)}] Converted {r['fil_id']} "
                  f"({r['boxes']} boxes, {r['pages']} pages, {r['negatives']} negatives)")

    print(f"Wrote {done} page-images with {total_boxes} boxes total, {negatives} negative pages ({negative_docs} fully negative docs)")
    if skipped_docs:
        print(f"Skipped {skipped_docs} already converted documents")

    # Scope guard: another uttrekk sharing the directory keeps its labels.
    rendered_now = {j["fil_id"] for j in jobs}
    scope = set(doc_pages) | (set(only_ids) if only_ids else set())
    refreshed = refreshed_boxes = orphan_rows = 0
    for png in sorted(images_dir.glob("*_p*.png")):
        doc, _, page_str = png.stem.rpartition("_p")
        if doc in rendered_now or doc not in scope:
            continue
        size = _png_size(png)
        if size is None:
            continue
        boxes = doc_pages.get(doc, {}).get(int(page_str), [])
        refreshed_boxes += _write_label(labels_dir, png.stem, boxes, *size)
        refreshed += 1
    for doc, pages in doc_pages.items():
        if doc in rendered_now or doc in missing:
            continue
        orphan_rows += sum(len(b) for p, b in pages.items()
                           if not (images_dir / f"{doc}_p{p}.png").exists())
    if refreshed:
        print(f"Refreshed labels for {refreshed} cached page-images "
              f"({refreshed_boxes} boxes)")
    if orphan_rows:
        print(f"WARNING: {orphan_rows} boxes reference pages with no rendered "
              f"image (cached documents rendered before those rows existed)")
    if missing:
        print(f"Missing PDFs: {len(missing)}")
    if skipped_pages:
        print(f"Skipped {skipped_pages} out-of-range pages")
    if failed:
        print(f"Failed: {len(failed)} documents")
        for file_id, err in failed[:10]:
            print(f"  {file_id}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to YOLO format")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", default="dataset", help="Output directory (default: dataset)")
    parser.add_argument("--ids", default=None, help="File with document IDs to convert (one per line). Converts all if not set.")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel rendering processes (0 = half the cores, max 32)")
    args = parser.parse_args()

    only_ids = None
    if args.ids:
        with open(args.ids) as f:
            only_ids = {line.strip().removesuffix(".pdf") for line in f
                        if line.strip() and not line.lstrip().startswith("#")}
        print(f"Loaded {len(only_ids)} IDs from {args.ids}")

    convert(args.csv, args.pdfs, args.output, only_ids=only_ids, workers=args.workers)
