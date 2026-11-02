"""
Rotated pages as images, for judging the orientation model by eye.

Reads the per-document OCR cache, picks every page the orientation step turned,
and writes it turned the same way the pipeline read it. A correct call is
therefore an upright image and a wrong one is not. The rotation is first in the
filename, so a sorted folder groups 90, 180 and 270.

Two edges. The tool reads the cache and never runs the model, so a document
without a cache entry is counted and skipped: fill the cache first with
precache.py --only ocr. And a page the model got wrong but scored below
ORIENTATION_MIN_CONFIDENCE was forced to 0 in the cache, so it is invisible
here. This shows the pages that were turned, not the ones that should have been.

Run:
    python utils/rotated_pages.py --folder /path/to/pdfs
    python utils/rotated_pages.py --folder /path/to/pdfs --dpi 150 --max 200
    python utils/rotated_pages.py --folder /path/to/pdfs --raw
"""

import argparse
import os
import sys
from collections import Counter

import fitz
from PIL import Image

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from file_selection import select_files
from ocr_cache import read_cache as read_ocr_cache


def _derive_ocr_cache(args):
    """OCR cache folder: --ocr-cache, else $SLADD_CACHE/<uttrekk>/ocr."""
    if args.ocr_cache:
        return args.ocr_cache
    base = os.environ.get("SLADD_CACHE")
    if base:
        uttrekk = os.path.basename(os.path.normpath(args.folder))
        return os.path.join(base, uttrekk, "ocr")
    return None


def _page_image(page, k, dpi, raw):
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.n == 4 else "RGB"
    image = Image.frombytes(mode, (pix.w, pix.h), pix.samples).convert("RGB")
    if raw or not k:
        return image
    # PIL rotates counterclockwise, same direction as the pipeline's np.rot90(b, k).
    return image.rotate(90 * k, expand=True)


def main():
    p = argparse.ArgumentParser(
        description="Write every page the orientation step turned as an image, "
                    "with the rotation in the filename.")
    p.add_argument("--folder", required=True, help="folder of PDF files")
    p.add_argument("--ocr-cache", default=None,
                   help="OCR cache folder (default: $SLADD_CACHE/<uttrekk>/ocr)")
    p.add_argument("--out-dir", default="rotated_pages",
                   help="output folder (default: rotated_pages)")
    p.add_argument("--dpi", type=int, default=100,
                   help="render dpi (default: 100). Changes file size only, "
                        "not the rotation")
    p.add_argument("--max", type=int, default=0,
                   help="stop after this many images (0 = no limit)")
    p.add_argument("--raw", action="store_true",
                   help="save the page as it lies in the PDF, without turning it")
    p.add_argument("--select", nargs="*", default=[],
                   help="specific files (filename/substring)")
    p.add_argument("--select-from-file", default=None,
                   help="read file IDs from a text file, one per line")
    p.add_argument("--count", default="all",
                   help="number of files when --select is empty (a number, or 'all')")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: --folder does not exist: {args.folder}")
        return 1
    if args.select_from_file and not os.path.isfile(args.select_from_file):
        print(f"ERROR: --select-from-file does not exist: {args.select_from_file}")
        return 1

    ocr_dir = _derive_ocr_cache(args)
    if not ocr_dir or not os.path.isdir(ocr_dir):
        print(f"ERROR: no OCR cache folder. Use --ocr-cache or set $SLADD_CACHE "
              f"(tried: {ocr_dir})")
        return 1

    select, from_file = args.select, False
    if args.select_from_file:
        with open(args.select_from_file, encoding="utf-8") as f:
            select = [line.strip() for line in f if line.strip()]
        from_file = True
        print(f"Read {len(select)} IDs from {args.select_from_file}")

    files = select_files(args.folder, select, args.count, exact=from_file)
    if not files:
        print("No files to process. Check --folder / --select / --count.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"OCR cache:    {ocr_dir}")
    print(f"Output:       {args.out_dir}")

    per_rotation = Counter()
    read_docs = pages_total = uncached = written = 0
    for path in files:
        name = os.path.basename(path)
        try:
            cached = read_ocr_cache(ocr_dir, name)
        except ValueError:
            cached = None
        if cached is None:
            uncached += 1
            continue
        read_docs += 1
        rotations = cached[0]
        pages_total += len(rotations)
        turned = [(si, k) for si, k in enumerate(rotations, start=1) if k]
        if not turned:
            continue

        stem = os.path.splitext(name)[0]
        try:
            with fitz.open(path) as doc:
                for si, k in turned:
                    if si > len(doc):
                        print(f"!! {name}: cache has page {si}, the PDF has {len(doc)}")
                        continue
                    image = _page_image(doc[si - 1], k, args.dpi, args.raw)
                    out = os.path.join(args.out_dir, f"{90 * k:03d}deg_{stem}_p{si}.jpg")
                    image.save(out, quality=80)
                    per_rotation[90 * k] += 1
                    written += 1
                    if args.max and written >= args.max:
                        break
        except Exception as e:
            print(f"!! {name}: could not render ({e!r})")
            continue
        if args.max and written >= args.max:
            print(f"Stopped at --max {args.max}.")
            break

    print()
    print(f"Documents read from cache   {read_docs:>8}")
    print(f"Documents without cache     {uncached:>8}")
    print(f"Pages in those documents    {pages_total:>8}")
    for deg in sorted(per_rotation):
        share = 100 * per_rotation[deg] / pages_total if pages_total else 0
        print(f"  turned {deg:>3} degrees        {per_rotation[deg]:>8}  ({share:.2f} % of pages)")
    print(f"Images written              {written:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
