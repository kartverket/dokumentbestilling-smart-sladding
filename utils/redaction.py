import os
from collections import defaultdict

import fitz


def _pixel_to_point(box, bw, bh, pw, ph):
    x0, y0, x1, y1 = box
    return fitz.Rect(x0 / bw * pw, y0 / bh * ph, x1 / bw * pw, y1 / bh * ph)


def sladd_file(in_path, out_path, pages):
    d = fitz.open(in_path)
    n = 0
    for (si, bw, bh, boxes) in pages:
        page = d[si - 1]
        pw, ph = page.rect.width, page.rect.height
        for box in boxes:
            page_rect = _pixel_to_point(box, bw, bh, pw, ph)
            page.add_redact_annot(page_rect * page.derotation_matrix, fill=(0, 0, 0))
            n += 1
        page.apply_redactions()
    d.save(out_path)
    d.close()
    return n


def sladd_files(sladd_boxes, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    per_file = defaultdict(list)
    for (name, si), (bw, bh, boxes) in sladd_boxes.items():
        per_file[name].append((si, bw, bh, boxes))

    sladdet, failed = 0, []
    for name in sorted(per_file):
        try:
            n = sladd_file(os.path.join(in_dir, name),
                          os.path.join(out_dir, name), sorted(per_file[name]))
            sladdet += 1
            print(f"   {name}: {n} box(es) sladdet")
        except Exception as e:
            failed.append((name, repr(e)))
            print(f"   {name}: ERROR {e!r}")
    return sladdet, failed
