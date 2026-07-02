import os
from collections import defaultdict

import fitz


def _piksel_til_punkt(boks, bw, bh, pw, ph):
    x0, y0, x1, y1 = boks
    return fitz.Rect(x0 / bw * pw, y0 / bh * ph, x1 / bw * pw, y1 / bh * ph)


def sladd_fil(inn_sti, ut_sti, sider):
    d = fitz.open(inn_sti)
    n = 0
    for (si, bw, bh, bokser) in sider:
        side = d[si - 1]
        pw, ph = side.rect.width, side.rect.height
        for boks in bokser:
            side.add_redact_annot(_piksel_til_punkt(boks, bw, bh, pw, ph), fill=(0, 0, 0))
            n += 1
        side.apply_redactions()
    d.save(ut_sti)
    d.close()
    return n


def sladd_alle(sladd_bokser, inn_mappe, ut_mappe):
    "Sladd alle PDF-ene i sladd_bokser. Skriver til ut_mappe (originalene urørt)."
    os.makedirs(ut_mappe, exist_ok=True)
    per_fil = defaultdict(list)
    for (navn, si), (bw, bh, bokser) in sladd_bokser.items():
        per_fil[navn].append((si, bw, bh, bokser))

    sladdet, feilet = 0, []
    for navn in sorted(per_fil):
        try:
            n = sladd_fil(os.path.join(inn_mappe, navn),
                          os.path.join(ut_mappe, navn), sorted(per_fil[navn]))
            sladdet += 1
            print(f"   {navn}: {n} boks(er) sladdet")
        except Exception as e:
            feilet.append((navn, repr(e)))
            print(f"   {navn}: FEIL {e!r}")
    return sladdet, feilet
