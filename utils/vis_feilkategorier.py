"""Visualiser labels fordelt på feilkategorier (false positive / true negative).

Tegner PDF-sider med fargede rammer rundt hver label:
  - Grønn:  ACCEPTED  (ml_generated=true, ml_status=ACCEPTED)
  - Oransje: REJECTED / false positive (ml_generated=true, ml_status=REJECTED)
  - Rød:    MANUELL / true negative   (ml_generated=false)

Filtrerer bort dokumenter der alt er ACCEPTED (ingen feil).
Lagrer i to mapper:
  - false_positive/  (sider med minst én REJECTED-boks)
  - true_negative/   (sider med minst én manuell boks)
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
SKALA = PDF_DPI / 72.0  # PDF-punkt -> piksel

# Farger
FARGE_ACCEPTED = (30, 160, 30)      # grønn
FARGE_REJECTED = (255, 140, 0)      # oransje (false positive)
FARGE_MANUELL = (200, 0, 0)         # rød (true negative)


def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def les_labels_csv(sti):
    """Les labels-CSV og returner {(dok_nr, side): [label-dict, ...]}."""
    per_side = defaultdict(list)
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                dok_nr = int(r["fil_revisjon_id"])
                side = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                continue

            ml = (r.get("ml_generated") or "").strip().lower() == "true"
            status = (r.get("ml_status") or "").strip().upper()

            if ml and status == "ACCEPTED":
                kategori = "accepted"
            elif ml and status == "REJECTED":
                kategori = "rejected"
            else:
                kategori = "manuell"

            per_side[(dok_nr, side)].append({
                "x": x, "y": y, "w": w, "h": h,
                "type": (r.get("type") or "").strip(),
                "kategori": kategori,
            })
    return per_side


def les_dokument_csv(sti):
    """Les dokument-CSV og returner set av fil_revisjon_id-er som skal prosesseres."""
    ids = set()
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                ids.add(int(r["fil_revisjon_id"]))
            except (TypeError, ValueError, KeyError):
                continue
    return ids


def _render_side(side):
    pix = side.get_pixmap(dpi=PDF_DPI)
    modus = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(modus, (pix.w, pix.h), pix.samples).convert("RGB")


def _farge_for(kategori):
    if kategori == "accepted":
        return FARGE_ACCEPTED
    elif kategori == "rejected":
        return FARGE_REJECTED
    return FARGE_MANUELL


def tegn_side(bilde, labels, side_bredde_pt, side_hoyde_pt):
    """Tegn fargede rammer for alle labels på bildet."""
    tegner = ImageDraw.Draw(bilde)
    bw, bh = bilde.width, bilde.height

    for lab in labels:
        # Konverter PDF-punkter til piksler
        x0 = lab["x"] * SKALA
        y0 = lab["y"] * SKALA
        x1 = (lab["x"] + lab["w"]) * SKALA
        y1 = (lab["y"] + lab["h"]) * SKALA

        # Normaliser (håndter negativ bredde/høyde)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        farge = _farge_for(lab["kategori"])
        tegner.rectangle([x0, y0, x1, y1], outline=farge, width=4)

    return bilde


def finn_pdf_for_dok(mappe, dok_nr):
    """Finn PDF-filen som matcher et gitt dok_nr."""
    for fn in os.listdir(mappe):
        if not fn.lower().endswith(".pdf"):
            continue
        if _dok_nr(fn) == dok_nr:
            return fn
    return None


def main():
    p = argparse.ArgumentParser(
        description="Visualiser labels med feilkategorier (false positive / true negative). "
                    "Tegner fargede rammer: grønn=accepted, oransje=rejected (FP), rød=manuell (TN).")
    p.add_argument("--mappe", required=True,
                   help="mappe med PDF-dokumentene")
    p.add_argument("--labels-csv", required=True,
                   help="labels-CSV med kolonnene fil_revisjon_id, sidetall, x, y, width, height, "
                        "ml_generated, ml_status")
    p.add_argument("--dokument-csv", default=None,
                   help="valgfri dokument-CSV med fil_revisjon_id for filtrering av dokumenter")
    p.add_argument("--ut-mappe", default="feilkategorier",
                   help="rotmappe for utdata (default: feilkategorier/)")
    p.add_argument("--fortsett", action="store_true",
                   help="hopp over sider som allerede er tegnet (default: overskriver alt)")
    args = p.parse_args()

    # --- Les labels ---
    per_side = les_labels_csv(args.labels_csv)
    n_labels = sum(len(v) for v in per_side.values())
    n_dok = len({k[0] for k in per_side})
    print(f"Leste {n_labels} labels for {n_dok} dokumenter fra {args.labels_csv}")

    # --- Filtrer på dokument-CSV hvis angitt ---
    if args.dokument_csv:
        dok_ids = les_dokument_csv(args.dokument_csv)
        print(f"Dokument-CSV: {len(dok_ids)} dokumenter fra {args.dokument_csv}")
        per_side = {k: v for k, v in per_side.items() if k[0] in dok_ids}
        n_labels = sum(len(v) for v in per_side.values())
        n_dok = len({k[0] for k in per_side})
        print(f"Etter filtrering: {n_labels} labels for {n_dok} dokumenter")

    # --- Filtrer bort dokumenter der alt er ACCEPTED ---
    dok_kategorier = defaultdict(set)
    for (dok_nr, side), labels in per_side.items():
        for lab in labels:
            dok_kategorier[dok_nr].add(lab["kategori"])

    interessante_dok = {
        dok_nr for dok_nr, kats in dok_kategorier.items()
        if kats != {"accepted"}
    }
    per_side = {k: v for k, v in per_side.items() if k[0] in interessante_dok}

    n_filtrert = n_dok - len(interessante_dok)
    print(f"Filtrerte bort {n_filtrert} dokumenter der alt var ACCEPTED")
    print(f"Behandler {len(interessante_dok)} dokumenter med feil")

    if not interessante_dok:
        print("Ingen dokumenter med feil — ferdig.")
        return

    # --- Finn hvilke sider som skal i hvilke mapper ---
    fp_mappe = os.path.join(args.ut_mappe, "false_positive")
    tn_mappe = os.path.join(args.ut_mappe, "true_negative")
    os.makedirs(fp_mappe, exist_ok=True)
    os.makedirs(tn_mappe, exist_ok=True)

    # --- Prosesser dokumenter ---
    n_tegnet = 0
    n_fp = 0
    n_tn = 0
    prosesserte_dok = set()

    for dok_nr in sorted(interessante_dok):
        pdf_navn = finn_pdf_for_dok(args.mappe, dok_nr)
        if not pdf_navn:
            print(f"  dok {dok_nr}: fant ikke PDF i {args.mappe}")
            continue

        pdf_sti = os.path.join(args.mappe, pdf_navn)
        try:
            dok = fitz.open(pdf_sti)
        except Exception as e:
            print(f"  {pdf_navn}: kunne ikke åpnes ({e!r})")
            continue

        prosesserte_dok.add(dok_nr)

        # Samle sider for dette dokumentet
        dok_sider = {side: labels for (d, side), labels in per_side.items() if d == dok_nr}

        for si in sorted(dok_sider):
            if not 1 <= si <= len(dok):
                print(f"  {pdf_navn} side {si}: finnes ikke ({len(dok)} sider)")
                continue

            labels = dok_sider[si]

            # Avgjør hvilke mapper denne siden skal i
            kategorier_paa_side = {lab["kategori"] for lab in labels}
            filnavn = f"{os.path.splitext(pdf_navn)[0]}_side{si}.png"

            har_fp = "rejected" in kategorier_paa_side
            har_tn = "manuell" in kategorier_paa_side

            # --fortsett: hopp over hvis utfilene allerede finnes
            if args.fortsett:
                fp_finnes = (not har_fp) or os.path.exists(os.path.join(fp_mappe, filnavn))
                tn_finnes = (not har_tn) or os.path.exists(os.path.join(tn_mappe, filnavn))
                if fp_finnes and tn_finnes:
                    n_tegnet += 1
                    if har_fp:
                        n_fp += 1
                    if har_tn:
                        n_tn += 1
                    continue

            # Render og tegn (bare hvis vi ikke hoppet over)
            side_obj = dok[si - 1]
            bilde = _render_side(side_obj)
            pw, ph = side_obj.rect.width, side_obj.rect.height
            tegn_side(bilde, labels, pw, ph)

            if har_fp:
                bilde.save(os.path.join(fp_mappe, filnavn))
                n_fp += 1
            if har_tn:
                bilde.save(os.path.join(tn_mappe, filnavn))
                n_tn += 1

            n_tegnet += 1

        dok.close()

    # --- Oppsummering ---
    print(f"\nFerdig!")
    print(f"  Dokumenter prosessert: {len(prosesserte_dok)}")
    print(f"  Sider tegnet:         {n_tegnet}")
    print(f"  False positive (FP):  {n_fp} sider -> {fp_mappe}/")
    print(f"  True negative (TN):   {n_tn} sider -> {tn_mappe}/")

    # Skriv en liten legende
    legende_sti = os.path.join(args.ut_mappe, "LEGENDE.txt")
    with open(legende_sti, "w", encoding="utf-8") as f:
        f.write("FARGEKODER\n")
        f.write("==========\n")
        f.write("Grønn ramme:   ACCEPTED  — ML fant den, saksbehandler godkjente\n")
        f.write("Oransje ramme: REJECTED  — ML fant den, saksbehandler avviste (false positive)\n")
        f.write("Rød ramme:     MANUELL   — Saksbehandler la til selv (true negative / model bom)\n")
        f.write("\n")
        f.write("MAPPER\n")
        f.write("======\n")
        f.write("false_positive/  — Sider med minst én REJECTED-boks (oransje)\n")
        f.write("true_negative/   — Sider med minst én manuell boks (rød)\n")
        f.write("\nMerk: En side kan ligge i begge mapper hvis den har både FP og TN.\n")
    print(f"  Legende:              {legende_sti}")


if __name__ == "__main__":
    main()

