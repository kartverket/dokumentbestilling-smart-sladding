"""Smoke test of the three VLM pilot tools on synthetic data.

Runs without a GPU, without a server and without access to the documents: it
builds three small PDFs with known numbers, a fasit CSV, a result CSV and an
OCR cache in the pipeline's own format, starts a stub speaking the OpenAI
protocol, and runs vlm_export → vlm_judge → vlm_evaluate end to end.

Run:
    python utils/vlm_selftest.py          # cleans up after itself
    python utils/vlm_selftest.py --keep   # leaves the files for inspection
"""

import argparse
import base64
import csv
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "app")))

import fitz
from PIL import Image

from filter_common import SCALE
from filter_review import rotate_box
from vlm_export import _ocr_context
from ocr_cache import Token, write_cache
from vlm_judge import parse_answer
from vlm_evaluate import poisson_upper_bound

FONT = 11
PAGE_B, PAGE_H = 595.0, 842.0

# (text, x, y, is_fnr); y is the baseline. The numbers are the pilot's own
# contrasts: real fnr against coordinate, account number and table cell.
DOCUMENTS = {
    "0100001.pdf": [
        ("Hjemmelshaver 060695 00000", 70.0, 200.0, True),
        ("Koordinat N 6626630.58", 70.0, 260.0, False),
        ("Dagboknr 900123 tinglyst 03.11.1998", 70.0, 320.0, False),
    ],
    "0100002.pdf": [
        ("Selger 07079600000 andel 1/2", 70.0, 180.0, True),
        ("Konto 1234 56 78903", 70.0, 240.0, False),
    ],
    "0100003.pdf": [                      # rotated in the OCR cache
        ("Kjoper 48089700000 d-nummer", 70.0, 300.0, True),
        ("gnr 12 bnr 345 snr 6", 70.0, 360.0, False),
    ],
}
ROTATION = {"0100003.pdf": 1}


def _box(line_text, x, y):
    """The text's box in PDF points, as insert_text places it."""
    b = fitz.get_text_length(line_text, fontname="helv", fontsize=FONT)
    return (x, y - FONT * 0.8, x + b, y + FONT * 0.25)


def _number_box(line_text, x, y):
    """The box around the number itself, which is what the model proposes."""
    m = re.search(r"\d[\d .,/-]*\d", line_text)
    if not m:
        return _box(line_text, x, y)
    before = fitz.get_text_length(line_text[:m.start()], fontname="helv", fontsize=FONT)
    box_width = fitz.get_text_length(m.group(0), fontname="helv", fontsize=FONT)
    return (x + before, y - FONT * 0.8, x + before + box_width, y + FONT * 0.25)


def build_data(rot):
    """Builds PDFs, fasit CSV, result CSV and OCR cache. Returns the paths."""
    pdf_dir = os.path.join(rot, "pdf")
    ocr_dir = os.path.join(rot, "ocr")
    os.makedirs(pdf_dir, exist_ok=True)

    bw, bh = int(PAGE_B * SCALE), int(PAGE_H * SCALE)
    truth_rows, pred_rows = [], []
    label_id = 5000
    for name, poster in DOCUMENTS.items():
        doc = fitz.open()
        page = doc.new_page(width=PAGE_B, height=PAGE_H)
        tokens = []
        for line_text, x, y, er_fnr in poster:
            page.insert_text((x, y), line_text, fontsize=FONT, fontname="helv")
            tx0, ty0, tx1, ty1 = _number_box(line_text, x, y)
            if er_fnr:
                label_id += 1
                truth_rows.append({
                    "id": label_id, "fil_revisjon_id": int(name.lstrip("0")[:6]),
                    "sidetall": 1, "x": tx0, "y": ty0,
                    "width": tx1 - tx0, "height": ty1 - ty0,
                    "ml_status": "ACCEPTED",
                })
            px = [tx0 * SCALE, ty0 * SCALE, tx1 * SCALE, ty1 * SCALE]
            pred_rows.append({
                "navn": name, "side": 1, "bilde_bredde": bw, "bilde_hoyde": bh,
                "x0": round(px[0], 2), "y0": round(px[1], 2),
                "x1": round(px[2], 2), "y1": round(px[3], 2),
                "kilde": "yolo", "yolo_conf": 0.71, "paddle_rec_score": "",
                "har_tokens": 1,
            })
            # One token per word, in the rotated space the pipeline OCR-ed in.
            k = ROTATION.get(name, 0)
            kx = x
            for ord_ in line_text.split(" "):
                ob = fitz.get_text_length(ord_, fontname="helv", fontsize=FONT)
                r = [kx * SCALE, (y - FONT * 0.8) * SCALE,
                     (kx + ob) * SCALE, (y + FONT * 0.25) * SCALE]
                r = rotate_box(r, k, bw, bh)
                tokens.append(Token(ord_, r[0], r[1], r[2], r[3], 0.97))
                kx += ob + fitz.get_text_length(" ", fontname="helv",
                                                fontsize=FONT)
        doc.save(os.path.join(pdf_dir, name))
        doc.close()
        write_cache(ocr_dir, name, [ROTATION.get(name, 0)], [tokens])

    truth_csv = os.path.join(rot, "fasit.csv")
    with open(truth_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "fil_revisjon_id", "sidetall",
                                          "x", "y", "width", "height",
                                          "ml_status"])
        w.writeheader()
        w.writerows(truth_rows)

    res_csv = os.path.join(rot, "resultat.csv")
    field = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1",
            "y1", "kilde", "yolo_conf", "paddle_rec_score", "har_tokens"]
    with open(res_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field, extrasaction="ignore")
        w.writeheader()
        w.writerows(pred_rows)
    return pdf_dir, ocr_dir, truth_csv, res_csv


# ── Stub endpoint ─────────────────────────────────────────────

def make_server(bad=False, thinks=False, reject_reasoning=False,
                answers=None, fixed=None):
    """Stub speaking /v1/chat/completions.

    bad             injects 500 errors and prose answers
    thinks          answers like a thinking model: empty «content», everything
                    in «reasoning», unless reasoning_effort=none was sent
    reject_reasoning
                    answers 400 to reasoning_effort, like older endpoints
    answers         {image-b64: svar} — the stub's "model". Unknown images
                    fall back to a ja/nei/usikker round-robin.
    fixed           a dict answered to everything, for tests that need to know
                    what the model said without knowing which crop it saw
    """
    counter = {"n": 0, "reasoning": [], "avvist": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            requirement = json.loads(self.rfile.read(n).decode("utf-8"))
            with lock:
                counter["n"] += 1
                i = counter["n"]
                counter["reasoning"].append(requirement.get("reasoning_effort"))

            if reject_reasoning and "reasoning_effort" in requirement:
                with lock:
                    counter["avvist"] += 1
                self.send_error(400, "unknown field reasoning_effort")
                return

            if bad and i % 5 == 0:
                self.send_error(500, "synthetic server error")
                return

            svar_bilde = None
            for m in requirement["messages"]:
                c = m["content"]
                if not isinstance(c, list):
                    continue
                for d in c:
                    if d.get("type") == "image_url":
                        b64 = d["image_url"]["url"].split("base64,", 1)[-1]
                        svar_bilde = (answers or {}).get(b64)

            if fixed is not None:
                content = json.dumps(fixed)
            elif bad and i % 7 == 0:
                content = "Dette ser ut som et fødselsnummer, ja."
            else:
                answer = svar_bilde or ("ja", "nei", "usikker")[i % 3]
                content = json.dumps({"svar": answer,
                                      "tall": "", "begrunnelse": "stub"})
            message = {"role": "assistant", "content": content}
            if thinks and requirement.get("reasoning_effort") != "none":
                message = {"role": "assistant", "content": "",
                           "reasoning": "Hmm, la meg tenke grundig ..."}
            body = json.dumps({"choices": [{"message": message}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", counter


# ── Run ───────────────────────────────────────────────────────

def run_step(*args):
    r = subprocess.run([sys.executable] + list(args), capture_output=True,
                       text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"ERROR: {' '.join(args[:2])} returned {r.returncode}")
    return r.stdout


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")
    print(f"  ok  {message}")


# The fnr guard asks find_fnr for the SHAPE, not the check digits, so a
# number with a real day and month and invalid control digits is enough.
SHAPE_FNR = "01019900000"


def check_verifier(rot):
    """The verifier inside model_main: stratum, «nei», fnr guard, failures.

    Two levels. The pipeline level stubs YOLO and Paddle out and checks the
    wiring in run_model_on_pdf_bytes; the verify_page level feeds boxes and a
    page image straight in, since a hand-built OCR token cannot carry a box
    through lenient_check and mod11 the way a real scan does. Every path that
    is not a clear, unprotected «nei» must leave the box standing.
    """
    import numpy as np
    from PIL import Image

    import model_main
    import vlm_verifier
    from paddle_ocr_model_fnr import Token

    box = (300.0, 400.0, 460.0, 440.0)
    original = (model_main.find_yolo_boxes, model_main.read_tokens_batched,
                model_main.find_rotations_batch)
    model_main.find_yolo_boxes = lambda image, conf=None: [(*box, 0.85)]
    model_main.read_tokens_batched = lambda images: [[] for _ in images]
    model_main.find_rotations_batch = lambda images: [0 for _ in images]

    doc = fitz.open()
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()
    image = Image.fromarray(np.full((900, 700, 3), 255, dtype=np.uint8))

    def boxes(*sources):
        return [(box, kilde, 0.85, None, None) for kilde in sources]

    nei = {"tall": "12345", "holdepunkt": "kontonummer", "svar": "nei"}
    srv, url, counter = make_server(fixed=nei)
    try:
        a = vlm_verifier.VlmConfig([url], "stub", timeout=10)

        before = counter["n"]
        check(len(model_main.run_model_on_pdf_bytes(pdf)) == 1
              and counter["n"] == before,
              "vlm=None: the box stands and nothing is sent")
        check(len(model_main.run_model_on_pdf_bytes(pdf, vlm=a)) == 0,
              "an unprotected «nei» removes the box")
        check(counter["n"] == before + 1, "exactly one box was judged")

        # Documents with a rule profile are outside the stratum: the profiles
        # already clean up there, and the measured gain is elsewhere.
        before = counter["n"]
        for code, name in (("SR_JOU", "koordfam"), ("SE_SEK", "seksjonering")):
            n = len(model_main.run_model_on_pdf_bytes(
                pdf, vlm=a, rettsstiftelsestyper=[code]))
            check(n == 1 and counter["n"] == before,
                  f"a {name} document is not judged at all")

        # Only kilde «yolo» is worth the GPU: the gain measured on the other
        # kilder was 1 box of 332.
        left, judged, dropped = vlm_verifier.verify_page(
            boxes("begge", "paddle", "yolo_vertikal"), image, [], a)
        check((judged, dropped, len(left)) == (0, 0, 3),
              "begge, paddle and yolo_vertikal never reach the model")

        # The fnr guard: the model says «nei» to digits that are shaped like a
        # fnr. Both readings of the line must overrule it.
        left, _, dropped = vlm_verifier.verify_page(boxes("yolo"), image, [], a)
        check((dropped, len(left)) == (1, 0),
              "no fnr anywhere: the «nei» stands")
        line = [Token(f"Selger {SHAPE_FNR} andel", 290.0, 395.0, 470.0,
                      445.0, 0.99)]
        left, _, dropped = vlm_verifier.verify_page(
            boxes("yolo"), image, [(line, line[0].text, None)], a)
        check((dropped, len(left)) == (0, 1),
              "PaddleOCR's line holds a fnr run: the «nei» is overruled")

        dead = vlm_verifier.VlmConfig(["http://127.0.0.1:1/v1"], "stub",
                                      timeout=1)
        check(len(model_main.run_model_on_pdf_bytes(pdf, vlm=dead)) == 1,
              "an unreachable endpoint cannot cost a sladd")
    finally:
        srv.shutdown()

    srv, url, _ = make_server(fixed={"tall": SHAPE_FNR, "svar": "nei"})
    try:
        a = vlm_verifier.VlmConfig([url], "stub", timeout=10)
        left, _, dropped = vlm_verifier.verify_page(boxes("yolo"), image, [], a)
        check((dropped, len(left)) == (0, 1),
              "the model transcribed a fnr and still said «nei»: overruled")
    finally:
        srv.shutdown()

    srv, url, _ = make_server(fixed={"tall": "12345", "svar": "ja"})
    try:
        a = vlm_verifier.VlmConfig([url], "stub", timeout=10)
        check(len(model_main.run_model_on_pdf_bytes(pdf, vlm=a)) == 1,
              "a «ja» leaves the box alone")
    finally:
        srv.shutdown()

    # An endpoint that answers 400 to reasoning_effort would otherwise make
    # every box «usikker», and the verifier would look like it was running
    # while removing nothing.
    import vlm_client
    before_thinking = vlm_client._THINKING["value"]
    vlm_client._THINKING["value"] = "none"
    srv, url, counter = make_server(reject_reasoning=True, fixed=nei)
    try:
        a = vlm_verifier.VlmConfig([url], "stub", timeout=10,
                                   cache_dir=os.path.join(rot, "vlm_400"))
        left, judged, dropped = vlm_verifier.verify_page(
            boxes("yolo"), image, [], a)
        check(counter["avvist"] == 1,
              "only the first call is rejected, then the field is dropped")
        check((judged, dropped, len(left)) == (1, 1, 0),
              "the verdict still comes through and the box is removed")
        check(not glob.glob(os.path.join(a.cache_dir, "*.json")),
              "answers from the fallback stay out of the cache")
    finally:
        srv.shutdown()
        vlm_client._THINKING["value"] = before_thinking

    # The judgement cache is keyed on the crop, so the second run must not
    # reach the endpoint at all.
    srv, url, counter = make_server(fixed=nei)
    try:
        a = vlm_verifier.VlmConfig([url], "stub", timeout=10,
                                   cache_dir=os.path.join(rot, "vlm_cache"))
        check(len(a.fingerprint) == 16,
              "the cache folder is the prompt version")
        model_main.run_model_on_pdf_bytes(pdf, vlm=a)
        after_first = counter["n"]
        model_main.run_model_on_pdf_bytes(pdf, vlm=a)
        check(counter["n"] == after_first,
              "the same crop is answered from the cache the second time")
    finally:
        srv.shutdown()
        (model_main.find_yolo_boxes, model_main.read_tokens_batched,
         model_main.find_rotations_batch) = original


def main(keep):
    rot = tempfile.mkdtemp(prefix="vlm_selftest_")
    try:
        print(f"Work directory: {rot}")
        pdf_dir, ocr_dir, truth_csv, res_csv = build_data(rot)

        # ── Unit pieces, no server ───────────────────────────
        # The prompt must actually ask for every field the judge PARSES.
        print("\n[0] the prompt asks for the fields we parse")
        from vlm_judge import STD_PROMPT
        for field in ("svar", "tall", "holdepunkt"):
            check(f'"{field}"' in STD_PROMPT,
                  f"the JSON template asks for «{field}»")
        # Autoregressive: what the model writes before «svar» is computation
        # the verdict can build on, what comes after is rationalisation.
        for field in ("tall", "holdepunkt"):
            check(STD_PROMPT.index(f'"{field}"') < STD_PROMPT.index('"svar"'),
                  f"«{field}» comes before «svar» in the JSON template")
        # «usikker» is a parse failure state, not an answer: doubt has to go
        # to «ja», or recall leaks out through a third alternative.
        check(re.search(r"i tvil,?\s*svar «ja»", STD_PROMPT),
              "the prompt sends doubt to «ja»")
        check("usikker" not in STD_PROMPT.lower(),
              "the prompt does not offer «usikker» as an answer")
        asymmetry = re.search(r"(\d+) ganger verre", STD_PROMPT)
        check(asymmetry and int(asymmetry.group(1)) > 1,
              "the prompt states the cost asymmetry "
              f"({asymmetry.group(1) + 'x' if asymmetry else 'missing'})")
        # A «nei» is only legal for a named alternative, and the contrasts
        # this test builds documents from have to be among them.
        for word in ("kontonummer", "organisasjonsnummer", "koordinat"):
            check(word in STD_PROMPT.lower(),
                  f"the prompt names «{word}» as a legal «nei»")

        print("\n[1] parse_answer")
        check(parse_answer('{"svar":"nei","tall":"66266","begrunnelse":"koordinat"}')
              [:1] == ("nei",), "plain JSON is read")
        check(parse_answer('```json\n{"svar": "ja"}\n```')[0] == "ja",
              "JSON in a code block is read")
        check(parse_answer("Jeg mener dette er nei, en koordinat.")[0] == "nei",
              "prose falls back on keywords")
        check(parse_answer("")[0] == "usikker", "an empty answer becomes usikker")
        check(parse_answer("^^^")[0] == "usikker", "junk becomes usikker")
        # A full checklist without «svar» is still a parse failure.
        without = parse_answer('{"linjen":"Dagboknr. 1234/1980","holdepunkt":"dagboknr",'
                         '"tall":"1234/1980"}')
        check(without[0] == "usikker" and "omitted the «svar»" in without[3],
              f"a missing «svar» is named precisely ({without[3]!r})")
        check(all(parse_answer(t)[0] != "nei" for t in ("", "^^^", "{}")),
              "no failure state can become «nei»")

        print("\n[1b] fnr candidates with the pipeline's digit confusion")
        from vlm_client import fnr_candidate as _fnr_candidate
        from vlm_client import has_fnr_caption as _has_fnr_caption
        check(_fnr_candidate("loo190-00000"),
              "«loo190-00000» is recognised, o→0 and l→1")
        # Strip the spaces and the run glues to «1f1g» from «Iflg», and
        # the boundary check fails.
        check(_fnr_candidate("030392S0000 Iflg fullmakt"),
              "«030392S0000 Iflg fullmakt» is recognised despite neighbours")
        # The date of birth is in another form than DDMMYY, so there is no
        # eleven-digit run to find.
        for line in ("f ø dt : 1.2.1950 Personnummer : . 00000",
                      "0la Nordmann , f . 12 / 3-1950 , pers . nr . 00000 ,"):
            check(_has_fnr_caption(line) and not _fnr_candidate(line),
                  f"ledetekst protects «{line[:28]}…»")
        for line in ("Dagboknr. 1234/1980", "Takstnr. 11,12,13,14.",
                      "N 6626630.58 632412.066"):
            check(not _has_fnr_caption(line),
                  f"the ledetekst guard leaves «{line[:24]}…» alone")
        check(_fnr_candidate("02029100000") and _fnr_candidate("020291 00000"),
              "plain and grouped fødselsnumre are recognised")
        check(not _fnr_candidate("N 6626630.58"),
              "a coordinate is NOT recognised as fnr")
        check(not _fnr_candidate("Personnummer: 00000"),
              "five digits alone are not recognised")
        check(not _fnr_candidate("99999999999"),
              "eleven digits without a valid date are not recognised")

        print("\n[2] poisson_upper")
        check(abs(poisson_upper_bound(0) - 3.0) < 0.01,
              f"zero observed gives the rule of three ({poisson_upper_bound(0):.2f})")
        check(poisson_upper_bound(1) > 4.7 and poisson_upper_bound(1) < 4.8,
              f"one observed gives ~4.74 ({poisson_upper_bound(1):.2f})")

        # ── Export ───────────────────────────────────────────
        print("\n[3] vlm_export")
        ut = os.path.join(rot, "export")
        run_step(os.path.join(HERE, "vlm_export.py"),
             "--res-csv", res_csv, "--truth-csv", truth_csv,
             "--folder", pdf_dir, "--out-dir", ut,
             "--hit-sample", "2", "--ocr-cache", ocr_dir, "--seed", "7")
        manifest = read(os.path.join(ut, "manifest.csv"))
        with open(os.path.join(ut, "utvalg.json"), encoding="utf-8") as f:
            sample = json.load(f)

        check(sample["n_bom_total"] == 4, f"4 BOM found ({sample['n_bom_total']})")
        check(sample["n_covering_total"] == 3,
              f"3 covering found ({sample['n_covering_total']})")
        check(sample["n_covering_exported"] == 2, "the sample became 2 covering")
        check(abs(sample["hit_factor"] - 1.5) < 1e-9,
              f"treff_faktor = 1.5 ({sample['hit_factor']})")
        check(len(manifest) == 6, f"6 manifest rows ({len(manifest)})")
        check(all(os.path.getsize(os.path.join(ut, "utsnitt", r["utsnitt"])) > 500
                  for r in manifest), "every crop written and not empty")
        for r in manifest:
            b = Image.open(os.path.join(ut, "utsnitt", r["utsnitt"]))
            check(b.size == (int(r["utsnitt_bredde"]), int(r["utsnitt_hoyde"]))
                  and 0 <= float(r["m_x0"]) and float(r["m_x1"]) <= b.width,
                  f"the marker sits inside {r['utsnitt']}")
            break
        check(all(r["ocr_linje"] for r in manifest),
              "the OCR line was fetched for every row")
        hit = [r for r in manifest if r["klasse"] == "TREFF"]
        check(all(r["label_id"] for r in hit),
              "covering rows carry label_id through")
        rotated = [r for r in manifest if r["fil"] == "0100003.pdf"]
        check(len(rotated) >= 1 and all(
            int(r["utsnitt_hoyde"]) > int(r["utsnitt_bredde"]) for r in rotated),
              "rotated pages are cropped upright (tall crop)")
        check(all(r["ocr_tekst"] in ("48089700000", "12") for r in rotated),
              "the numbers are read from the OCR cache on the rotated page "
              f"({[r['ocr_tekst'] for r in rotated]})")

        # Band logic in _ocr_context, in the upright space it sees: neighbour
        # lines stay out, a partly overlapping token comes in.
        rect = [210.0, 500.0, 400.0, 520.0]
        toks = [Token("Selger", 100, 500, 200, 520, 0.9),
                Token("07079600000", 210, 500, 400, 520, 0.9),
                Token("andel", 410, 502, 470, 518, 0.9),
                Token("overlinje", 100, 470, 470, 488, 0.9),
                Token("underlinje", 100, 532, 470, 550, 0.9),
                Token("", 210, 500, 400, 520, 0.9)]
        i_box, i_line, block = _ocr_context(toks, rect)
        check(i_box == "07079600000", f"_ocr_context finds the box ({i_box!r})")
        check(i_line == "Selger 07079600000 andel",
              f"_ocr_context finds the line without neighbours ({i_line!r})")
        check(block == "", "without --ocr-lines the block is empty")
        _, _, b1 = _ocr_context(toks, rect, n_lines=1)
        check(b1.split("\n") == ["overlinje",
                                 "Selger 07079600000 andel",
                                 "underlinje"],
              f"n_lines=1 gives the line plus one neighbour each way ({b1!r})")
        check(_ocr_context([], rect) == ("", "", ""),
              "an empty token list is harmless")

        # Asymmetric margins and full page width.
        wide = os.path.join(rot, "eksport_bred")
        run_step(os.path.join(HERE, "vlm_export.py"),
             "--res-csv", res_csv, "--truth-csv", truth_csv,
             "--folder", pdf_dir, "--out-dir", wide, "--hit-sample", "0",
             "--margin-x", "200", "--margin-y", "20", "--max-px", "0")
        m_wide = {r["fil"] + r["nr"]: r for r in
                  read(os.path.join(wide, "manifest.csv"))}
        narrow = {r["fil"] + r["nr"]: r for r in manifest
                if r["klasse"] == "BOM"}
        check(all(int(m_wide[k]["utsnitt_bredde"]) > int(v["utsnitt_bredde"])
                  and int(m_wide[k]["utsnitt_hoyde"]) < int(v["utsnitt_hoyde"])
                  for k, v in narrow.items() if k in m_wide),
              "--margin-x/-y work independently per axis")

        full = os.path.join(rot, "eksport_full")
        run_step(os.path.join(HERE, "vlm_export.py"),
             "--res-csv", res_csv, "--truth-csv", truth_csv,
             "--folder", pdf_dir, "--out-dir", full, "--hit-sample", "0",
             "--full-width", "--max-px", "0")
        page_width = round(PAGE_B * SCALE)
        check(all(abs(int(r["utsnitt_bredde"]) - page_width) <= 2
                  for r in read(os.path.join(full, "manifest.csv"))
                  if r["fil"] != "0100003.pdf"),
              "--full-width gives crops the full page width")

        # nr is assigned before the work is distributed, so the manifests must
        # be bit-identical no matter how many processes.
        serial = os.path.join(rot, "eksport_j1")
        parallel = os.path.join(rot, "eksport_j4")
        for dir_out, n in ((serial, "1"), (parallel, "4")):
            run_step(os.path.join(HERE, "vlm_export.py"),
                 "--res-csv", res_csv, "--truth-csv", truth_csv,
                 "--folder", pdf_dir, "--out-dir", dir_out,
                 "--hit-sample", "2", "--ocr-cache", ocr_dir,
                 "--seed", "7", "--workers", n)
        a = open(os.path.join(serial, "manifest.csv"), "rb").read()
        b = open(os.path.join(parallel, "manifest.csv"), "rb").read()
        check(a == b, "--workers 1 and --workers 4 give an identical manifest")
        check(sorted(os.listdir(os.path.join(serial, "utsnitt")))
              == sorted(os.listdir(os.path.join(parallel, "utsnitt"))),
              "and identical crop names")
        progress = run_step(os.path.join(HERE, "vlm_export.py"),
                         "--res-csv", res_csv, "--truth-csv", truth_csv,
                         "--folder", pdf_dir, "--hit-sample", "0",
                         "--out-dir", os.path.join(rot, "eksport_frem"),
                         "--workers", "2")
        check("doc/s" in progress and "ETA" in progress,
              "progress is printed along the way")

        top = os.path.join(rot, "eksport_topp")
        run_step(os.path.join(HERE, "vlm_export.py"),
             "--res-csv", res_csv, "--truth-csv", truth_csv,
             "--folder", pdf_dir, "--out-dir", top, "--hit-sample", "0",
             "--from-top", "--full-width", "--max-px", "0")
        m_top = read(os.path.join(top, "manifest.csv"))
        check(all(abs(int(r["utsnitt_bredde"]) - round(PAGE_B * SCALE)) <= 2
                  for r in m_top if r["fil"] != "0100003.pdf"),
              "--from-top + --full-width gives the full page width")
        check(all(float(r["m_y1"]) <= int(r["utsnitt_hoyde"])
                  and float(r["m_y0"]) >= 0 for r in m_top),
              "the marker stays inside the image in full-page crops too")
        check(all(int(r["utsnitt_hoyde"]) > int(r["utsnitt_bredde"]) * 0.4
                  for r in m_top if r["fil"] != "0100003.pdf"),
              "crops reach down from the top, not just around the box")

        # ── Judging: friendly stub (semantics) ───────────────
        print("\n[4] vlm_judge  (friendly stub)")
        # The stub's "model": it knows which crop shows what.
        answers = {}
        for r in manifest:
            with open(os.path.join(ut, "utsnitt", r["utsnitt"]), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            answers[b64] = "ja" if r["klasse"] != "BOM" else "nei"
        srv, url, _ = make_server(bad=False, answers=answers)
        try:
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--concurrent", "2",
                 "--out-csv", os.path.join(ut, "judge_ok.csv"))
        finally:
            srv.shutdown()
        judge = {d["nr"]: d for d in read(os.path.join(ut, "judge_ok.csv"))}
        check(len(judge) == 6, f"6 judgements written ({len(judge)})")
        check(not any(d["feil"] for d in judge.values()),
              "no errors against the friendly stub")
        for r in manifest:
            truth_answer = "ja" if r["klasse"] != "BOM" else "nei"
            check(judge[r["nr"]]["svar"] == truth_answer,
                  f"{r['utsnitt']}: verdict «{judge[r['nr']]['svar']}» "
                  f"as expected")

        # ── Judging: nasty stub, image mode (robustness) ─────
        print("\n[5] vlm_judge  (nasty stub: 500s + prose)")
        srv, url, counter = make_server(bad=True)
        try:
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--concurrent", "1", "--attempt", "1", "--timeout", "20")
            image_csv = os.path.join(ut, "judge_image.csv")
            d1 = read(image_csv)
            check(len(d1) == 6, f"all 6 rows written despite errors ({len(d1)})")
            check(any(r["feil"] for r in d1), "at least one error was logged")
            check(all(r["svar"] in ("ja", "nei", "usikker") for r in d1),
                  "every answer is a valid value")
            check(all(r["svar"] == "usikker" for r in d1
                      if r["feil"].startswith(("HTTPError", "URLError"))),
                  "network errors became «usikker», never «nei»")
            n_error = sum(1 for r in d1 if r["feil"])

            # Resuming: the failed rows must be retried
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--concurrent", "1", "--resume")
            d2 = read(image_csv)
            check(len(d2) == 6 + n_error,
                  f"--resume added exactly the {n_error} failed rows "
                  f"({len(d2) - 6})")
        finally:
            srv.shutdown()

        # ── Thinking: qwen3-vl:8b IS the thinking variant ────
        print("\n[5b] --thinking against a model that wants to think")
        srv, url, counter = make_server(thinks=True)
        try:
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--out-csv", os.path.join(ut, "d_tenk.csv"), "--concurrent", "1")
            check(set(counter["reasoning"]) == {"none"},
                  "reasoning_effort=none is sent by default")
            d = read(os.path.join(ut, "d_tenk.csv"))
            check(all(not r["feil"] for r in d),
                  "thinking was turned off. Real answers, no errors")

            # With --thinking auto the field is omitted and content comes back
            # empty: the error must SAY that thinking was the cause.
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--out-csv", os.path.join(ut, "d_auto.csv"),
                 "--concurrent", "1", "--attempt", "1", "--thinking", "auto",
                 "--max-items", "2")
            d = read(os.path.join(ut, "d_auto.csv"))
            check(all("thought" in r["feil"] for r in d),
                  "empty «content» is diagnosed as thinking, not an anonymous "
                  "error")
            check(all(r["svar"] == "usikker" for r in d),
                  "and becomes «usikker», not «nei»")
        finally:
            srv.shutdown()

        print("\n[5e] --no-file picks out rows")
        srv, url, _ = make_server(bad=False, answers=answers)
        try:
            no_file = os.path.join(ut, "harde.txt")
            with open(no_file, "w", encoding="utf-8") as f:
                f.write("# the hard ones\n2\n5\n")
            path = os.path.join(ut, "d_nr.csv")
            printout = run_step(os.path.join(HERE, "vlm_judge.py"), "--manifest",
                            os.path.join(ut, "manifest.csv"), "--url", url,
                            "--model", "stub", "--out-csv", path, "--concurrent", "1",
                            "--no-file", no_file)
            d = read(path)
            check("--no-file: 2 of 6" in printout and len(d) == 2,
                  "only the two listed rows were judged")
            check({r["nr"] for r in d} == {"2", "5"},
                  "and they were the right rows")
        finally:
            srv.shutdown()

        print("\n[5d] resuming is the default")
        srv, url, _ = make_server(bad=False, answers=answers)
        try:
            path = os.path.join(ut, "d_std.csv")
            run_step(os.path.join(HERE, "vlm_judge.py"), "--manifest",
                 os.path.join(ut, "manifest.csv"), "--url", url, "--model",
                 "stub", "--out-csv", path, "--concurrent", "1")
            check(len(read(path)) == 6, "the first run judges all 6")
            ut2 = run_step(os.path.join(HERE, "vlm_judge.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--url", url,
                       "--model", "stub", "--out-csv", path,
                       "--concurrent", "1")
            check("Nothing to do" in ut2 and len(read(path)) == 6,
                  "a second run without flags does NOTHING. Finished work is "
                  "not overwritten")
            ut3 = run_step(os.path.join(HERE, "vlm_judge.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--url", url,
                       "--model", "stub", "--out-csv", path,
                       "--concurrent", "1", "--restart")
            check("overwriting" in ut3 and len(read(path)) == 6,
                  "--restart overwrites, and says that it does")
        finally:
            srv.shutdown()

        print("\n[5c] endpoint that does not know reasoning_effort")
        srv, url, counter = make_server(reject_reasoning=True)
        try:
            run_step(os.path.join(HERE, "vlm_judge.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--model", "stub", "--out-csv", os.path.join(ut, "d_400.csv"), "--concurrent", "1")
            d = read(os.path.join(ut, "d_400.csv"))
            check(counter["avvist"] == 1,
                  f"only the FIRST call was rejected ({counter['avvist']}), "
                  f"the field is dropped for the rest")
            check(len(d) == 6 and all(not r["feil"] for r in d),
                  "all 6 judgements came through after the fallback")
        finally:
            srv.shutdown()

        # ── Evaluation ───────────────────────────────────────
        print("\n[6] vlm_evaluate")
        out_text = run_step(os.path.join(HERE, "vlm_evaluate.py"),
                        "--manifest", os.path.join(ut, "manifest.csv"),
                        "--judge", os.path.join(ut, "judge_ok.csv"),
                        "--out-dir", os.path.join(ut, "evaluation"))
        with open(os.path.join(ut, "evaluation", "oppsummering.json"),
                  encoding="utf-8") as f:
            up = json.load(f)
        check(up["n_bom_nei"] == 4, f"4 BOM got nei ({up['n_bom_nei']})")
        check(up["n_dek_nei"] == 0, f"0 covering got nei ({up['n_dek_nei']})")
        check(abs(up["gain"] - 4.0) < 1e-6,
              f"gain scaled to 4 ({up['gain']})")
        # Partial run: if the loss is scaled up, the gain must be scaled too.
        half = os.path.join(ut, "d_halv.csv")
        rows_every = read(os.path.join(ut, "judge_ok.csv"))
        with open(half, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows_every[0].keys())
            w.writeheader()
            w.writerows([r for r in rows_every
                         if r["klasse"] == "BOM"][:2]
                        + [r for r in rows_every if r["klasse"] != "BOM"][:1])
        out_half = run_step(os.path.join(HERE, "vlm_evaluate.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--judge", half,
                       "--out-dir", os.path.join(ut, "ev_halv"))
        check("BOM × 2.00" in out_half,
              "BOM scales 4/2 = 2.00 in a partial run")
        check("covering × 3.00" in out_half,
              "covering scales 3/1 = 3.00, not the export factor 1.5")
        check(abs(up["loss_upper"] - 3.0 * 1.5) < 0.05,
              f"upper loss bound = 3 × factor 1.5 = 4.5 ({up['loss_upper']:.2f})")
        # Zero losses observed, so the point estimate is infinite while the
        # upper bound is not: two boxes prove nothing.
        check("∞ at face value" in out_text,
              "zero loss in the sample shows as an infinite face value, and "
              "the 95 % bound stands next to it")
        lost = read(os.path.join(ut, "evaluation", "lost.csv"))
        gain = read(os.path.join(ut, "evaluation", "gain.csv"))
        check(len(lost) == 0 and len(gain) == 4,
              "lost.csv empty, gain.csv has 4 rows")
        check("label_id" in read(os.path.join(ut, "evaluation",
                                             "gain.csv"))[0],
              "the manifests have a label_id column (same format as "
              "filter_review)")

        # The model may say nei all it likes: with eleven digits and a valid
        # date in its OWN transcription, the code overrules it.
        false = os.path.join(ut, "d_overstyr.csv")
        with open(false, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "nr", "svar", "tall", "begrunnelse", "linjen",
                "sifre_paa_linjen", "dato_gyldig", "holdepunkt", "feil"])
            w.writeheader()
            for r in manifest:
                fnr = r["klasse"] != "BOM"
                w.writerow({
                    "nr": r["nr"], "svar": "nei",
                    "tall": "", "begrunnelse": "claimed org.nr",
                    "linjen": "Kari Nordmann 010190 00000" if fnr else "N 6626630.58",
                    "sifre_paa_linjen": "01019000000" if fnr else "662663058",
                    "dato_gyldig": "true" if fnr else "false",
                    "holdepunkt": "", "feil": ""})
        out_ov = run_step(os.path.join(HERE, "vlm_evaluate.py"), "--manifest",
                     os.path.join(ut, "manifest.csv"), "--judge", false,
                     "--out-dir", os.path.join(ut, "ev_ov"),
                     "--fnr-override")
        # The guard must also fire when only PADDLE read the date half, from
        # the manifest's ocr_linje, without help from the model.
        only_ocr = os.path.join(ut, "d_kunocr.csv")
        with open(only_ocr, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["nr", "svar", "tall",
                                              "begrunnelse", "feil"])
            w.writeheader()
            for r in manifest:
                w.writerow({"nr": r["nr"], "svar": "nei",
                            "tall": "", "begrunnelse": "", "feil": ""})
        out_ocr = run_step(os.path.join(HERE, "vlm_evaluate.py"), "--manifest",
                      os.path.join(ut, "manifest.csv"), "--judge", only_ocr,
                      "--out-dir", os.path.join(ut, "ev_kunocr"),
                      "--fnr-override")
        check("--fnr-override: 2 verdicts changed" in out_ocr,
              "the manifest OCR line alone protects the covering boxes")

        check("--fnr-override: 2 verdicts changed" in out_ov,
              "the override saves both covering boxes")
        with open(os.path.join(ut, "ev_ov", "oppsummering.json"),
                  encoding="utf-8") as f:
            o = json.load(f)
        check(o["n_dek_nei"] == 0 and o["n_bom_nei"] == 4,
              f"zero loss, gain untouched ({o['n_dek_nei']}/{o['n_bom_nei']})")

        # Judge only the BOM rows and the tool must say the scale-up does not
        # apply, rather than quietly getting it wrong.
        only_miss = os.path.join(ut, "bare_bom.txt")
        with open(only_miss, "w", encoding="utf-8") as f:
            f.write("\n".join(r["nr"] for r in manifest
                               if r["klasse"] == "BOM"))
        srv, url, _ = make_server(bad=False, answers=answers)
        try:
            run_step(os.path.join(HERE, "vlm_judge.py"), "--manifest",
                 os.path.join(ut, "manifest.csv"), "--url", url, "--model",
                 "stub", "--concurrent", "1",
                 "--no-file", only_miss,
                 "--out-csv", os.path.join(ut, "d_skjev.csv"))
        finally:
            srv.shutdown()
        out_skewed = run_step(os.path.join(HERE, "vlm_evaluate.py"), "--manifest",
                        os.path.join(ut, "manifest.csv"), "--judge",
                        os.path.join(ut, "d_skjev.csv"), "--out-dir",
                        os.path.join(ut, "ev_skjev"))
        check("WARNING" in out_skewed and "does not look random" in out_skewed,
              "a skewed sample triggers the warning")
        check("NB: the result above rests" in out_skewed,
              "and the warning is repeated at the result")
        check("WARNING" not in out_text,
              "a complete sample gives NO warning")

        # Rule baseline: same accounts, no model. On the synthetic data only
        # the fnr lines have a valid 11-digit run.
        out_rule = run_step(os.path.join(HERE, "vlm_evaluate.py"),
                        "--manifest", os.path.join(ut, "manifest.csv"),
                        "--judge", "regel:fnr-kandidat",
                        "--out-dir", os.path.join(ut, "ev_regel"))
        check("Baseline" in out_rule, "the regel: baseline runs without a model")
        with open(os.path.join(ut, "ev_regel", "oppsummering.json"),
                  encoding="utf-8") as f:
            o = json.load(f)
        check(o["n_bom_nei"] == 4 and o["n_dek_nei"] == 0,
              f"the fnr-kandidat rule takes 4 BOM and zero covering "
              f"({o['n_bom_nei']}/{o['n_dek_nei']})")

        # har_tokens=1 on every synthetic row, so the subset is the whole set;
        # a condition that matches nothing must say so.
        out_split = run_step(os.path.join(HERE, "vlm_evaluate.py"),
                      "--manifest", os.path.join(ut, "manifest.csv"),
                      "--judge", os.path.join(ut, "judge_ok.csv"),
                      "--out-dir", os.path.join(ut, "ev_del"),
                      "--split-by", "kilde", "--only", "har_tokens=1")
        check("SPLIT BY kilde" in out_split and "Subset" in out_split,
              "--split-by and --only run together")
        check("6 of 6 rows" in out_split, "har_tokens=1 keeps every row")
        out_empty = run_step(os.path.join(HERE, "vlm_evaluate.py"),
                      "--manifest", os.path.join(ut, "manifest.csv"),
                      "--judge", os.path.join(ut, "judge_ok.csv"),
                      "--out-dir", os.path.join(ut, "ev_tom"),
                      "--only", "kilde=finnesikke")
        check("No rows left" in out_empty,
              "an empty condition says so instead of counting on nothing")

        # With --uncertain-remover the image arm's usikre must show as loss
        run_step(os.path.join(HERE, "vlm_evaluate.py"),
             "--manifest", os.path.join(ut, "manifest.csv"),
             "--judge", os.path.join(ut, "judge_image.csv"),
             "--out-dir", os.path.join(ut, "evaluering_bilde"),
             "--uncertain-remover")
        check(os.path.isfile(os.path.join(ut, "evaluering_bilde", "lost.csv")),
              "--uncertain-remover runs and writes lost.csv")

        # ── The verifier in the pipeline ─────────────────────
        # vlm_judge above is the offline tool. This is the same prompt and
        # the same parsing, but inside run_model_on_pdf_bytes, where a wrong
        # «nei» removes a real sladd.
        print("\n[7] vlm_verifier in the pipeline")
        check_verifier(rot)

        print("\nALT OK.")
        if keep:
            print(f"The files are in {rot}")
        return rot
    finally:
        if not keep:
            shutil.rmtree(rot, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--keep", action="store_true",
                   help="Do not delete the work directory afterwards")
    main(p.parse_args().keep)
