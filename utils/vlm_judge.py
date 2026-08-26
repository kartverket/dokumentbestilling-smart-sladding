"""Sends the crops from vlm_export to a local VLM and stores the judgements.

Step 2 of the VLM verifier pilot. Speaks OpenAI-compatible
/v1/chat/completions, which vLLM, llama.cpp-server, LM Studio and Ollama all
offer — switch model with --url/--model, not by editing the code.

Resuming is the DEFAULT: finished rows in an existing judgement CSV are
skipped and only the failed ones retried. Use --restart to judge everything
again.

Judgements are also cached on disk (see vlm_cache.py): with the same prompt,
config and model, a box already judged anywhere is answered from the cache
instead of the GPU — across runs, manifests and export names. The prompt
version (the cache fingerprint) is printed at startup. --no-cache turns the
cache off; --restart alone rewrites the CSV but still reuses cached answers.

The model judges the crop image alone. Feeding it the OCR text as well made
the judgements worse, so the old tekst and begge modes are gone; the
manifest's OCR is still used after the fact, by vlm_evaluate --fnr-override.

Anything that cannot be interpreted — timeout, HTTP error, unparsable answer —
is logged in the «feil» column and becomes «usikker», NEVER «nei»: «nei» is
the answer that costs recall.

Run:
    python utils/vlm_judge.py \
        --manifest /data2/vlm/uttrekk6_kalibrering/manifest.csv \
        --url http://localhost:8000/v1 \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --concurrent 4
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import vlm_cache
from filter_common import reclassify_invalid_covering

STD_URL = "http://localhost:8000/v1"
STD_TIMEOUT = 120
# Measured answers land on 56-76 tokens, so this is roughly double the
# longest one seen. A truncated answer is unparsable and becomes «usikker»,
# never «nei», so the cap costs recall nothing when it does bite.
STD_MAX_TOKENS = 150

# Set to None if the endpoint rejects the field — shared across the threads.
_THINKING = {"value": "none"}
# WITHOUT_CONTENT_FIELD plus the sensitive and technical columns. The model's
# checklist is stored so vlm_evaluate can re-apply the fnr rule without a GPU.
OUT_FIELD = ["utsnitt", "riktig", "klasse", "svar", "begrunnelse",
           "dato_gyldig", "holdepunkt", "sekunder", "label_id", "nr", "kilde",
           "tall", "linjen", "sifre_paa_linjen", "feil", "raatekst"]

# The judgement CSV above transcribes REAL fødselsnumre and is itself
# sensitive; this sister file holds only the verdicts and can be shared.
WITHOUT_CONTENT_FIELD = ["utsnitt", "riktig", "klasse", "svar",
                     "begrunnelse", "dato_gyldig", "holdepunkt", "sekunder",
                     "label_id", "nr", "kilde"]


def _correct(klasse, answer):
    wanted = "nei" if klasse == "BOM" else "ja"
    if answer not in ("ja", "nei"):
        return "⚪"
    if answer == wanted:
        return "✅"
    # 🟡 (ja on a BOM) only leaves a needless sladd standing; ❌ (nei on a
    # covering box) would unmask a real fnr — those are what review looks for.
    return "🟡" if klasse == "BOM" else "❌"


def _without_content_row(row):
    row = dict(row)
    row["utsnitt"] = os.path.splitext(
        os.path.basename(row.get("utsnitt", "")))[0]
    row["riktig"] = _correct(row.get("klasse", ""),
                            (row.get("svar") or "").strip().lower())
    return row


def _md_row(row, crop_dir, md_dir):
    """One review line. The link is relative so VSCode resolves it over SSH
    and it survives an rsync."""
    file = row.get("utsnitt", "")
    rel = os.path.relpath(os.path.join(crop_dir, file), md_dir)
    correct = row.get("riktig") or _correct(
        row.get("klasse", ""), (row.get("svar") or "").strip().lower())
    cells = [f"[{os.path.splitext(file)[0]}]({rel})", correct,
              row.get("klasse", ""), row.get("svar", ""),
              row.get("holdepunkt", ""),
              row.get("begrunnelse", "")]
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |\n"


def _page_file(out_path, name, flat_suffix):
    """Path to a derived file: canonical names in the per-run directory
    layout, stem+suffix in the flat one, so old runs stay untouched."""
    folder = os.path.dirname(os.path.abspath(out_path))
    if os.path.basename(out_path) == "full_info.csv":
        return os.path.join(folder, name)
    stem, _ = os.path.splitext(out_path)
    return stem + flat_suffix


def _md_label_row(row, crop_dir, md_dir):
    """One line for the label review — only covering boxes judged «nei» (❌).
    Everything else returns an empty string and stays out of the file."""
    correct = row.get("riktig") or _correct(
        row.get("klasse", ""), (row.get("svar") or "").strip().lower())
    if correct != "❌":
        return ""
    file = row.get("utsnitt", "")
    rel = os.path.relpath(os.path.join(crop_dir, file), md_dir)
    cells = [f"[{os.path.splitext(file)[0]}]({rel})", correct,
              row.get("klasse", ""), row.get("svar", ""),
              row.get("label_id", ""), row.get("holdepunkt", ""),
              row.get("begrunnelse", "")]
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |\n"


def write_review_label_md(out_path, crop_dir):
    """Derives the label review: only the rows where the model says «nei» to a
    box fasit covers. Each row carries label_id, so a verdict that turns out
    to be right goes straight into fasit maintenance. Returns the path."""
    path = _page_file(out_path, "gjennomgang_label.md", "_gjennomgang_label.md")
    md_dir = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(os.path.dirname(os.path.abspath(out_path))
                            if os.path.basename(out_path) == "full_info.csv"
                            else os.path.splitext(out_path)[0])
    with open(out_path, newline="", encoding="utf-8") as f_inn, \
         open(path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# Disagreements with fasit: {name}\n\n"
                   f"The model says «nei» where fasit has a covered box — "
                   f"either it is wrong (expensive in prod), or the label is "
                   f"noise (label_id goes into fasit maintenance).\n\n"
                   f"| utsnitt | riktig | klasse | svar | label_id "
                   f"| holdepunkt | begrunnelse |\n"
                   f"|---|---|---|---|---|---|---|\n")
        for row in csv.DictReader(f_inn):
            f_out.write(_md_label_row(row, crop_dir, md_dir))
    return path


def write_review_md(out_path, crop_dir):
    """Derives the review markdown from the judgement CSV. Returns the path."""
    path = _page_file(out_path, "gjennomgang.md", "_gjennomgang.md")
    md_dir = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(os.path.dirname(os.path.abspath(out_path))
                            if os.path.basename(out_path) == "full_info.csv"
                            else os.path.splitext(out_path)[0])
    with open(out_path, newline="", encoding="utf-8") as f_inn, \
         open(path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# Review: {name}\n\n"
                   f"❌ marks the rows that threaten recall.\n\n"
                   f"| utsnitt | riktig | klasse | svar "
                   f"| holdepunkt | begrunnelse |\n"
                   f"|---|---|---|---|---|---|\n")
        for row in csv.DictReader(f_inn):
            f_out.write(_md_row(row, crop_dir, md_dir))
    return path


def write_without_content(out_path):
    """Derives the content-free CSV from the judgement CSV. Returns the path."""
    path = _page_file(out_path, "uten_innhold.csv", "_uten_innhold.csv")
    with open(out_path, newline="", encoding="utf-8") as f_inn, \
         open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=WITHOUT_CONTENT_FIELD,
                                 extrasaction="ignore")
        writer.writeheader()
        for row in csv.DictReader(f_inn):
            writer.writerow(_without_content_row(row))
    return path

# The prompt is the real experiment of the pilot: it is built around the
# contrasts both YOLO and the rules fall for, so treat edits as experiments.
STD_PROMPT = """\
Du ser et utsnitt fra et skannet norsk tinglysingsdokument. Den røde rammen markerer et område som er foreslått sladdet.

Norske fødselsnumre har elleve sifre: fødselsdato (DDMMÅÅ) fulgt av fem sifre personnummer. Sladdingen skal som regel bare dekke de fem siste sifrene, så rammen inneholder ofte bare en bit av nummeret. Datoen kan stå foran på linjen eller på linjen over.

Spørsmålet: berører rammen et fødselsnummer, helt eller delvis?

- Svar «ja» når tallet er eller sannsynligvis er et fødselsnummer.
- Svar «nei» BARE når du tydelig ser at tallet er noe annet: kontonummer, organisasjonsnummer, koordinat, beløp, dato alene, matrikkel-/saks-/dokumentnummer, og skriv hva det er i «holdepunkt».
- Å si nei på et fødselsnummer er 100 ganger verre enn å si ja på et annet tall. Når du er i tvil, svar «ja» og skriv hvorfor i begrunnelsen.

Svar kun med JSON:
{"tall": "tallene du ser i og rundt rammen", "holdepunkt": "hva tallet er, hvis det er noe annet enn et fødselsnummer — ellers tom", "svar": "ja", "begrunnelse": "maks 10 ord"}\
"""


# ── Answer parsing ────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_ANSWER_RE = re.compile(r"\b(ja|nei|usikker)\b", re.I)


def _checklist(d):
    """The checklist fields as text. Unknown or missing become empty."""
    ut = {}
    for field in ("linjen", "sifre_paa_linjen", "dato_gyldig", "holdepunkt"):
        v = d.get(field)
        ut[field] = "" if v is None else str(v).strip().replace("\n", " ")[:300]
    return ut


def parse_answer(text):
    """Raw model answer -> (svar, tall, begrunnelse, feil, checklist).

    Strict to lenient: plain JSON, then the first JSON object in the text
    (models like to wrap it in ```json), then a keyword search. Anything left
    over becomes «usikker», never «nei».
    """
    if not text or not text.strip():
        return "usikker", "", "", "empty answer", {}
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", clean).strip()

    candidates = [clean]
    m = _JSON_RE.search(clean)
    if m:
        candidates.append(m.group(0))
    for kand in candidates:
        try:
            d = json.loads(kand)
        except (ValueError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        answer = str(d.get("svar", "")).strip().lower()
        if answer in ("ja", "nei", "usikker"):
            return (answer, str(d.get("tall", "")).strip(),
                    str(d.get("begrunnelse", "")).strip(), "", _checklist(d))
        missing = "svar" not in d or not str(d.get("svar", "")).strip()
        return ("usikker", str(d.get("tall", "")).strip(),
                str(d.get("begrunnelse", "")).strip(),
                "the model omitted the «svar» field — the rest of the JSON came"
                if missing else f"unknown answer {answer!r}", _checklist(d))

    m = _ANSWER_RE.search(clean)
    if m:
        return (m.group(1).lower(), "", clean[:120],
                "not JSON, read by keyword", {})
    return "usikker", "", clean[:120], "unparsable answer", {}


# ── Calls ─────────────────────────────────────────────────────

def _build_melding(prompt, image_b64):
    """One chat message for one crop, from the inputs judge_one loaded."""
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]}]


def call_model(url, model, messages, api_key=None, timeout=STD_TIMEOUT,
                temperature=0.0, max_tokens=STD_MAX_TOKENS, thinking="none"):
    """One call. thinking=None omits reasoning_effort entirely.

    Ollama turns thinking ON by itself when the field is missing, and then the
    whole token budget goes to an inner monologue while «content» stays empty.
    """
    body = {"model": model, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens}
    if thinking:
        body["reasoning_effort"] = thinking
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key or 'none'}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as answer:
            d = json.loads(answer.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # urllib throws away the response body, which is exactly where the
        # endpoint explains what it did not like.
        try:
            explanation = e.read().decode("utf-8", "replace").strip()[:300]
        except Exception:
            explanation = ""
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason} — {explanation}" if explanation
            else str(e.reason), e.headers, None)
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    if not content.strip():
        # Empty content with reasoning beside it means thinking ate the answer.
        for field in ("reasoning", "reasoning_content"):
            if (m.get(field) or "").strip():
                raise ValueError(
                    "empty «content» — the model thought instead of "
                    "answering. Qwen3-VL ships as two checkpoints: on a "
                    "THINKING one (Ollama tag «qwen3-vl:8b») --thinking none "
                    "does not help, the thinking is trained in. Use "
                    "«qwen3-vl:8b-instruct», or raise --max-tokens to let it "
                    "finish thinking (expensive)")
    return content


def judge_one(row, a, folder, prompt, cache_dir=None):
    """Judges one box. Never raises — errors land in the «feil» column.

    With cache_dir set, a box already judged with the same prompt, config
    and model is answered from the cache, and a fresh answer that parses is
    written back. The extra «_cache» key marks a hit and is dropped before
    the CSV.
    """
    t0 = time.monotonic()

    try:
        with open(os.path.join(folder, row["utsnitt"]), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        return {"svar": "usikker", "tall": "",
                "begrunnelse": "",
                "sekunder": round(time.monotonic() - t0, 2),
                "feil": f"{type(e).__name__}: {e}"[:200], "raatekst": ""}

    key = vlm_cache.item_key(image_b64, None) if cache_dir else None
    if key:
        cached = vlm_cache.read_cache(cache_dir, key)
        if cached is not None:
            answer, number, rationale, parse_error, check = \
                parse_answer(cached)
            if not parse_error:
                return {"svar": answer,
                        "tall": number, "begrunnelse": rationale, **check,
                        "sekunder": round(time.monotonic() - t0, 2),
                        "feil": "", "raatekst": "", "_cache": True}

    error = ""
    raw = ""
    urler = a.url if isinstance(a.url, list) else [a.url]
    try:
        i_url = int(row.get("nr", 0)) % len(urler)
    except (TypeError, ValueError):
        i_url = 0
    for attempt in range(1, a.attempt + 1):
        try:
            messages = _build_melding(prompt, image_b64)
            # A retry goes to the next backend, so one dead instance does not
            # cost the row.
            url = urler[(i_url + attempt - 1) % len(urler)]
            raw = call_model(url, a.model, messages,
                              api_key=a.api_key, timeout=a.timeout,
                              temperature=a.temperature,
                              max_tokens=a.max_tokens,
                              thinking=_THINKING["value"])
            error = ""
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError, TimeoutError) as e:
            error = f"{type(e).__name__}: {e}"[:200]
            # Older endpoints answer 400 to reasoning_effort. Only when the
            # endpoint names the field: any other 400 has another cause.
            if (isinstance(e, urllib.error.HTTPError) and e.code == 400
                    and _THINKING["value"]
                    and "reasoning" in str(e).lower()):
                _THINKING["value"] = None
                print("  ⚠ Endpoint rejected reasoning_effort — "
                      "continuing without it", flush=True)
                continue
            if attempt < a.attempt:
                time.sleep(min(2 ** attempt, 10))
    sec = time.monotonic() - t0

    if error:
        return {"svar": "usikker", "tall": "",
                "begrunnelse": "", "sekunder": round(sec, 2), "feil": error,
                "raatekst": ""}
    answer, number, rationale, parse_error, check = parse_answer(raw)
    if key and not parse_error:
        vlm_cache.write_cache(cache_dir, key, row.get("utsnitt", ""), raw, sec)
    return {"svar": answer, "tall": number,
            "begrunnelse": rationale, **check,
            "sekunder": round(sec, 2), "feil": parse_error,
            "raatekst": raw.replace("\n", " ")[:400] if parse_error else ""}


# ── Run ───────────────────────────────────────────────────────

def _sync_klasse(out_path, klasse_per_nr):
    """Re-labels stored judgement rows whose klasse disagrees with the
    manifest — i.e. labels added to ugyldige_labels.txt after the rows were
    judged. The verdicts stand; only klasse and «riktig» change, so the
    derived review files stop flagging known noise as recall threats.
    Returns the number of rows changed."""
    if not os.path.isfile(out_path):
        return 0
    with open(out_path, newline="", encoding="utf-8-sig") as f:
        leser = csv.DictReader(f)
        old_rows = list(leser)
        field = leser.fieldnames
    changed = 0
    for r in old_rows:
        want = klasse_per_nr.get(r.get("nr"))
        if want and r.get("klasse") != want:
            r["klasse"] = want
            r["riktig"] = _correct(want, (r.get("svar") or "").strip().lower())
            changed += 1
    if changed:
        tmp = out_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=field, extrasaction="ignore")
            w.writeheader()
            w.writerows(old_rows)
        os.replace(tmp, out_path)
    return changed


def resolve_sources(a):
    """Fills --res-csv/--truth-csv/--pdf-dir from utvalg.json.

    vlm_export already writes the three absolute paths next to the manifest,
    so --missing-candidates normally needs no arguments of its own. The flags
    stay as overrides for a manifest whose sources have since moved.
    Returns the names still unresolved.
    """
    sidecar = os.path.join(os.path.dirname(os.path.abspath(a.manifest)),
                           "utvalg.json")
    known = {}
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as f:
                d = json.load(f)
            known = {"res_csv": d.get("res_csv"),
                     "truth_csv": d.get("fasit_csv"),
                     "pdf_dir": d.get("folder")}
        except (ValueError, OSError):
            pass
    missing = []
    for flag in ("res_csv", "truth_csv", "pdf_dir"):
        if not getattr(a, flag):
            setattr(a, flag, known.get(flag))
        if not getattr(a, flag):
            missing.append("--" + flag.replace("_", "-"))
    return missing


def write_missing_candidates(out_path, rows, a):
    """Pages for BOM boxes the model called a fødselsnummer.

    A BOM box covers no fasit label. When the model says «ja» anyway, either
    it is wrong or the labelling missed a number, and only a human can tell
    the two apart. The page is drawn the way valider_full draws its errors:
    every prediction framed and numbered, every fasit label at half opacity,
    so the «#N» in the corner goes straight into note_missing_label.py.

    The index deliberately holds no transcription. The number is on the page
    already, and this file does not need to be another copy of it.
    """
    from csv_export import read_result_csv
    from evaluation import read_truth_xywh
    from visualization import draw_and_save

    verdict = {}
    with open(out_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            verdict[r["nr"]] = (r.get("svar") or "").strip().lower()

    hits = [r for r in rows
            if r.get("klasse") == "BOM" and verdict.get(r["nr"]) == "ja"]
    if not hits:
        print("  --missing-candidates: no BOM box was judged «ja».")
        return None

    out_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                           "manglende_kandidater")
    boxes = read_result_csv(a.res_csv)
    truth = read_truth_xywh(a.truth_csv) or {}
    pages = {(r["fil"], int(r["side"])) for r in hits}
    doc_pages = {(int(r["doc_no"]), int(r["side"])) for r in hits}
    on_page = {k: v for k, v in boxes.items() if k in pages}

    # Only the boxes judged «ja» get the oversladd colour, so the candidate
    # stands out from the other predictions on the same page.
    flagged = {}
    for r in hits:
        key = (r["fil"], int(r["side"]))
        bw, bh, _ = on_page.get(key, (0, 0, []))
        flagged.setdefault(key, (bw, bh, []))[2].append(
            tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")))

    # miss_indices=set() puts every label in the faint colour: this is not a
    # recall review, the fasit is here as context for the box in question.
    draw_and_save(on_page, {k: v for k, v in truth.items() if k in doc_pages},
                  a.pdf_dir, out_dir, write_log=False, clean=False,
                  sources=on_page, oversladd_boxes=flagged,
                  miss_indices=set())

    def index_of(row):
        _, _, page_boxes = on_page.get((row["fil"], int(row["side"])),
                                       (0, 0, []))
        want = tuple(round(float(row[k]), 1) for k in ("x0", "y0", "x1", "y1"))
        for i, b in enumerate(page_boxes):
            if tuple(round(v, 1) for v in b[:4]) == want:
                return i
        return None

    lines = ["# BOM judged «ja» — candidates for a missing label", "",
             f"{len(hits)} box(es). Open the PNG, check the frame with the "
             f"matching number, and run the command under it if the fasit "
             f"really did miss a fødselsnummer.", ""]
    unmatched = 0
    for r in sorted(hits, key=lambda r: (r["fil"], int(r["side"]))):
        i = index_of(r)
        png = f"{os.path.splitext(r['fil'])[0]}_side{int(r['side'])}.png"
        if i is None:
            unmatched += 1
            lines += [f"## {png}", "", "Box not found in the result CSV — "
                      "wrong --res-csv for this manifest?", ""]
            continue
        lines += [f"## {png}  box #{i}  ({r.get('kilde', '')})", "",
                  f"![]({os.path.join('.', png)})", "", "```bash",
                  f"python utils/note_missing_label.py --png "
                  f"{os.path.join(out_dir, png)} --box {i} "
                  f"--res-csv {a.res_csv} --truth-csv {a.truth_csv}",
                  "```", ""]

    index = os.path.join(out_dir, "kandidater.md")
    with open(index, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  --missing-candidates: {len(hits)} box(es) in {out_dir}")
    if unmatched:
        print(f"    ⚠ {unmatched} could not be matched against --res-csv")
    return index


def run(a):
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    reclassified = reclassify_invalid_covering(rows)
    if reclassified:
        print(f"  {reclassified} covering rows treated as BOM — their labels "
              f"are listed in ugyldige_labels.txt")
    folder = a.crop_dir or os.path.join(
        os.path.dirname(os.path.abspath(a.manifest)), "utsnitt")
    prompt = STD_PROMPT
    if a.prompt_file:
        with open(a.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()

    cache_dir = None
    if a.cache:
        fp = vlm_cache.fingerprint(prompt, "", a.model, "image",
                                   a.temperature, a.max_tokens,
                                   _THINKING["value"])
        cache_dir = os.path.join(a.cache, fp)
        vlm_cache.write_meta(cache_dir, {
            "model": a.model, "mode": "image", "temperature": a.temperature,
            "max_tokens": a.max_tokens, "thinking": _THINKING["value"],
            "prompt_file": a.prompt_file or "(built-in)",
            "templates": "", "prompt": prompt})
        print(f"  Prompt version {fp}  (cache: {cache_dir})")

    out_path = a.out_csv or os.path.join(os.path.dirname(
        os.path.abspath(a.manifest)), "judge_image.csv")
    if not out_path.lower().endswith(".csv"):
        # One directory per run: all four files under the run name.
        kat = out_path
        os.makedirs(kat, exist_ok=True)
        out_path = os.path.join(kat, "full_info.csv")
        old = kat + ".csv"
        if not os.path.isfile(out_path) and os.path.isfile(old):
            os.replace(old, out_path)
            print(f"  Moved {old} -> {out_path} (directory layout)")
    if a.resume:
        synced = _sync_klasse(out_path,
                              {r["nr"]: r.get("klasse", "") for r in rows})
        if synced:
            print(f"  {synced} stored rows re-labelled from the manifest "
                  f"(ugyldige_labels.txt grew since they were judged)")
    done = set()
    if not a.resume and os.path.isfile(out_path):
        print(f"  ⚠ --restart: overwriting {out_path}")
    if a.resume and os.path.isfile(out_path):
        with open(out_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                # Failed rows are judged again — that is the point of resuming.
                if r.get("svar") and not r.get("feil"):
                    done.add(r["nr"])
        print(f"  Resuming: {len(done)} rows already judged")

    left = [r for r in rows if r["nr"] not in done]
    if a.no_file:
        with open(a.no_file, encoding="utf-8-sig") as f:
            selected = {lin.strip() for lin in f
                      if lin.strip() and not lin.startswith("#")}
        before = len(left)
        left = [r for r in left if r["nr"] in selected]
        print(f"  --no-file: {len(left)} of {before} rows selected "
              f"({len(selected)} nr in the file)")
    if a.max_items:
        left = left[:a.max_items]
    if not left:
        print("  Nothing to do.")
        if os.path.isfile(out_path):
            print(f"  Without content: {write_without_content(out_path)}")
            print(f"  Review:          {write_review_md(out_path, folder)}")
            print(f"  With label-id:   "
                  f"{write_review_label_md(out_path, folder)}")
        return out_path

    print(f"  {len(left)} boxes to judge  "
          f"({a.concurrent} concurrent, model {a.model})")
    new_file = not (a.resume and os.path.isfile(out_path))
    if not new_file:
        with open(out_path, newline="", encoding="utf-8-sig") as f:
            leser = csv.DictReader(f)
            old_rows = list(leser) if leser.fieldnames != OUT_FIELD else None
        if old_rows is not None:
            print(f"  Column layout changed — rewriting {out_path} in the "
                  f"new order ({len(old_rows)} rows)")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=OUT_FIELD,
                                   extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    r.setdefault("riktig", _correct(
                        r.get("klasse", ""),
                        (r.get("svar") or "").strip().lower()))
                    w.writerow({k: r.get(k, "") for k in OUT_FIELD})
    f_out = open(out_path, "w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=OUT_FIELD, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        f_out.flush()
    # The derived files are written as the batch runs, so review can start
    # before it finishes; deriving them first catches up resumed rows.
    ui_path = write_without_content(out_path)
    md_path = write_review_md(out_path, folder)
    f_md = open(md_path, "a", encoding="utf-8")
    md_dir = os.path.dirname(os.path.abspath(md_path))
    model_path = write_review_label_md(out_path, folder)
    f_model = open(model_path, "a", encoding="utf-8")
    f_ui = open(ui_path, "a", newline="", encoding="utf-8")
    writer_ui = csv.DictWriter(f_ui, fieldnames=WITHOUT_CONTENT_FIELD,
                                extrasaction="ignore")
    laas = threading.Lock()
    tally = {"n": 0, "feil": 0, "cache": 0, "judged": 0}
    timings = []
    t_start = time.monotonic()
    # Cache hits are free, so the rate is counted over the judged boxes only.
    mark = {"t": t_start, "judged": 0}

    def work(row):
        res = judge_one(row, a, folder, prompt, cache_dir)
        from_cache = res.pop("_cache", False)
        row_out = {k: row.get(k, "") for k in
                  ("nr", "utsnitt", "klasse", "kilde", "label_id")}
        row_out.update(res)
        row_out["riktig"] = _correct(row_out.get("klasse", ""),
                                   (row_out.get("svar") or "").strip().lower())
        with laas:
            writer.writerow(row_out)
            f_out.flush()
            writer_ui.writerow(_without_content_row(row_out))
            f_ui.flush()
            f_md.write(_md_row(row_out, folder, md_dir))
            f_md.flush()
            f_model.write(_md_label_row(row_out, folder, md_dir))
            f_model.flush()
            tally["n"] += 1
            if res["feil"]:
                tally["feil"] += 1
            if from_cache:
                tally["cache"] += 1
            else:
                tally["judged"] += 1
                timings.append(res["sekunder"])
            if tally["n"] % 25 == 0 or tally["n"] == len(left):
                now = time.monotonic()
                gone = now - t_start
                block = tally["judged"] - mark["judged"]
                rate = (f"{(now - mark['t']) / block:5.2f}" if block
                        else "    -")
                print(f"    {tally['n']:>6}/{len(left)}  "
                      f"{gone:6.0f}s  {rate} s/box  "
                      f"{tally['feil']} errors  "
                      f"{tally['cache']} from cache", flush=True)
                mark["t"], mark["judged"] = now, tally["judged"]

    try:
        with ThreadPoolExecutor(max_workers=a.concurrent) as pool:
            list(pool.map(work, left))
    finally:
        f_out.close()
        f_ui.close()
        f_md.close()
        f_model.close()

    gone = time.monotonic() - t_start
    timings.sort()
    if tally["judged"]:
        print(f"\n  Done: {tally['n']} judgements in {gone:.0f}s "
              f"({gone / tally['judged']:.2f} s/box over the {tally['judged']} "
              f"judged, {a.concurrent} concurrent)")
    else:
        print(f"\n  Done: {tally['n']} judgements in {gone:.0f}s "
              f"(everything came from the cache)")
    if cache_dir:
        print(f"  Cache: {tally['cache']} of {tally['n']} answers reused "
              f"({cache_dir})")
    if a.missing_candidates:
        write_missing_candidates(out_path, rows, a)
    if timings:
        print(f"  Latency per call: median {timings[len(timings) // 2]:.2f}s, "
              f"p90 {timings[int(len(timings) * 0.9)]:.2f}s, "
              f"max {timings[-1]:.2f}s")
    if tally["feil"]:
        print(f"  ⚠ {tally['feil']} rows with errors/unparsable answers "
              f"— all counted as «usikker». Run again with --resume.")
    print(f"  Judgements:      {out_path}")
    print(f"  Without content: {write_without_content(out_path)}")
    print(f"  Review:          {write_review_md(out_path, folder)}")
    print(f"  With label-id:   {write_review_label_md(out_path, folder)}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Judges crops from vlm_export with a local, "
                    "OpenAI-compatible VLM (vLLM / llama.cpp / Ollama).")
    p.add_argument("--manifest", default=None,
                   help="manifest.csv from vlm_export (required)")
    p.add_argument("--crop-dir", default=None,
                   help="Directory with the PNGs (default: utsnitt/ next to "
                        "the manifest)")
    p.add_argument("--out-csv", default=None,
                   help="Judgement CSV (default: judge_image.csv next to "
                        "the manifest)")

    p.add_argument("--url", nargs="+", default=[STD_URL],
                   help=f"OpenAI-compatible base URL (default {STD_URL}). "
                        "Several URLs spread the boxes round-robin — run one "
                        "Ollama instance per URL, since qwen35/qwen3vl "
                        "serialise calls within an instance. "
                        "Ollama: http://localhost:11434/v1")
    p.add_argument("--model", default=None,
                   help="Model name the endpoint knows (required)")
    p.add_argument("--api-key", default=None, help="Bearer token if needed")

    p.add_argument("--concurrent", type=int, default=4, metavar="N",
                   help="Parallel calls (default 4)")
    p.add_argument("--timeout", type=float, default=STD_TIMEOUT,
                   help=f"Seconds per call (default {STD_TIMEOUT})")
    p.add_argument("--attempt", type=int, default=2, metavar="N",
                   help="Attempts per box on failure (default 2)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Default 0.0 — judgements must be reproducible")
    p.add_argument("--max-tokens", type=int, default=STD_MAX_TOKENS,
                   help=f"Maximum answer length (default {STD_MAX_TOKENS})")
    p.add_argument("--thinking", default="none",
                   choices=("none", "low", "medium", "high", "auto"),
                   help="reasoning_effort sent to the endpoint. Only works on "
                        "HYBRID models: Qwen3-VL has separate Instruct and "
                        "Thinking checkpoints, and a Thinking checkpoint "
                        "thinks regardless — pick the «-instruct» tag "
                        "instead. «auto» omits the field.")
    p.add_argument("--no-file", default=None, metavar="FIL",
                   help="Judge only the rows with these nr, one per line. "
                        "Build it with awk from an earlier judgement CSV to "
                        "iterate on the prompt against the cases that went "
                        "wrong.")
    p.add_argument("--max-items", type=int, default=None, metavar="N",
                   help="Judge only the first N rows (prompt testing)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="(default on) Continue a started run: skip rows "
                        "already judged without error, retry the failed ones.")
    p.add_argument("--restart", dest="resume", action="store_false",
                   help="OVERWRITE the judgement CSV and judge everything "
                        "again. A finished run is data, and overwriting it by "
                        "accident costs GPU hours.")
    p.add_argument("--cache", default=None, metavar="DIR",
                   help="Judgement cache directory (default $SLADD_CACHE/vlm "
                        "when SLADD_CACHE is set). Same prompt, config and "
                        "model reuse earlier answers; a change in any of "
                        "them gets a fresh cache folder.")
    p.add_argument("--no-cache", action="store_true",
                   help="Judge without reading or writing the cache. "
                        "--restart alone still reuses cached answers.")
    p.add_argument("--prompt-file", default=None, metavar="FIL",
                   help="Read the prompt from a file instead of the built-in")
    p.add_argument("--missing-candidates", action="store_true",
                   help="After judging, draw the pages holding BOM boxes the "
                        "model called a fødselsnummer, next to the "
                        "judgements. Those are candidates for a label the "
                        "fasit is missing, and the images feed "
                        "note_missing_label.py. Sources are read from "
                        "utvalg.json next to the manifest.")
    p.add_argument("--res-csv", default=None,
                   help="Override the result CSV from utvalg.json")
    p.add_argument("--truth-csv", default=None,
                   help="Override the labels CSV from utvalg.json")
    p.add_argument("--pdf-dir", default=None,
                   help="Override the PDF directory from utvalg.json")
    p.add_argument("--write-prompt", action="store_true",
                   help="Print the built-in prompt to stdout and exit")
    a = p.parse_args()

    if a.write_prompt:
        print(STD_PROMPT)
        return
    _THINKING["value"] = None if a.thinking == "auto" else a.thinking
    if a.no_cache:
        a.cache = None
    elif a.cache is None and os.environ.get("SLADD_CACHE"):
        a.cache = os.path.join(os.environ["SLADD_CACHE"], "vlm")
    for flag in ("manifest", "model"):
        if not getattr(a, flag):
            p.error(f"--{flag} is required")
    # Resolved before the run, not after: the drawing happens at the end, and
    # a missing path should not surface after hours of GPU time.
    if a.missing_candidates:
        missing = resolve_sources(a)
        if missing:
            p.error("--missing-candidates found no "
                    + ", ".join(missing)
                    + " in utvalg.json next to the manifest. Give them.")
    run(a)


if __name__ == "__main__":
    main()
