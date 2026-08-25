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

Three modes: --mode bilde (the crop as an image), tekst (ocr_tekst/ocr_linje
from the manifest, needs an export run with --ocr-cache) and begge. Having
both measures whether sight is worth the cost: where PaddleOCR read
correctly the task is pure text, but on handwriting the OCR text is missing.

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
STD_MAX_TOKENS = 700

# Set to None if the endpoint rejects the field — shared across the threads.
_THINKING = {"value": "none"}
# WITHOUT_CONTENT_FIELD plus the sensitive and technical columns. The model's
# checklist is stored so vlm_evaluate can re-apply the fnr rule without a GPU.
OUT_FIELD = ["utsnitt", "riktig", "klasse", "svar", "sikkerhet", "begrunnelse",
           "dato_gyldig", "holdepunkt", "sekunder", "label_id", "nr", "kilde",
           "tall", "linjen", "sifre_paa_linjen", "feil", "raatekst"]

# The judgement CSV above transcribes REAL fødselsnumre and is itself
# sensitive; this sister file holds only the verdicts and can be shared.
WITHOUT_CONTENT_FIELD = ["utsnitt", "riktig", "klasse", "svar", "sikkerhet",
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
              str(row.get("sikkerhet", "")), row.get("holdepunkt", ""),
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
                   f"| utsnitt | riktig | klasse | svar | sikkerhet "
                   f"| holdepunkt | begrunnelse |\n"
                   f"|---|---|---|---|---|---|---|\n")
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
Du ser et utsnitt fra et skannet norsk tinglysingsdokument. Den RØDE RAMMEN markerer et område en automatisk modell foreslår å sladde.

Spørsmålet er IKKE «er dette et fullstendig fødselsnummer». Det er: BERØRER RAMMEN ET FØDSELSNUMMER? Rammen er som regel liten MED VILJE: sladdingen skal typisk bare dekke personnummeret — de fem siste sifrene — mens fødselsdatoen står usladdet rett foran på linjen, eller på linjen over. At rammen bare inneholder fem sifre er altså det NORMALE for en riktig sladd, aldri i seg selv et argument for nei. Dommen gjelder hele sifferrekken som berører rammen, medregnet sifrene som står utenfor den.

FREMGANGSMÅTE — følg den i rekkefølge og stopp så snart du får svar:
1. Les hele teksten du ser, ikke bare det som står inne i rammen.
2. Skriv av HELE tekstlinjen rammen står på, fra venstre til høyre, slik du ser den. Ikke bare det som er inne i rammen — hele linjen.
3. Finn i den avskriften den lengste sammenhengende sifferrekken som berører rammen — OGSÅ sifrene som står utenfor rammen. Rekken slutter ikke ved rammekanten. Se bort fra mellomrom, punktum og bindestrek mellom sifrene.
4. Har den rekken elleve sifre, og er de seks første en gyldig dato (dag 01-31 eller 41-71, måned 01-12 eller 41-52)? → svar JA. Du er ferdig.
5. LET ETTER LEDETEKST som navngir tallet: «fødselsnummer», «fødselsnr», «f.nr», «personnr», «fødsel- og personnr» — også i kombinasjoner som «organisasjonsnr/fødselsnr». I skjemaer og tabeller står ledeteksten som overskrift OVER feltet, gjerne to-fire linjer over rammen — let til du finner den. Fant du en slik ledetekst: har rekken ni sifre og starter på 8 eller 9 er det org.nr-delen av en kombinert ledetekst → NEI. Ellers → svar JA.
6. Har rekken fem-seks sifre: SE PÅ LINJEN RETT OVER og teksten rett foran. Står det en sekssifret gyldig dato der, hører de sammen — et fødselsnummer delt over to linjer, med datoen øverst og personnummeret under. → svar JA.
7. Ellers: står det et ORD ELLER TEGN fra listen under som forteller hva tallet er? → svar NEI med ordet i «holdepunkt». Er det ENESTE som taler mot tallet din egen telling eller dato-utregning — ingen ledetekst, ingen forklaring — er NEI for dyrt: svar NEI bare når rekken åpenbart ikke kan være del av et fødselsnummer (f.eks. tre-fire sifre alene), ellers USIKKER.
8. Bare hvis sifrene ikke lar seg lese i det hele tatt — uskarpt, avskåret, tomt — svar USIKKER.

ORD OG TEGN som bekrefter et nei. Du kan også svare nei uten et slikt ord, når sifferrekken ikke har elleve sifre — men dikt aldri opp en kategori som ikke står her:
  - ordet konto, bank eller IBAN i nærheten, og elleve sifre gruppert 4-2-5
  - desimaltall: punktum eller komma INNE I selve tallet, som «6626630.58» eller «256843,12»
  - gnr, bnr, fnr, snr, matrikkel eller seksjon, ofte med skråstrek
  - org.nr, AS, ANS eller foretak i nærheten, og ni sifre som starter på 8 eller 9
  - dagboknr, journalnr eller saksnr
  - årstall, beløp, areal, sidetall eller tall på en målestokk

FELLER SOM HAR GITT FEIL FØR:
  - «020291 00000» er gruppert 6-5. Det ER fødselsnummerformatet. Kall det aldri kontonummer.
  - Fødselsnumre begynner ofte med 0, fordi dagen er 01-09. Helt normalt.
  - «12.03.50» er en dato. Punktum MELLOM datoledd er ikke desimalskille.
  - «301 / 10000» er matrikkel. Skråstrek er ikke desimalskille.
  - Et smalt 1-tall er lett å lese som skråstrek i et skann. Skråstrek ALENE er aldri holdepunkt for nei — krev ordet gnr, bnr, snr eller matrikkel i tillegg. Og teller du elleve sifre når skråstreken leses som 1, ER det elleve sifre.
  - Fødselsnummer deles ofte over to linjer: dato øverst, fem sifre personnummer under. En femsifret rekke er bare «for kort» når heller ikke linjen eller nabolinjene har dato-halvdelen.
  - «bare fem sifre i rammen» eller «mangler resten av fødselsnummeret» er ALDRI en gyldig begrunnelse for nei. Rammen skal bare dekke personnummeret — resten av rekken står utenfor rammen: foran på linjen, eller på linjen over. Let den opp før du konkluderer.
  - Skjemaer setter ledeteksten OVER feltet, ofte to-tre linjer over rammen. «Fødselsnr.» langt over rammen gjelder fortsatt tallet i rammen.
  - Lister og tabeller: står rammen i en kolonne der radene over er fødselsnumre, er denne raden det også. Døm kolonnen, ikke cellen alene.
  - Din egen dato-utregning er UPÅLITELIG — «1607» er 16. juli, en helt gyldig dato. Avvis aldri en ellevesifret rekke fordi du regnet dato-delen ugyldig; uten et ord-holdepunkt i tillegg → USIKKER.
  - Ikke regn på kontrollsifre. Du kan ikke gjøre mod11 pålitelig, og et feilregnestykke er ingen grunn til nei.
  - Dikt aldri opp en kategori som ikke står i listen over.

FIRE EKSEMPLER:
Rammen dekker «00000», og linjen sier «Kari Nordmann 010190 00000»:
{"linjen": "Kari Nordmann 010190 00000", "sifre_i_rammen": 5, "sifre_paa_linjen": "01019000000", "dato_gyldig": true, "holdepunkt": "", "elleve_og_dato_ok": true, "svar": "ja", "sikkerhet": 95, "tall": "010190 00000", "begrunnelse": "rammen dekker halve fnr-et på linjen"}
Rammen dekker «6626630.58» på et målebrevkart:
{"linjen": "N 6626630.58 Ø 256843.12", "sifre_i_rammen": 9, "sifre_paa_linjen": "662663058", "dato_gyldig": false, "holdepunkt": "desimalpunktum i tallet", "elleve_og_dato_ok": false, "svar": "nei", "sikkerhet": 95, "tall": "6626630.58", "begrunnelse": "koordinat med desimaler"}
Rammen dekker «00000», linjen over sier «Kari Nordmann f. 010190»:
{"linjen": "00000", "sifre_i_rammen": 5, "sifre_paa_linjen": "00000", "dato_gyldig": false, "holdepunkt": "", "elleve_og_dato_ok": false, "svar": "ja", "sikkerhet": 90, "tall": "010190 00000", "begrunnelse": "dato på linjen over — fnr delt over to linjer"}
Rammen dekker «48526», og verken linjen eller nabolinjene har en dato:
{"linjen": "48526", "sifre_i_rammen": 5, "sifre_paa_linjen": "48526", "dato_gyldig": false, "holdepunkt": "", "elleve_og_dato_ok": false, "svar": "nei", "sikkerhet": 80, "tall": "48526", "begrunnelse": "fem sifre, ingen dato-halvdel på nabolinjene"}

Rammen er for uskarp til at sifrene kan leses:
{"linjen": "", "sifre_i_rammen": 0, "sifre_paa_linjen": "", "dato_gyldig": false, "holdepunkt": "", "elleve_og_dato_ok": false, "svar": "usikker", "sikkerhet": 20, "tall": "", "begrunnelse": "kan ikke lese sifrene"}

Sjekklistefeltene skal stå FØR «svar». Rekkefølgen er ikke tilfeldig: skriver du kontrollen først, har du gjort den når du konkluderer.
  «linjen»            hele tekstlinjen rammen står på, avskrevet slik du ser den
  «sifre_i_rammen»    antall sifre du ser innenfor rammen
  «sifre_paa_linjen»  lengste sammenhengende sifferrekke i «linjen» som berører rammen, uten skilletegn.
                      Den skal kunne finnes igjen i «linjen» — dikt den ikke opp.
  «dato_gyldig»       true hvis de seks første sifrene i den rekken er en gyldig DDMMÅÅ
  «holdepunkt»        ordet eller tegnet som begrunner et nei, eller "" hvis du ikke fant noe
  «elleve_og_dato_ok» true hvis «sifre_paa_linjen» har nøyaktig elleve sifre OG «dato_gyldig» er true.
                      Er dette true, MÅ «svar» være "ja". Ingen unntak, uansett hva tallet ellers ligner på.
  «svar»              "ja", "nei" eller "usikker". DETTE FELTET ER OBLIGATORISK — svaret er ubrukelig uten det.
  «tall»              tallet du leser i og rundt rammen
  «sikkerhet»         helt tall 0-100 for hvor sikker du er. Vær ærlig — et lavt tall er nyttig og koster ingenting.
  «begrunnelse»       høyst 15 ord

Å svare nei på et ekte fødselsnummer er tjue ganger så dyrt som å la en unødvendig sladding stå. Er du i tvil, svar USIKKER.

Svar KUN med JSON på samme form som eksemplene, uten annen tekst. Kontroller før du sender at «svar» er med.\
"""

TEXT_MEASURE = """\

OCR-en leste dette inne i rammen: «{ocr_tekst}»
Teksten OCR-en leste rundt rammen (linjen, og nabolinjene om de finnes):
«{ocr_linje}»
Merk at OCR-en kan ha lest feil, særlig på håndskrift.\
"""

TEXT_ONLY_MEASURE = """\
Du får ikke se bildet. OCR-en leste dette inne i rammen: «{ocr_tekst}»
Teksten OCR-en leste rundt rammen (linjen, og nabolinjene om de finnes):
«{ocr_linje}»
Merk at OCR-en kan ha lest feil, særlig på håndskrift — svar USIKKER hvis \
teksten er for ødelagt til å avgjøre.\
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


def _confidence(d):
    """«sikkerhet» as 0-100, or "" if the model gave nothing usable."""
    raw = d.get("sikkerhet")
    if raw is None or raw == "":
        return ""
    try:
        return max(0, min(100, int(round(float(raw)))))
    except (TypeError, ValueError):
        return ""


def parse_answer(text):
    """Raw model answer -> (svar, sikkerhet, tall, begrunnelse, feil, checklist).

    Strict to lenient: plain JSON, then the first JSON object in the text
    (models like to wrap it in ```json), then a keyword search. Anything left
    over becomes «usikker», never «nei».
    """
    if not text or not text.strip():
        return "usikker", "", "", "", "empty answer", {}
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
            return (answer, _confidence(d), str(d.get("tall", "")).strip(),
                    str(d.get("begrunnelse", "")).strip(), "", _checklist(d))
        missing = "svar" not in d or not str(d.get("svar", "")).strip()
        return ("usikker", _confidence(d), str(d.get("tall", "")).strip(),
                str(d.get("begrunnelse", "")).strip(),
                "the model omitted the «svar» field — the rest of the JSON came"
                if missing else f"unknown answer {answer!r}", _checklist(d))

    m = _ANSWER_RE.search(clean)
    if m:
        return (m.group(1).lower(), "", "", clean[:120],
                "not JSON, read by keyword", {})
    return "usikker", "", "", clean[:120], "unparsable answer", {}


# ── Calls ─────────────────────────────────────────────────────

def _build_melding(prompt, mode, ocr, image_b64):
    """One chat message for one crop, from the inputs judge_one loaded."""
    text = prompt
    if mode == "text":
        return [{"role": "user",
                 "content": text + "\n\n" + TEXT_ONLY_MEASURE.format(**ocr)}]
    if mode == "both":
        text += "\n" + TEXT_MEASURE.format(**ocr)
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
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

    ocr = None
    if a.mode in ("text", "both"):
        ocr = {"ocr_tekst": row.get("ocr_tekst", "") or "(ingenting)",
               "ocr_linje": (row.get("ocr_blokk") or row.get("ocr_linje") or
                             "(ingenting)")}
    image_b64 = None
    if a.mode in ("image", "both"):
        try:
            with open(os.path.join(folder, row["utsnitt"]), "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            return {"svar": "usikker", "sikkerhet": "", "tall": "",
                    "begrunnelse": "",
                    "sekunder": round(time.monotonic() - t0, 2),
                    "feil": f"{type(e).__name__}: {e}"[:200], "raatekst": ""}

    key = vlm_cache.item_key(image_b64, ocr) if cache_dir else None
    if key:
        cached = vlm_cache.read_cache(cache_dir, key)
        if cached is not None:
            answer, confidence, number, rationale, parse_error, check = \
                parse_answer(cached)
            if not parse_error:
                return {"svar": answer, "sikkerhet": confidence,
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
            messages = _build_melding(prompt, a.mode, ocr, image_b64)
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
        return {"svar": "usikker", "sikkerhet": "", "tall": "",
                "begrunnelse": "", "sekunder": round(sec, 2), "feil": error,
                "raatekst": ""}
    answer, confidence, number, rationale, parse_error, check = parse_answer(raw)
    if key and not parse_error:
        vlm_cache.write_cache(cache_dir, key, row.get("utsnitt", ""), raw, sec)
    return {"svar": answer, "sikkerhet": confidence, "tall": number,
            "begrunnelse": rationale, **check,
            "sekunder": round(sec, 2), "feil": parse_error,
            "raatekst": raw.replace("\n", " ")[:400] if parse_error else ""}


# ── Run ───────────────────────────────────────────────────────

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
        templates = {"image": "", "both": TEXT_MEASURE,
                     "text": TEXT_ONLY_MEASURE}[a.mode]
        fp = vlm_cache.fingerprint(prompt, templates, a.model, a.mode,
                                   a.temperature, a.max_tokens,
                                   _THINKING["value"])
        cache_dir = os.path.join(a.cache, fp)
        vlm_cache.write_meta(cache_dir, {
            "model": a.model, "mode": a.mode, "temperature": a.temperature,
            "max_tokens": a.max_tokens, "thinking": _THINKING["value"],
            "prompt_file": a.prompt_file or "(built-in)",
            "templates": templates, "prompt": prompt})
        print(f"  Prompt version {fp}  (cache: {cache_dir})")

    if a.mode in ("text", "both") and not any(
            r.get("ocr_linje") for r in rows):
        print("  ⚠ Manifest has no ocr_linje — run vlm_export with "
              "--ocr-cache, or the text arm judges blind.")

    out_path = a.out_csv or os.path.join(os.path.dirname(
        os.path.abspath(a.manifest)), f"judge_{a.mode}.csv")
    if not out_path.lower().endswith(".csv"):
        # One directory per run: all four files under the run name.
        kat = out_path
        os.makedirs(kat, exist_ok=True)
        out_path = os.path.join(kat, "full_info.csv")
        old = kat + ".csv"
        if not os.path.isfile(out_path) and os.path.isfile(old):
            os.replace(old, out_path)
            print(f"  Moved {old} -> {out_path} (directory layout)")
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

    print(f"  {len(left)} boxes to judge  ({a.mode} mode, "
          f"{a.concurrent} concurrent, model {a.model})")
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
    tally = {"n": 0, "feil": 0, "cache": 0}
    timings = []
    t_start = time.monotonic()

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
                timings.append(res["sekunder"])
            if tally["n"] % 25 == 0 or tally["n"] == len(left):
                gone = time.monotonic() - t_start
                print(f"    {tally['n']:>6}/{len(left)}  "
                      f"{gone:6.0f}s  {gone / tally['n']:5.2f} s/box  "
                      f"{tally['feil']} errors  "
                      f"{tally['cache']} from cache", flush=True)

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
    print(f"\n  Done: {tally['n']} judgements in {gone:.0f}s "
          f"({gone / max(tally['n'], 1):.2f} s/box wall clock, "
          f"{a.concurrent} concurrent)")
    if cache_dir:
        print(f"  Cache: {tally['cache']} of {tally['n']} answers reused "
              f"({cache_dir})")
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
                   help="Judgement CSV (default: judge_<mode>.csv next to "
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
    p.add_argument("--mode", default="image", choices=("image", "text", "both"),
                   help="bilde = VLM on the crop, tekst = LLM on the OCR "
                        "text, begge = crop + OCR line (default bilde)")

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
    run(a)


if __name__ == "__main__":
    main()
