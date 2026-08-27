"""OpenAI-compatible VLM client: prompt, call, answer parsing, fnr guard.

Shared by the pilot tooling in utils/ and by the verifier that runs inside
the pipeline (vlm_verifier.py), so prod and the offline runs judge with the
exact same prompt and parse the answers the same way. Lives in app/ because
only app/*.py is copied into the container.

The endpoint is /v1/chat/completions, which llama-server, vLLM and LM Studio
all offer, so the model is switched with a URL and a name instead of a code
change. llama-server is what runs on the GPU host: see docs/VLM-ISOLATION.md
for why the smallest server wins here.

The model answers «ja» or «nei» and nothing else. Anything that cannot be
interpreted — timeout, HTTP error, unparsable answer — becomes «ja», NEVER
«nei»: «nei» is the answer that costs recall. The «feil» field is what tells a
forced «ja» apart from one the model actually gave.
"""

import json
import os
import re
import urllib.error
import urllib.request

from paddle_ocr_model_fnr import find_fnr

# The container runs with HTTP_PROXY set for the outside world, and urllib
# would send the VLM call through it — the endpoint is on the inside and the
# proxy either refuses it or never reaches it. SLADD_VLM_PROXY is the way back
# out for an endpoint that really does sit behind the proxy.
_PROXY = os.environ.get("SLADD_VLM_PROXY", "").strip()
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler(
    {"http": _PROXY, "https": _PROXY} if _PROXY else {}))

STD_URL = "http://127.0.0.1:8080/v1"   # llama-server default port
STD_TIMEOUT = 120
# Roughly double a full answer. A truncated answer is unparsable and
# becomes «ja», never «nei», so the cap costs recall nothing when it bites.
STD_MAX_TOKENS = 150

# Set to None if the endpoint rejects the field — shared across the threads.
_THINKING = {"value": "none"}
# The prompt is the real experiment of the pilot: it is built around the
# contrasts both YOLO and the rules fall for, so treat edits as experiments.
STD_PROMPT = """\
Du ser et utsnitt fra et skannet norsk tinglysingsdokument. Den røde rammen markerer et område som er foreslått sladdet.

Norske fødselsnumre har elleve sifre: fødselsdato (DDMMÅÅ) fulgt av fem sifre personnummer. Sladdingen skal som regel bare dekke de fem siste sifrene, så rammen inneholder ofte bare en bit av nummeret. Datoen kan stå foran på linjen eller på linjen over.

Spørsmålet: berører rammen et fødselsnummer, helt eller delvis?

- Svar «ja» når tallet er eller sannsynligvis er et fødselsnummer.
- Svar «nei» BARE når du tydelig ser at tallet er noe annet: kontonummer, organisasjonsnummer, koordinat, beløp, dato alene, matrikkel-/saks-/dokumentnummer, og skriv hva det er i «holdepunkt».
- Å si nei på et fødselsnummer er 100 ganger verre enn å si ja på et annet tall. Når du er i tvil, svar «ja».

Svar kun med JSON:
{"tall": "tallene du ser i og rundt rammen", "holdepunkt": "hva tallet er, hvis det er noe annet enn et fødselsnummer — ellers tom", "svar": "ja"}\
"""

# ── Answer parsing ────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_ANSWER_RE = re.compile(r"\b(ja|nei)\b", re.I)


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
    over becomes «ja», never «nei»: a box we could not read a verdict for
    keeps its sladd. «feil» says why.
    """
    if not text or not text.strip():
        return "ja", "", "", "empty answer", {}
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
        if answer in ("ja", "nei"):
            return (answer, str(d.get("tall", "")).strip(),
                    str(d.get("begrunnelse", "")).strip(), "", _checklist(d))
        missing = "svar" not in d or not str(d.get("svar", "")).strip()
        return ("ja", str(d.get("tall", "")).strip(),
                str(d.get("begrunnelse", "")).strip(),
                "the model omitted the «svar» field — the rest of the JSON came"
                if missing else f"unknown answer {answer!r}", _checklist(d))

    m = _ANSWER_RE.search(clean)
    if m:
        return (m.group(1).lower(), "", clean[:120],
                "not JSON, read by keyword", {})
    return "ja", "", clean[:120], "unparsable answer", {}


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

    Servers that default thinking ON spend the whole token budget on an inner
    monologue and return an empty «content», so the field is sent explicitly.
    An endpoint that does not know it answers 400, and the caller drops it.
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
        with _OPENER.open(req, timeout=timeout) as answer:
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
                    "answering. Some checkpoints have the thinking trained "
                    "in, and there --thinking none does not help: use the "
                    "instruct variant of the same model, or raise "
                    "--max-tokens to let it finish thinking (expensive)")
    return content



# ── fnr guard ─────────────────────────────────────────────────
# The model reads better than it infers: a verdict of «nei» is overruled when
# any transcription of the line holds a valid fnr run. Every measured loss so
# far was an absence argument («missing date», «too short»), and this is the
# deterministic backstop against them.

_FNR_WORD = re.compile(
    r"(f\s*[øo]dsels\s*n|pers\s*[.\s]*n|p\s*\.\s*nr|f\s*\.\s*nr|fnr|personnummer)",
    re.I)
_FIVE_RUN = re.compile(r"(?<!\d)\d{5}(?!\d)")


def fnr_candidate(line):
    """Does the line hold an 11-digit run shaped like a fødselsnummer?

    Uses the pipeline's own find_fnr, which works on digit positions: strip
    the separators first and «030392S0000 Iflg fullmakt» glues to «1f1g», the
    boundary check fails, and a real fnr is lost.
    """
    return bool(find_fnr(line or "", require_mod11=False))


def has_fnr_caption(line):
    """A fnr ledetekst plus a five-digit run on the same line.

    Catches documents where the date of birth is written in some other form
    than DDMMYY, so there is no eleven-digit run to find at all.
    """
    text = line or ""
    return bool(_FNR_WORD.search(text)) and bool(_FIVE_RUN.search(text))


def fnr_protects(texts, caption=True):
    """Whether any of the readings of the line protects the box from «nei».

    `texts` are independent readings of the same line — the model's own
    transcription and PaddleOCR's — since they rarely fail together.
    """
    readings = [t for t in (str(v).strip() for v in texts if v) if t]
    return (any(fnr_candidate(t) for t in readings)
            or (caption and any(has_fnr_caption(t) for t in readings)))
