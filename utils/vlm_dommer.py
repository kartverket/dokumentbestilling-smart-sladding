"""Sender utsnittene fra vlm_eksport til en lokal VLM og lagrer dommene.

Kjøringen FORTSETTER som standard: finnes dommer-CSV-en fra før, hoppes
ferdige rader over og bare de feilede prøves på nytt. Det er nesten alltid
det man vil — en avbrutt batch skal kunne tas opp igjen, og en ferdig batch
skal ikke kastes ved et uhell. Vil du dømme alt om igjen, si det eksplisitt
med --start-paa-nytt.

Steg 2 av VLM-verifikator-piloten. Snakker OpenAI-kompatibelt
/v1/chat/completions, som både vLLM, llama.cpp-server, LM Studio og Ollama
tilbyr — bytt modell ved å bytte --url/--modell, ikke ved å endre koden.

Tre modi, fordi det ikke er avgjort om oppgaven trenger syn:
  --modus bilde   utsnittet som bilde (VLM)             — ser håndskrift
  --modus tekst   ocr_tekst/ocr_linje fra manifestet    — ren tekst-LLM, billig
  --modus begge   bilde + OCR-linjen som ekstra kontekst
Tekst-armen krever at eksporten ble kjørt med --ocr-cache. Poenget med å ha
begge er å måle om synet er verdt kostnaden: der PaddleOCR leste riktig, er
kontrasten koordinat/kontonummer/fnr et rent tekstproblem — men på håndskrift
er det nettopp OCR-teksten som ikke finnes (se rec-score-kalibreringen).

Robusthet: hver boks er uavhengig. Timeout, HTTP-feil og uparsbare svar logges
i «feil»-kolonnen og kjøringen fortsetter. Alt som ikke kan tolkes blir
«usikker», ALDRI «nei» — «nei» er det svaret som koster recall, og en
nettverksfeil er ikke et argument for å fjerne en sladding. Raden skrives
fortløpende, så en avbrutt kjøring kan gjenopptas med --gjenoppta.

Eksempel:
    python utils/vlm_dommer.py \
        --manifest /data2/vlm/uttrekk6_kalibrering/manifest.csv \
        --url http://localhost:8000/v1 \
        --modell Qwen/Qwen3-VL-8B-Instruct \
        --samtidige 4
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

STD_URL = "http://localhost:8000/v1"
STD_TIMEOUT = 120
STD_MAKS_TOKENS = 700

# Settes til None hvis endepunktet avviser feltet — delt mellom trådene.
_TENKNING = {"verdi": "none"}
# Samme kolonneorden som UTEN_INNHOLD_FELT, med det sensitive og tekniske
# LAGT TIL på slutten — de to filene skal kunne leses side om side.
# «tall», «linjen», «sifre_paa_linjen» og «raatekst» er sjekklisten modellen
# fyller ut FØR dommen. Den lagres fordi avlesningen er langt mer pålitelig
# enn slutningen: med «linjen» og «sifre_paa_linjen» på disk kan vlm_evaluer
# anvende fnr-regelen deterministisk i etterkant, uten ny GPU-kjøring.
UT_FELT = ["utsnitt", "riktig", "klasse", "svar", "sikkerhet", "begrunnelse",
           "dato_gyldig", "holdepunkt", "sekunder", "label_id", "nr", "kilde",
           "tall", "linjen", "sifre_paa_linjen", "feil", "raatekst"]

# Innholdsfri søsterfil: dommer-CSV-en over inneholder avskrifter av EKTE
# fødselsnumre («tall», «linjen», «sifre_paa_linjen», «raatekst») og er
# dermed selv sensitiv. *_uten_innhold.csv har bare dommene og kan åpnes og
# deles fritt under gjennomgang; «utsnitt» er kun stammen
# (00002_BOM_1000040863_s3) — klikkbare lenker bor i *_gjennomgang.md,
# der VSCode:s markdown-støtte løser relative stier også over SSH. «riktig»
# holder dommen mot fasit-klassen: ✅ = dommen stemmer (BOM fikk nei, dekkende
# fikk ja), 🟡 = ja på en BOM (tapt gevinst, ufarlig), ❌ = nei på en
# dekkende boks (ville avsladdet et ekte fnr), ❓ = usikker/feilet kall.
# Regenereres i sin helhet fra hovedfilen etter hver kjøring — også en
# gjenopptatt eller ferdig en.
UTEN_INNHOLD_FELT = ["utsnitt", "riktig", "klasse", "svar", "sikkerhet",
                     "begrunnelse", "dato_gyldig", "holdepunkt", "sekunder",
                     "label_id", "nr", "kilde"]


def _riktig(klasse, svar):
    onsket = "nei" if klasse == "BOM" else "ja"
    if svar not in ("ja", "nei"):
        return "⚪"
    if svar == onsket:
        return "✅"
    # De to gale svarene er ikke like gale: «ja» på en BOM lar bare en
    # unødvendig sladding stå (🟡), «nei» på en dekkende boks ville
    # avsladdet et ekte fnr (❌) — det er dem gjennomgangen skal finne.
    return "🟡" if klasse == "BOM" else "❌"


def _uten_innhold_rad(rad):
    rad = dict(rad)
    rad["utsnitt"] = os.path.splitext(
        os.path.basename(rad.get("utsnitt", "")))[0]
    rad["riktig"] = _riktig(rad.get("klasse", ""),
                            (rad.get("svar") or "").strip().lower())
    return rad


def _md_rad(rad, utsnitt_mappe, md_mappe):
    """Én gjennomgangslinje. Relativ lenke: VSCode:s markdown-støtte løser
    [tekst](relativ/sti.png) selv, gjennom remote-filsystemet — cmd+klikk
    virker i editoren der CSV-lenker ikke gjør det, og lenkene overlever
    en rsync fordi de er relative til dokumentet."""
    fil = rad.get("utsnitt", "")
    rel = os.path.relpath(os.path.join(utsnitt_mappe, fil), md_mappe)
    riktig = rad.get("riktig") or _riktig(
        rad.get("klasse", ""), (rad.get("svar") or "").strip().lower())
    celler = [f"[{os.path.splitext(fil)[0]}]({rel})", riktig,
              rad.get("klasse", ""), rad.get("svar", ""),
              str(rad.get("sikkerhet", "")), rad.get("holdepunkt", ""),
              rad.get("begrunnelse", "")]
    return "| " + " | ".join(c.replace("|", "/") for c in celler) + " |\n"


def skriv_gjennomgang_md(ut_sti, utsnitt_mappe):
    """Deriverer *_gjennomgang.md fra dommer-CSV-en. Returnerer stien."""
    stem, _ = os.path.splitext(ut_sti)
    sti = stem + "_gjennomgang.md"
    md_mappe = os.path.dirname(os.path.abspath(sti))
    with open(ut_sti, newline="", encoding="utf-8") as f_inn, \
         open(sti, "w", encoding="utf-8") as f_ut:
        f_ut.write(f"# Gjennomgang: {os.path.basename(stem)}\n\n"
                   f"❌ er radene som truer recall.\n\n"
                   f"| utsnitt | riktig | klasse | svar | sikkerhet "
                   f"| holdepunkt | begrunnelse |\n"
                   f"|---|---|---|---|---|---|---|\n")
        for rad in csv.DictReader(f_inn):
            f_ut.write(_md_rad(rad, utsnitt_mappe, md_mappe))
    return sti


def skriv_uten_innhold(ut_sti):
    """Deriverer *_uten_innhold.csv fra dommer-CSV-en. Returnerer stien."""
    stem, _ = os.path.splitext(ut_sti)
    sti = stem + "_uten_innhold.csv"
    with open(ut_sti, newline="", encoding="utf-8") as f_inn, \
         open(sti, "w", newline="", encoding="utf-8") as f_ut:
        skriver = csv.DictWriter(f_ut, fieldnames=UTEN_INNHOLD_FELT,
                                 extrasaction="ignore")
        skriver.writeheader()
        for rad in csv.DictReader(f_inn):
            skriver.writerow(_uten_innhold_rad(rad))
    return sti

# Prompten er pilotens egentlige eksperiment. Den er bygget rundt kontrastene
# fra oversladdingsanalysen — koordinater, kontonumre, dagboknumre,
# gårds-/bruks-/seksjonsnumre — fordi det er DE som ligner et fnr nok til at
# både YOLO og reglene går på dem. Kalibreringskjøringen på TR_MAS finnes for
# å teste akkurat denne teksten før hovedbatchen.
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

TEKST_MAL = """\

OCR-en leste dette inne i rammen: «{ocr_tekst}»
Teksten OCR-en leste rundt rammen (linjen, og nabolinjene om de finnes):
«{ocr_linje}»
Merk at OCR-en kan ha lest feil, særlig på håndskrift.\
"""

TEKST_KUN_MAL = """\
Du får ikke se bildet. OCR-en leste dette inne i rammen: «{ocr_tekst}»
Teksten OCR-en leste rundt rammen (linjen, og nabolinjene om de finnes):
«{ocr_linje}»
Merk at OCR-en kan ha lest feil, særlig på håndskrift — svar USIKKER hvis \
teksten er for ødelagt til å avgjøre.\
"""


# ── Svar-parsing ──────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_SVAR_RE = re.compile(r"\b(ja|nei|usikker)\b", re.I)


def _sjekkliste(d):
    """Sjekklistefeltene, som tekst. Ukjente eller manglende blir tomme."""
    ut = {}
    for felt in ("linjen", "sifre_paa_linjen", "dato_gyldig", "holdepunkt"):
        v = d.get(felt)
        ut[felt] = "" if v is None else str(v).strip().replace("\n", " ")[:300]
    return ut


def _sikkerhet(d):
    """«sikkerhet» som 0-100, eller "" hvis modellen ikke oppga noe brukbart."""
    raa = d.get("sikkerhet")
    if raa is None or raa == "":
        return ""
    try:
        return max(0, min(100, int(round(float(raa)))))
    except (TypeError, ValueError):
        return ""


def tolk_svar(tekst):
    """Rått modellsvar -> (svar, sikkerhet, tall, begrunnelse, feil, sjekkliste).

    Rekkefølgen er streng-til-slapp: ren JSON, så første JSON-objekt i teksten
    (modeller pakker gjerne svaret i ```json), så et nøkkelordsøk. Alt som
    fortsatt ikke gir mening blir «usikker» med en feilmerknad — se
    modulens docstring om hvorfor det aldri blir «nei».
    """
    if not tekst or not tekst.strip():
        return "usikker", "", "", "", "tomt svar", {}
    rens = tekst.strip()
    if rens.startswith("```"):
        rens = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", rens).strip()

    kandidater = [rens]
    m = _JSON_RE.search(rens)
    if m:
        kandidater.append(m.group(0))
    for kand in kandidater:
        try:
            d = json.loads(kand)
        except (ValueError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        svar = str(d.get("svar", "")).strip().lower()
        if svar in ("ja", "nei", "usikker"):
            return (svar, _sikkerhet(d), str(d.get("tall", "")).strip(),
                    str(d.get("begrunnelse", "")).strip(), "", _sjekkliste(d))
        mangler = "svar" not in d or not str(d.get("svar", "")).strip()
        return ("usikker", _sikkerhet(d), str(d.get("tall", "")).strip(),
                str(d.get("begrunnelse", "")).strip(),
                "modellen utelot «svar»-feltet — resten av JSON-en kom"
                if mangler else f"ukjent svar {svar!r}", _sjekkliste(d))

    m = _SVAR_RE.search(rens)
    if m:
        return (m.group(1).lower(), "", "", rens[:120],
                "ikke-JSON, nøkkelordtolket", {})
    return "usikker", "", "", rens[:120], "uparsbart svar", {}


# ── Kall ──────────────────────────────────────────────────────

def _bygg_melding(rad, mappe, prompt, modus):
    """Én chat-melding for én manifest-rad."""
    tekst = prompt
    ocr = {"ocr_tekst": rad.get("ocr_tekst", "") or "(ingenting)",
           "ocr_linje": (rad.get("ocr_blokk") or rad.get("ocr_linje") or
                         "(ingenting)")}
    if modus == "tekst":
        return [{"role": "user",
                 "content": tekst + "\n\n" + TEKST_KUN_MAL.format(**ocr)}]
    if modus == "begge":
        tekst += "\n" + TEKST_MAL.format(**ocr)

    sti = os.path.join(mappe, rad["utsnitt"])
    with open(sti, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": tekst},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]


def kall_modell(url, modell, meldinger, api_nokkel=None, timeout=STD_TIMEOUT,
                temperatur=0.0, maks_tokens=STD_MAKS_TOKENS, tenkning="none"):
    """Ett kall. tenkning=None utelater reasoning_effort helt.

    Ollama slår PÅ tenkning av seg selv for modeller som kan det når feltet
    mangler — og «qwen3-vl:8b» er thinking-varianten. Da havner resonnementet
    i «reasoning», «content» blir tomt, og hele token-budsjettet går med til
    en indre monolog vi ikke har bruk for. Vi ber om en JSON-dom, ikke en
    utredning, så standarden her er å skru det av.
    """
    kropp = {"model": modell, "messages": meldinger,
             "temperature": temperatur, "max_tokens": maks_tokens}
    if tenkning:
        kropp["reasoning_effort"] = tenkning
    data = json.dumps(kropp).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_nokkel or 'ingen'}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as svar:
            d = json.loads(svar.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # urllib kaster bort svarkroppen, og det er nettopp der endepunktet
        # forklarer HVA det ikke likte. Uten denne står det bare
        # «HTTP Error 400: Bad Request» i feilkolonnen, og da må man gjette.
        try:
            forklaring = e.read().decode("utf-8", "replace").strip()[:300]
        except Exception:
            forklaring = ""
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason} — {forklaring}" if forklaring
            else str(e.reason), e.headers, None)
    m = d["choices"][0]["message"]
    innhold = m.get("content") or ""
    if not innhold.strip():
        # Tomt content med resonnement ved siden av betyr at tenkningen spiste
        # svaret. Si det rett ut i stedet for å la det bli en anonym «usikker».
        for felt in ("reasoning", "reasoning_content"):
            if (m.get(felt) or "").strip():
                raise ValueError(
                    "tomt «content» — modellen tenkte i stedet for å svare. "
                    "Qwen3-VL er delt i to sjekkpunkter: på et THINKING-"
                    "sjekkpunkt (Ollama-taggen «qwen3-vl:8b») hjelper ikke "
                    "--tenkning none, for tenkningen er trent inn. Bruk "
                    "«qwen3-vl:8b-instruct», eller hev --maks-tokens for å "
                    "la den tenke ferdig (dyrt)")
    return innhold


def dom_en(rad, a, mappe, prompt):
    """Dømmer én boks. Kaster aldri — feil havner i «feil»-kolonnen."""
    t0 = time.monotonic()
    feil = ""
    raa = ""
    for forsok in range(1, a.forsok + 1):
        try:
            meldinger = _bygg_melding(rad, mappe, prompt, a.modus)
            raa = kall_modell(a.url, a.modell, meldinger,
                              api_nokkel=a.api_nokkel, timeout=a.timeout,
                              temperatur=a.temperatur,
                              maks_tokens=a.maks_tokens,
                              tenkning=_TENKNING["verdi"])
            feil = ""
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError, TimeoutError) as e:
            feil = f"{type(e).__name__}: {e}"[:200]
            # Eldre endepunkter kjenner ikke reasoning_effort og svarer 400.
            # Slå det av for resten av kjøringen i stedet for å feile alle
            # radene på samme måte.
            # Bare når endepunktet FAKTISK klager på feltet. Ellers ville en
            # hvilken som helst 400 blitt bortforklart som en tenknings-sak,
            # og den ekte årsaken forsvunnet i en misvisende melding.
            if (isinstance(e, urllib.error.HTTPError) and e.code == 400
                    and _TENKNING["verdi"]
                    and "reasoning" in str(e).lower()):
                _TENKNING["verdi"] = None
                print("  ⚠ Endepunktet avviste reasoning_effort — "
                      "fortsetter uten", flush=True)
                continue
            if forsok < a.forsok:
                time.sleep(min(2 ** forsok, 10))
    sek = time.monotonic() - t0

    if feil:
        return {"svar": "usikker", "sikkerhet": "", "tall": "",
                "begrunnelse": "", "sekunder": round(sek, 2), "feil": feil,
                "raatekst": ""}
    svar, sikkerhet, tall, begrunnelse, parsefeil, sjekk = tolk_svar(raa)
    return {"svar": svar, "sikkerhet": sikkerhet, "tall": tall,
            "begrunnelse": begrunnelse, **sjekk,
            "sekunder": round(sek, 2), "feil": parsefeil,
            "raatekst": raa.replace("\n", " ")[:400] if parsefeil else ""}


# ── Kjøring ───────────────────────────────────────────────────

def kjor(a):
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rader = list(csv.DictReader(f))
    mappe = a.utsnitt_mappe or os.path.join(
        os.path.dirname(os.path.abspath(a.manifest)), "utsnitt")
    prompt = STD_PROMPT
    if a.prompt_fil:
        with open(a.prompt_fil, encoding="utf-8") as f:
            prompt = f.read().strip()

    if a.modus in ("tekst", "begge") and not any(
            r.get("ocr_linje") for r in rader):
        print("  ⚠ Manifestet har ingen ocr_linje — kjør vlm_eksport med "
              "--ocr-cache, ellers dømmer tekst-armen i blinde.")

    ut_sti = a.ut_csv or os.path.join(os.path.dirname(
        os.path.abspath(a.manifest)), f"dommer_{a.modus}.csv")
    ferdige = set()
    if not a.gjenoppta and os.path.isfile(ut_sti):
        print(f"  ⚠ --start-paa-nytt: overskriver {ut_sti}")
    if a.gjenoppta and os.path.isfile(ut_sti):
        with open(ut_sti, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                # Rader som feilet dømmes på nytt — det er hele poenget med
                # å gjenoppta etter at endepunktet var nede.
                if r.get("svar") and not r.get("feil"):
                    ferdige.add(r["nr"])
        print(f"  Gjenopptar: {len(ferdige)} rader allerede dømt")

    igjen = [r for r in rader if r["nr"] not in ferdige]
    if a.nr_fil:
        with open(a.nr_fil, encoding="utf-8-sig") as f:
            valgte = {lin.strip() for lin in f
                      if lin.strip() and not lin.startswith("#")}
        foer = len(igjen)
        igjen = [r for r in igjen if r["nr"] in valgte]
        print(f"  --nr-fil: {len(igjen)} av {foer} rader valgt "
              f"({len(valgte)} nr i filen)")
    if a.maks:
        igjen = igjen[:a.maks]
    if not igjen:
        print("  Ingenting å gjøre.")
        if os.path.isfile(ut_sti):
            print(f"  Uten innhold: {skriv_uten_innhold(ut_sti)}")
            print(f"  Gjennomgang:  {skriv_gjennomgang_md(ut_sti, mappe)}")
        return ut_sti

    print(f"  {len(igjen)} bokser å dømme  ({a.modus}-modus, "
          f"{a.samtidige} samtidige, modell {a.modell})")
    ny_fil = not (a.gjenoppta and os.path.isfile(ut_sti))
    if not ny_fil:
        with open(ut_sti, newline="", encoding="utf-8-sig") as f:
            leser = csv.DictReader(f)
            gamle = list(leser) if leser.fieldnames != UT_FELT else None
        if gamle is not None:
            print(f"  Kolonneoppsettet er endret — skriver {ut_sti} om "
                  f"i ny orden ({len(gamle)} rader)")
            with open(ut_sti, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=UT_FELT,
                                   extrasaction="ignore")
                w.writeheader()
                for r in gamle:
                    r.setdefault("riktig", _riktig(
                        r.get("klasse", ""),
                        (r.get("svar") or "").strip().lower()))
                    w.writerow({k: r.get(k, "") for k in UT_FELT})
    f_ut = open(ut_sti, "w" if ny_fil else "a", newline="", encoding="utf-8")
    skriver = csv.DictWriter(f_ut, fieldnames=UT_FELT, extrasaction="ignore")
    if ny_fil:
        skriver.writeheader()
        f_ut.flush()
    # Den innholdsfrie søsterfilen skrives like fortløpende som hovedfilen —
    # gjennomgangen skal kunne starte mens batchen går. Deriveringen først
    # bringer den i takt med allerede dømte rader ved gjenopptak.
    ui_sti = skriv_uten_innhold(ut_sti)
    md_sti = skriv_gjennomgang_md(ut_sti, mappe)
    f_md = open(md_sti, "a", encoding="utf-8")
    md_mappe = os.path.dirname(os.path.abspath(md_sti))
    f_ui = open(ui_sti, "a", newline="", encoding="utf-8")
    skriver_ui = csv.DictWriter(f_ui, fieldnames=UTEN_INNHOLD_FELT,
                                extrasaction="ignore")
    laas = threading.Lock()
    telling = {"n": 0, "feil": 0}
    tider = []
    t_start = time.monotonic()

    def arbeid(rad):
        res = dom_en(rad, a, mappe, prompt)
        rad_ut = {k: rad.get(k, "") for k in
                  ("nr", "utsnitt", "klasse", "kilde", "label_id")}
        rad_ut.update(res)
        rad_ut["riktig"] = _riktig(rad_ut.get("klasse", ""),
                                   (rad_ut.get("svar") or "").strip().lower())
        with laas:
            skriver.writerow(rad_ut)
            f_ut.flush()
            skriver_ui.writerow(_uten_innhold_rad(rad_ut))
            f_ui.flush()
            f_md.write(_md_rad(rad_ut, mappe, md_mappe))
            f_md.flush()
            telling["n"] += 1
            if res["feil"]:
                telling["feil"] += 1
            tider.append(res["sekunder"])
            if telling["n"] % 25 == 0 or telling["n"] == len(igjen):
                gaatt = time.monotonic() - t_start
                print(f"    {telling['n']:>6}/{len(igjen)}  "
                      f"{gaatt:6.0f}s  {gaatt / telling['n']:5.2f} s/boks  "
                      f"{telling['feil']} feil", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=a.samtidige) as pool:
            list(pool.map(arbeid, igjen))
    finally:
        f_ut.close()
        f_ui.close()
        f_md.close()

    gaatt = time.monotonic() - t_start
    tider.sort()
    print(f"\n  Ferdig: {telling['n']} dommer på {gaatt:.0f}s "
          f"({gaatt / max(telling['n'], 1):.2f} s/boks veggklokke, "
          f"{a.samtidige} samtidige)")
    if tider:
        print(f"  Latens per kall: median {tider[len(tider) // 2]:.2f}s, "
              f"p90 {tider[int(len(tider) * 0.9)]:.2f}s, "
              f"maks {tider[-1]:.2f}s")
    if telling["feil"]:
        print(f"  ⚠ {telling['feil']} rader med feil/uparsbart svar "
              f"— alle talt som «usikker». Kjør på nytt med --gjenoppta.")
    print(f"  Dommer: {ut_sti}")
    print(f"  Uten innhold: {skriv_uten_innhold(ut_sti)}")
    print(f"  Gjennomgang:  {skriv_gjennomgang_md(ut_sti, mappe)}")
    return ut_sti


def main():
    p = argparse.ArgumentParser(
        description="Dømmer utsnitt fra vlm_eksport med en lokal, "
                    "OpenAI-kompatibel VLM (vLLM / llama.cpp / Ollama).")
    p.add_argument("--manifest", default=None,
                   help="manifest.csv fra vlm_eksport (kreves)")
    p.add_argument("--utsnitt-mappe", default=None,
                   help="Mappe med PNG-ene (default: utsnitt/ ved manifestet)")
    p.add_argument("--ut-csv", default=None,
                   help="Dommer-CSV (default: dommer_<modus>.csv ved manifestet)")

    p.add_argument("--url", default=STD_URL,
                   help=f"OpenAI-kompatibel base-URL (default {STD_URL}). "
                        "Ollama: http://localhost:11434/v1")
    p.add_argument("--modell", default=None,
                   help="Modellnavn endepunktet kjenner (kreves)")
    p.add_argument("--api-nokkel", default=None, help="Bearer-token om nødvendig")
    p.add_argument("--modus", default="bilde", choices=("bilde", "tekst", "begge"),
                   help="bilde = VLM på utsnittet, tekst = LLM på OCR-teksten, "
                        "begge = bilde + OCR-linje (default bilde)")

    p.add_argument("--samtidige", type=int, default=4, metavar="N",
                   help="Parallelle kall (default 4)")
    p.add_argument("--timeout", type=float, default=STD_TIMEOUT,
                   help=f"Sekunder per kall (default {STD_TIMEOUT})")
    p.add_argument("--forsok", type=int, default=2, metavar="N",
                   help="Antall forsøk per boks ved feil (default 2)")
    p.add_argument("--temperatur", type=float, default=0.0,
                   help="Default 0.0 — dommene skal være reproduserbare")
    p.add_argument("--maks-tokens", type=int, default=STD_MAKS_TOKENS,
                   help=f"Maks svarlengde (default {STD_MAKS_TOKENS})")
    p.add_argument("--tenkning", default="none",
                   choices=("none", "low", "medium", "high", "auto"),
                   help="reasoning_effort mot endepunktet. Default «none». "
                        "NB: dette virker bare på HYBRIDE modeller. Qwen3-VL "
                        "har egne Instruct- og Thinking-sjekkpunkter, og på "
                        "et Thinking-sjekkpunkt tenker modellen uansett — "
                        "velg «-instruct»-taggen i stedet. «auto» utelater "
                        "feltet.")
    p.add_argument("--nr-fil", default=None, metavar="FIL",
                   help="Døm bare radene med disse nr-ene, ett per linje. "
                        "Lag den med awk fra en tidligere dommer-CSV for å "
                        "iterere på prompten mot nettopp de tilfellene som "
                        "gikk galt — sekunder i stedet for en time.")
    p.add_argument("--maks", type=int, default=None, metavar="N",
                   help="Døm bare de N første radene (prompt-testing)")
    p.add_argument("--gjenoppta", action="store_true", default=True,
                   help="(default på) Fortsett en påbegynt kjøring: hopp over "
                        "rader som allerede er dømt uten feil, og prøv de "
                        "feilede på nytt.")
    p.add_argument("--start-paa-nytt", dest="gjenoppta", action="store_false",
                   help="OVERSKRIV dommer-CSV-en og døm alt om igjen. Bruk "
                        "dette bevisst — en ferdig kjøring er data, og en "
                        "utilsiktet overskriving koster GPU-timer.")
    p.add_argument("--prompt-fil", default=None, metavar="FIL",
                   help="Les prompten fra fil i stedet for den innebygde")
    p.add_argument("--skriv-prompt", action="store_true",
                   help="Skriv den innebygde prompten til stdout og avslutt")
    a = p.parse_args()

    if a.skriv_prompt:
        print(STD_PROMPT)
        return
    _TENKNING["verdi"] = None if a.tenkning == "auto" else a.tenkning
    for flagg in ("manifest", "modell"):
        if not getattr(a, flagg):
            p.error(f"--{flagg} er påkrevd")
    kjor(a)


if __name__ == "__main__":
    main()
