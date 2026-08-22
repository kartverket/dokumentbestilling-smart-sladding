"""Røyktest av VLM-pilotens tre verktøy på syntetiske data.

Kjøres uten GPU, uten server og uten tilgang til dokumentene: den bygger tre
små PDF-er med kjente tall, en fasit-CSV, en resultat-CSV og en OCR-cache i
pipelinens eget format, starter en stub som snakker OpenAI-protokollen, og
kjører vlm_eksport → vlm_dommer → vlm_evaluer ende til ende.

Testen dekker det som faktisk kan gå galt uten å oppdages på serveren:
  * at TREFF/BOM-klassifiseringen og utvalgsfaktorene stemmer
  * at utsnittene faktisk skrives, med markøren inne i bildet
  * at OCR-konteksten hentes riktig — også på en side pipelinen roterte
  * at timeout, HTTP-500 og prosa-svar ikke stopper kjøringen, og at de
    alltid ender som «usikker», aldri «nei»
  * at gjenopptagelse plukker opp nettopp de radene som feilet
  * at regnskapet i vlm_evaluer skalerer tap med utvalgsfaktoren

    python utils/vlm_selvtest.py            # rydder etter seg
    python utils/vlm_selvtest.py --behold   # lar filene ligge for øyesyn
"""

import argparse
import base64
import csv
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

HER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HER)
sys.path.insert(0, os.path.normpath(os.path.join(HER, "..", "app")))

import fitz
from PIL import Image

from filter_felles import SKALA
from filter_review import _rekt_frem
from vlm_eksport import _ocr_kontekst
from ocr_cache import Token, skriv_cache
from vlm_dommer import tolk_svar
from vlm_evaluer import poisson_ovre

FONT = 11
SIDE_B, SIDE_H = 595.0, 842.0

# (tekst, x, y, er_fnr) — y er grunnlinjen. Tallene er valgt som pilotens
# egne kontraster: ekte fnr mot koordinat, kontonummer og tabellcelle.
DOKUMENTER = {
    "0100001.pdf": [
        ("Hjemmelshaver 060695 00000", 70.0, 200.0, True),
        ("Koordinat N 6626630.58", 70.0, 260.0, False),
        ("Dagboknr 900123 tinglyst 03.11.1998", 70.0, 320.0, False),
    ],
    "0100002.pdf": [
        ("Selger 07079600000 andel 1/2", 70.0, 180.0, True),
        ("Konto 1234 56 78903", 70.0, 240.0, False),
    ],
    "0100003.pdf": [                      # roteres i OCR-cachen (rotasjon 1)
        ("Kjoper 48089700000 d-nummer", 70.0, 300.0, True),
        ("gnr 12 bnr 345 snr 6", 70.0, 360.0, False),
    ],
}
ROTASJON = {"0100003.pdf": 1}


def _boks(tekst, x, y):
    """Tekstens boks i PDF-punkt, slik insert_text plasserer den."""
    b = fitz.get_text_length(tekst, fontname="helv", fontsize=FONT)
    return (x, y - FONT * 0.8, x + b, y + FONT * 0.25)


def _tall_boks(tekst, x, y):
    """Boksen rundt SELVE tallet i strengen — det er den modellen foreslår."""
    m = re.search(r"\d[\d .,/-]*\d", tekst)
    if not m:
        return _boks(tekst, x, y)
    foer = fitz.get_text_length(tekst[:m.start()], fontname="helv", fontsize=FONT)
    bredde = fitz.get_text_length(m.group(0), fontname="helv", fontsize=FONT)
    return (x + foer, y - FONT * 0.8, x + foer + bredde, y + FONT * 0.25)


def bygg_data(rot):
    """Lager PDF-er, fasit-CSV, resultat-CSV og OCR-cache. Returnerer stier."""
    pdf_mappe = os.path.join(rot, "pdf")
    ocr_mappe = os.path.join(rot, "ocr")
    os.makedirs(pdf_mappe, exist_ok=True)

    bw, bh = int(SIDE_B * SKALA), int(SIDE_H * SKALA)
    fasit_rader, pred_rader = [], []
    label_id = 5000
    for navn, poster in DOKUMENTER.items():
        dok = fitz.open()
        side = dok.new_page(width=SIDE_B, height=SIDE_H)
        tokens = []
        for tekst, x, y, er_fnr in poster:
            side.insert_text((x, y), tekst, fontsize=FONT, fontname="helv")
            tx0, ty0, tx1, ty1 = _tall_boks(tekst, x, y)
            if er_fnr:
                label_id += 1
                fasit_rader.append({
                    "id": label_id, "fil_revisjon_id": int(navn.lstrip("0")[:6]),
                    "sidetall": 1, "x": tx0, "y": ty0,
                    "width": tx1 - tx0, "height": ty1 - ty0,
                    "ml_status": "ACCEPTED",
                })
            px = [tx0 * SKALA, ty0 * SKALA, tx1 * SKALA, ty1 * SKALA]
            pred_rader.append({
                "navn": navn, "side": 1, "bilde_bredde": bw, "bilde_hoyde": bh,
                "x0": round(px[0], 2), "y0": round(px[1], 2),
                "x1": round(px[2], 2), "y1": round(px[3], 2),
                "kilde": "yolo", "yolo_conf": 0.71, "paddle_rec_score": "",
                "har_tokens": 1,
            })
            # Ett token per ord, i det roterte rommet pipelinen OCR-et i.
            k = ROTASJON.get(navn, 0)
            kx = x
            for ord_ in tekst.split(" "):
                ob = fitz.get_text_length(ord_, fontname="helv", fontsize=FONT)
                r = [kx * SKALA, (y - FONT * 0.8) * SKALA,
                     (kx + ob) * SKALA, (y + FONT * 0.25) * SKALA]
                r = _rekt_frem(r, k, bw, bh)
                tokens.append(Token(ord_, r[0], r[1], r[2], r[3], 0.97))
                kx += ob + fitz.get_text_length(" ", fontname="helv",
                                                fontsize=FONT)
        dok.save(os.path.join(pdf_mappe, navn))
        dok.close()
        skriv_cache(ocr_mappe, navn, [ROTASJON.get(navn, 0)], [tokens])

    fasit_csv = os.path.join(rot, "fasit.csv")
    with open(fasit_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "fil_revisjon_id", "sidetall",
                                          "x", "y", "width", "height",
                                          "ml_status"])
        w.writeheader()
        w.writerows(fasit_rader)

    res_csv = os.path.join(rot, "resultat.csv")
    felt = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1",
            "y1", "kilde", "yolo_conf", "paddle_rec_score", "har_tokens"]
    with open(res_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felt, extrasaction="ignore")
        w.writeheader()
        w.writerows(pred_rader)
    return pdf_mappe, ocr_mappe, fasit_csv, res_csv


# ── Stub-endepunkt ────────────────────────────────────────────

_SITAT = re.compile(r"leste dette inne i rammen: «(.*?)»")
_FNR = re.compile(r"(?<!\d)(\d[\d]{5}[ .-]?\d{5})(?!\d)")


def _svar_fra_tekst(tekst):
    """Stubbens «modell»: 11 siffer uten desimal og uten 4-2-5-gruppering."""
    m = _SITAT.search(tekst)
    if not m:
        return None
    sitat = m.group(1)
    if re.search(r"\d[.,]\d", sitat) or re.match(r"^\d{4} \d{2} \d{5}$",
                                                 sitat.strip()):
        return "nei"
    siffer = re.sub(r"\D", "", sitat)
    return "ja" if len(siffer) == 11 else "nei"


def lag_server(slem=False, tenker=False, avvis_reasoning=False):
    """Stub som snakker /v1/chat/completions.

    slem            injiserer 500-feil og prosa-svar
    tenker          svarer som en thinking-modell: tomt «content», alt i
                    «reasoning» — med mindre reasoning_effort=none er sendt
    avvis_reasoning svarer 400 på reasoning_effort, som eldre endepunkter
    """
    teller = {"n": 0, "reasoning": [], "avvist": 0}
    laas = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            krav = json.loads(self.rfile.read(n).decode("utf-8"))
            with laas:
                teller["n"] += 1
                i = teller["n"]
                teller["reasoning"].append(krav.get("reasoning_effort"))

            if avvis_reasoning and "reasoning_effort" in krav:
                with laas:
                    teller["avvist"] += 1
                self.send_error(400, "unknown field reasoning_effort")
                return

            if slem and i % 5 == 0:
                self.send_error(500, "syntetisk serverfeil")
                return

            biter = []
            for m in krav["messages"]:
                c = m["content"]
                if isinstance(c, str):
                    biter.append(c)
                else:
                    biter += [d.get("text", "") for d in c
                              if d.get("type") == "text"]
            tekst = "\n".join(biter)

            if slem and i % 7 == 0:
                innhold = "Dette ser ut som et fødselsnummer, ja."
            else:
                svar = _svar_fra_tekst(tekst) or ("ja", "nei", "usikker")[i % 3]
                innhold = json.dumps({"svar": svar, "sikkerhet": 90,
                                      "tall": "", "begrunnelse": "stub"})
            melding = {"role": "assistant", "content": innhold}
            if tenker and krav.get("reasoning_effort") != "none":
                melding = {"role": "assistant", "content": "",
                           "reasoning": "Hmm, la meg tenke grundig ..."}
            kropp = json.dumps({"choices": [{"message": melding}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(kropp)))
            self.end_headers()
            self.wfile.write(kropp)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", teller


# ── Kjøring ───────────────────────────────────────────────────

def kjor(*args):
    r = subprocess.run([sys.executable] + list(args), capture_output=True,
                       text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"FEIL: {' '.join(args[:2])} ga returkode {r.returncode}")
    return r.stdout


def les(sti):
    with open(sti, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def kryss(betingelse, melding):
    if not betingelse:
        raise SystemExit(f"FEIL: {melding}")
    print(f"  ok  {melding}")


def hoved(behold):
    rot = tempfile.mkdtemp(prefix="vlm_selvtest_")
    try:
        print(f"Arbeidsmappe: {rot}")
        pdf_mappe, ocr_mappe, fasit_csv, res_csv = bygg_data(rot)

        # ── Enhetsbiter uten server ──────────────────────────
        # Prompten er et produkt, ikke bare tekst: felter dommeren PARSER må
        # den faktisk be om. En stille mislykket erstatning her ga en hel
        # kjøring med tom sikkerhet-kolonne før den ble oppdaget.
        print("\n[0] prompten ber om feltene vi parser")
        from vlm_dommer import STD_PROMPT
        # Ikke «finnes ordet et sted i prompten» — det var eksemplene nok
        # til å oppfylle mens FELTLISTA manglet «svar», og modellen fulgte
        # lista. Kravet er at hvert felt parseren leser står forklart i lista.
        feltliste = STD_PROMPT[STD_PROMPT.index("Sjekklistefeltene skal stå"):
                               STD_PROMPT.index("Å svare nei")]
        for felt in ("svar", "sikkerhet", "tall", "begrunnelse"):
            kryss(f"«{felt}»" in feltliste,
                  f"feltlista forklarer «{felt}»")
            kryss(f'"{felt}"' in STD_PROMPT,
                  f"og eksemplene viser «{felt}»")
        kryss("usikker" in STD_PROMPT.lower(),
              "prompten tilbyr «usikker» som utvei")
        # Sjekklisten virker bare hvis den står FØR dommen: modellen er
        # autoregressiv, så et felt som kommer etter «svar» er en
        # etterrasjonalisering, ikke en kontroll.
        eksempel = STD_PROMPT[STD_PROMPT.index('{"linjen"'):]
        for felt in ("linjen", "sifre_i_rammen", "sifre_paa_linjen",
                     "dato_gyldig", "holdepunkt"):
            kryss(eksempel.index(f'"{felt}"') < eksempel.index('"svar"'),
                  f"sjekklistefeltet «{felt}» kommer før «svar» i JSON-en")
        kryss("12.03.50" in STD_PROMPT,
              "prompten sier eksplisitt at dato-punktum ikke er desimalskille")
        # Listen over nei-grunner må være LUKKET. Da modellen sto fritt,
        # fant den på «identifikasjonsnummer for elektronisk signatur».
        kryss("dikt aldri opp en kategori" in STD_PROMPT.lower(),
              "kategorilista er lukket mot oppdiktede grunner")
        # Ett eksempel per utfall — fem-siffer-fragmentet er hovedtapet.
        for utfall in ('"svar": "ja"', '"svar": "nei"', '"svar": "usikker"'):
            kryss(utfall in STD_PROMPT, f"prompten har et eksempel med {utfall}")

        print("\n[1] tolk_svar")
        kryss(tolk_svar('{"svar":"nei","tall":"66266","begrunnelse":"koordinat"}')
              [:1] == ("nei",), "ren JSON tolkes")
        kryss(tolk_svar('```json\n{"svar": "ja"}\n```')[0] == "ja",
              "JSON i kodeblokk tolkes")
        kryss(tolk_svar("Jeg mener dette er nei, en koordinat.")[0] == "nei",
              "prosa faller tilbake på nøkkelord")
        kryss(tolk_svar("")[0] == "usikker", "tomt svar blir usikker")
        kryss(tolk_svar("^^^")[0] == "usikker", "søppel blir usikker")
        # Den dyreste feilen så langt: full sjekkliste, men ingen «svar».
        uten = tolk_svar('{"linjen":"Dagboknr. 1234/1980","holdepunkt":"dagboknr",'
                         '"sikkerhet":95,"tall":"1234/1980"}')
        kryss(uten[0] == "usikker" and "utelot «svar»" in uten[4],
              f"manglende «svar» navngis presist ({uten[4]!r})")
        kryss(all(tolk_svar(t)[0] != "nei" for t in ("", "^^^", "{}")),
              "ingen feiltilstand kan bli «nei»")

        print("\n[1b] fnr-kandidat med pipelinens sifferforveksling")
        from vlm_evaluer import _fnr_kandidat
        from vlm_evaluer import _fnr_ledetekst
        kryss(_fnr_kandidat("loo190-00000"),
              "«loo190-00000» gjenkjennes — o→0 og l→1")
        # Denne kostet oss en ekte boks: fjernes mellomrommene limes løpet
        # sammen med «1f1g» fra «Iflg», og grensesjekken feiler. Pipelinens
        # finn_fnr arbeider på sifferposisjoner og unngår det.
        kryss(_fnr_kandidat("030392S0000 Iflg fullmakt"),
              "«030392S0000 Iflg fullmakt» gjenkjennes tross naboord")
        # Ledetekst-vernet: fødselsdatoen står i en annen form enn DDMMÅÅ,
        # så det finnes ikke noe ellevesifret løp å finne.
        for linje in ("f ø dt : 1.2.1950 Personnummer : . 00000",
                      "0la Nordmann , f . 12 / 3-1950 , pers . nr . 00000 ,"):
            kryss(_fnr_ledetekst(linje) and not _fnr_kandidat(linje),
                  f"ledetekst verner «{linje[:28]}…»")
        for linje in ("Dagboknr. 1234/1980", "Takstnr. 11,12,13,14.",
                      "N 6626630.58 632412.066"):
            kryss(not _fnr_ledetekst(linje),
                  f"ledetekst-vernet rører ikke «{linje[:24]}…»")
        kryss(_fnr_kandidat("02029100000") and _fnr_kandidat("020291 00000"),
              "rene og grupperte fødselsnumre gjenkjennes")
        kryss(not _fnr_kandidat("N 6626630.58"),
              "koordinat gjenkjennes IKKE som fnr")
        kryss(not _fnr_kandidat("Personnummer: 00000"),
              "fem sifre alene gjenkjennes ikke")
        kryss(not _fnr_kandidat("99999999999"),
              "elleve sifre uten gyldig dato gjenkjennes ikke")

        print("\n[2] poisson_ovre")
        kryss(abs(poisson_ovre(0) - 3.0) < 0.01,
              f"null observerte gir tre-regelen ({poisson_ovre(0):.2f})")
        kryss(poisson_ovre(1) > 4.7 and poisson_ovre(1) < 4.8,
              f"én observert gir ~4.74 ({poisson_ovre(1):.2f})")

        # ── Eksport ──────────────────────────────────────────
        print("\n[3] vlm_eksport")
        ut = os.path.join(rot, "eksport")
        kjor(os.path.join(HER, "vlm_eksport.py"),
             "--res-csv", res_csv, "--fasit-csv", fasit_csv,
             "--mappe", pdf_mappe, "--ut-mappe", ut,
             "--treff-utvalg", "2", "--ocr-cache", ocr_mappe, "--seed", "7")
        manifest = les(os.path.join(ut, "manifest.csv"))
        with open(os.path.join(ut, "utvalg.json"), encoding="utf-8") as f:
            utvalg = json.load(f)

        kryss(utvalg["n_bom_total"] == 4, f"4 BOM funnet ({utvalg['n_bom_total']})")
        kryss(utvalg["n_dekkende_total"] == 3,
              f"3 dekkende funnet ({utvalg['n_dekkende_total']})")
        kryss(utvalg["n_dekkende_eksportert"] == 2, "utvalget ble 2 dekkende")
        kryss(abs(utvalg["treff_faktor"] - 1.5) < 1e-9,
              f"treff_faktor = 1.5 ({utvalg['treff_faktor']})")
        kryss(len(manifest) == 6, f"6 manifestrader ({len(manifest)})")
        kryss(all(os.path.getsize(os.path.join(ut, "utsnitt", r["utsnitt"])) > 500
                  for r in manifest), "alle utsnitt skrevet og ikke tomme")
        for r in manifest:
            b = Image.open(os.path.join(ut, "utsnitt", r["utsnitt"]))
            kryss(b.size == (int(r["utsnitt_bredde"]), int(r["utsnitt_hoyde"]))
                  and 0 <= float(r["m_x0"]) and float(r["m_x1"]) <= b.width,
                  f"markøren ligger inne i {r['utsnitt']}")
            break
        kryss(all(r["ocr_linje"] for r in manifest),
              "OCR-linjen hentet for alle rader")
        treff = [r for r in manifest if r["klasse"] == "TREFF"]
        kryss(all(r["label_id"] for r in treff),
              "dekkende rader bærer label_id videre")
        rotert = [r for r in manifest if r["fil"] == "0100003.pdf"]
        kryss(len(rotert) >= 1 and all(
            int(r["utsnitt_hoyde"]) > int(r["utsnitt_bredde"]) for r in rotert),
              "roterte sider croppes i det oppreiste rommet (høyt utsnitt)")
        kryss(all(r["ocr_tekst"] in ("48089700000", "12") for r in rotert),
              "tallene leses ut av OCR-cachen på den roterte siden "
              f"({[r['ocr_tekst'] for r in rotert]})")

        # Bånd-logikken i _ocr_kontekst, målt direkte i det rommet den
        # faktisk ser: tokens fra pipelinen ligger alltid i den OPPREISTE
        # siden, der tekstlinjer er vannrette. Naboer over og under skal ikke
        # dras inn i linjen, og en delvis overlappende token skal.
        rekt = [210.0, 500.0, 400.0, 520.0]
        toks = [Token("Selger", 100, 500, 200, 520, 0.9),
                Token("07079600000", 210, 500, 400, 520, 0.9),
                Token("andel", 410, 502, 470, 518, 0.9),
                Token("overlinje", 100, 470, 470, 488, 0.9),
                Token("underlinje", 100, 532, 470, 550, 0.9),
                Token("", 210, 500, 400, 520, 0.9)]
        i_boks, i_linje, blokk = _ocr_kontekst(toks, rekt)
        kryss(i_boks == "07079600000", f"_ocr_kontekst finner boksen ({i_boks!r})")
        kryss(i_linje == "Selger 07079600000 andel",
              f"_ocr_kontekst finner linjen uten naboer ({i_linje!r})")
        kryss(blokk == "", "uten --ocr-linjer er blokken tom")
        _, _, b1 = _ocr_kontekst(toks, rekt, n_linjer=1)
        kryss(b1.split("\n") == ["overlinje",
                                 "Selger 07079600000 andel",
                                 "underlinje"],
              f"n_linjer=1 gir linjen med én nabo på hver side ({b1!r})")
        kryss(_ocr_kontekst([], rekt) == ("", "", ""),
              "tom token-liste er ufarlig")

        # Asymmetrisk margin og full sidebredde: bredere utsnitt skal
        # faktisk bli bredere, og --full-bredde skal treffe hele siden.
        bred = os.path.join(rot, "eksport_bred")
        kjor(os.path.join(HER, "vlm_eksport.py"),
             "--res-csv", res_csv, "--fasit-csv", fasit_csv,
             "--mappe", pdf_mappe, "--ut-mappe", bred, "--treff-utvalg", "0",
             "--margin-x", "200", "--margin-y", "20", "--maks-px", "0")
        m_bred = {r["fil"] + r["nr"]: r for r in
                  les(os.path.join(bred, "manifest.csv"))}
        smal = {r["fil"] + r["nr"]: r for r in manifest
                if r["klasse"] == "BOM"}
        kryss(all(int(m_bred[k]["utsnitt_bredde"]) > int(v["utsnitt_bredde"])
                  and int(m_bred[k]["utsnitt_hoyde"]) < int(v["utsnitt_hoyde"])
                  for k, v in smal.items() if k in m_bred),
              "--margin-x/-y virker uavhengig per akse")

        full = os.path.join(rot, "eksport_full")
        kjor(os.path.join(HER, "vlm_eksport.py"),
             "--res-csv", res_csv, "--fasit-csv", fasit_csv,
             "--mappe", pdf_mappe, "--ut-mappe", full, "--treff-utvalg", "0",
             "--full-bredde", "--maks-px", "0")
        sidebredde = round(SIDE_B * SKALA)
        kryss(all(abs(int(r["utsnitt_bredde"]) - sidebredde) <= 2
                  for r in les(os.path.join(full, "manifest.csv"))
                  if r["fil"] != "0100003.pdf"),
              "--full-bredde gir utsnitt i hele sidens bredde")

        # Parallellisering endrer HVEM som gjør jobben, ikke hva som kommer
        # ut. nr tildeles før arbeidet fordeles, så manifestene skal være
        # bit-identiske uansett antall prosesser.
        seriell = os.path.join(rot, "eksport_j1")
        parallell = os.path.join(rot, "eksport_j4")
        for mappe_ut, n in ((seriell, "1"), (parallell, "4")):
            kjor(os.path.join(HER, "vlm_eksport.py"),
                 "--res-csv", res_csv, "--fasit-csv", fasit_csv,
                 "--mappe", pdf_mappe, "--ut-mappe", mappe_ut,
                 "--treff-utvalg", "2", "--ocr-cache", ocr_mappe,
                 "--seed", "7", "--jobber", n)
        a = open(os.path.join(seriell, "manifest.csv"), "rb").read()
        b = open(os.path.join(parallell, "manifest.csv"), "rb").read()
        kryss(a == b, "--jobber 1 og --jobber 4 gir identisk manifest")
        kryss(sorted(os.listdir(os.path.join(seriell, "utsnitt")))
              == sorted(os.listdir(os.path.join(parallell, "utsnitt"))),
              "og identiske utsnittsnavn")
        fremdrift = kjor(os.path.join(HER, "vlm_eksport.py"),
                         "--res-csv", res_csv, "--fasit-csv", fasit_csv,
                         "--mappe", pdf_mappe, "--treff-utvalg", "0",
                         "--ut-mappe", os.path.join(rot, "eksport_frem"),
                         "--jobber", "2")
        kryss("dok/s" in fremdrift and "ETA" in fremdrift,
              "fremdriften skrives ut underveis")

        topp = os.path.join(rot, "eksport_topp")
        kjor(os.path.join(HER, "vlm_eksport.py"),
             "--res-csv", res_csv, "--fasit-csv", fasit_csv,
             "--mappe", pdf_mappe, "--ut-mappe", topp, "--treff-utvalg", "0",
             "--fra-toppen", "--full-bredde", "--maks-px", "0")
        m_topp = les(os.path.join(topp, "manifest.csv"))
        kryss(all(abs(int(r["utsnitt_bredde"]) - round(SIDE_B * SKALA)) <= 2
                  for r in m_topp if r["fil"] != "0100003.pdf"),
              "--fra-toppen + --full-bredde gir full sidebredde")
        kryss(all(float(r["m_y1"]) <= int(r["utsnitt_hoyde"])
                  and float(r["m_y0"]) >= 0 for r in m_topp),
              "markøren ligger innenfor bildet også i helsides utsnitt")
        # Boksens overkant i punkt = markørens y0, siden utsnittet nå
        # begynner på side-toppen. Da er høyden ~ boksens y1 + margin.
        kryss(all(int(r["utsnitt_hoyde"]) > int(r["utsnitt_bredde"]) * 0.4
                  for r in m_topp if r["fil"] != "0100003.pdf"),
              "utsnittene strekker seg ned fra toppen, ikke bare rundt boksen")

        # ── Dommer: snill stub, tekst-modus (semantikk) ──────
        print("\n[4] vlm_dommer --modus tekst  (snill stub)")
        srv, url, _ = lag_server(slem=False)
        try:
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "tekst",
                 "--samtidige", "2")
        finally:
            srv.shutdown()
        dommer = {d["nr"]: d for d in les(os.path.join(ut, "dommer_tekst.csv"))}
        kryss(len(dommer) == 6, f"6 dommer skrevet ({len(dommer)})")
        kryss(not any(d["feil"] for d in dommer.values()),
              "ingen feil mot snill stub")
        for r in manifest:
            fasit_svar = "ja" if r["klasse"] != "BOM" else "nei"
            kryss(dommer[r["nr"]]["svar"] == fasit_svar,
                  f"{r['utsnitt']}: dom «{dommer[r['nr']]['svar']}» "
                  f"som ventet")

        # ── Dommer: slem stub, bilde-modus (robusthet) ───────
        print("\n[5] vlm_dommer --modus bilde  (slem stub: 500-feil + prosa)")
        srv, url, teller = lag_server(slem=True)
        try:
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "bilde",
                 "--samtidige", "1", "--forsok", "1", "--timeout", "20")
            bilde_csv = os.path.join(ut, "dommer_bilde.csv")
            d1 = les(bilde_csv)
            kryss(len(d1) == 6, f"alle 6 rader skrevet tross feil ({len(d1)})")
            kryss(any(r["feil"] for r in d1), "minst én feil ble logget")
            kryss(all(r["svar"] in ("ja", "nei", "usikker") for r in d1),
                  "alle svar er gyldige verdier")
            kryss(all(r["svar"] == "usikker" for r in d1
                      if r["feil"].startswith(("HTTPError", "URLError"))),
                  "nettverksfeil ble «usikker», aldri «nei»")
            n_feil = sum(1 for r in d1 if r["feil"])

            # Gjenopptagelse: de feilede radene skal prøves på nytt
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "bilde",
                 "--samtidige", "1", "--gjenoppta")
            d2 = les(bilde_csv)
            kryss(len(d2) == 6 + n_feil,
                  f"--gjenoppta la til nøyaktig de {n_feil} feilede radene "
                  f"({len(d2) - 6})")
        finally:
            srv.shutdown()

        # ── Tenkning: qwen3-vl:8b ER thinking-varianten ──────
        print("\n[5b] --tenkning mot en modell som vil tenke")
        srv, url, teller = lag_server(tenker=True)
        try:
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "tekst",
                 "--ut-csv", os.path.join(ut, "d_tenk.csv"), "--samtidige", "1")
            kryss(set(teller["reasoning"]) == {"none"},
                  "reasoning_effort=none sendes som standard")
            d = les(os.path.join(ut, "d_tenk.csv"))
            kryss(all(not r["feil"] for r in d),
                  "tenkningen ble slått av — ekte svar, ingen feil")

            # Med --tenkning auto utelates feltet, stubben tenker, content
            # blir tomt: da skal feilen SI at det var tenkningen.
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "tekst",
                 "--ut-csv", os.path.join(ut, "d_auto.csv"),
                 "--samtidige", "1", "--forsok", "1", "--tenkning", "auto",
                 "--maks", "2")
            d = les(os.path.join(ut, "d_auto.csv"))
            kryss(all("tenkte" in r["feil"] for r in d),
                  "tom «content» diagnostiseres som tenkning, ikke anonym feil")
            kryss(all(r["svar"] == "usikker" for r in d),
                  "og blir «usikker», ikke «nei»")
        finally:
            srv.shutdown()

        print("\n[5e] --nr-fil velger ut rader")
        srv, url, _ = lag_server(slem=False)
        try:
            nr_fil = os.path.join(ut, "harde.txt")
            with open(nr_fil, "w", encoding="utf-8") as f:
                f.write("# de vanskelige\n2\n5\n")
            sti = os.path.join(ut, "d_nr.csv")
            utskrift = kjor(os.path.join(HER, "vlm_dommer.py"), "--manifest",
                            os.path.join(ut, "manifest.csv"), "--url", url,
                            "--modell", "stub", "--modus", "tekst",
                            "--ut-csv", sti, "--samtidige", "1",
                            "--nr-fil", nr_fil)
            d = les(sti)
            kryss("--nr-fil: 2 av 6" in utskrift and len(d) == 2,
                  "bare de to oppgitte radene ble dømt")
            kryss({r["nr"] for r in d} == {"2", "5"},
                  "og det var riktige rader")
        finally:
            srv.shutdown()

        print("\n[5d] gjenopptagelse er standard")
        srv, url, _ = lag_server(slem=False)
        try:
            sti = os.path.join(ut, "d_std.csv")
            kjor(os.path.join(HER, "vlm_dommer.py"), "--manifest",
                 os.path.join(ut, "manifest.csv"), "--url", url, "--modell",
                 "stub", "--modus", "tekst", "--ut-csv", sti, "--samtidige", "1")
            kryss(len(les(sti)) == 6, "første kjøring dømmer alle 6")
            ut2 = kjor(os.path.join(HER, "vlm_dommer.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--url", url,
                       "--modell", "stub", "--modus", "tekst", "--ut-csv", sti,
                       "--samtidige", "1")
            kryss("Ingenting å gjøre" in ut2 and len(les(sti)) == 6,
                  "andre kjøring uten flagg gjør INGENTING — ferdig arbeid "
                  "overskrives ikke")
            ut3 = kjor(os.path.join(HER, "vlm_dommer.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--url", url,
                       "--modell", "stub", "--modus", "tekst", "--ut-csv", sti,
                       "--samtidige", "1", "--start-paa-nytt")
            kryss("overskriver" in ut3 and len(les(sti)) == 6,
                  "--start-paa-nytt overskriver, og sier fra at den gjør det")
        finally:
            srv.shutdown()

        print("\n[5c] endepunkt som ikke kjenner reasoning_effort")
        srv, url, teller = lag_server(avvis_reasoning=True)
        try:
            kjor(os.path.join(HER, "vlm_dommer.py"),
                 "--manifest", os.path.join(ut, "manifest.csv"),
                 "--url", url, "--modell", "stub", "--modus", "tekst",
                 "--ut-csv", os.path.join(ut, "d_400.csv"), "--samtidige", "1")
            d = les(os.path.join(ut, "d_400.csv"))
            kryss(teller["avvist"] == 1,
                  f"bare FØRSTE kall ble avvist ({teller['avvist']}) — "
                  f"feltet slås av for resten")
            kryss(len(d) == 6 and all(not r["feil"] for r in d),
                  "alle 6 dommer kom gjennom etter fallbacken")
        finally:
            srv.shutdown()

        # ── Evaluering ───────────────────────────────────────
        print("\n[6] vlm_evaluer")
        ut_tekst = kjor(os.path.join(HER, "vlm_evaluer.py"),
                        "--manifest", os.path.join(ut, "manifest.csv"),
                        "--dommer", os.path.join(ut, "dommer_tekst.csv"),
                        "--ut-mappe", os.path.join(ut, "evaluering"))
        with open(os.path.join(ut, "evaluering", "oppsummering.json"),
                  encoding="utf-8") as f:
            opp = json.load(f)
        kryss(opp["n_bom_nei"] == 4, f"4 BOM fikk nei ({opp['n_bom_nei']})")
        kryss(opp["n_dek_nei"] == 0, f"0 dekkende fikk nei ({opp['n_dek_nei']})")
        kryss(abs(opp["gevinst"] - 4.0) < 1e-6,
              f"gevinst skalert til 4 ({opp['gevinst']})")
        # Delvis kjøring: skaleres tapet opp må gevinsten skaleres like mye,
        # ellers ser ov/tapt katastrofalt ut av rene bokføringsgrunner.
        halv = os.path.join(ut, "d_halv.csv")
        rader_alle = les(os.path.join(ut, "dommer_tekst.csv"))
        with open(halv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rader_alle[0].keys())
            w.writeheader()
            w.writerows([r for r in rader_alle
                         if r["klasse"] == "BOM"][:2]
                        + [r for r in rader_alle if r["klasse"] != "BOM"][:1])
        ut_halv = kjor(os.path.join(HER, "vlm_evaluer.py"), "--manifest",
                       os.path.join(ut, "manifest.csv"), "--dommer", halv,
                       "--ut-mappe", os.path.join(ut, "ev_halv"))
        kryss("Oppskalering BOM:          2.00" in ut_halv,
              "BOM skaleres 4/2 = 2.00 ved delvis kjøring")
        kryss("Oppskalering dekkende:     3.00" in ut_halv,
              "dekkende skaleres 3/1 = 3.00, ikke eksportfaktoren 1.5")
        kryss(abs(opp["tap_ovre"] - 3.0 * 1.5) < 0.05,
              f"øvre tapsgrense = 3 × faktor 1.5 = 4.5 ({opp['tap_ovre']:.2f})")
        # Null tap observert => punktestimatet er uendelig, men den øvre
        # grensen (4.5 tap) gjør ikke jobben. Nettopp derfor skal dommen bli
        # USIKKER og ikke BESTÅTT: et utvalg på to bokser beviser ingenting.
        kryss("DOM: USIKKER" in ut_tekst,
              "dommen er USIKKER — null tap i et bitte lite utvalg beviser "
              "ingenting")
        kryss("∞" in ut_tekst, "punktestimatet vises som uendelig")
        tapt = les(os.path.join(ut, "evaluering", "tapt.csv"))
        gevinst = les(os.path.join(ut, "evaluering", "gevinst.csv"))
        kryss(len(tapt) == 0 and len(gevinst) == 4,
              "tapt.csv tom, gevinst.csv har 4 rader")
        kryss("label_id" in les(os.path.join(ut, "evaluering",
                                             "gevinst.csv"))[0],
              "manifestene har label_id-kolonne (samme format som "
              "filter_review)")

        kryss(all(d["sikkerhet"] == "90" for d in dommer.values()),
              "sikkerhet bæres gjennom til dommer-CSV-en")
        ut_kurve = kjor(os.path.join(HER, "vlm_evaluer.py"),
                        "--manifest", os.path.join(ut, "manifest.csv"),
                        "--dommer", os.path.join(ut, "dommer_tekst.csv"),
                        "--ut-mappe", os.path.join(ut, "ev_kurve"))
        kryss("SIKKERHETSKURVE" in ut_kurve, "sikkerhetskurven skrives ut")
        ut_terskel = kjor(os.path.join(HER, "vlm_evaluer.py"),
                          "--manifest", os.path.join(ut, "manifest.csv"),
                          "--dommer", os.path.join(ut, "dommer_tekst.csv"),
                          "--ut-mappe", os.path.join(ut, "ev_terskel"),
                          "--min-sikkerhet", "95")
        kryss("4 «nei»-dommer nedgradert" in ut_terskel,
              "--min-sikkerhet nedgraderer «nei» under terskelen")

        # --fnr-overstyring: modellen kan si nei så mye den vil — er det
        # elleve sifre med gyldig dato i dens EGEN avskrift, skal koden
        # overprøve. Her forfalskes en dommer-CSV med nettopp det tilfellet.
        falsk = os.path.join(ut, "d_overstyr.csv")
        with open(falsk, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "nr", "svar", "sikkerhet", "tall", "begrunnelse", "linjen",
                "sifre_paa_linjen", "dato_gyldig", "holdepunkt", "feil"])
            w.writeheader()
            for r in manifest:
                fnr = r["klasse"] != "BOM"
                w.writerow({
                    "nr": r["nr"], "svar": "nei", "sikkerhet": 95,
                    "tall": "", "begrunnelse": "påstått org.nr",
                    "linjen": "Kari Nordmann 010190 00000" if fnr else "N 6626630.58",
                    "sifre_paa_linjen": "01019000000" if fnr else "662663058",
                    "dato_gyldig": "true" if fnr else "false",
                    "holdepunkt": "", "feil": ""})
        ut_ov = kjor(os.path.join(HER, "vlm_evaluer.py"), "--manifest",
                     os.path.join(ut, "manifest.csv"), "--dommer", falsk,
                     "--ut-mappe", os.path.join(ut, "ev_ov"),
                     "--fnr-overstyring")
        # Vernet skal også slå inn når det bare er PADDLE som har lest
        # datohalvdelen — manifestets ocr_linje, uten hjelp fra modellen.
        kun_ocr = os.path.join(ut, "d_kunocr.csv")
        with open(kun_ocr, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["nr", "svar", "sikkerhet", "tall",
                                              "begrunnelse", "feil"])
            w.writeheader()
            for r in manifest:
                w.writerow({"nr": r["nr"], "svar": "nei", "sikkerhet": 95,
                            "tall": "", "begrunnelse": "", "feil": ""})
        ut_ocr = kjor(os.path.join(HER, "vlm_evaluer.py"), "--manifest",
                      os.path.join(ut, "manifest.csv"), "--dommer", kun_ocr,
                      "--ut-mappe", os.path.join(ut, "ev_kunocr"),
                      "--fnr-overstyring")
        kryss("--fnr-overstyring: 2 dommer endret" in ut_ocr,
              "manifestets OCR-linje alene verner de dekkende boksene")

        kryss("--fnr-overstyring: 2 dommer endret" in ut_ov,
              "overstyringen redder begge de dekkende boksene")
        with open(os.path.join(ut, "ev_ov", "oppsummering.json"),
                  encoding="utf-8") as f:
            o = json.load(f)
        kryss(o["n_dek_nei"] == 0 and o["n_bom_nei"] == 4,
              f"null tap, gevinsten urørt ({o['n_dek_nei']}/{o['n_bom_nei']})")

        # Skjevt utvalg: dømmer man bare BOM-radene, skal verktøyet si at
        # oppskaleringen ikke gjelder — ikke stille regne feil.
        bare_bom = os.path.join(ut, "bare_bom.txt")
        with open(bare_bom, "w", encoding="utf-8") as f:
            f.write("\n".join(r["nr"] for r in manifest
                               if r["klasse"] == "BOM"))
        srv, url, _ = lag_server(slem=False)
        try:
            kjor(os.path.join(HER, "vlm_dommer.py"), "--manifest",
                 os.path.join(ut, "manifest.csv"), "--url", url, "--modell",
                 "stub", "--modus", "tekst", "--samtidige", "1",
                 "--nr-fil", bare_bom,
                 "--ut-csv", os.path.join(ut, "d_skjev.csv"))
        finally:
            srv.shutdown()
        ut_skjev = kjor(os.path.join(HER, "vlm_evaluer.py"), "--manifest",
                        os.path.join(ut, "manifest.csv"), "--dommer",
                        os.path.join(ut, "d_skjev.csv"), "--ut-mappe",
                        os.path.join(ut, "ev_skjev"))
        kryss("ADVARSEL" in ut_skjev and "ikke tilfeldig" in ut_skjev,
              "skjevt utvalg utløser advarsel")
        kryss("NB: regnskapet over hviler" in ut_skjev,
              "og advarselen gjentas ved regnskapet")
        kryss("ADVARSEL" not in ut_tekst,
              "et fullstendig utvalg gir INGEN advarsel")

        # Regelbasislinje: samme regnskap, ingen modell. På de syntetiske
        # dataene har bare fnr-linjene et gyldig 11-sifret løp, så regelen
        # skal treffe nøyaktig som en perfekt modell.
        ut_regel = kjor(os.path.join(HER, "vlm_evaluer.py"),
                        "--manifest", os.path.join(ut, "manifest.csv"),
                        "--dommer", "regel:fnr-kandidat",
                        "--ut-mappe", os.path.join(ut, "ev_regel"))
        kryss("Basislinje" in ut_regel, "regel:-basislinjen kjører uten modell")
        with open(os.path.join(ut, "ev_regel", "oppsummering.json"),
                  encoding="utf-8") as f:
            o = json.load(f)
        kryss(o["n_bom_nei"] == 4 and o["n_dek_nei"] == 0,
              f"fnr-kandidat-regelen tar 4 BOM og null dekkende "
              f"({o['n_bom_nei']}/{o['n_dek_nei']})")

        # --del-etter og --bare: verktøyet for å lete etter delmengder der
        # modellen er trygg. har_tokens=1 på alle syntetiske rader, så
        # delmengden skal være hele settet — og et vilkår som ikke treffer
        # skal si fra i stedet for å regne på tomme tall.
        ut_del = kjor(os.path.join(HER, "vlm_evaluer.py"),
                      "--manifest", os.path.join(ut, "manifest.csv"),
                      "--dommer", os.path.join(ut, "dommer_tekst.csv"),
                      "--ut-mappe", os.path.join(ut, "ev_del"),
                      "--del-etter", "kilde", "--bare", "har_tokens=1")
        kryss("DELT ETTER kilde" in ut_del and "Delmengde" in ut_del,
              "--del-etter og --bare kjører sammen")
        kryss("6 av 6 rader" in ut_del, "har_tokens=1 beholder alle radene")
        ut_tom = kjor(os.path.join(HER, "vlm_evaluer.py"),
                      "--manifest", os.path.join(ut, "manifest.csv"),
                      "--dommer", os.path.join(ut, "dommer_tekst.csv"),
                      "--ut-mappe", os.path.join(ut, "ev_tom"),
                      "--bare", "kilde=finnesikke")
        kryss("Ingen rader igjen" in ut_tom,
              "tomt vilkår gir beskjed, ikke et regnestykke på null")

        # Med --usikker-fjerner skal bilde-armens usikre slå ut som tap
        kjor(os.path.join(HER, "vlm_evaluer.py"),
             "--manifest", os.path.join(ut, "manifest.csv"),
             "--dommer", os.path.join(ut, "dommer_bilde.csv"),
             "--ut-mappe", os.path.join(ut, "evaluering_bilde"),
             "--usikker-fjerner")
        kryss(os.path.isfile(os.path.join(ut, "evaluering_bilde", "tapt.csv")),
              "--usikker-fjerner kjører og skriver tapt.csv")

        print("\nALT OK.")
        if behold:
            print(f"Filene ligger i {rot}")
        return rot
    finally:
        if not behold:
            shutil.rmtree(rot, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--behold", action="store_true",
                   help="Ikke slett arbeidsmappa etterpå")
    hoved(p.parse_args().behold)
