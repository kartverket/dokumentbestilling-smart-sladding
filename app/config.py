# ---- PDF-rendering -----------------------------------------
PDF_DPI = 300                  # oppløsning ved rendering (høyere = tregere, mer nøyaktig)

# ---- Dimensjonsfiltre ------------------------------------------------------
# Grensene er UNIVERSELLE: samme verdier for alle kilder (paddle, yolo, begge).
# Eneste unntak er høy konfidens, se YOLO_CONF_GEOMETRI_TERSKEL — det gjelder
# også likt for alle kilder, men paddle har ingen konfidens fra OCR og blir
# derfor alltid filtrert.
#
# Grensene er orienteringsuavhengige: en sladding av 5 sifre kan stå loddrett,
# og da er «høyde» den lange siden. Tidligere MAKS_BOKS_HOYDE_PT/…_BREDDE_PT
# målte høyde og bredde hver for seg og forkastet derfor stående sladdinger
# systematisk — 30 av 34 feilaktige fjerninger i gjennomgangen av uttrekk 4.
#
# Verdiene er utledet fra 41 manuelt gjennomgåtte tap på uttrekk 4
# (utils/filter_sweep.py + filter_review.py). Resultat med disse: 64 fjernede
# oversladdinger, 7 tapte fasit-bokser, alle 7 bekreftet som bokser som ikke
# skulle vært sladdet. Holdout-validert på uavhengige dokumenter.
MIN_BOKS_AREAL     = 965       # bokser med mindre areal regnes som støy (px²) — gjelder alle
MIN_ELONGATION     = 1.44      # min max(w/h, h/w) — forkaster nesten-kvadratiske bokser
                               # (1.5 tok 3 ekte sladdinger på 1.47-1.49)
MAKS_ELONGATION    = 9         # maks max(w/h, h/w) — bokser 3-4x bredere enn feltet
MIN_KORTSIDE_PT    = 6         # min korteste side i punkt — for tynn til å være tekst

# Strengere formkrav for rene YOLO-bokser (kun kilde «yolo» — «begge» og
# paddle har et Paddle-funn bak seg og rammes ikke, «yolo_vertikal» heller
# ikke). Utledet fra uttrekk 6 på samme måte som grensene over: 4 tapte
# fasit-bokser, alle bekreftet som referansenumre som ikke skulle vært
# sladdet, mot ~53 fjernede oversladdinger. Kortsidekravet gjelder uansett
# konfidens (runde 2: 0 tapt uten conf-port); langsidekravet fritas ved
# conf ≥ YOLO_CONF_GEOMETRI_TERSKEL som resten av geometrien.
MIN_KORTSIDE_YOLO_PT = 7       # smalere enn dette er støy
MIN_LANGSIDE_YOLO_PT = 20      # for kort til å romme 5 sifre

# Strengere formkrav for paddle-bokser (kun kilde «paddle», ikke «begge»).
# Paddle fritas aldri av konfidens — OCR-konfidens er lesekvalitet, ikke
# deteksjonssikkerhet. Utledet fra uttrekk 6 runde 2 (etter runde 1-
# filtrene): 0 tapte fasit-bokser på både trening og holdout, ~20 fjernede
# oversladdinger. Paddle-BOM er tynne streker: kortside p99.9 = 14,6pt der
# minste ekte paddle-treff er 8,6pt.
MIN_KORTSIDE_PADDLE_PT = 7
MIN_LANGSIDE_PADDLE_PT = 20
MAKS_ELONGATION_PADDLE = 6

# Bokser med conf ≥ dette hopper over geometrifiltrene. Gjelder alle kilder;
# paddle-bokser har conf=None og fritas aldri. «begge»-bokser var tidligere
# fritatt uansett konfidens — det er fjernet, se _hopp_over_geometrifilter.
YOLO_CONF_GEOMETRI_TERSKEL = 0.5

# Kun default for --maks-bredde i utils/tegn.py; ikke del av filterstien.
MAKS_BREDDE_ELEKTRONISK_PT = 50

# ---- YOLO-vekter -------------------------------------------
# Vektene bor ikke i repoet. I containeren har ./deploy.sh bygget inn den
# valgte modellen som weights/modell.pt; utenfor containeren peker
# SLADD_PRODVEKTER (server.env) på en modell i vektlageret. YOLO_VEKTER
# overstyrer begge, og --yolo-vekter overstyrer alt (utils/run.py).
import os as _os


def standard_vekter():
    for var in ("YOLO_VEKTER", "SLADD_PRODVEKTER"):
        sti = _os.environ.get(var)
        if sti:
            return sti
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "weights", "modell.pt")


# ---- YOLO --------------------------------------------------
YOLO_CONF          = 0.12      # predict-terskel
YOLO_CONF_UTEN_TEKST = 0.40   # krav når Paddle ikke leste noe i boksen
YOLO_CONF_VERTIKAL = 0.90     # vertikale bokser (stående tekst)
VERTIKAL_FAKTOR    = 1.3       # høyde > 1.3 × bredde regnes som vertikal
YOLO_IMGSZ         = 1280      # bildestørrelse inn til YOLO
MIN_SIFFER         = 1         # minst så mange siffer i boksen (snill-sjekk)
MAKS_BOKSTAVER     = 1         # 2+ bokstaver, ikke FNR, uansett YOLO

# Desimalregelen (uttrekk 6): står det et desimalskille i tallet OG Paddle
# leste teksten sikkert, er boksen en koordinat eller et beløp — fnr har
# aldri desimalskille. Høy deteksjonskonfidens fritar: manuell gjennomgang av
# samtlige tap viste at ekte fnr som rammes nesten alle har conf ≥ 0.5
# (typisk fnr skrevet «ddmmåå.xxxxx»), mens koordinater ligger ≤ 0.37.
# Anvendes på ENDELIG kilde etter dedup — bare rene «yolo»-bokser. Målt på
# uttrekk 6: ~1250 fjernede oversladdinger mot 3 tapte ekte fnr (alle i
# conf-båndet 0.49–0.57). Se _desimalregel_forkaster i model_main.py og
# _ocr_grunn i utils/filter_felles.py.
AVVIS_DESIMAL_REC_VETO    = 0.98   # regelen gjelder først når rec_min ≥ dette
AVVIS_DESIMAL_CONF_FRITAK = 0.6    # conf ≥ dette overstyrer regelen
# Lavere tier (uttrekk 6, full fasit): rec_min i [0.95, 0.98) er også bevis
# nok NÅR deteksjonen selv er svak — målt +38 fjernede oversladdinger mot
# 1 tapt.
AVVIS_DESIMAL_REC_VETO_LAV  = 0.95
AVVIS_DESIMAL_CONF_TAK_LAV  = 0.4  # lav-tier gjelder bare når conf < dette

# Linjebevis-reglene (uttrekk 6 med fasit som dekker hele uttrekket; manuelt
# labelet i tre runder — pakke C): når HELE linjen er sikkert lest, kan
# tallet bevises å ikke være et fnr. To bevistyper:
#   * sifferløp på 6-8 (med luker): for langt for nakent personnummer, for
#     kort for fnr — dagboknr, beløp, koordinat. 9-løp er BEVISST utenfor:
#     manuell gjennomgang viste at de ofte er ekte fnr der OCR selvsikkert
#     har mistet to tegn (samme sykdom som felte 10-løpene).
#   * gyldig orgnr-mod11 i boksen.
# Målt: 79 fjernede oversladdinger mot 1 ekte fnr (conf 0.463, rett under
# fritaket) + 1 fasit-støy.
LINJEBEVIS_LINJE_VETO  = 0.99  # rec_min_linje ≥ dette for at reglene gjelder
LINJEBEVIS_CONF_FRITAK = 0.5   # conf ≥ dette overstyrer reglene
LINJEBEVIS_RUN_MAKS    = 8     # sifferløp 6..8 forkastes

# Paddle-vindu-reglene: forkast paddle-bokser der 11-siffer-vinduet boksen
# ble bygget fra er sydd sammen over et desimalskille eller en stor fysisk
# luke UTENFOR de lovlige posisjonene (etter siffer 2/4/6 — datoformatets
# punktum og skilletegnet/feltskillet). Koordinat- og målekolonner
# («6626630.58 549810.29») og spredte skissemål er kilden; et ekte fnr har
# aldri slike luker. Gjelder KUN endelig kilde «paddle» — «begge» er
# yolo-bekreftet og fritas (posisjonsblind variant på begge-bokser målte
# 974 tapte fnr: datoformat/OCR-prikker/håndskriftsgap er vanlige i ekte
# fnr). Terskel 8, ikke 3: håndskrevne fnr kan ha luker opp mot ~6 siffer-
# bredder (målt 5.94 på ekte fnr), koordinat-gap ligger typisk ≥10.
# Målt uttrekk 6 (vindu2): 72 oversladdinger fjernet / 0 tapte fnr
# (luke 3: 91/1 — netto ved kostnad 20: 71 mot 72, og null tap vinner).
VINDU_MAKS_LUKE          = 8.0   # sifferbredder, luker utenfor posisjon 2/4/6
VINDU_AVVIS_DESIMAL_LUKE = True  # desimalskille i luke utenfor 2/4/6

# ── Regelprofil per rettsstiftelsestype ──────────────────────────
# Koordinat-familien: jordskifte, målebrev, grensejustering, massetransport,
# skjønn og jordsameie — dokumenter som er kart, måltabeller og koordinat-
# lister. Målt på uttrekk 6: 221 dokumenter, 865 prediksjoner, presisjon
# 21 % — mot 88,5 % ellers. I disse dokumentene aktiveres koordfam-regelen
# (_koordfam_forkaster): YOLO-bokser med lest tekst forkastes når linjen
# mangler 11-sifret fnr-kandidat eller tallet har desimalskille. Globalt
# koster den regelen hundrevis av ekte fnr; innenfor familien er den målt
# til 576 fjernede oversladdinger mot 0 ekte fnr (5 «tap» var fasit-støy,
# manuelt bekreftet). Dokumentets koder kommer fra skip-jobben via
# dokumentbestilling-API-et; mangler de, gjelder dagens globale oppførsel.
KOORDFAM_KODER = frozenset((
    "SR_JOU",   # Jordskifte
    "AH_JOU",   # Jordskifte (annen hjemmel)
    "KA_MOB",   # Målebrev
    "KA_GRE",   # Grensejustering
    "TR_MAS",   # Massetransport
    "SR_SKN",   # Skjønn
    "JS_JSA",   # Opprettelse av jordsameie
    "FR_REG",   # Registrering av grunn — oppmålingsdokumenter; målt
                # marginalt: 183 ov.fj / 1 ekte fnr + 1 usikker (ov/tapt ≥91)
    "SR_UTS",   # Utskifting — historisk jordskifte
))

# Seksjonering-profilen: SE_SEK-dokumenter er tabelltunge (eierbrøker,
# arealer, seksjonsnumre) — presisjon 72 % mot 92 % ellers, og BOM-ene er
# tabellceller: for store bokser eller korte talløp. Målt på uttrekk 6:
# 143 oversladdinger fjernet / 2 ekte fnr tapt (ov/tapt 71). De to tapene
# er fnr der Paddle bare leste 5 av 11 siffer selvsikkert (kjent klasse,
# jf. rec-score-kalibreringen) — smin 5 ville reddet dem, men koster ~80
# oversladdinger (40 per fnr > kostnad 20). Geometrien er ORIENTERINGSFRI
# (kortside/langside), så prod (rotert rom) og analysen (side-rom) er
# bit-like også på roterte sider. NB: koordfam-reglene er DØDELIGE her
# (fnr står i samme tabellinjer som desimal-arealene: 130 tapt globalt) —
# derfor egne profiler per dokumenttype. Kun SE_SEK er målt;
# reseksjonering-slektningene (RS_RES, SB_SEB, …) er fnr-tette og holdes
# utenfor til de eventuelt måles for seg.
# Tokenløse bokser i koordfam-dokumenter er kart-/grafikkdeteksjoner —
# tekstreglene over ser dem aldri (har_tokens=0), så de krever i stedet
# høy deteksjons-conf. Målt marginalt (uttrekk6_sesek, med profilen aktiv):
# 26 oversladdinger fjernet / 0 tapt ved 0.7 — samme tall som før profilen,
# gevinsten er uavhengig av fnr-kandidat-regelen. Globalt er samme regel
# tapsgivende (tokenløse bokser over conf 0.4 er oftere ekte fnr enn støy —
# uten_tekst_conf-hypotesen døde globalt); den er trygg KUN i koordfam.
KOORDFAM_UTEN_TEKST_CONF = 0.7

SEKSJONERING_KODER = frozenset(("SE_SEK",))
SEKSJONERING_MAKS_KORTSIDE_PT = 40.0  # yolo+paddle: fnr-sladd p99.9 = 29.5
SEKSJONERING_MAKS_LANGSIDE_PT = 80.0  # yolo+paddle: fnr-sladd p99.9 = 78.9
SEKSJONERING_PADDLE_MIN_ELONG = 3.0   # paddle: kvadratiske celler forkastes
SEKSJONERING_MIN_SIFFER  = 6          # yolo: færre lest siffer = brøk/snr
SEKSJONERING_REC_VETO    = 0.98       # sifferkravet kun ved sikker lesning
SEKSJONERING_CONF_FRITAK = 0.5        # høy deteksjons-conf overstyrer

# Trekkene boks_trekk beregner per YOLO-boks og som skrives til resultat-CSV-en.
# Navnet bor her, ikke i boks_trekk, fordi utils/csv_export.py og
# utils/filter_felles.py trenger listen uten å dra inn PaddleOCR.
# De to siste er VINDU-trekk og finnes kun for paddle/begge-bokser: de
# beskriver 11-siffer-vinduet boksen ble bygget fra (paddle_ocr_model_fnr.
# _vindu_trekk). maks_luke = største fysiske avstand mellom to nabosiffer i
# vinduet, i median sifferbredde; har_desimal_luke = 1 når en luke inneholder
# desimalskille (. eller ,). Ekte fnr har verken store luker eller desimaler —
# koordinat- og målekolonner («6626630.58 549810.29») har begge.
TREKK_FELT = ("har_tokens", "n_siffer", "n_bokstaver", "rec_min", "rec_median",
              "rec_min_linje", "n_siffer_linje", "siffer_run",
              "har_fnr_kandidat", "har_desimal_naer",
              "har_00_run", "har_orgnr", "har_org_ord", "lang_run",
              "maks_luke", "har_desimal_luke")

# Konfidens-gulv YOLO-cachen skrives med (utils/run.py --yolo-cache). Må ligge
# under alle terskler man vil kunne endre uten å invalidere cachen: bokser
# lagres ned til gulvet og filtreres mot YOLO_CONF ved lesing. Merk at et lavere
# gulv sender flere kandidater inn i NMS enn en ren predict på YOLO_CONF gjør;
# boksene som overlever YOLO_CONF blir de samme, siden NMS alltid undertrykker
# med en høyere-skårende boks.
YOLO_CACHE_CONF_GULV = 0.05

# ---- Paddle OCR --------------------------------------------
MODELL_SETT        = "v6"      # "v5" eller "v6"
DET_SIDE_LEN       = 2048      # deteksjon: maks sidelengde i piksler
REC_BATCH          = 64        # tekstlinjer per gjenkjennings-batch
SIDER_PER_OCR_BATCH = 8        # sider per predict-kall (GPU-utnyttelse)

# ---- Sladde-boks -------------------------------------------
SLADDE_SIFFER      = 5         # antall sifre som vises i sladde-boksen
LUFT_X             = 0.35      # horisontal utvidelse (andel av sifferbredde)
LUFT_Y             = 0.0       # vertikal utvidelse
MAKS_HOYDE_FAKTOR  = 3.0       # sladde-høyde maks N × median sifferbredde

# ---- Pipeline ----------------------------------------------
DEDUP_OVERLAPP     = 0.5       # YOLO-boks regnes som "samme" når den dekker Paddle-boks så mye

# ---- Orientering -------------------------------------------
NEDSKALERING       = 4         # nedskalering av bilde før orienteringssjekk
MIN_KONFIDENS      = 0.7       # under dette: stol ikke på gjetningen, ikke roter
