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
# sladdet, mot ~53 fjernede oversladdinger. Conf ≥ YOLO_CONF_GEOMETRI_TERSKEL
# fritar, som for grensene over.
MIN_KORTSIDE_YOLO_PT = 7       # smalere enn dette er støy
MIN_LANGSIDE_YOLO_PT = 20      # for kort til å romme 5 sifre

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

# Trekkene boks_trekk beregner per YOLO-boks og som skrives til resultat-CSV-en.
# Navnet bor her, ikke i boks_trekk, fordi utils/csv_export.py og
# utils/filter_felles.py trenger listen uten å dra inn PaddleOCR.
TREKK_FELT = ("har_tokens", "n_siffer", "n_bokstaver", "rec_min", "rec_median",
              "rec_min_linje", "n_siffer_linje", "siffer_run",
              "har_fnr_kandidat", "har_desimal_naer")

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
