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

# Bokser med conf ≥ dette hopper over geometrifiltrene. Gjelder alle kilder;
# paddle-bokser har conf=None og fritas aldri. «begge»-bokser var tidligere
# fritatt uansett konfidens — det er fjernet, se _hopp_over_geometrifilter.
YOLO_CONF_GEOMETRI_TERSKEL = 0.5

# Kun default for --maks-bredde i utils/tegn.py; ikke del av filterstien.
MAKS_BREDDE_ELEKTRONISK_PT = 50

# ---- YOLO --------------------------------------------------
YOLO_CONF          = 0.12      # predict-terskel
YOLO_CONF_UTEN_TEKST = 0.40   # krav når Paddle ikke leste noe i boksen
YOLO_CONF_VERTIKAL = 0.90     # vertikale bokser (stående tekst)
VERTIKAL_FAKTOR    = 1.3       # høyde > 1.3 × bredde regnes som vertikal
YOLO_IMGSZ         = 1280      # bildestørrelse inn til YOLO
MIN_SIFFER         = 1         # minst så mange siffer i boksen (snill-sjekk)
MAKS_BOKSTAVER     = 1         # 2+ bokstaver, ikke FNR, uansett YOLO

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
