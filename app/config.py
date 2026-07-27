# ---- PDF-rendering -----------------------------------------
PDF_DPI = 300                  # oppløsning ved rendering (høyere = tregere, mer nøyaktig)

# ---- YOLO --------------------------------------------------
YOLO_CONF          = 0.12      # predict-terskel
YOLO_CONF_UTEN_TEKST = 0.40   # krav når Paddle ikke leste noe i boksen
YOLO_CONF_VERTIKAL = 0.90     # vertikale bokser (stående tekst)
VERTIKAL_FAKTOR    = 1.3       # høyde > 1.3 × bredde regnes som vertikal
MIN_BOKS_AREAL     = 965      # bokser med mindre areal regnes som støy
YOLO_IMGSZ         = 1280      # bildestørrelse inn til YOLO
MIN_SIFFER         = 1         # minst så mange siffer i boksen (snill-sjekk)
MAKS_BOKSTAVER     = 1         # 2+ bokstaver, ikke FNR, uansett YOLO

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
MAKS_BREDDE_PT     = 50       # elektronisk tinglyst: bokser bredere enn dette (PDF-punkt) er feil-deteksjon

# ---- Pipeline ----------------------------------------------
DEDUP_OVERLAPP     = 0.5       # YOLO-boks regnes som "samme" når den dekker Paddle-boks så mye

# ---- Orientering -------------------------------------------
NEDSKALERING       = 4         # nedskalering av bilde før orienteringssjekk
MIN_KONFIDENS      = 0.7       # under dette: stol ikke på gjetningen, ikke roter
