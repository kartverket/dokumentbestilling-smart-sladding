import glob
import os


def velg_filer(mappe, velg_dokumenter, antall):
    alle = sorted(glob.glob(os.path.join(mappe, "*")))
    print(f"Mappe «{mappe}» inneholder {len(alle)} fil(er) totalt.")

    if velg_dokumenter:
        valg = [str(v).strip() for v in velg_dokumenter if str(v).strip()]
        filer = [f for f in alle if any(v in os.path.basename(f) for v in valg)]
        print(f"Modus: SPESIFIKKE — {len(filer)} fil(er) matchet {len(valg)} søk:")
        for f in filer:
            print("   ", os.path.basename(f))
        mangler = [v for v in valg if not any(v in os.path.basename(f) for f in alle)]
        if mangler:
            print("!! Fant ingen treff for:", mangler)
    elif antall in (None, 0) or str(antall).strip().lower() in ("alle", "alt"):
        filer = alle
        print(f"Modus: ALLE — kjører alle {len(filer)} filene.")
    else:
        n = int(antall)
        filer = alle[:n]
        print(f"Modus: ANTALL — kjører de {len(filer)} første av {len(alle)} (ba om {n}).")

    return filer
