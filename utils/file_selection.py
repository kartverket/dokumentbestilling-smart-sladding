import glob
import os


def velg_filer(mappe, velg_dokumenter, antall, eksakt=False):
    alle = sorted(glob.glob(os.path.join(mappe, "*")))
    print(f"Mappe «{mappe}» inneholder {len(alle)} fil(er) totalt.")

    if velg_dokumenter:
        valg = [str(v).strip() for v in velg_dokumenter if str(v).strip()]
        if eksakt:
            valg_set = {os.path.splitext(v)[0] for v in valg}
            filer = [f for f in alle if os.path.splitext(os.path.basename(f))[0] in valg_set]
            mangler = valg_set - {os.path.splitext(os.path.basename(f))[0] for f in alle}
        else:
            filer = [f for f in alle if any(v in os.path.basename(f) for v in valg)]
            mangler = [v for v in valg if not any(v in os.path.basename(f) for f in alle)]
        print(f"Valgt: {len(filer)} fil(er) matchet {len(valg)} søk ({'eksakt' if eksakt else 'delstreng'})")
        if len(filer) <= 5:
            for f in filer:
                print("   ", os.path.basename(f))
        if mangler:
            mangler_liste = sorted(mangler) if isinstance(mangler, set) else mangler
            print(f"!! Fant ingen treff for {len(mangler_liste)} av søkene:", mangler_liste[:5])
    elif antall in (None, 0) or str(antall).strip().lower() in ("alle", "alt"):
        filer = alle
        print(f"Modus: ALLE — kjører alle {len(filer)} filene.")
    else:
        n = int(antall)
        filer = alle[:n]
        print(f"Modus: ANTALL — kjører de {len(filer)} første av {len(alle)} (ba om {n}).")

    return filer
