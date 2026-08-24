"""Skriver listen over kjørte dokumenter som IKKE får regelprofil i prod.

VLM-verifikatoren skal måles på residualet etter ALT annet regelverk. De
globale etterfiltrene er allerede anvendt i kjøringen (resultat-CSV-en
inneholder bare overlevere), men per-type-profilene (KOORDFAM_KODER,
SEKSJONERING_KODER) aktiveres av dokumentkoder — og en global kjøring har
ikke anvendt dem. I stedet for å gjenskape profilene offline, avgrenses
VLM-strataet til dokumentene der prod IKKE aktiverer noen profil: der er
en global kjøring og prod identiske bak etterfilteret.

Speiler prod-oppførselen eksakt, med begge dens kanter:
  * Dokumenter uten metadata-rad får ingen koder i prod → ingen profil →
    de HØRER TIL i strataet og tas med (antall rapporteres).
  * Legacy-koder uten prefiks (JOU, REG, …) matcher ikke frozenset-ene i
    prod i dag → heller ingen profil → tas med. Tettes det hullet, må
    denne listen genereres på nytt.

Kjør:
    python utils/lag_profilfri_liste.py \
        --metadata-csv $SLADD_METADATA/uttrekk_4.csv \
        --res-csv $SLADD_VALIDERING/<run>/resultat.csv \
        --ut /data2/tmp/vlm_profilfri_uttrekk4.txt
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import KOORDFAM_KODER, SEKSJONERING_KODER
from filter_felles import dok_nr
from rettsstiftelse_stat import les_metadata


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV — definerer hvilke dokumenter som er "
                        "kjørt (dokumenter uten prediksjoner bidrar uansett "
                        "ingen bokser til VLM-eksporten)")
    p.add_argument("--ut", required=True, metavar="FIL",
                   help="Utfil: ett dokumentnummer per linje")
    args = p.parse_args()

    profil_koder = KOORDFAM_KODER | SEKSJONERING_KODER

    kjorte = set()
    with open(args.res_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nr = dok_nr(r.get("navn", ""))
            if nr is not None:
                kjorte.add(nr)

    meta, _besk = les_metadata(args.metadata_csv)

    med, profil_ekskludert, uten_meta = [], 0, 0
    for dok in sorted(kjorte):
        koder = meta.get(dok)
        if koder is None:
            uten_meta += 1
            med.append(dok)
        elif profil_koder & set(koder[0]):
            profil_ekskludert += 1
        else:
            med.append(dok)

    with open(args.ut, "w", encoding="utf-8") as f:
        f.write("\n".join(str(d) for d in med) + "\n")

    print(f"Kjørte dokumenter i resultat-CSV: {len(kjorte)}")
    print(f"  ekskludert (profil-kode):       {profil_ekskludert}")
    print(f"  med, uten metadata-rad:         {uten_meta}  (prod har ingen "
          f"koder for dem — ingen profil)")
    print(f"Skrevet: {len(med)} dokumenter -> {args.ut}")


if __name__ == "__main__":
    main()
