"""Writes the list of processed documents that get NO rule profile in prod.

The VLM verifier must be measured on the residual left after every other
rule. Global postfilters are already applied in the run, but the per-type
profiles (KOORDFAM_CODES, SEKSJONERING_CODES) are activated by document
codes a global run never applied. Rather than rebuilding the profiles
offline, the stratum is narrowed to documents where prod activates no
profile at all. There a global run and prod behave identically.

Both prod edges are mirrored: documents with no metadata row, and legacy
codes without a prefix (JOU, REG, …) that match no frozenset today, get no
profile and are therefore included. Regenerate this list if that gap closes.

Run:
    python utils/make_profile_free_list.py \
        --metadata-csv $SLADD_METADATA/uttrekk_4.csv \
        --res-csv $SLADD_VALIDATION/<run>/resultat.csv \
        --out /data2/tmp/vlm_profile_free_extract4.txt
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import KOORDFAM_CODES, SEKSJONERING_CODES
from filter_common import doc_no
from rettsstiftelse_stat import read_metadata


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--res-csv", required=True,
                   help="Result CSV. Defines which documents were processed "
                        "(documents without predictions contribute no boxes "
                        "to the VLM export anyway)")
    p.add_argument("--out", required=True, metavar="FILE",
                   help="Output file: one document number per line")
    args = p.parse_args()

    profile_codes = KOORDFAM_CODES | SEKSJONERING_CODES

    processed = set()
    with open(args.res_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            no = doc_no(r.get("navn", ""))
            if no is not None:
                processed.add(no)

    meta, _desc = read_metadata(args.metadata_csv)

    included, profile_excluded, without_meta = [], 0, 0
    for doc in sorted(processed):
        codes = meta.get(doc)
        if codes is None:
            without_meta += 1
            included.append(doc)
        elif profile_codes & set(codes[0]):
            profile_excluded += 1
        else:
            included.append(doc)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(str(d) for d in included) + "\n")

    print(f"Processed documents in result CSV: {len(processed)}")
    print(f"  excluded (profile code):         {profile_excluded}")
    print(f"  included, no metadata row:       {without_meta}  (prod has no "
          f"codes for them, so no profile)")
    print(f"Written: {len(included)} documents -> {args.out}")


if __name__ == "__main__":
    main()
