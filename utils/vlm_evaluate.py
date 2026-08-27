"""Evaluates the VLM judgements against the fasit classes: cost against gain.

Step 3 of the VLM verifier pilot. Joins judge_image.csv against manifest.csv
and counts BOM boxes judged «nei» (gain: oversladdinger we can drop) against
covering boxes judged «nei» (risk: real fnr). The tool reports the numbers.
Whether they are good enough is a call the reader makes.

Two things make that less trivial than it looks. The export takes ALL BOM but
only a sample of the covering boxes, so the factors in utvalg.json scale the
loss up to the full uttrekk. And with 0 or 1 losses observed a point estimate
is worthless, so oversladdinger per lost fnr is also computed against a 95 %
Poisson upper bound.

Every covering box judged «nei» is written to lost.csv with label_id: fasit
can be noise, and the ids go straight into ugyldige_labels.txt if the verdict
turns out to be right.

Run:
    python utils/vlm_evaluate.py \
        --manifest /data2/vlm/uttrekk6_kalibrering/manifest.csv \
        --judge   /data2/vlm/uttrekk6_kalibrering/judge_image.csv
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "app")))

# Imported, not copied, so a change in prod cannot leave a silently diverging
# copy of the digit-confusion rules here. The fnr guard is the same code the
# in-pipeline verifier runs, so --fnr-override measures what prod does.
from vlm_client import fnr_candidate as _fnr_candidate
from vlm_client import fnr_protects
from filter_common import (reclassify_invalid_covering,
                           reclassify_missing_covered)

ANSWER = ("ja", "nei", "usikker")


# ── Statistics ────────────────────────────────────────────────

def poisson_upper_bound(k, confidence=0.95):
    """Upper (95 %) bound on the expected count when k has been observed.

    Bisects P(X <= k | lambda) = 1 - tillit. k = 0 gives ~3.0, the rule of
    three: without it, zero losses in the sample reads as "no losses".
    """
    if k < 0:
        return 0.0
    target = 1.0 - confidence

    def cdf(lam):
        if lam <= 0:
            return 1.0
        s, ledd = 0.0, math.exp(-lam)
        for i in range(k + 1):
            if i:
                ledd *= lam / i
            s += ledd
        return s

    lo, hi = 0.0, max(10.0, k + 10.0)
    while cdf(hi) > target:
        hi *= 2
        if hi > 1e6:
            return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── Rule baseline ─────────────────────────────────────────────
# A VLM verdict only counts if it beats the rule it would replace, hence
# --judge regel:<name>, judged from the OCR line through the same accounts.

def _has_decimal(text):
    return bool(re.search(r"\d[.,]\d", text or ""))


RULES = {
    # Keep the box if the line has a fnr candidate.
    "fnr-kandidat": lambda r: "ja" if _fnr_candidate(r.get("ocr_linje")) else "nei",
    # Drop the box if its number has a decimal separator.
    "desimal": lambda r: "nei" if _has_decimal(r.get("ocr_tekst")) else "ja",
    # Drop only when the line lacks a fnr candidate AND the number is decimal.
    "fnr-kandidat+desimal": lambda r: (
        "nei" if not _fnr_candidate(r.get("ocr_linje"))
        and _has_decimal(r.get("ocr_tekst")) else "ja"),
}


def judge_from_rule(manifest, name):
    try:
        rule = RULES[name]
    except KeyError:
        raise SystemExit(f"unknown rule {name!r}, valid: "
                         + ", ".join(sorted(RULES)))
    return [{"nr": r["nr"], "svar": rule(r), "tall": r.get("ocr_tekst", ""),
             "begrunnelse": f"regel:{name}", "feil": "", "sekunder": ""}
            for r in manifest]


def read_result_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def join(manifest, judge):
    """manifest rows + verdict, keyed on nr. Returns (rows, unjudged)."""
    judgement_per_nr = {}
    for d in judge:
        # With --resume the same nr can appear twice, a failed attempt and a
        # successful one. The row without an error wins.
        nr = d["nr"]
        if nr in judgement_per_nr and judgement_per_nr[nr].get("feil") and not d.get("feil"):
            judgement_per_nr[nr] = d
        elif nr not in judgement_per_nr:
            judgement_per_nr[nr] = d
    row_list, unjudged = [], 0
    for m in manifest:
        d = judgement_per_nr.get(m["nr"])
        if d is None:
            unjudged += 1
            continue
        row = dict(m)
        row["svar"] = (d.get("svar") or "usikker").strip().lower()
        if row["svar"] not in ANSWER:
            row["svar"] = "usikker"
        row["tall"] = d.get("tall", "")
        row["begrunnelse"] = d.get("begrunnelse", "")
        row["feil"] = d.get("feil", "")
        row["sekunder"] = d.get("sekunder", "")
        # The model's own reading; --fnr-override has nothing without it.
        for field in ("linjen", "sifre_paa_linjen", "dato_gyldig", "holdepunkt"):
            if field in d:
                row[field] = d[field]
        row_list.append(row)
    return row_list, unjudged


# ── Subsets ───────────────────────────────────────────────────

_OPS = (("!=", lambda a, b: a != b), (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b), ("=", lambda a, b: a == b),
        (">", lambda a, b: a > b), ("<", lambda a, b: a < b))


def parse_condition(spec):
    """«har_fnr_kandidat=1» or «conf>=0.7» -> a predicate on a manifest row.

    Compared as numbers when both sides parse as numbers, else as text. Empty
    fields never match: "not computed" is not the same as zero.
    """
    for sym, op in _OPS:
        if sym in spec:
            col, _, value = spec.partition(sym)
            col, value = col.strip(), value.strip()

            def test(row, col=col, value=value, op=op):
                raw = (row.get(col) or "").strip()
                if raw == "":
                    return False
                try:
                    return op(float(raw), float(value))
                except ValueError:
                    return op(raw, value)
            return test
    raise ValueError(f"invalid condition {spec!r}, e.g. har_fnr_kandidat=1")


def write_deling(row_list, column, write=print):
    """Confusion matrix per value of a manifest column.

    Looks for subsets where the model is safe: a rule that is dangerous
    globally can be free in one family, as with the rettsstiftelse profiles.
    """
    tab = defaultdict(lambda: defaultdict(int))
    for r in row_list:
        value = (r.get(column) or "").strip() or "(empty)"
        tab[value][(_group(r), r["svar"])] += 1
    write(f"\nSPLIT BY {column}")
    write(f"  {column:>16} {'BOM nei':>9} {'BOM tot':>9} "
          f"{'DEK nei':>9} {'DEK tot':>9} {'DEK nei%':>9}")
    for value in sorted(tab):
        t = tab[value]
        bn = t[("BOM", "nei")]
        bt = sum(t[("BOM", s)] for s in ANSWER)
        dn = t[("DEKKENDE", "nei")]
        dt = sum(t[("DEKKENDE", s)] for s in ANSWER)
        write(f"  {value:>16} {bn:>9} {bt:>9} {dn:>9} {dt:>9} "
              f"{(dn / dt * 100 if dt else 0):>8.1f}%")
    write("  Look for a row with high «BOM nei» and «DEK nei%» near zero, "
          "that is a profile.")


def warn_om_skewed_sample(manifest, row_list, write=print):
    """Says so when the judged rows do not look like a random sample.

    The factors in utvalg.json assume they are. Run with --no-file on rows an
    earlier prompt got wrong and the set is adversarial, the scale-up is
    nonsense, and nothing in the numbers shows it.
    """
    if not row_list or len(row_list) >= 0.9 * len(manifest):
        return False
    def share(rows):
        n_miss = sum(1 for r in rows if r["klasse"] == "BOM")
        return n_miss / len(rows) if rows else 0.0
    expected, actual = share(manifest), share(row_list)
    if expected <= 0 or abs(actual - expected) <= 0.15:
        return False
    write("")
    write("  " + "!" * 66)
    write(f"  WARNING: the {len(row_list)} judged rows are {actual*100:.0f} % BOM, "
          f"while the manifest is {expected*100:.0f} %.")
    write("  The sample does not look random. Did you use --no-file or a "
          "hand-picked list?")
    write("  Then the sampling factors do NOT apply, and gain, loss and "
          "ov/lost below are meaningless.")
    write("  The confusion matrix still reads as «what the model did with "
          "exactly these».")
    write("  " + "!" * 66)
    return True


def _group(row):
    """BOM = oversladding, DEKKENDE = TREFF or SLURV (covers a fasit box)."""
    return "BOM" if row["klasse"] == "BOM" else "DEKKENDE"


# ── Report ────────────────────────────────────────────────────

def _columns(tab):
    """The verdicts to show. usikker only appears when something landed there."""
    return [s for s in ANSWER
            if s != "usikker" or any(tab[k]["usikker"] for k in tab)]


def write_matrix(row_list, write=print):
    tab = defaultdict(lambda: defaultdict(int))
    for r in row_list:
        tab[r["klasse"]][r["svar"]] += 1
    column = _columns(tab)
    write("\nCONFUSION MATRIX  (row = fasit class, column = VLM verdict)")
    write(f"  {'klasse':>10} " + " ".join(f"{s:>8}" for s in column)
          + f" {'sum':>8}")
    for klasse in sorted(tab):
        row = tab[klasse]
        write(f"  {klasse:>10} " + " ".join(f"{row[s]:>8}" for s in column)
              + f" {sum(row.values()):>8}")
    write(f"  {'SUM':>10} "
          + " ".join(f"{sum(tab[k][s] for k in tab):>8}" for s in column)
          + f" {len(row_list):>8}")


def write_per_source(row_list, write=print):
    tab = defaultdict(lambda: defaultdict(int))
    for r in row_list:
        tab[r["kilde"]][(_group(r), r["svar"])] += 1
    write("\nBY kilde")
    write(f"  {'kilde':>10} {'BOM nei':>9} {'BOM tot':>9} {'share':>7}"
          f" {'DEK nei':>9} {'DEK tot':>9} {'share':>7}")
    for source in sorted(tab):
        t = tab[source]
        bn = t[("BOM", "nei")]
        bt = sum(t[("BOM", s)] for s in ANSWER)
        dn = t[("DEKKENDE", "nei")]
        dt = sum(t[("DEKKENDE", s)] for s in ANSWER)
        write(f"  {source:>10} {bn:>9} {bt:>9} "
              f"{(bn / bt * 100 if bt else 0):>6.1f}% "
              f"{dn:>9} {dt:>9} {(dn / dt * 100 if dt else 0):>6.1f}%")


def _factors(sample, n_bom_judged, n_cov_judged):
    """Scale-up factors from JUDGED rows to the whole uttrekk.

    Without totals in utvalg.json this falls back to 1.0, "what you see is all
    there is". That is the only assumption that does not lie when the basis is gone.
    """
    miss_tot = float(sample.get("n_bom_total") or 0)
    cov_tot = float(sample.get("n_covering_total") or 0)
    miss = miss_tot / n_bom_judged if miss_tot and n_bom_judged else 1.0
    cov = cov_tot / n_cov_judged if cov_tot and n_cov_judged else 1.0
    return miss, cov


def tally(row_list, sample, uncertain_remover=False, write=print):
    """Gain against loss risk, scaled up to the full uttrekk."""
    remove = {"nei", "usikker"} if uncertain_remover else {"nei"}

    miss = [r for r in row_list if _group(r) == "BOM"]
    cov = [r for r in row_list if _group(r) == "DEKKENDE"]
    miss_removed = [r for r in miss if r["svar"] in remove]
    cov_remove = [r for r in cov if r["svar"] in remove]

    # Against the number JUDGED, not the number exported: in a partial run the
    # export factors would scale the loss but not the gain.
    miss_factor, hit_factor = _factors(sample, len(miss), len(cov))

    gain = len(miss_removed) * miss_factor
    loss_point = len(cov_remove) * hit_factor
    loss_upper = poisson_upper_bound(len(cov_remove)) * hit_factor

    width = max(len(str(len(miss))), len(str(len(cov))))
    lower_bound = gain / loss_upper if loss_upper > 0 else float("inf")
    point = gain / loss_point if loss_point > 0 else float("inf")

    write("\n" + "=" * 68)
    write("RESULT")
    write("=" * 68)
    write(f"  {len(miss_removed):>{width}} of {len(miss):>{width}} BOM said «nei»"
          f"      {_share(len(miss_removed), len(miss)):>5}"
          f"   →  {_estimate(gain):>7} oversladdinger dropped")
    write(f"  {len(cov_remove):>{width}} of {len(cov):>{width}} covering said «nei»"
          f" {_share(len(cov_remove), len(cov)):>5}"
          f"   →  {_estimate(loss_point):>7} fnr lost, "
          f"{_estimate(loss_upper)} at the 95 % bound")
    write(f"  Sample to uttrekk: BOM × {miss_factor:.2f}, "
          f"covering × {hit_factor:.2f}.")
    if uncertain_remover:
        write("  «usikker» counts as «nei» here (--uncertain-remover).")
    write("")

    if not cov:
        write("  No covering boxes in the sample, so there is nothing to "
              "weigh the gain against.")
    else:
        write(f"  Oversladdinger per lost fnr:  "
              f"{_estimate(lower_bound)} against the 95 % bound, "
              f"{_estimate(point)} at face value.")

    more_coverers = sum(1 for r in cov_remove
                        if _int(r.get("dekkere")) > 1)
    write("")
    if more_coverers:
        write(f"  {more_coverers} of the losses cover a fasit box that other "
              f"boxes also cover, so the")
        write("  loss is an upper bound.")
    write("  Every covering «nei» must be reviewed by hand before the number "
          "is trusted:")
    write("  fasit can be noise.")
    return {"gain": gain, "loss_point": loss_point, "loss_upper": loss_upper,
            "ov_per_tapt": lower_bound, "n_bom_nei": len(miss_removed),
            "n_dek_nei": len(cov_remove), "n_bom": len(miss), "n_dek": len(cov)}


def _share(part, whole):
    return f"{part / whole * 100:.1f}%" if whole else "-"


def _estimate(v):
    """Estimates: one decimal below 10, whole numbers above, ∞ when no loss."""
    if v == float("inf"):
        return "∞"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# ── Manifests for manual review ───────────────────────────────

LOST_FIELD = ["nr", "fil", "side", "label_id", "reason", "kilde", "conf",
             "klasse", "dekkere_foer", "vlm_tall", "ocr_linje",
             "elongation", "kortside_pt", "langside_pt",
             "pred_bredde_pt", "pred_hoyde_pt", "utsnitt", "vurdering"]


def write_manifest(row_list, path, field=LOST_FIELD):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field, extrasaction="ignore")
        w.writeheader()
        for r in row_list:
            w.writerow({
                "nr": r["nr"], "fil": r["fil"], "side": r["side"],
                "label_id": r.get("label_id", ""),
                "reason": r.get("begrunnelse", ""),
                "kilde": r.get("kilde", ""), "conf": r.get("conf", ""),
                "klasse": r.get("klasse", ""),
                "dekkere_foer": r.get("dekkere", ""),
                "vlm_tall": r.get("tall", ""),
                "ocr_linje": r.get("ocr_linje", ""),
                "elongation": r.get("elongation", ""),
                "kortside_pt": r.get("kortside_pt", ""),
                "langside_pt": r.get("langside_pt", ""),
                "pred_bredde_pt": r.get("bredde_pt", ""),
                "pred_hoyde_pt": r.get("hoyde_pt", ""),
                "utsnitt": r.get("utsnitt", ""),
                "vurdering": "",
            })


def main():
    p = argparse.ArgumentParser(
        description="Joins VLM judgements against fasit classes and counts "
                    "gain against loss risk.")
    p.add_argument("--manifest", required=True, help="manifest.csv from vlm_export")
    p.add_argument("--judge", required=True,
                   help="judge_*.csv from vlm_judge, OR «regel:<name>» to "
                        "measure a plain rule on the manifest's OCR line "
                        "instead, and see whether the model beats the rule it "
                        "would replace. Valid: "
                        + ", ".join(f"regel:{n}" for n in sorted(RULES)))
    p.add_argument("--sample", default=None,
                   help="utvalg.json (default: next to the manifest)")
    p.add_argument("--out-dir", default=None,
                   help="Directory for lost.csv/gain.csv (default: next to "
                        "the judgements)")
    p.add_argument("--fnr-override", action="store_true",
                   help="Force the verdict to «ja» when the model's own "
                        "transcription OR PaddleOCR's line from the manifest "
                        "holds a valid 11-digit fnr run. The model reads "
                        "better than it infers, so let the code infer.")
    p.add_argument("--without-caption", action="store_true",
                   help="Turn off the ledetekst guard in --fnr-override, "
                        "so only real 11-digit runs protect a box. Use it to "
                        "measure what the guard costs in gain.")
    p.add_argument("--split-by", default=None, metavar="KOLONNE",
                   help="Show the confusion matrix per value of a manifest "
                        "column (e.g. har_fnr_kandidat, kilde, "
                        "har_desimal_naer), to find subsets where the model "
                        "is safe.")
    p.add_argument("--only", nargs="+", default=None, metavar="VILKÅR",
                   help="Count only rows meeting every condition, e.g. --only "
                        "har_fnr_kandidat=0 conf<0.7. The sampling factors "
                        "still hold: the covering boxes were drawn at random "
                        "and independently of the features, so a subset is "
                        "still a random sample of its own subpopulation.")
    p.add_argument("--uncertain-remover", action="store_true",
                   help="Count «usikker» as «nei». The default is to keep, "
                        "an uncertain verdict is no proof of oversladding.")
    a = p.parse_args()

    manifest = read_result_csv(a.manifest)
    # Conservative on gain: reclassified rows keep the covering sample's
    # draw rate, so their contribution is not scaled up.
    reclassified = reclassify_invalid_covering(manifest)
    if reclassified:
        print(f"{reclassified} covering rows reclassified to BOM — all their "
              f"labels are listed in ugyldige_labels.txt")
    remapped = reclassify_missing_covered(manifest)
    if remapped:
        print(f"{remapped} BOM rows reclassified to TREFF — their boxes "
              f"cover a row in manglende_labels.csv")
    if os.path.isdir(a.judge):
        a.judge = os.path.join(a.judge, "full_info.csv")
    if a.judge.startswith("regel:"):
        judge = judge_from_rule(manifest, a.judge.split(":", 1)[1])
        print(f"Baseline «{a.judge}», no model involved")
    else:
        judge = read_result_csv(a.judge)
    sample_path = a.sample or os.path.join(
        os.path.dirname(os.path.abspath(a.manifest)), "utvalg.json")
    sample = {}
    if os.path.isfile(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            sample = json.load(f)
    else:
        print(f"  ⚠ Could not find {sample_path}, using factor 1.0, as if "
              f"the WHOLE uttrekk had been judged. The loss estimate comes "
              f"out too low if the covering boxes are a sample.")

    row_list, unjudged = join(manifest, judge)
    print(f"Judged {len(row_list)} of {len(manifest)} manifest rows"
          + (f"  ({unjudged} without a verdict)" if unjudged else ""))
    n_error = sum(1 for r in row_list if r.get("feil"))
    if n_error:
        print(f"  ⚠ {n_error} rows had errors/unparsable answers, counted "
              f"as usikker")

    if a.fnr_override:
        # Four independent readings of the same line; one valid 11-digit run
        # in ANY of them protects the box, since they rarely fail together.
        n, missing = 0, 0
        caption = not a.without_caption
        for r in row_list:
            sources = [r.get("sifre_paa_linjen"), r.get("linjen"),
                      r.get("ocr_linje"), r.get("ocr_blokk")]
            sources = [k for k in (t.strip() for t in sources if t) if k]
            if not sources:
                missing += 1
                continue
            verner = fnr_protects(sources, caption=caption)
            if r["svar"] != "ja" and verner:
                r["svar"] = "ja"
                n += 1
        print(f"  --fnr-override: {n} verdicts changed to «ja», a valid "
              f"11-digit fnr run in the transcription")
        if missing:
            print(f"    ({missing} rows have neither checklist fields nor OCR "
                  f"text, left untouched)")

    if a.only:
        condition = [parse_condition(v) for v in a.only]
        before = len(row_list)
        row_list = [r for r in row_list if all(t(r) for t in condition)]
        print(f"  Subset «{' '.join(a.only)}»: {len(row_list)} of {before} rows")
        if not row_list:
            print("  No rows left. Check column names and values.")
            return

    skewed = warn_om_skewed_sample(manifest, row_list)
    write_matrix(row_list)
    write_per_source(row_list)
    if a.split_by:
        write_deling(row_list, a.split_by)
    res = tally(row_list, sample, uncertain_remover=a.uncertain_remover)
    if skewed:
        print("  NB: the result above rests on a sample that does not look "
              "random, see the warning at the top.")

    out_dir = a.out_dir or (
        os.path.dirname(os.path.abspath(a.manifest)) if a.judge.startswith("regel:")
        else os.path.dirname(os.path.abspath(a.judge)))
    os.makedirs(out_dir, exist_ok=True)
    remove = {"nei", "usikker"} if a.uncertain_remover else {"nei"}
    lost = [r for r in row_list if _group(r) == "DEKKENDE" and r["svar"] in remove]
    gain = [r for r in row_list if _group(r) == "BOM" and r["svar"] in remove]
    lost.sort(key=lambda r: (r["fil"], int(r["side"]), int(r["nr"])))
    gain.sort(key=lambda r: (r["fil"], int(r["side"]), int(r["nr"])))

    lost_path = os.path.join(out_dir, "lost.csv")
    write_manifest(lost, lost_path)
    gain_path = os.path.join(out_dir, "gain.csv")
    write_manifest(gain, gain_path)
    print(f"\n  {len(lost):>6} rows in {lost_path}"
          f"   ← review these manually (label_id included)")
    print(f"  {len(gain):>6} rows in {gain_path}"
          f"   ← spot check on the gain")

    with open(os.path.join(out_dir, "oppsummering.json"), "w",
              encoding="utf-8") as f:
        json.dump({**res, "usikker_fjerner": a.uncertain_remover,
                   "n_domt": len(row_list), "n_manifest": len(manifest),
                   "n_feil": n_error}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
