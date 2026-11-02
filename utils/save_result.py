import csv
from datetime import datetime
from pathlib import Path


def write_result_files(result, folder=".", description=None, log=None):
    if result is None:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dir_name = f"result-{timestamp}"
    if description:
        dir_name += f"-{description}"

    run_dir = Path(folder) / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_file = run_dir / "sammendrag.csv"
    details_file = run_dir / "detaljer.csv"

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["## Overall"])
        w.writerow(["recall_pct", "hit", "fasit", "pred",
                    "oversladd", "total_overlap_pct", "threshold_pct"])
        w.writerow([
            round(result["recall"] * 100, 1),
            result["hit"],
            result["fasit"],
            result["pred"],
            result["oversladd"],
            round(result.get("total_overlap", 0.0) * 100, 1),
            round(result.get("threshold", 0.0) * 100, 1),
        ])

        w.writerow([])
        w.writerow(["## Recall per type"])
        w.writerow(["type", "hit", "truth_total", "recall_pct"])
        for t, (tr, tot) in sorted(result["pr_type"].items()):
            w.writerow([t or "(empty)", tr, tot, round(tr / tot * 100, 1) if tot else 0])

        kilde = result.get("pr_kilde") or {}
        if kilde:
            w.writerow([])
            w.writerow(["## Per kilde"])
            w.writerow(["kilde", "hit", "oversladd", "pred", "oversladd_pct"])
            for k, (tr, ov) in sorted(kilde.items()):
                tot = tr + ov
                w.writerow([k, tr, ov, tot,
                            round(ov / tot * 100, 1) if tot else 0])

        timings = result.get("timings") or {}
        if timings:
            total = sum(timings.values())
            w.writerow([])
            w.writerow(["## Time"])
            w.writerow(["phase", "seconds", "pct"])
            for phase, sec in sorted(timings.items(), key=lambda kv: -kv[1]):
                w.writerow([phase, round(sec, 1),
                            round(sec / total * 100, 1) if total else 0])
            w.writerow(["sum", round(total, 1), 100.0])
            wall = result.get("wall_time")
            if wall is not None:
                w.writerow(["wall_clock", round(wall, 1), ""])

        vlm = result.get("vlm") or {}
        if vlm:
            w.writerow([])
            w.writerow(["## VLM"])
            w.writerow(["model", "documents", "documents_judged",
                        "boxes_judged",
                        "boxes_removed", "cache_hits", "seconds"])
            w.writerow([vlm.get("model", ""), vlm.get("docs", 0),
                        vlm.get("docs_judged", 0),
                        vlm.get("judged", 0), vlm.get("dropped", 0),
                        vlm.get("cache_hits", 0),
                        round(vlm.get("seconds", 0.0), 1)])
            not_cached = vlm.get("not_cached") or {}
            if not_cached:
                w.writerow([])
                w.writerow(["## VLM not cached"])
                w.writerow(["reason", "count"])
                for reason in sorted(not_cached, key=lambda r: -not_cached[r]):
                    w.writerow([reason, not_cached[reason]])

            judged_per_kilde = vlm.get("judged_per_kilde") or {}
            if judged_per_kilde:
                dropped_per_kilde = vlm.get("dropped_per_kilde") or {}
                w.writerow([])
                w.writerow(["## VLM per kilde"])
                w.writerow(["kilde", "judged", "removed"])
                for k in sorted(judged_per_kilde):
                    w.writerow([k, judged_per_kilde[k],
                                dropped_per_kilde.get(k, 0)])

        miss = result.get("miss_files", [])
        if miss:
            w.writerow([])
            w.writerow(["## Filer med bom"])
            w.writerow(["fil", "side", "bom", "truth_total"])
            for row in miss:
                w.writerow([row["fil"], row["side"], row["bom"], row["truth_total"]])

    # One row per truth box
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "fil", "side", "fasit_nr", "type",
            "coverage_pct", "result", "kilde", "conf",
            "fasit_x0", "fasit_y0", "fasit_x1", "fasit_y1",
        ])
        w.writeheader()
        w.writerows(result.get("details", []))

    if log is not None:
        log_file = run_dir / "logg.txt"
        log_file.write_text(log, encoding="utf-8")
        print(f"Log:      {log_file}")

    print(f"Summary:  {summary_file}")
    print(f"Details:  {details_file}")
    return run_dir
