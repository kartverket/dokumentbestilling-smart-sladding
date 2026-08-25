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
                    "surplus", "total_overlap_pct", "threshold_pct"])
        w.writerow([
            round(result["recall"] * 100, 1),
            result["hit"],
            result["fasit"],
            result["pred"],
            result["surplus"],
            round(result.get("total_overlap", 0.0) * 100, 1),
            round(result.get("threshold", 0.0) * 100, 1),
        ])

        w.writerow([])
        w.writerow(["## Recall per type"])
        w.writerow(["type", "hit", "truth_total", "recall_pct"])
        for t, (tr, tot) in sorted(result["pr_type"].items()):
            w.writerow([t or "(empty)", tr, tot, round(tr / tot * 100, 1) if tot else 0])

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
