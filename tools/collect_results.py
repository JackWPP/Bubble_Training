"""Collect Ultralytics run metrics into experiment_summary.csv/json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def read_best_row(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        return {}
    key = METRIC_KEYS["map5095"]
    return max(rows, key=lambda row: float(row.get(key, 0.0) or 0.0))


def collect(project: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in project.iterdir() if path.is_dir() and path.name.startswith("E")):
        summary = read_summary(run_dir)
        best = read_best_row(run_dir / "results.csv")
        model_info = summary.get("model_info", {})
        row = {
            "exp_id": summary.get("exp_id", run_dir.name.split("_", 1)[0]),
            "name": summary.get("name", run_dir.name),
            "model": summary.get("model", ""),
            "modules": summary.get("modules", ""),
            "nwd_weight": summary.get("nwd_weight", 0.0),
            "params": model_info.get("params", ""),
            "flops": model_info.get("flops", ""),
            "precision": best.get(METRIC_KEYS["precision"], ""),
            "recall": best.get(METRIC_KEYS["recall"], ""),
            "map50": best.get(METRIC_KEYS["map50"], ""),
            "map5095": best.get(METRIC_KEYS["map5095"], ""),
            "best_epoch": best.get("epoch", ""),
            "run_dir": str(run_dir),
            "best_pt": summary.get("best_pt", str(run_dir / "weights" / "best.pt")),
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bubble")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    project = args.project if args.project.is_absolute() else ROOT / args.project
    project.mkdir(parents=True, exist_ok=True)
    rows = collect(project)
    out_csv = args.out_csv or project / "experiment_summary.csv"
    out_json = args.out_json or project / "experiment_summary.json"
    fieldnames = [
        "exp_id",
        "name",
        "model",
        "modules",
        "nwd_weight",
        "params",
        "flops",
        "precision",
        "recall",
        "map50",
        "map5095",
        "best_epoch",
        "run_dir",
        "best_pt",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
