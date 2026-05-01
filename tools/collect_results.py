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


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    return rows


def read_best_row(results_csv: Path) -> dict[str, str]:
    rows = read_rows(results_csv)
    if not rows:
        return {}
    key = METRIC_KEYS["map5095"]
    return max(rows, key=lambda row: float(row.get(key, 0.0) or 0.0))


def read_last_row(results_csv: Path) -> dict[str, str]:
    rows = read_rows(results_csv)
    return rows[-1] if rows else {}


def metrics_from(summary: dict[str, Any], section: str) -> dict[str, Any]:
    return summary.get(section, {}) or {}


def metric(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(METRIC_KEYS[key], "")


def collect(project: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in project.iterdir() if path.is_dir() and path.name[:1] in {"B", "E"}):
        summary = read_summary(run_dir)
        best = read_best_row(run_dir / "results.csv")
        last = read_last_row(run_dir / "results.csv")
        model_info = summary.get("model_info", {})
        selection = metrics_from(summary, "selection_val_metrics")
        official_val = metrics_from(summary, "official_val_metrics")
        official_test = metrics_from(summary, "official_test_metrics")
        checkpoint_metrics = summary.get("checkpoint_metrics", {})
        best_official_test = checkpoint_metrics.get("best", {}).get("official_test_metrics", {})
        last_official_test = checkpoint_metrics.get("last", {}).get("official_test_metrics", {})
        row = {
            "exp_id": summary.get("exp_id", run_dir.name.split("_", 1)[0]),
            "name": summary.get("name", run_dir.name),
            "model": summary.get("model", ""),
            "modules": summary.get("modules", ""),
            "nwd_weight": summary.get("nwd_weight", 0.0),
            "params": model_info.get("params", ""),
            "flops": model_info.get("flops", ""),
            "selection_precision": metric(selection, "precision") or best.get(METRIC_KEYS["precision"], ""),
            "selection_recall": metric(selection, "recall") or best.get(METRIC_KEYS["recall"], ""),
            "selection_map50": metric(selection, "map50") or best.get(METRIC_KEYS["map50"], ""),
            "selection_map5095": metric(selection, "map5095") or best.get(METRIC_KEYS["map5095"], ""),
            "official_val_map50": metric(official_val, "map50"),
            "official_val_map5095": metric(official_val, "map5095"),
            "official_test_map50": metric(official_test, "map50"),
            "official_test_map5095": metric(official_test, "map5095"),
            "best_official_test_map5095": metric(best_official_test, "map5095"),
            "last_official_test_map5095": metric(last_official_test, "map5095"),
            "precision": metric(selection, "precision") or best.get(METRIC_KEYS["precision"], ""),
            "recall": metric(selection, "recall") or best.get(METRIC_KEYS["recall"], ""),
            "map50": metric(selection, "map50") or best.get(METRIC_KEYS["map50"], ""),
            "map5095": metric(selection, "map5095") or best.get(METRIC_KEYS["map5095"], ""),
            "best_epoch": best.get("epoch", ""),
            "last_epoch": last.get("epoch", ""),
            "run_dir": str(run_dir),
            "best_pt": summary.get("best_pt", str(run_dir / "weights" / "best.pt")),
            "last_pt": summary.get("last_pt", str(run_dir / "weights" / "last.pt")),
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
        "selection_precision",
        "selection_recall",
        "selection_map50",
        "selection_map5095",
        "official_val_map50",
        "official_val_map5095",
        "official_test_map50",
        "official_test_map5095",
        "best_official_test_map5095",
        "last_official_test_map5095",
        "best_epoch",
        "last_epoch",
        "run_dir",
        "best_pt",
        "last_pt",
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
