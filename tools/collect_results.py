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
        return json.loads(path.read_text(encoding="utf-8-sig"))
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
    key = METRIC_KEYS["map50"]
    return max(rows, key=lambda row: float(row.get(key, 0.0) or 0.0))


def read_last_row(results_csv: Path) -> dict[str, str]:
    rows = read_rows(results_csv)
    return rows[-1] if rows else {}


def metrics_from(summary: dict[str, Any], section: str) -> dict[str, Any]:
    return summary.get(section, {}) or {}


def metric(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(METRIC_KEYS[key], "")


def iter_run_dirs(project: Path, recursive: bool) -> list[Path]:
    candidates = project.rglob("*") if recursive else project.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_dir() and ((path / "summary.json").exists() or (path / "results.csv").exists())
    )


def collect(project: Path, recursive: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(project, recursive):
        summary = read_summary(run_dir)
        best = read_best_row(run_dir / "results.csv")
        last = read_last_row(run_dir / "results.csv")
        model_info = summary.get("model_info", {})
        selection = metrics_from(summary, "selection_val_metrics")
        official_val = metrics_from(summary, "official_val_metrics")
        official_test = metrics_from(summary, "official_test_metrics")
        main_test = metrics_from(summary, "main_test_metrics")
        ood_val = metrics_from(summary, "ood_val_metrics") or official_val
        ood_test = metrics_from(summary, "ood_test_metrics") or official_test
        map50_selected = summary.get("map50_selected", {}) or {}
        selector_selection = map50_selected.get("selection_val_metrics", {}) or selection
        selector_main_test = map50_selected.get("main_test_metrics", {}) or main_test
        conf_sweep = summary.get("conf_sweep", {}) or {}
        conf_best = conf_sweep.get("best", {}) or {}
        checkpoint_metrics = summary.get("checkpoint_metrics", {})
        best_official_test = checkpoint_metrics.get("best", {}).get("ood_test_metrics", checkpoint_metrics.get("best", {}).get("official_test_metrics", {}))
        last_official_test = checkpoint_metrics.get("last", {}).get("ood_test_metrics", checkpoint_metrics.get("last", {}).get("official_test_metrics", {}))
        train_curve = summary.get("train_curve", {}) or {}
        row = {
            "exp_id": summary.get("exp_id", run_dir.name.split("_", 1)[0]),
            "name": summary.get("name", run_dir.name),
            "model": summary.get("model", ""),
            "pretrained_weight": summary.get("pretrained_weight", ""),
            "data_config": summary.get("data_config", ""),
            "official_eval_data_config": summary.get("official_eval_data_config", ""),
            "modules": summary.get("modules", ""),
            "nwd_weight": summary.get("nwd_weight", 0.0),
            "params": model_info.get("params", ""),
            "flops": model_info.get("flops", ""),
            "selection_precision": metric(selection, "precision") or best.get(METRIC_KEYS["precision"], ""),
            "selection_recall": metric(selection, "recall") or best.get(METRIC_KEYS["recall"], ""),
            "selection_map50": metric(selection, "map50") or best.get(METRIC_KEYS["map50"], ""),
            "selection_map5095": metric(selection, "map5095") or best.get(METRIC_KEYS["map5095"], ""),
            "selector_label": map50_selected.get("label", ""),
            "selector_selected_from": map50_selected.get("selected_from", ""),
            "selector_eval_mode": summary.get("selector_eval_mode", ""),
            "selector_selection_source": map50_selected.get("selection_source", ""),
            "selector_precision": metric(selector_selection, "precision"),
            "selector_recall": metric(selector_selection, "recall"),
            "selector_map50": metric(selector_selection, "map50"),
            "selector_map5095": metric(selector_selection, "map5095"),
            "main_test_precision": metric(selector_main_test, "precision"),
            "main_test_recall": metric(selector_main_test, "recall"),
            "main_test_map50": metric(selector_main_test, "map50"),
            "main_test_map5095": metric(selector_main_test, "map5095"),
            "ood_val_map50": metric(ood_val, "map50"),
            "ood_val_map5095": metric(ood_val, "map5095"),
            "ood_test_map50": metric(ood_test, "map50"),
            "ood_test_map5095": metric(ood_test, "map5095"),
            "official_val_map50": metric(official_val, "map50"),
            "official_val_map5095": metric(official_val, "map5095"),
            "official_test_map50": metric(official_test, "map50"),
            "official_test_map5095": metric(official_test, "map5095"),
            "best_official_test_map5095": metric(best_official_test, "map5095"),
            "last_official_test_map5095": metric(last_official_test, "map5095"),
            "best_conf": conf_best.get("conf", ""),
            "best_conf_f1": conf_best.get("f1", ""),
            "best_conf_precision": conf_best.get("precision", ""),
            "best_conf_recall": conf_best.get("recall", ""),
            "curve_best_epoch": train_curve.get("best_epoch", ""),
            "curve_last_epoch": train_curve.get("last_epoch", ""),
            "curve_first_map50": train_curve.get("first_map50", ""),
            "curve_map50_gain_first_to_best": train_curve.get("map50_gain_first_to_best", ""),
            "curve_map50_drop": train_curve.get("map50_drop_best_to_last", ""),
            "curve_bad_values": train_curve.get("bad_values", ""),
            "curve_train_loss_down": train_curve.get("train_loss_continued_down", ""),
            "curve_val_loss_improved_after_best": train_curve.get("val_box_loss_improved_after_best", ""),
            "precision": metric(selector_selection, "precision") or best.get(METRIC_KEYS["precision"], ""),
            "recall": metric(selector_selection, "recall") or best.get(METRIC_KEYS["recall"], ""),
            "map50": metric(selector_selection, "map50") or best.get(METRIC_KEYS["map50"], ""),
            "map5095": metric(selector_selection, "map5095") or best.get(METRIC_KEYS["map5095"], ""),
            "best_epoch": best.get("epoch", ""),
            "last_epoch": last.get("epoch", ""),
            "run_dir": str(run_dir),
            "best_pt": summary.get("best_pt", str(run_dir / "weights" / "best.pt")),
            "last_pt": summary.get("last_pt", str(run_dir / "weights" / "last.pt")),
            "map50_selected_pt": summary.get("map50_selected_pt", ""),
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bubble")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Collect runs below project recursively")
    args = parser.parse_args()

    project = args.project if args.project.is_absolute() else ROOT / args.project
    project.mkdir(parents=True, exist_ok=True)
    rows = collect(project, recursive=args.recursive)
    out_csv = args.out_csv or project / "experiment_summary.csv"
    out_json = args.out_json or project / "experiment_summary.json"
    fieldnames = [
        "exp_id",
        "name",
        "model",
        "pretrained_weight",
        "data_config",
        "official_eval_data_config",
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
        "selector_label",
        "selector_selected_from",
        "selector_eval_mode",
        "selector_selection_source",
        "selector_precision",
        "selector_recall",
        "selector_map50",
        "selector_map5095",
        "main_test_precision",
        "main_test_recall",
        "main_test_map50",
        "main_test_map5095",
        "ood_val_map50",
        "ood_val_map5095",
        "ood_test_map50",
        "ood_test_map5095",
        "official_val_map50",
        "official_val_map5095",
        "official_test_map50",
        "official_test_map5095",
        "best_official_test_map5095",
        "last_official_test_map5095",
        "best_conf",
        "best_conf_f1",
        "best_conf_precision",
        "best_conf_recall",
        "curve_best_epoch",
        "curve_last_epoch",
        "curve_first_map50",
        "curve_map50_gain_first_to_best",
        "curve_map50_drop",
        "curve_bad_values",
        "curve_train_loss_down",
        "curve_val_loss_improved_after_best",
        "best_epoch",
        "last_epoch",
        "run_dir",
        "best_pt",
        "last_pt",
        "map50_selected_pt",
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
