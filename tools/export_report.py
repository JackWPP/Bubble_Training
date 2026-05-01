"""Export a Markdown report from collected Bubble-YOLO experiment results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def values_for(rows: list[dict], key: str) -> str:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value and value not in values:
            values.append(str(value))
    return ", ".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bubble")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    project = args.project if args.project.is_absolute() else ROOT / args.project
    summary_json = args.summary_json or project / "experiment_summary.json"
    out = args.out or project / "TRAINING_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else []

    lines = [
        "# Bubble-YOLO11s Training Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Project: `{project}`",
        f"- Training dataset: `{values_for(rows, 'data_config') or 'unknown'}`",
        f"- Official eval dataset: `{values_for(rows, 'official_eval_data_config') or 'unknown'}`",
        "- Primary selector: selection mAP@50 with precision/recall balance constraints.",
        "",
        "## Experiment Summary",
        "",
        "| Exp | Model | Modules | Sel P | Sel R | Sel mAP@50 | Main Test mAP@50 | OOD Test mAP@50 | mAP@50-95 diag | Selector | Best Conf/F1 | Best Epoch | Last Epoch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {exp_id} | {model} | {modules} | {selector_precision} | {selector_recall} | {selector_map50} | {main_test_map50} | {ood_test_map50} | {selector_map5095} | {selector_label} | {best_conf} / {best_conf_f1} | {best_epoch} | {last_epoch} |".format(
                exp_id=row.get("exp_id", ""),
                model=Path(str(row.get("model", ""))).name,
                modules=row.get("modules", ""),
                selector_precision=fmt(row.get("selector_precision", row.get("precision", ""))),
                selector_recall=fmt(row.get("selector_recall", row.get("recall", ""))),
                selector_map50=fmt(row.get("selector_map50", row.get("map50", ""))),
                main_test_map50=fmt(row.get("main_test_map50", "")),
                ood_test_map50=fmt(row.get("ood_test_map50", row.get("official_test_map50", ""))),
                selector_map5095=fmt(row.get("selector_map5095", row.get("map5095", ""))),
                selector_label=row.get("selector_label", ""),
                best_conf=fmt(row.get("best_conf", "")),
                best_conf_f1=fmt(row.get("best_conf_f1", "")),
                best_epoch=row.get("best_epoch", ""),
                last_epoch=row.get("last_epoch", ""),
            )
        )

    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir:
            continue
        lines.append(f"- `{row.get('exp_id', run_dir.name)}`: `{run_dir}`")
        lines.append(f"  - weights: `{row.get('best_pt', '')}`")
        lines.append(f"  - last weights: `{row.get('last_pt', '')}`")
        lines.append(f"  - mAP50-selected weights: `{row.get('map50_selected_pt', '')}`")
        lines.append(f"  - curves: `{run_dir / 'results.png'}`")
        lines.append(f"  - summary: `{run_dir / 'summary.json'}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Selection metrics come from the train-domain validation split and are used for checkpoint selection only.",
            "- Main test metrics come from the same primary dataset test split.",
            "- OOD grouped val/test results are retained as stress tests, not checkpoint-selection inputs.",
            "- Report best.pt, last.pt, and mAP50-selected weights together when their metrics diverge.",
            "- mAP@50 is the primary detector metric for this phase; mAP@50-95 is a strict localization diagnostic for dense and fuzzy bubbles.",
            "- NWD is a training loss change only in this implementation; NMS and assignment remain Ultralytics defaults.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
