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
        f"- Dataset: `yolo_dataset_grouped/bubble.yaml`",
        "",
        "## Experiment Summary",
        "",
        "| Exp | Model | Modules | NWD | Params | FLOPs | P | R | mAP@50 | mAP@50-95 | Best Epoch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {exp_id} | {model} | {modules} | {nwd_weight} | {params} | {flops} | {precision} | {recall} | {map50} | {map5095} | {best_epoch} |".format(
                exp_id=row.get("exp_id", ""),
                model=Path(str(row.get("model", ""))).name,
                modules=row.get("modules", ""),
                nwd_weight=fmt(row.get("nwd_weight", "")),
                params=fmt(row.get("params", "")),
                flops=fmt(row.get("flops", "")),
                precision=fmt(row.get("precision", "")),
                recall=fmt(row.get("recall", "")),
                map50=fmt(row.get("map50", "")),
                map5095=fmt(row.get("map5095", "")),
                best_epoch=row.get("best_epoch", ""),
            )
        )

    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir:
            continue
        lines.append(f"- `{row.get('exp_id', run_dir.name)}`: `{run_dir}`")
        lines.append(f"  - weights: `{row.get('best_pt', '')}`")
        lines.append(f"  - curves: `{run_dir / 'results.png'}`")
        lines.append(f"  - summary: `{run_dir / 'summary.json'}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Grouped dataset results should be treated as the official generalization results.",
            "- Compare both mAP@50 and mAP@50-95; dense bubbles may trade precision and recall differently.",
            "- NWD is a training loss change only in this implementation; NMS and assignment remain Ultralytics defaults.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
