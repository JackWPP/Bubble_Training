"""Run the Bubble-YOLO11s experiment matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_matrix(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full", "debug_overfit"), default="full")
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--data")
    parser.add_argument("--weights")
    parser.add_argument("--project", default=str(ROOT / "runs" / "bubble"))
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "train" / "experiments.yaml")
    parser.add_argument("--experiments", nargs="+", help="Explicit experiment ids")
    parser.add_argument("--compressed", action="store_true", help="Run E0,E1,E3,E5 only")
    parser.add_argument("--resume-missing", action="store_true", help="Skip experiments that already have best.pt")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    matrix = load_matrix(args.matrix if args.matrix.is_absolute() else ROOT / args.matrix)["experiments"]
    exp_ids = args.experiments or (["E0", "E1", "E3", "E5"] if args.compressed else ["E0", "E1", "E2", "E3", "E4", "E5"])

    for exp_id in exp_ids:
        name = matrix[exp_id]["name"]
        run_name = name if args.preset == "full" else f"{name}_{args.preset}"
        best_pt = Path(args.project) / run_name / "weights" / "best.pt"
        if args.resume_missing and best_pt.exists():
            print(f"[skip] {exp_id} already has {best_pt}")
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_experiment.py"),
            "--exp",
            exp_id,
            "--preset",
            args.preset,
            "--device",
            args.device,
            "--project",
            args.project,
        ]
        if args.data:
            cmd.extend(["--data", args.data])
        if args.weights:
            cmd.extend(["--weights", args.weights])
        if args.exist_ok:
            cmd.append("--exist-ok")
        if args.skip_predict:
            cmd.append("--skip-predict")

        print(f"[run] {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            print(f"[fail] {exp_id} exited with {completed.returncode}")
            if not args.keep_going:
                return completed.returncode

    collect_cmd = [sys.executable, str(ROOT / "tools" / "collect_results.py"), "--project", args.project]
    report_cmd = [sys.executable, str(ROOT / "tools" / "export_report.py"), "--project", args.project]
    subprocess.run(collect_cmd, cwd=ROOT, check=False)
    subprocess.run(report_cmd, cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
