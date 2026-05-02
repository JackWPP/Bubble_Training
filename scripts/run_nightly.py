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
    parser.add_argument(
        "--preset",
        choices=("smoke", "full", "full_conservative", "full_conservative_freeze", "debug_overfit"),
        default="full_conservative",
    )
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--batch", type=int, help="Override training batch size")
    parser.add_argument("--workers", type=int, help="Override dataloader workers")
    parser.add_argument("--data")
    parser.add_argument("--weights")
    parser.add_argument("--project", default=str(ROOT / "runs" / "bubble"))
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "train" / "experiments.yaml")
    parser.add_argument("--experiments", nargs="+", help="Explicit experiment ids")
    parser.add_argument("--baseline-fix", action="store_true", help="Run B0,B1,B2 only")
    parser.add_argument("--compressed", action="store_true", help="Run E0,E1,E3,E5 only")
    parser.add_argument("--balanced-v3", action="store_true", help="Run BV2S_CTL,BV3S_A,BV3S_B,BV3S_768")
    parser.add_argument(
        "--paper-v4-ablation",
        action="store_true",
        help="Run PV4S_768_LR0010,PV4E1_SSB_P3,PV4E2_GLRB_P3,PV4E3_SSB_GLRB_P3",
    )
    parser.add_argument(
        "--paper-v4-rescue",
        action="store_true",
        help="Run PV4E1Z_SSB_P3_ID,PV4E1Z_SSB_P3_LOWLR,PV4E1Z_SSB_P3_FREEZE17",
    )
    parser.add_argument("--resume-missing", action="store_true", help="Skip experiments that already have best.pt")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    matrix = load_matrix(args.matrix if args.matrix.is_absolute() else ROOT / args.matrix)["experiments"]
    if args.experiments:
        exp_ids = args.experiments
    elif args.baseline_fix:
        exp_ids = ["B0", "B1", "B2"]
    elif args.compressed:
        exp_ids = ["E0", "E1", "E3", "E5"]
    elif args.balanced_v3:
        exp_ids = ["BV2S_CTL", "BV3S_A", "BV3S_B", "BV3S_768"]
    elif args.paper_v4_ablation:
        exp_ids = ["PV4S_768_LR0010", "PV4E1_SSB_P3", "PV4E2_GLRB_P3", "PV4E3_SSB_GLRB_P3"]
    elif args.paper_v4_rescue:
        exp_ids = ["PV4E1Z_SSB_P3_ID", "PV4E1Z_SSB_P3_LOWLR", "PV4E1Z_SSB_P3_FREEZE17"]
    else:
        exp_ids = ["B0", "B1", "E0", "E1", "E2", "E3", "E4", "E5"]

    if (args.paper_v4_ablation or args.paper_v4_rescue) and Path(args.project).name != "bubble_paper_v4":
        raise ValueError("--paper-v4-ablation/--paper-v4-rescue must use --project runs/bubble_paper_v4")

    for exp_id in exp_ids:
        name = matrix[exp_id]["name"]
        has_exp_config = bool(matrix[exp_id].get("train_config"))
        run_name = name if args.preset == "full" or has_exp_config else f"{name}_{args.preset}"
        if args.preset in {"full_conservative", "full_conservative_freeze"}:
            run_name = name
        best_pt = Path(args.project) / run_name / "weights" / "best.pt"
        summary_json = Path(args.project) / run_name / "summary.json"
        if args.resume_missing:
            if best_pt.exists():
                print(f"[skip] {exp_id} already has {best_pt}")
                continue
            if matrix[exp_id].get("eval_only") and summary_json.exists():
                print(f"[skip] {exp_id} already has {summary_json}")
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
        if args.batch is not None:
            cmd.extend(["--batch", str(args.batch)])
        if args.workers is not None:
            cmd.extend(["--workers", str(args.workers)])
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
