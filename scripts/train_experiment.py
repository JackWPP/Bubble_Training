"""Train one Bubble-YOLO11s experiment and export a compact summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules
from ultralytics_custom.bubble_loss import enable_nwd_loss


TRAIN_KEYS = {
    "data",
    "imgsz",
    "epochs",
    "batch",
    "workers",
    "device",
    "seed",
    "patience",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "cos_lr",
    "close_mosaic",
    "save_period",
    "plots",
    "amp",
    "val",
    "mosaic",
    "mixup",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "fliplr",
    "flipud",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_project_path(value: str | os.PathLike[str]) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def resolve_model_path(value: str | os.PathLike[str]) -> str:
    text = str(value)
    candidate = ROOT / text
    if candidate.exists() or "/" in text or "\\" in text:
        return str(candidate if not Path(text).is_absolute() else Path(text))
    return text


def parse_device(value: Any) -> Any:
    if isinstance(value, str) and "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def metric_dict(metrics: Any) -> dict[str, Any]:
    return jsonable(getattr(metrics, "results_dict", {}) or {})


def model_info_dict(model: Any) -> dict[str, Any]:
    try:
        info = model.info(verbose=False)
    except Exception as exc:
        return {"error": str(exc)}
    if isinstance(info, tuple):
        keys = ("layers", "params", "gradients", "flops")
        return {keys[i]: jsonable(v) for i, v in enumerate(info[: len(keys)])}
    return {"raw": jsonable(info)}


def load_experiment(exp_id: str, matrix_path: Path) -> dict[str, Any]:
    matrix = load_yaml(matrix_path).get("experiments", {})
    if exp_id not in matrix:
        available = ", ".join(sorted(matrix))
        raise KeyError(f"Unknown experiment {exp_id!r}. Available: {available}")
    exp = dict(matrix[exp_id])
    exp["exp_id"] = exp_id
    return exp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="E0", help="Experiment id from configs/train/experiments.yaml")
    parser.add_argument("--preset", choices=("smoke", "full", "debug_overfit"), default="smoke")
    parser.add_argument("--config", type=Path, help="Training preset YAML override")
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "train" / "experiments.yaml")
    parser.add_argument("--model", help="Override model yaml/weights")
    parser.add_argument("--name", help="Override Ultralytics run name")
    parser.add_argument("--data", help="Override dataset yaml")
    parser.add_argument("--project", help="Override Ultralytics project directory")
    parser.add_argument("--weights", default=os.getenv("BUBBLE_PRETRAINED_WEIGHTS", "yolo11s.pt"))
    parser.add_argument("--device", default=os.getenv("BUBBLE_DEVICE"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--use-nwd", action="store_true")
    parser.add_argument("--nwd-weight", type=float)
    parser.add_argument("--nwd-constant", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    register_bubble_modules()

    preset_path = args.config or ROOT / "configs" / "train" / f"{args.preset}.yaml"
    config = load_yaml(preset_path)
    exp = load_experiment(args.exp, args.matrix if args.matrix.is_absolute() else ROOT / args.matrix)

    model_path = resolve_model_path(args.model or exp["model"])
    data_path = resolve_project_path(args.data or config["data"])
    project = resolve_project_path(args.project or config.get("project", "runs/bubble"))
    run_name = args.name or exp["name"]
    if args.preset != "full" and not args.name:
        run_name = f"{run_name}_{args.preset}"

    overrides = {
        "data": data_path,
        "project": project,
        "name": run_name,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
    }
    for key in ("device", "epochs", "batch", "workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.device:
        config["device"] = args.device

    train_args = {key: config[key] for key in TRAIN_KEYS if key in config}
    train_args.update(overrides)
    train_args["device"] = parse_device(train_args.get("device"))

    use_nwd = bool(exp.get("use_nwd", False) or args.use_nwd)
    nwd_weight = args.nwd_weight if args.nwd_weight is not None else float(exp.get("nwd_weight", 0.4))
    nwd_constant = args.nwd_constant if args.nwd_constant is not None else float(exp.get("nwd_constant", 12.8))
    if use_nwd:
        enable_nwd_loss(nwd_weight=nwd_weight, nwd_constant=nwd_constant)

    from ultralytics import YOLO

    model = YOLO(model_path)
    if not args.no_pretrained and model_path.endswith((".yaml", ".yml")):
        train_args["pretrained"] = resolve_model_path(args.weights)
    elif not args.no_pretrained:
        train_args["pretrained"] = True
    else:
        train_args["pretrained"] = False

    print(f"[train] exp={args.exp} name={run_name} model={model_path}")
    print(f"[train] data={data_path} project={project} device={train_args.get('device')}")
    if use_nwd:
        print(f"[train] NWD enabled weight={nwd_weight} constant={nwd_constant}")

    results = model.train(**train_args)
    run_dir = Path(getattr(results, "save_dir", Path(project) / run_name))
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    eval_weight = best_pt if best_pt.exists() else last_pt if last_pt.exists() else Path(model_path)

    val_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    info: dict[str, Any] = {}
    if not args.skip_val and eval_weight.exists():
        eval_model = YOLO(str(eval_weight))
        info = model_info_dict(eval_model)
        val_metrics = metric_dict(
            eval_model.val(
                data=data_path,
                imgsz=config.get("imgsz", 640),
                device=train_args.get("device"),
                split="val",
                project=str(Path(project) / "validation"),
                name=f"{run_name}_val",
                exist_ok=True,
            )
        )
        test_metrics = metric_dict(
            eval_model.val(
                data=data_path,
                imgsz=config.get("imgsz", 640),
                device=train_args.get("device"),
                split="test",
                project=str(Path(project) / "validation"),
                name=f"{run_name}_test",
                exist_ok=True,
            )
        )
        if config.get("predict", False) and not args.skip_predict:
            source = Path(data_path).parent / "test" / "images"
            eval_model.predict(
                source=str(source),
                conf=float(config.get("predict_conf", 0.25)),
                iou=float(config.get("predict_iou", 0.60)),
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(Path(project) / "predictions"),
                name=run_name,
                exist_ok=True,
            )

    summary = {
        "exp_id": args.exp,
        "name": run_name,
        "modules": exp.get("modules", ""),
        "model": model_path,
        "pretrained_weight": resolve_model_path(args.weights),
        "data_config": data_path,
        "preset": args.preset,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt) if best_pt.exists() else "",
        "last_pt": str(last_pt) if last_pt.exists() else "",
        "use_nwd": use_nwd,
        "nwd_weight": nwd_weight if use_nwd else 0.0,
        "nwd_constant": nwd_constant if use_nwd else 0.0,
        "model_info": info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_args": jsonable(train_args),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_dir = ROOT / "runs" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"{run_name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
