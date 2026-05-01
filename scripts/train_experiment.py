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
    "freeze",
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
    torch_model = getattr(model, "model", None)
    if torch_model is not None:
        params = sum(p.numel() for p in torch_model.parameters())
        gradients = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        return {"params": params, "gradients": gradients, "flops": ""}
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
    parser.add_argument(
        "--preset",
        choices=("smoke", "full", "full_conservative", "full_conservative_freeze", "debug_overfit"),
        default="smoke",
    )
    parser.add_argument("--config", type=Path, help="Training preset YAML override")
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "train" / "experiments.yaml")
    parser.add_argument("--model", help="Override model yaml/weights")
    parser.add_argument("--name", help="Override Ultralytics run name")
    parser.add_argument("--data", help="Override dataset yaml")
    parser.add_argument("--eval-data", help="Official grouped eval dataset yaml")
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
    parser.add_argument("--eval-only", action="store_true", help="Evaluate model without training")
    return parser


def val_model(
    model_ref: str,
    data: str,
    split: str,
    imgsz: int,
    device: Any,
    project: Path,
    name: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_ref)
    return metric_dict(
        model.val(
            data=data,
            imgsz=imgsz,
            device=device,
            split=split,
            project=str(project),
            name=name,
            exist_ok=True,
        )
    )


def eval_weight_bundle(
    weight_ref: str,
    data_path: str,
    official_data_path: str,
    imgsz: int,
    device: Any,
    project: Path,
    run_name: str,
    label: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on selection val and official grouped val/test."""
    bundle = {
        "weight": weight_ref,
        "selection_val_metrics": val_model(
            weight_ref, data_path, "val", imgsz, device, project / "validation", f"{run_name}_{label}_selection_val"
        ),
        "official_val_metrics": val_model(
            weight_ref, official_data_path, "val", imgsz, device, project / "validation", f"{run_name}_{label}_official_val"
        ),
        "official_test_metrics": val_model(
            weight_ref, official_data_path, "test", imgsz, device, project / "validation", f"{run_name}_{label}_official_test"
        ),
    }
    return bundle


def main() -> int:
    args = build_parser().parse_args()
    register_bubble_modules()

    exp = load_experiment(args.exp, args.matrix if args.matrix.is_absolute() else ROOT / args.matrix)
    has_exp_config = bool(args.config or exp.get("train_config"))
    preset_path = args.config or exp.get("train_config") or ROOT / "configs" / "train" / f"{args.preset}.yaml"
    preset_path = preset_path if isinstance(preset_path, Path) else Path(preset_path)
    config = load_yaml(preset_path if preset_path.is_absolute() else ROOT / preset_path)

    model_path = resolve_model_path(args.model or exp["model"])
    data_path = resolve_project_path(args.data or config["data"])
    official_eval_data_path = resolve_project_path(args.eval_data or config.get("eval_data", data_path))
    project = resolve_project_path(args.project or config.get("project", "runs/bubble"))
    run_name = args.name or exp["name"]
    if not has_exp_config and args.preset not in {"full", "full_conservative"} and not args.name:
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

    eval_only = bool(args.eval_only or exp.get("eval_only", False))
    if not args.no_pretrained and model_path.endswith((".yaml", ".yml")):
        effective_pretrained_weight = resolve_model_path(args.weights)
        train_args["pretrained"] = effective_pretrained_weight
    elif not args.no_pretrained:
        effective_pretrained_weight = model_path if model_path.endswith(".pt") else ""
        train_args["pretrained"] = True
    else:
        effective_pretrained_weight = ""
        train_args["pretrained"] = False

    print(f"[train] exp={args.exp} name={run_name} model={model_path}")
    print(f"[train] data={data_path} official_eval_data={official_eval_data_path}")
    print(f"[train] project={project} device={train_args.get('device')}")
    if use_nwd:
        print(f"[train] NWD enabled weight={nwd_weight} constant={nwd_constant}")

    run_dir = Path(project) / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    checkpoint_metrics: dict[str, Any] = {}
    selection_val_metrics: dict[str, Any] = {}
    official_val_metrics: dict[str, Any] = {}
    official_test_metrics: dict[str, Any] = {}
    info: dict[str, Any] = {}

    if eval_only:
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_weight_ref = model_path
        eval_model = YOLO(eval_weight_ref)
        info = model_info_dict(eval_model)
        if not args.skip_val:
            checkpoint_metrics["pretrained"] = eval_weight_bundle(
                eval_weight_ref,
                data_path,
                official_eval_data_path,
                int(config.get("imgsz", 640)),
                train_args.get("device"),
                Path(project),
                run_name,
                "pretrained",
            )
            selection_val_metrics = checkpoint_metrics["pretrained"]["selection_val_metrics"]
            official_val_metrics = checkpoint_metrics["pretrained"]["official_val_metrics"]
            official_test_metrics = checkpoint_metrics["pretrained"]["official_test_metrics"]
    else:
        model = YOLO(model_path)
        results = model.train(**train_args)
        run_dir = Path(getattr(results, "save_dir", run_dir))
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        eval_weights: dict[str, str] = {}
        if best_pt.exists():
            eval_weights["best"] = str(best_pt)
        if last_pt.exists() and last_pt != best_pt:
            eval_weights["last"] = str(last_pt)
        if not eval_weights:
            eval_weights["model"] = model_path

        if not args.skip_val:
            for label, weight_ref in eval_weights.items():
                checkpoint_metrics[label] = eval_weight_bundle(
                    weight_ref,
                    data_path,
                    official_eval_data_path,
                    int(config.get("imgsz", 640)),
                    train_args.get("device"),
                    Path(project),
                    run_name,
                    label,
                )
            preferred = checkpoint_metrics.get("best") or next(iter(checkpoint_metrics.values()), {})
            selection_val_metrics = preferred.get("selection_val_metrics", {})
            official_val_metrics = preferred.get("official_val_metrics", {})
            official_test_metrics = preferred.get("official_test_metrics", {})
            eval_model = YOLO(preferred.get("weight", next(iter(eval_weights.values()))))
            info = model_info_dict(eval_model)

        if config.get("predict", False) and not args.skip_predict and best_pt.exists():
            eval_model = YOLO(str(best_pt))
            source = Path(official_eval_data_path).parent / "test" / "images"
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
        "pretrained_weight": effective_pretrained_weight,
        "data_config": data_path,
        "official_eval_data_config": official_eval_data_path,
        "preset": args.preset,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt) if best_pt.exists() else "",
        "last_pt": str(last_pt) if last_pt.exists() else "",
        "eval_only": eval_only,
        "use_nwd": use_nwd,
        "nwd_weight": nwd_weight if use_nwd else 0.0,
        "nwd_constant": nwd_constant if use_nwd else 0.0,
        "model_info": info,
        "selection_val_metrics": selection_val_metrics,
        "official_val_metrics": official_val_metrics,
        "official_test_metrics": official_test_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "val_metrics": official_val_metrics,
        "test_metrics": official_test_metrics,
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
