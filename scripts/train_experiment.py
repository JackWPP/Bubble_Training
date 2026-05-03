"""Train one Bubble-YOLO11s experiment and export a compact summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules
from ultralytics_custom.bubble_loss import enable_nwd_loss
from ultralytics_custom.trainer import BubbleDetectionTrainer, BubbleNWDDetectionTrainer


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
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
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

METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}

LOSS_KEYS = {
    "train_box": "train/box_loss",
    "val_box": "val/box_loss",
    "train_cls": "train/cls_loss",
    "val_cls": "val/cls_loss",
    "train_dfl": "train/dfl_loss",
    "val_dfl": "val/dfl_loss",
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


def requested_cuda_device(value: Any) -> bool:
    """Return True when the requested Ultralytics device requires CUDA."""
    if value is None:
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, (list, tuple)):
        return any(requested_cuda_device(item) for item in value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "cpu", "mps"}:
            return False
        if text.startswith("cuda"):
            return True
        if "," in text:
            return any(requested_cuda_device(part.strip()) for part in text.split(","))
        try:
            return int(text) >= 0
        except ValueError:
            return False
    return False


def ensure_requested_device_available(value: Any) -> None:
    if not requested_cuda_device(value):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested, but torch.cuda.is_available() is False. "
            "Refusing to continue because this would risk a slow CPU training/evaluation run. "
            "Check the NVIDIA driver with nvidia-smi or rerun with --device cpu explicitly."
        )


def ensure_repo_pythonpath() -> None:
    """Make repo-local modules importable from Ultralytics DDP temp scripts."""
    root = str(ROOT)
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root, *parts])


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


def metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(METRIC_KEYS[key], 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def float_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selector-eval-mode", choices=("online", "offline"))
    parser.add_argument("--use-nwd", action="store_true")
    parser.add_argument("--nwd-weight", type=float)
    parser.add_argument("--nwd-constant", type=float)
    parser.add_argument("--iou-type", type=str, default="CIoU", choices=["CIoU", "WIoU"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate model without training")
    parser.add_argument("--postprocess-only", action="store_true", help="Select/evaluate an existing run without training")
    return parser


def val_model(
    model_ref: str,
    data: str,
    split: str,
    imgsz: int,
    device: Any,
    project: Path,
    name: str,
    conf: float | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_ref)
    val_args = {
        "data": data,
        "imgsz": imgsz,
        "device": device,
        "split": split,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "plots": False,
    }
    if conf is not None:
        val_args["conf"] = conf
    return metric_dict(model.val(**val_args))


def optional_val_model(
    model_ref: str,
    data: str,
    split: str,
    imgsz: int,
    device: Any,
    project: Path,
    name: str,
) -> dict[str, Any]:
    try:
        return val_model(model_ref, data, split, imgsz, device, project, name)
    except FileNotFoundError as exc:
        print(f"[warn] skipped optional validation {name}: {exc}", file=sys.stderr)
        return {"skipped": True, "error": str(exc)}


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
    """Evaluate one checkpoint on selection val/main test and OOD grouped val/test."""
    bundle = {
        "weight": weight_ref,
        "selection_val_metrics": val_model(
            weight_ref, data_path, "val", imgsz, device, project / "validation", f"{run_name}_{label}_selection_val"
        ),
        "main_test_metrics": val_model(
            weight_ref, data_path, "test", imgsz, device, project / "validation", f"{run_name}_{label}_main_test"
        ),
        "ood_val_metrics": optional_val_model(
            weight_ref, official_data_path, "val", imgsz, device, project / "validation", f"{run_name}_{label}_ood_val"
        ),
        "ood_test_metrics": optional_val_model(
            weight_ref, official_data_path, "test", imgsz, device, project / "validation", f"{run_name}_{label}_ood_test"
        ),
    }
    bundle["official_val_metrics"] = bundle["ood_val_metrics"]
    bundle["official_test_metrics"] = bundle["ood_test_metrics"]
    return bundle


def checkpoint_candidates(run_dir: Path, model_path: str) -> dict[str, str]:
    weights_dir = run_dir / "weights"
    candidates: dict[str, str] = {}
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    if best_pt.exists():
        candidates["best"] = str(best_pt)
    if last_pt.exists():
        candidates["last"] = str(last_pt)
    for path in sorted(weights_dir.glob("epoch*.pt")):
        candidates[path.stem] = str(path)
    if not candidates:
        candidates["model"] = model_path
    return candidates


def read_results_rows(run_dir: Path) -> list[dict[str, str]]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def metrics_from_results_row(row: dict[str, Any]) -> dict[str, Any]:
    return {metric_key: float_value(row, metric_key) for metric_key in METRIC_KEYS.values()}


def epoch_label_from_row(row: dict[str, Any]) -> str:
    epoch = int(float_value(row, "epoch", 0.0))
    return f"epoch{max(epoch - 1, 0)}"


def select_online_checkpoint(
    run_dir: Path,
    precision_min: float,
    recall_min: float,
) -> dict[str, Any]:
    rows = read_results_rows(run_dir)
    if not rows:
        return {}

    viable: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    fallback: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for row in rows:
        metrics = metrics_from_results_row(row)
        precision = metric_value(metrics, "precision")
        recall = metric_value(metrics, "recall")
        map50 = metric_value(metrics, "map50")
        map5095 = metric_value(metrics, "map5095")
        score = (map50, min(precision, recall), map5095)
        target = viable if precision >= precision_min and recall >= recall_min else fallback
        target.append((score, row))

    selected_from = "online_thresholded" if viable else "online_fallback"
    pool = viable or fallback
    score, row = max(pool, key=lambda item: item[0])
    label = epoch_label_from_row(row)
    weight = run_dir / "weights" / f"{label}.pt"
    if not weight.exists():
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        weight = best if best.exists() else last
        label = weight.stem if weight.exists() else label
    metrics = metrics_from_results_row(row)
    return {
        "label": label,
        "weight": str(weight) if weight.exists() else "",
        "selected_from": selected_from,
        "selection_source": "results.csv",
        "online_results_epoch": int(float_value(row, "epoch", 0.0)),
        "precision_min": precision_min,
        "recall_min": recall_min,
        "score": {
            "map50": score[0],
            "min_precision_recall": score[1],
            "map5095": score[2],
        },
        "selection_val_metrics": metrics,
    }


def curve_diagnostics(run_dir: Path) -> dict[str, Any]:
    rows = read_results_rows(run_dir)
    if not rows:
        return {}

    metric_columns = list(METRIC_KEYS.values()) + list(LOSS_KEYS.values())
    bad_values = 0
    for row in rows:
        for column in metric_columns:
            value = str(row.get(column, ""))
            if not value or value.lower() in {"nan", "inf", "-inf"}:
                bad_values += 1

    best_row = max(rows, key=lambda row: float_value(row, METRIC_KEYS["map50"]))
    first_row = rows[0]
    last_row = rows[-1]
    first_map50 = float_value(first_row, METRIC_KEYS["map50"])
    best_map50 = float_value(best_row, METRIC_KEYS["map50"])
    last_map50 = float_value(last_row, METRIC_KEYS["map50"])
    best_val_box = float_value(best_row, LOSS_KEYS["val_box"])
    last_val_box = float_value(last_row, LOSS_KEYS["val_box"])
    best_train_box = float_value(best_row, LOSS_KEYS["train_box"])
    last_train_box = float_value(last_row, LOSS_KEYS["train_box"])
    return {
        "rows": len(rows),
        "bad_values": bad_values,
        "first_epoch": int(float_value(first_row, "epoch")),
        "best_epoch": int(float_value(best_row, "epoch")),
        "last_epoch": int(float_value(last_row, "epoch")),
        "first_map50": first_map50,
        "best_map50": best_map50,
        "last_map50": last_map50,
        "map50_gain_first_to_best": best_map50 - first_map50,
        "map50_drop_best_to_last": best_map50 - last_map50,
        "first_precision": float_value(first_row, METRIC_KEYS["precision"]),
        "first_recall": float_value(first_row, METRIC_KEYS["recall"]),
        "first_map5095": float_value(first_row, METRIC_KEYS["map5095"]),
        "best_precision": float_value(best_row, METRIC_KEYS["precision"]),
        "best_recall": float_value(best_row, METRIC_KEYS["recall"]),
        "best_map5095": float_value(best_row, METRIC_KEYS["map5095"]),
        "first_val_dfl_loss": float_value(first_row, LOSS_KEYS["val_dfl"]),
        "best_val_box_loss": best_val_box,
        "last_val_box_loss": last_val_box,
        "val_box_loss_delta_best_to_last": last_val_box - best_val_box,
        "first_train_dfl_loss": float_value(first_row, LOSS_KEYS["train_dfl"]),
        "best_train_box_loss": best_train_box,
        "last_train_box_loss": last_train_box,
        "train_box_loss_delta_best_to_last": last_train_box - best_train_box,
        "train_loss_continued_down": last_train_box < best_train_box,
        "val_box_loss_improved_after_best": last_val_box < best_val_box,
    }


def select_map50_checkpoint(
    checkpoint_metrics: dict[str, Any],
    precision_min: float,
    recall_min: float,
) -> dict[str, Any]:
    viable: list[tuple[tuple[float, float, float], str, dict[str, Any]]] = []
    fallback: list[tuple[tuple[float, float, float], str, dict[str, Any]]] = []
    for label, bundle in checkpoint_metrics.items():
        metrics = bundle.get("selection_val_metrics", {}) or {}
        precision = metric_value(metrics, "precision")
        recall = metric_value(metrics, "recall")
        map50 = metric_value(metrics, "map50")
        map5095 = metric_value(metrics, "map5095")
        score = (map50, min(precision, recall), map5095)
        target = viable if precision >= precision_min and recall >= recall_min else fallback
        target.append((score, label, bundle))

    selected_from = "thresholded" if viable else "fallback"
    pool = viable or fallback
    if not pool:
        return {}
    score, label, bundle = max(pool, key=lambda item: item[0])
    return {
        "label": label,
        "weight": bundle.get("weight", ""),
        "selected_from": selected_from,
        "precision_min": precision_min,
        "recall_min": recall_min,
        "score": {
            "map50": score[0],
            "min_precision_recall": score[1],
            "map5095": score[2],
        },
        "selection_val_metrics": bundle.get("selection_val_metrics", {}),
        "main_test_metrics": bundle.get("main_test_metrics", {}),
        "ood_val_metrics": bundle.get("ood_val_metrics", bundle.get("official_val_metrics", {})),
        "ood_test_metrics": bundle.get("ood_test_metrics", bundle.get("official_test_metrics", {})),
    }


def attach_final_evaluations(
    selected: dict[str, Any],
    data_path: str,
    official_data_path: str,
    imgsz: int,
    device: Any,
    project: Path,
    run_name: str,
) -> dict[str, Any]:
    weight = selected.get("weight", "")
    label = selected.get("label", "selected")
    if not weight:
        return selected
    selected["main_test_metrics"] = val_model(
        weight, data_path, "test", imgsz, device, project / "validation", f"{run_name}_{label}_main_test"
    )
    selected["ood_val_metrics"] = optional_val_model(
        weight, official_data_path, "val", imgsz, device, project / "validation", f"{run_name}_{label}_ood_val"
    )
    selected["ood_test_metrics"] = optional_val_model(
        weight, official_data_path, "test", imgsz, device, project / "validation", f"{run_name}_{label}_ood_test"
    )
    selected["official_val_metrics"] = selected["ood_val_metrics"]
    selected["official_test_metrics"] = selected["ood_test_metrics"]
    return selected


def run_conf_sweep(
    weight_ref: str,
    data_path: str,
    imgsz: int,
    device: Any,
    project: Path,
    run_name: str,
    conf_values: list[float],
) -> dict[str, Any]:
    if not weight_ref or not conf_values:
        return {}
    rows = []
    for conf in conf_values:
        metrics = val_model(
            weight_ref,
            data_path,
            "test",
            imgsz,
            device,
            project / "conf_sweep",
            f"{run_name}_conf_{str(conf).replace('.', '_')}",
            conf=conf,
        )
        precision = metric_value(metrics, "precision")
        recall = metric_value(metrics, "recall")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append(
            {
                "conf": conf,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": metric_value(metrics, "map50"),
                "map5095": metric_value(metrics, "map5095"),
                "metrics": metrics,
            }
        )
    best = max(rows, key=lambda row: (row["f1"], min(row["precision"], row["recall"]), row["map50"]))
    return {"rows": rows, "best": best}


def main() -> int:
    args = build_parser().parse_args()
    ensure_repo_pythonpath()
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
    for key in ("device", "epochs", "batch", "workers", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.device:
        config["device"] = args.device
    if args.selector_eval_mode:
        config["selector_eval_mode"] = args.selector_eval_mode

    train_args = {key: config[key] for key in TRAIN_KEYS if key in config}
    train_args.update(overrides)
    train_args["device"] = parse_device(train_args.get("device"))
    ensure_requested_device_available(train_args.get("device"))

    use_nwd = bool(exp.get("use_nwd", False) or args.use_nwd)
    nwd_weight = args.nwd_weight if args.nwd_weight is not None else float(exp.get("nwd_weight", 0.4))
    nwd_constant = args.nwd_constant if args.nwd_constant is not None else float(exp.get("nwd_constant", 12.8))
    iou_type = args.iou_type if args.iou_type != "CIoU" else exp.get("iou_type", "CIoU")
    if use_nwd:
        enable_nwd_loss(nwd_weight=nwd_weight, nwd_constant=nwd_constant, iou_type=iou_type)
        os.environ["BUBBLE_NWD_WEIGHT"] = str(nwd_weight)
        os.environ["BUBBLE_NWD_CONSTANT"] = str(nwd_constant)
        os.environ["BUBBLE_IOU_TYPE"] = iou_type

    from ultralytics import YOLO

    eval_only = bool(args.eval_only or exp.get("eval_only", False))
    postprocess_only = bool(args.postprocess_only)
    if eval_only and postprocess_only:
        raise ValueError("--eval-only and --postprocess-only are mutually exclusive")
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
        print(f"[train] NWD enabled weight={nwd_weight} constant={nwd_constant} iou_type={iou_type}")

    run_dir = Path(project) / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    checkpoint_metrics: dict[str, Any] = {}
    selection_val_metrics: dict[str, Any] = {}
    official_val_metrics: dict[str, Any] = {}
    official_test_metrics: dict[str, Any] = {}
    main_test_metrics: dict[str, Any] = {}
    ood_val_metrics: dict[str, Any] = {}
    ood_test_metrics: dict[str, Any] = {}
    map50_selected: dict[str, Any] = {}
    conf_sweep: dict[str, Any] = {}
    info: dict[str, Any] = {}
    train_curve: dict[str, Any] = {}
    selector_eval_mode = str(config.get("selector_eval_mode", "offline")).lower()
    if selector_eval_mode not in {"online", "offline"}:
        raise ValueError(f"selector_eval_mode must be 'online' or 'offline', got {selector_eval_mode!r}")

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
            main_test_metrics = checkpoint_metrics["pretrained"]["main_test_metrics"]
            ood_val_metrics = checkpoint_metrics["pretrained"]["ood_val_metrics"]
            ood_test_metrics = checkpoint_metrics["pretrained"]["ood_test_metrics"]
            official_val_metrics = ood_val_metrics
            official_test_metrics = ood_test_metrics
            map50_selected = select_map50_checkpoint(
                checkpoint_metrics,
                float(config.get("selector_precision_min", 0.75)),
                float(config.get("selector_recall_min", 0.72)),
            )
    elif postprocess_only:
        if not run_dir.exists():
            raise FileNotFoundError(f"Cannot postprocess missing run directory: {run_dir}")
        train_curve = curve_diagnostics(run_dir)
        if train_curve:
            print(
                "[curve] best_epoch={best} last_epoch={last} map50_drop={drop:.4f} bad_values={bad}".format(
                    best=train_curve.get("best_epoch", ""),
                    last=train_curve.get("last_epoch", ""),
                    drop=float(train_curve.get("map50_drop_best_to_last", 0.0) or 0.0),
                    bad=train_curve.get("bad_values", ""),
                )
            )

        if not args.skip_val:
            if selector_eval_mode == "online":
                map50_selected = select_online_checkpoint(
                    run_dir,
                    float(config.get("selector_precision_min", 0.75)),
                    float(config.get("selector_recall_min", 0.72)),
                )
                map50_selected = attach_final_evaluations(
                    map50_selected,
                    data_path,
                    official_eval_data_path,
                    int(config.get("imgsz", 640)),
                    train_args.get("device"),
                    Path(project),
                    run_name,
                )
                selected_weight = map50_selected.get("weight", str(best_pt if best_pt.exists() else last_pt))
                preferred = map50_selected
                if selected_weight:
                    checkpoint_metrics[map50_selected.get("label", "selected")] = preferred
            else:
                eval_weights = checkpoint_candidates(run_dir, model_path)
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
                map50_selected = select_map50_checkpoint(
                    checkpoint_metrics,
                    float(config.get("selector_precision_min", 0.75)),
                    float(config.get("selector_recall_min", 0.72)),
                )
                selected_label = map50_selected.get("label", "")
                selected_bundle = checkpoint_metrics.get(selected_label, {}) if selected_label else {}
                preferred = selected_bundle or checkpoint_metrics.get("best") or next(iter(checkpoint_metrics.values()), {})
                selected_weight = preferred.get("weight", next(iter(eval_weights.values())))

            selection_val_metrics = preferred.get("selection_val_metrics", {})
            main_test_metrics = preferred.get("main_test_metrics", {})
            ood_val_metrics = preferred.get("ood_val_metrics", preferred.get("official_val_metrics", {}))
            ood_test_metrics = preferred.get("ood_test_metrics", preferred.get("official_test_metrics", {}))
            official_val_metrics = ood_val_metrics
            official_test_metrics = ood_test_metrics
            selected_path = run_dir / "weights" / "map50_selected.pt"
            if selected_weight and Path(selected_weight).exists() and Path(selected_weight) != selected_path:
                shutil.copy2(selected_weight, selected_path)
                map50_selected["copied_weight"] = str(selected_path)
            conf_values = [float(value) for value in config.get("conf_sweep", [])]
            conf_sweep = run_conf_sweep(
                selected_weight,
                data_path,
                int(config.get("imgsz", 640)),
                train_args.get("device"),
                Path(project),
                run_name,
                conf_values,
            )
            eval_model = YOLO(selected_weight)
            info = model_info_dict(eval_model)

    else:
        model = YOLO(model_path)
        trainer_cls = BubbleNWDDetectionTrainer if use_nwd else BubbleDetectionTrainer
        results = model.train(trainer=trainer_cls, **train_args)
        run_dir = Path(getattr(results, "save_dir", run_dir))
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        eval_weights = checkpoint_candidates(run_dir, model_path)
        train_curve = curve_diagnostics(run_dir)
        if train_curve:
            print(
                "[curve] best_epoch={best} last_epoch={last} map50_drop={drop:.4f} bad_values={bad}".format(
                    best=train_curve.get("best_epoch", ""),
                    last=train_curve.get("last_epoch", ""),
                    drop=float(train_curve.get("map50_drop_best_to_last", 0.0) or 0.0),
                    bad=train_curve.get("bad_values", ""),
                )
            )

        if not args.skip_val:
            if selector_eval_mode == "online":
                map50_selected = select_online_checkpoint(
                    run_dir,
                    float(config.get("selector_precision_min", 0.75)),
                    float(config.get("selector_recall_min", 0.72)),
                )
                map50_selected = attach_final_evaluations(
                    map50_selected,
                    data_path,
                    official_eval_data_path,
                    int(config.get("imgsz", 640)),
                    train_args.get("device"),
                    Path(project),
                    run_name,
                )
                selected_weight = map50_selected.get("weight", str(best_pt if best_pt.exists() else last_pt))
                preferred = map50_selected
                if selected_weight:
                    checkpoint_metrics[map50_selected.get("label", "selected")] = preferred
            else:
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
                map50_selected = select_map50_checkpoint(
                    checkpoint_metrics,
                    float(config.get("selector_precision_min", 0.75)),
                    float(config.get("selector_recall_min", 0.72)),
                )
                selected_label = map50_selected.get("label", "")
                selected_bundle = checkpoint_metrics.get(selected_label, {}) if selected_label else {}
                preferred = selected_bundle or checkpoint_metrics.get("best") or next(iter(checkpoint_metrics.values()), {})
                selected_weight = preferred.get("weight", next(iter(eval_weights.values())))
            selection_val_metrics = preferred.get("selection_val_metrics", {})
            main_test_metrics = preferred.get("main_test_metrics", {})
            ood_val_metrics = preferred.get("ood_val_metrics", preferred.get("official_val_metrics", {}))
            ood_test_metrics = preferred.get("ood_test_metrics", preferred.get("official_test_metrics", {}))
            official_val_metrics = ood_val_metrics
            official_test_metrics = ood_test_metrics
            selected_path = run_dir / "weights" / "map50_selected.pt"
            if selected_weight and Path(selected_weight).exists() and Path(selected_weight) != selected_path:
                shutil.copy2(selected_weight, selected_path)
                map50_selected["copied_weight"] = str(selected_path)
            conf_values = [float(value) for value in config.get("conf_sweep", [])]
            conf_sweep = run_conf_sweep(
                selected_weight,
                data_path,
                int(config.get("imgsz", 640)),
                train_args.get("device"),
                Path(project),
                run_name,
                conf_values,
            )
            eval_model = YOLO(selected_weight)
            info = model_info_dict(eval_model)

        if config.get("predict", False) and not args.skip_predict and best_pt.exists():
            predict_weight = map50_selected.get("copied_weight") or map50_selected.get("weight") or str(best_pt)
            eval_model = YOLO(str(predict_weight))
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
        "map50_selected_pt": map50_selected.get("copied_weight", map50_selected.get("weight", "")),
        "eval_only": eval_only,
        "use_nwd": use_nwd,
        "nwd_weight": nwd_weight if use_nwd else 0.0,
        "nwd_constant": nwd_constant if use_nwd else 0.0,
        "iou_type": iou_type if use_nwd else "CIoU",
        "model_info": info,
        "selection_val_metrics": selection_val_metrics,
        "main_test_metrics": main_test_metrics,
        "ood_val_metrics": ood_val_metrics,
        "ood_test_metrics": ood_test_metrics,
        "official_val_metrics": official_val_metrics,
        "official_test_metrics": official_test_metrics,
        "map50_selected": map50_selected,
        "conf_sweep": conf_sweep,
        "checkpoint_metrics": checkpoint_metrics,
        "selector_eval_mode": selector_eval_mode,
        "train_curve": train_curve,
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
