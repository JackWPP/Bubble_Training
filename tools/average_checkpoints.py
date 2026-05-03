"""Average compatible Ultralytics checkpoints into one eval-ready checkpoint."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules


def model_from_checkpoint(checkpoint: dict[str, Any]):
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError("Checkpoint has neither 'ema' nor 'model'")
    return model


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts supplied")

    keys = list(state_dicts[0].keys())
    for index, state in enumerate(state_dicts[1:], start=1):
        if list(state.keys()) != keys:
            missing = set(keys).symmetric_difference(state.keys())
            raise ValueError(f"State dict {index} has different keys: {sorted(missing)[:10]}")

    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key].detach().cpu() for state in state_dicts]
        first = tensors[0]
        if any(tensor.shape != first.shape for tensor in tensors):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise ValueError(f"Shape mismatch for {key}: {shapes}")
        if torch.is_floating_point(first):
            value = torch.stack([tensor.float() for tensor in tensors], dim=0).mean(dim=0).to(dtype=first.dtype)
        else:
            value = first.clone()
        averaged[key] = value
    return averaged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Output .pt path")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Compatible .pt checkpoints")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise ValueError("At least two checkpoints are required")

    register_bubble_modules()
    loaded = [torch.load(path, map_location="cpu") for path in args.checkpoints]
    models = [model_from_checkpoint(checkpoint).float().cpu() for checkpoint in loaded]
    states = [model.state_dict() for model in models]
    averaged_state = average_state_dicts(states)

    out_model = deepcopy(models[0]).float().cpu()
    out_model.load_state_dict(averaged_state, strict=True)
    out_model.eval()
    for parameter in out_model.parameters():
        parameter.requires_grad_(False)

    out_checkpoint = dict(loaded[0])
    out_checkpoint["model"] = None
    out_checkpoint["ema"] = out_model
    out_checkpoint["optimizer"] = None
    out_checkpoint["scaler"] = None
    out_checkpoint["updates"] = 0
    out_checkpoint["epoch"] = -1
    out_checkpoint["best_fitness"] = None
    out_checkpoint["date"] = datetime.now().isoformat(timespec="seconds")
    out_checkpoint["soup"] = {
        "method": "uniform_average",
        "sources": [str(path) for path in args.checkpoints],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_checkpoint, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
