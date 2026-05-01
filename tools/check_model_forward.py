"""Check that a YOLO model YAML builds and forwards one dummy image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules


def iter_tensors(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    register_bubble_modules()
    from ultralytics import YOLO

    model_path = Path(args.model)
    model_ref = str(model_path if model_path.is_absolute() else ROOT / model_path)
    model = YOLO(model_ref)
    model.info(verbose=True)
    torch_model = model.model.to(args.device)
    torch_model.eval()
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        y = torch_model(x)
    tensors = list(iter_tensors(y))
    if not tensors:
        raise RuntimeError("Model forward produced no tensors")
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Model forward produced NaN or Inf")
    print(f"forward ok: {args.model} ({len(tensors)} output tensor groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
