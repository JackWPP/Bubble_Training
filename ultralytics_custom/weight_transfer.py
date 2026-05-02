"""Weight transfer helpers for Bubble-YOLO structure ablations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch.nn as nn

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER


_MODEL_KEY_RE = re.compile(r"^model\.(\d+)\.(.+)$")


def _class_name(model: nn.Module, index: int) -> str:
    return model.model[index].__class__.__name__  # type: ignore[attr-defined]


def _layer_mapping(model: nn.Module) -> dict[int, int] | None:
    """Return baseline YOLO11s layer index mapping for supported Bubble structures."""
    layers = getattr(model, "model", None)
    if layers is None:
        return None

    count = len(layers)
    names = [_class_name(model, i) for i in range(count)]
    if count < 24 or names[0] != "Conv" or names[16] != "C3k2":
        return None

    mapping = {i: i for i in range(17)}

    # One P3 refinement block inserted after baseline layer 16.
    p3_refine_names = {
        "SSBRefine",
        "GLRB",
        "MSLRefine",
        "P3CAGate",
        "P3SAGate",
        "P3LCRefine",
        "P3MLCRefine",
        "ECAGate",
        "CoordGate",
        "SimAMGate",
    }

    if count == 25 and names[24] == "Detect" and names[17] in p3_refine_names:
        mapping.update({17: 18, 19: 20, 20: 21, 22: 23, 23: 24})
        return mapping

    # Two P3 refinement blocks inserted after baseline layer 16.
    if count == 26 and names[25] == "Detect" and names[17] in p3_refine_names and names[18] in p3_refine_names:
        mapping.update({17: 19, 19: 21, 20: 22, 22: 24, 23: 25})
        return mapping

    # P3/P4/P5 each get SSBRefine+GLRB after the original detection inputs.
    if (
        count == 30
        and names[29] == "Detect"
        and names[17:19] == ["SSBRefine", "GLRB"]
        and names[22:24] == ["SSBRefine", "GLRB"]
        and names[27:29] == ["SSBRefine", "GLRB"]
    ):
        mapping.update({17: 19, 19: 21, 20: 24, 22: 26, 23: 29})
        return mapping

    return None


def supports_bubble_remap(model: nn.Module) -> bool:
    """Return True when the target model is a supported inserted Bubble topology."""
    return _layer_mapping(model) is not None


def load_bubble_remapped_weights(model: nn.Module, weights: str | Path | nn.Module, verbose: bool = True) -> dict[str, Any]:
    """Load YOLO11s weights into a Bubble topology with inserted refinement layers.

    Ultralytics' default loader matches state_dict keys by name and shape. Inserting
    a module before P3 shifts later layer numbers, so the default path drops the
    neck and Detect weights. This loader remaps baseline layer indices explicitly.
    """
    mapping = _layer_mapping(model)
    if mapping is None:
        raise ValueError("Unsupported Bubble topology for remapped weight transfer")

    if isinstance(weights, (str, Path)):
        source_model, _ = load_checkpoint(weights)
        weights_label = str(weights)
    else:
        source_model = weights
        weights_label = weights.__class__.__name__
    source_state = source_model.float().state_dict()
    target_state = model.state_dict()

    updates = {}
    mismatched = []
    missing = []
    for source_key, value in source_state.items():
        match = _MODEL_KEY_RE.match(source_key)
        if not match:
            continue
        source_index = int(match.group(1))
        suffix = match.group(2)
        target_index = mapping.get(source_index)
        if target_index is None:
            continue
        target_key = f"model.{target_index}.{suffix}"
        if target_key not in target_state:
            missing.append((source_key, target_key))
            continue
        if target_state[target_key].shape != value.shape:
            mismatched.append((source_key, target_key, tuple(value.shape), tuple(target_state[target_key].shape)))
            continue
        updates[target_key] = value

    model.load_state_dict(updates, strict=False)
    stats = {
        "transferred": len(updates),
        "total": len(target_state),
        "source_total": len(source_state),
        "missing": len(missing),
        "mismatched": len(mismatched),
        "weights": weights_label,
    }
    if verbose:
        LOGGER.info(
            "Bubble remap transferred %s/%s items from %s (%s missing, %s mismatched)",
            stats["transferred"],
            stats["total"],
            stats["weights"],
            stats["missing"],
            stats["mismatched"],
        )
    return stats
