"""Weight transfer helpers for Bubble-YOLO structure ablations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch.nn as nn

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER


_MODEL_KEY_RE = re.compile(r"^model\.(\d+)\.(.+)$")
_YOLO11S_BASELINE_NAMES = [
    "Conv",
    "Conv",
    "C3k2",
    "Conv",
    "C3k2",
    "Conv",
    "C3k2",
    "Conv",
    "C3k2",
    "SPPF",
    "C2PSA",
    "Upsample",
    "Concat",
    "C3k2",
    "Upsample",
    "Concat",
    "C3k2",
    "Conv",
    "Concat",
    "C3k2",
    "Conv",
    "Concat",
    "C3k2",
    "Detect",
]
_INSERTED_MODULE_NAMES = {
    "SSBRefine",
    "GLRB",
    "LCRefine",
    "MSLRefine",
    "P3CAGate",
    "P3SAGate",
    "P3LCRefine",
    "P3MLCRefine",
    "ECAGate",
    "CoordGate",
    "SimAMGate",
    "WeightedConcat",
    "ChannelWeightedConcat",
}


def _class_name(model: nn.Module, index: int) -> str:
    return model.model[index].__class__.__name__  # type: ignore[attr-defined]


def _inserted_module_mapping(names: list[str]) -> dict[int, int] | None:
    """Map baseline YOLO11s layers through inserted custom modules."""
    mapping: dict[int, int] = {}
    target_index = 0
    skipped = 0
    for source_index, expected_name in enumerate(_YOLO11S_BASELINE_NAMES):
        while target_index < len(names) and names[target_index] != expected_name:
            if names[target_index] not in _INSERTED_MODULE_NAMES:
                return None
            skipped += 1
            target_index += 1
        if target_index >= len(names):
            return None
        mapping[source_index] = target_index
        target_index += 1

    if any(name not in _INSERTED_MODULE_NAMES for name in names[target_index:]):
        return None
    skipped += len(names) - target_index
    return mapping if skipped else None


def _layer_mapping(model: nn.Module) -> dict[int, int] | None:
    """Return baseline YOLO11s layer index mapping for supported Bubble structures."""
    layers = getattr(model, "model", None)
    if layers is None:
        return None

    count = len(layers)
    names = [_class_name(model, i) for i in range(count)]
    if count < 24 or names[0] != "Conv":
        return None

    mapping = {i: i for i in range(17)}

    # One P3 refinement block inserted after baseline layer 16.
    p3_refine_names = {
        "SSBRefine",
        "GLRB",
        "LCRefine",
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

    # One refined P3 backbone skip inserted after baseline layer 4. The
    # downstream stride-16 path still reads the original layer 4 output.
    if count == 25 and names[24] == "Detect" and names[5] in p3_refine_names:
        mapping.update({5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 13: 14, 16: 17, 17: 18, 19: 20, 20: 21, 22: 23, 23: 24})
        return mapping

    # One refined P4 backbone skip inserted after baseline layer 6. The
    # downstream stride-32 path still reads the original layer 6 output.
    if count == 25 and names[24] == "Detect" and names[7] in p3_refine_names:
        mapping.update({7: 8, 8: 9, 9: 10, 10: 11, 13: 14, 16: 17, 17: 18, 19: 20, 20: 21, 22: 23, 23: 24})
        return mapping

    # One P4 refinement block inserted after baseline layer 19.
    if count == 25 and names[24] == "Detect" and names[20] in p3_refine_names:
        mapping.update({17: 17, 19: 19, 20: 21, 22: 23, 23: 24})
        return mapping

    # Two P3 refinement blocks inserted after baseline layer 16.
    if count == 26 and names[25] == "Detect" and names[17] in p3_refine_names and names[18] in p3_refine_names:
        mapping.update({17: 19, 19: 21, 20: 22, 22: 24, 23: 25})
        return mapping

    # Refined P3 and P4 backbone skips inserted after baseline layers 4 and 6.
    if count == 26 and names[25] == "Detect" and names[5] in p3_refine_names and names[8] in p3_refine_names:
        mapping.update({5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12, 13: 15, 16: 18, 17: 19, 19: 21, 20: 22, 22: 24, 23: 25})
        return mapping

    # One P3 and one P4 refinement block inserted after baseline layers 16 and 19.
    if count == 26 and names[25] == "Detect" and names[17] in p3_refine_names and names[21] in p3_refine_names:
        mapping.update({17: 18, 19: 20, 20: 22, 22: 24, 23: 25})
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

    return _inserted_module_mapping(names)


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
