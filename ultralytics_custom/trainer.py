"""Ultralytics trainers that keep Bubble extensions visible in DDP workers."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from . import register_bubble_modules
from .bubble_loss import enable_nwd_loss
from .weight_transfer import load_bubble_remapped_weights, supports_bubble_remap


def _remap_source(weights, pretrained):
    if isinstance(pretrained, (str, Path)) and str(pretrained).endswith(".pt"):
        return pretrained
    return weights


class BubbleDetectionTrainer(DetectionTrainer):
    """Detection trainer that registers custom modules before model parsing."""

    def __init__(self, *args, **kwargs) -> None:
        register_bubble_modules()
        super().__init__(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        register_bubble_modules()
        cfg = kwargs.get("cfg", args[0] if len(args) > 0 else None)
        weights = kwargs.get("weights", args[1] if len(args) > 1 else None)
        verbose = kwargs.get("verbose", args[2] if len(args) > 2 else True)
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            if supports_bubble_remap(model):
                load_bubble_remapped_weights(
                    model,
                    _remap_source(weights, getattr(self.args, "pretrained", None)),
                    verbose=verbose and RANK == -1,
                )
            else:
                model.load(weights)
        return model


class BubbleNWDDetectionTrainer(BubbleDetectionTrainer):
    """Detection trainer that also enables Bubble NWD loss in DDP workers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enable_bubble_nwd()

    def get_model(self, *args, **kwargs):
        self._enable_bubble_nwd()
        return super().get_model(*args, **kwargs)

    def _enable_bubble_nwd(self) -> None:
        args = getattr(self, "args", None)
        weight = float(os.getenv("BUBBLE_NWD_WEIGHT", getattr(args, "bubble_nwd_weight", 0.4)) or 0.4)
        constant = float(os.getenv("BUBBLE_NWD_CONSTANT", getattr(args, "bubble_nwd_constant", 12.8)) or 12.8)
        enable_nwd_loss(nwd_weight=weight, nwd_constant=constant)
