"""Ultralytics trainers that keep Bubble extensions visible in DDP workers."""

from __future__ import annotations

import os

from ultralytics.models.yolo.detect.train import DetectionTrainer

from . import register_bubble_modules
from .bubble_loss import enable_nwd_loss


class BubbleDetectionTrainer(DetectionTrainer):
    """Detection trainer that registers custom modules before model parsing."""

    def __init__(self, *args, **kwargs) -> None:
        register_bubble_modules()
        super().__init__(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        register_bubble_modules()
        return super().get_model(*args, **kwargs)


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
