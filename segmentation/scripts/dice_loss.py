"""Dice+BCE hybrid mask loss for YOLO-seg via monkey-patch.

Usage:
    import dice_loss
    dice_loss.enable_dice_loss(dice_weight=0.3)

This replaces v8SegmentationLoss.single_mask_loss with a hybrid
BCE + Dice Loss that better penalizes boundary errors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss

_original_single_mask_loss = v8SegmentationLoss.single_mask_loss
_dice_weight = 0.3


def _patched_single_mask_loss(
    gt_mask: torch.Tensor,
    pred: torch.Tensor,
    proto: torch.Tensor,
    xyxy: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    """Hybrid BCE + Dice mask loss. Dice directly optimizes IoU overlap."""
    pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
    bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

    # Dice loss: soft version, directly optimizes overlap
    pred_prob = pred_mask.sigmoid()
    intersection = (pred_prob * gt_mask).sum(dim=(1, 2))
    union = pred_prob.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)  # smooth
    dice_loss = 1.0 - dice  # (N,) per-object

    from ultralytics.utils.loss import crop_mask
    bce_cropped = crop_mask(bce, xyxy).mean(dim=(1, 2)) / area  # (N,)

    return ((1.0 - _dice_weight) * bce_cropped + _dice_weight * dice_loss).sum()


def enable_dice_loss(dice_weight: float = 0.3) -> None:
    """Enable hybrid Dice+BCE mask loss with given Dice weight."""
    global _dice_weight
    _dice_weight = dice_weight
    v8SegmentationLoss.single_mask_loss = staticmethod(_patched_single_mask_loss)
    print(f"Dice+BCE hybrid loss enabled: dice_weight={dice_weight}")


def disable_dice_loss() -> None:
    """Restore original BCE-only mask loss."""
    v8SegmentationLoss.single_mask_loss = _original_single_mask_loss
    print("Dice loss disabled, original BCE restored.")
