"""NWD loss utilities and an optional Ultralytics BboxLoss monkey patch."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy boxes to xywh boxes in the same scale."""
    xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2.0
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0.0)
    return torch.cat((xy, wh), dim=-1)


def bbox_nwd_xywh(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute normalized Gaussian Wasserstein distance similarity."""
    pxy = pred_xywh[..., 0:2]
    txy = target_xywh[..., 0:2]
    pwh = pred_xywh[..., 2:4].clamp(min=eps)
    twh = target_xywh[..., 2:4].clamp(min=eps)
    center_dist = ((pxy - txy) ** 2).sum(dim=-1)
    size_dist = ((pwh - twh) ** 2).sum(dim=-1) / 4.0
    wasserstein = torch.sqrt(center_dist + size_dist + eps)
    return torch.exp(-wasserstein / constant)


def bbox_nwd_xyxy(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute NWD similarity for xyxy boxes."""
    return bbox_nwd_xywh(xyxy_to_xywh(pred_xyxy), xyxy_to_xywh(target_xyxy), constant=constant, eps=eps)


def enable_nwd_loss(nwd_weight: float = 0.4, nwd_constant: float = 12.8) -> None:
    """Patch Ultralytics BboxLoss so box loss blends CIoU and NWD.

    The patch is process-local and should be called before creating the YOLO
    trainer. Calling it repeatedly updates the NWD parameters without stacking
    wrappers.
    """
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    if not hasattr(BboxLoss, "_bubble_original_forward"):
        BboxLoss._bubble_original_forward = BboxLoss.forward  # type: ignore[attr-defined]

    BboxLoss._bubble_nwd_weight = float(nwd_weight)  # type: ignore[attr-defined]
    BboxLoss._bubble_nwd_constant = float(nwd_constant)  # type: ignore[attr-defined]

    def forward(
        self: Any,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]
        iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        nwd = bbox_nwd_xyxy(pred_fg, target_fg, constant=self._bubble_nwd_constant).unsqueeze(-1)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
        loss_box = (1.0 - self._bubble_nwd_weight) * loss_iou + self._bubble_nwd_weight * loss_nwd

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_box, loss_dfl

    BboxLoss.forward = forward  # type: ignore[method-assign]


def disable_nwd_loss() -> None:
    """Restore the original Ultralytics BboxLoss forward if patched."""
    from ultralytics.utils.loss import BboxLoss

    original = getattr(BboxLoss, "_bubble_original_forward", None)
    if original is not None:
        BboxLoss.forward = original  # type: ignore[method-assign]
