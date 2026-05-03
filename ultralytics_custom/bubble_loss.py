"""NWD loss and WIoU v3 utilities with an optional Ultralytics BboxLoss monkey patch."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# WIoU v3 — Wise-IoU with dynamic non-monotonic focusing
# Reference: https://arxiv.org/abs/2301.10051
# ---------------------------------------------------------------------------


class WIoUv3Loss:
    """EMA-tracked WIoU v3 loss computer.

    Maintains a running mean of L_IoU across batches so that the outlier degree
    ``beta`` adapts as training progresses.  Safe to call repeatedly from
    different forward passes.

    Parameters
    ----------
    alpha : float
        Shape parameter of the non-monotonic focusing curve (default 1.9).
    delta : float
        Peak position of the curve.  When ``beta == delta`` the focusing
        coefficient ``r == 1`` (default 3.0).
    momentum : float or None
        EMA momentum for the IoU-mean tracker.  ``None`` defaults to a sensible
        value for 30-epoch training (~0.01).
    """

    def __init__(
        self,
        alpha: float = 1.9,
        delta: float = 3.0,
        momentum: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum if momentum is not None else 0.01
        self.iou_mean: float = 1.0  # EMA of per-batch mean L_IoU

    def compute(
        self,
        pred_xyxy: torch.Tensor,
        target_xyxy: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """Return WIoU v3 loss for each box in *pred_xyxy*.

        Boxes are in ``(x1, y1, x2, y2)`` format.  The returned tensor has
        shape ``[N]`` where 0 means perfect alignment.
        """
        # --- geometry -------------------------------------------------------
        px1, py1, px2, py2 = pred_xyxy.chunk(4, -1)
        tx1, ty1, tx2, ty2 = target_xyxy.chunk(4, -1)

        pw = (px2 - px1).clamp(min=eps)
        ph = (py2 - py1).clamp(min=eps)
        tw = (tx2 - tx1).clamp(min=eps)
        th = (ty2 - ty1).clamp(min=eps)

        # Intersection / union / IoU
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)
        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        union = pw * ph + tw * th - inter + eps
        iou = inter / union
        liou = 1.0 - iou.squeeze(-1)  # [N]

        # Smallest enclosing box
        cx1 = torch.min(px1, tx1)
        cy1 = torch.min(py1, ty1)
        cx2 = torch.max(px2, tx2)
        cy2 = torch.max(py2, ty2)
        cw = (cx2 - cx1).clamp(min=eps)
        ch = (cy2 - cy1).clamp(min=eps)

        # Centre distance
        pcx = (px1 + px2) * 0.5
        pcy = (py1 + py2) * 0.5
        tcx = (tx1 + tx2) * 0.5
        tcy = (ty1 + ty2) * 0.5
        centre_dist_sq = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2).squeeze(-1)

        # Diagonal of enclosing box (detached — see §3.1 of the paper)
        diag_sq = (cw**2 + ch**2).squeeze(-1).detach()

        # --- distance attention R_WIoU -------------------------------------
        r_wiou = torch.exp(centre_dist_sq / diag_sq.clamp(min=eps))

        # --- WIoU v1 (attention-based) -------------------------------------
        l_wiou_v1 = r_wiou * liou  # [N]

        # --- dynamic non-monotonic focusing (v3) ---------------------------
        with torch.no_grad():
            batch_mean = liou.detach().mean().item()
            self.iou_mean = (1.0 - self.momentum) * self.iou_mean + self.momentum * batch_mean
            mean = max(self.iou_mean, eps)
            beta = liou.detach() / mean  # outlier degree
            r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))

        return (r.detach() * l_wiou_v1).clamp(min=0.0)  # [N]


def bbox_wiou_v3(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    alpha: float = 1.9,
    delta: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Stateless WIoU v3 loss for a single batch.

    Convenience wrapper that creates a temporary ``WIoUv3Loss``.  For training
    you should use the class so that the EMA persists across batches.
    """
    return WIoUv3Loss(alpha=alpha, delta=delta).compute(pred_xyxy, target_xyxy, eps=eps)


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


def enable_nwd_loss(
    nwd_weight: float = 0.4,
    nwd_constant: float = 12.8,
    iou_type: str = "CIoU",
) -> None:
    """Patch Ultralytics BboxLoss so box loss blends an IoU variant and NWD.

    The patch is process-local and should be called before creating the YOLO
    trainer. Calling it repeatedly updates the NWD parameters without stacking
    wrappers.

    Parameters
    ----------
    nwd_weight : float
        Weight of NWD in the blended loss (0‒1).
    nwd_constant : float
        NWD constant ``C`` in ``exp(-Wasserstein / C)``.
    iou_type : str
        ``"CIoU"`` (default) or ``"WIoU"``.
    """
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    if not hasattr(BboxLoss, "_bubble_original_forward"):
        BboxLoss._bubble_original_forward = BboxLoss.forward  # type: ignore[attr-defined]

    BboxLoss._bubble_nwd_weight = float(nwd_weight)  # type: ignore[attr-defined]
    BboxLoss._bubble_nwd_constant = float(nwd_constant)  # type: ignore[attr-defined]
    BboxLoss._bubble_iou_type = iou_type  # type: ignore[attr-defined]

    _wiou = WIoUv3Loss() if iou_type == "WIoU" else None
    BboxLoss._bubble_wiou = _wiou  # type: ignore[attr-defined]

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

        if getattr(self, "_bubble_iou_type", "CIoU") == "WIoU":
            wiou = self._bubble_wiou
            loss_iou = (wiou.compute(pred_fg, target_fg) * weight.squeeze(-1)).sum() / target_scores_sum
        else:
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
