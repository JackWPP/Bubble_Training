from __future__ import annotations

import torch

from ultralytics_custom.bubble_loss import bbox_nwd_xywh, disable_nwd_loss, enable_nwd_loss
from ultralytics_custom.bubble_modules import (
    ChannelWeightedConcat,
    CoordGate,
    ECAGate,
    GLRB,
    LCRefine,
    MSLRefine,
    P3CAGate,
    P3LCRefine,
    P3MLCRefine,
    P3SAGate,
    SSBRefine,
    SimAMGate,
    WeightedConcat,
)


def test_ssb_refine_shape_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = SSBRefine(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_glrb_shape_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = GLRB(128, num_heads=4)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_msl_refine_shape_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = MSLRefine(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_p3_ca_gate_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = P3CAGate(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_p3_sa_gate_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = P3SAGate(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_p3_lc_refine_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = P3LCRefine(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_lc_refine_shape_identity_and_backward():
    x = torch.randn(2, 256, 10, 10, requires_grad=True)
    module = LCRefine(256)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_weighted_concat_identity_and_backward():
    x1 = torch.randn(2, 256, 20, 20, requires_grad=True)
    x2 = torch.randn(2, 256, 20, 20, requires_grad=True)
    module = WeightedConcat([256, 256])
    y = module(torch.cat((x1, x2), dim=1))
    expected = torch.cat((x1, x2), dim=1)
    assert y.shape == expected.shape
    assert torch.allclose(y, expected, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x1.grad is not None
    assert x2.grad is not None
    assert torch.isfinite(x1.grad).all()
    assert torch.isfinite(x2.grad).all()


def test_channel_weighted_concat_identity_and_backward():
    x1 = torch.randn(2, 128, 20, 20, requires_grad=True)
    x2 = torch.randn(2, 256, 20, 20, requires_grad=True)
    module = ChannelWeightedConcat([128, 256])
    y = module(torch.cat((x1, x2), dim=1))
    expected = torch.cat((x1, x2), dim=1)
    assert y.shape == expected.shape
    assert torch.allclose(y, expected, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x1.grad is not None
    assert x2.grad is not None
    assert torch.isfinite(x1.grad).all()
    assert torch.isfinite(x2.grad).all()


def test_p3_mlc_refine_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = P3MLCRefine(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_eca_gate_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = ECAGate(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_coord_gate_shape_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = CoordGate(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_simam_gate_shape_near_identity_and_backward():
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    module = SimAMGate(128)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-2)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_nwd_similarity_range_and_identity():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [20.0, 20.0, 10.0, 12.0]])
    same = bbox_nwd_xywh(boxes, boxes)
    shifted = bbox_nwd_xywh(boxes, boxes + 5.0)
    assert torch.allclose(same, torch.ones_like(same), atol=1e-3)
    assert torch.all(shifted < same)
    assert torch.all((shifted >= 0.0) & (shifted <= 1.0))


def test_nwd_loss_patch_matches_current_ultralytics_signature():
    from ultralytics.utils.loss import BboxLoss

    class FakeDFL(torch.nn.Module):
        reg_max = 4

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return pred.mean(dim=1, keepdim=True) * 0.0 + target.mean(dim=1, keepdim=True) * 0.0

    try:
        enable_nwd_loss(nwd_weight=0.2, nwd_constant=12.8)
        loss = BboxLoss(reg_max=4)
        loss.dfl_loss = FakeDFL()
        pred_dist = torch.randn(1, 3, 4, requires_grad=True)
        pred_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 12.0, 12.0], [4.0, 4.0, 14.0, 14.0]]])
        target_bboxes = pred_bboxes.clone()
        anchor_points = torch.zeros(3, 2)
        target_scores = torch.ones(1, 3, 1)
        fg_mask = torch.tensor([[True, True, False]])
        loss_box, loss_dfl = loss(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            torch.tensor(2.0),
            fg_mask,
        )
        assert torch.isfinite(loss_box)
        assert torch.isfinite(loss_dfl)
    finally:
        disable_nwd_loss()
