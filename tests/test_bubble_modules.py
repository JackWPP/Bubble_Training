from __future__ import annotations

import torch

from ultralytics_custom.bubble_loss import bbox_nwd_xywh
from ultralytics_custom.bubble_modules import GLRB, SSBRefine


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


def test_nwd_similarity_range_and_identity():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [20.0, 20.0, 10.0, 12.0]])
    same = bbox_nwd_xywh(boxes, boxes)
    shifted = bbox_nwd_xywh(boxes, boxes + 5.0)
    assert torch.allclose(same, torch.ones_like(same), atol=1e-3)
    assert torch.all(shifted < same)
    assert torch.all((shifted >= 0.0) & (shifted <= 1.0))
