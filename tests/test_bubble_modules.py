from __future__ import annotations

import torch

from ultralytics_custom.bubble_loss import bbox_nwd_xywh
from ultralytics_custom.bubble_modules import (
    CoordGate,
    ECAGate,
    GLRB,
    MSLRefine,
    P3CAGate,
    P3LCRefine,
    P3MLCRefine,
    P3SAGate,
    SSBRefine,
    SimAMGate,
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
