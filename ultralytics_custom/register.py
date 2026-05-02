"""Runtime registration for Bubble-YOLO modules."""

from __future__ import annotations

from .bubble_modules import (
    CoordGate,
    ECAGate,
    GDFN,
    GLRB,
    MDTA,
    MSLRefine,
    P3CAGate,
    P3LCRefine,
    P3MLCRefine,
    P3SAGate,
    SSBRefine,
    SimAMGate,
    LayerNorm2d,
)


def register_bubble_modules() -> None:
    """Expose custom modules to Ultralytics' YAML parser globals."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    custom = {
        "SSBRefine": SSBRefine,
        "MSLRefine": MSLRefine,
        "P3CAGate": P3CAGate,
        "P3SAGate": P3SAGate,
        "P3LCRefine": P3LCRefine,
        "P3MLCRefine": P3MLCRefine,
        "ECAGate": ECAGate,
        "CoordGate": CoordGate,
        "SimAMGate": SimAMGate,
        "LayerNorm2d": LayerNorm2d,
        "MDTA": MDTA,
        "GDFN": GDFN,
        "GLRB": GLRB,
    }
    for name, cls in custom.items():
        setattr(tasks, name, cls)
        setattr(modules, name, cls)
