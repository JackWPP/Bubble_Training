"""Runtime registration for Bubble-YOLO modules."""

from __future__ import annotations

from .bubble_modules import GDFN, GLRB, MDTA, SSBRefine, LayerNorm2d


def register_bubble_modules() -> None:
    """Expose custom modules to Ultralytics' YAML parser globals."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    custom = {
        "SSBRefine": SSBRefine,
        "LayerNorm2d": LayerNorm2d,
        "MDTA": MDTA,
        "GDFN": GDFN,
        "GLRB": GLRB,
    }
    for name, cls in custom.items():
        setattr(tasks, name, cls)
        setattr(modules, name, cls)
