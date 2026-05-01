"""Patch an editable Ultralytics checkout to expose Bubble custom modules.

This is optional. The repository training scripts register modules at runtime,
but patching an editable checkout makes direct `yolo train model=...` CLI usage
work for YAML files that reference SSBRefine or GLRB.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER_START = "# BUBBLE_YOLO_CUSTOM_START"
MARKER_END = "# BUBBLE_YOLO_CUSTOM_END"
IMPORT_BLOCK = f"""{MARKER_START}
from ultralytics.nn.modules.bubble_modules import GDFN, GLRB, MDTA, SSBRefine, LayerNorm2d
{MARKER_END}
"""


def inject_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_START in text:
        before = text.split(MARKER_START)[0].rstrip()
        after = text.split(MARKER_END, 1)[1].lstrip()
        text = f"{before}\n{block}\n{after}"
    else:
        text = f"{text.rstrip()}\n\n{block}\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultralytics-src", type=Path, default=ROOT / "ultralytics_src")
    args = parser.parse_args()

    src = args.ultralytics_src
    package = src / "ultralytics"
    modules_dir = package / "nn" / "modules"
    tasks_py = package / "nn" / "tasks.py"
    init_py = modules_dir / "__init__.py"
    if not tasks_py.exists() or not init_py.exists():
        raise FileNotFoundError(f"Not an Ultralytics source checkout: {src}")

    shutil.copy2(ROOT / "ultralytics_custom" / "bubble_modules.py", modules_dir / "bubble_modules.py")
    inject_block(init_py, IMPORT_BLOCK)
    inject_block(tasks_py, IMPORT_BLOCK)
    print(f"patched {src}")
    print("run: pip install -e ultralytics_src")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
