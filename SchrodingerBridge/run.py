from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Delegate to src/run.py (the actual training entry). Use runpy with the
    # src/run.py file path so the module is loaded by file location, avoiding
    # the self-import recursion bug where `import run` resolves to this file.
    sys.argv[0] = str(src_dir / "run.py")
    runpy.run_path(str(src_dir / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
