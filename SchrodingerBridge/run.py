from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Import by module name so multiprocessing can re-import in worker processes.
    module = importlib.import_module("run")
    module.main()


if __name__ == "__main__":
    main()
