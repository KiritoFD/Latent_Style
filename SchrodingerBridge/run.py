from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    # Delegate to src/run.py (the actual training entry). Use subprocess to
    # run `python src/run.py <args>`, which sets sys.path[0] to src/ so all
    # internal imports resolve correctly. This avoids the self-import
    # recursion bug where `import run` resolves to this file, and avoids
    # runpy path conflicts with the trainer's _assert_active_source_modules
    # check. On Windows os.execv does not reliably inherit the console, so
    # subprocess.run with inherit_handles is used instead.
    target = src_dir / "run.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
