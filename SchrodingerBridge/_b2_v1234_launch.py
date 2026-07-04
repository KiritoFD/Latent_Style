"""Launch B2 V1-V4 batch runner as a detached Windows process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ_ROOT = ROOT
SCRIPT = PROJ_ROOT / "_b2_v1234_batch.py"
LOG = PROJ_ROOT / "_b2_v1234_master.log"

env = dict()
import os
env.update(os.environ)

log_file = open(LOG, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(
    ["python", "-u", str(SCRIPT)],
    cwd=str(PROJ_ROOT),
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
)
log_file.close()
print(f"[launch] detached PID={proc.pid}, log={LOG}", flush=True)
