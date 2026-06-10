from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SB_ROOT = ROOT.parent
WORKSPACE = SB_ROOT.parent
RUNNER = SB_ROOT / "tools" / "experiments" / "run_vlm_snapshot_compare.py"
BOARD = SB_ROOT / "tools" / "build_vlm_external_baseline_board.py"


def _run(cmd: list[str]) -> int:
    print("[round1_attn_sa_mod_vlm_snapshot205] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    prefix_e08 = ROOT / "round1_attn_sa_mod_vlm_snapshot205_e08_vs_seedream_samam_20260610"
    prefix_e24 = ROOT / "round1_attn_sa_mod_vlm_snapshot205_e24_vs_seedream_samam_20260610"

    commands = [
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(ROOT / "round1_attn_sa_mod_vlm_manifest_e08_vs_seedream_samam_20260610.csv"),
            "--runs",
            "Seedream_repaired750",
            "SaMAM_2250",
            "AttnSA_e08",
            "--output-prefix",
            str(prefix_e08),
            "--limit",
            "205",
            "--resume",
        ],
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(ROOT / "round1_attn_sa_mod_vlm_manifest_e24_vs_seedream_samam_20260610.csv"),
            "--runs",
            "Seedream_repaired750",
            "SaMAM_2250",
            "AttnSA_e24",
            "--output-prefix",
            str(prefix_e24),
            "--limit",
            "205",
            "--resume",
        ],
        [
            sys.executable,
            str(BOARD),
            "--input",
            f"e08={prefix_e08}.method_summary.csv",
            f"e24={prefix_e24}.method_summary.csv",
            "--output-csv",
            str(ROOT / "round1_attn_sa_mod_vlm_snapshot205_board_20260610.csv"),
            "--output-md",
            str(ROOT / "round1_attn_sa_mod_vlm_snapshot205_board_20260610.md"),
        ],
    ]

    for cmd in commands:
        rc = _run(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
