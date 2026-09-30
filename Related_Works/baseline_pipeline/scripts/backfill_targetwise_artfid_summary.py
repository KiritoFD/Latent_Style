from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SB_SRC = WORKSPACE_ROOT / "SchrodingerBridge" / "src"
if str(SB_SRC) not in sys.path:
    sys.path.insert(0, str(SB_SRC))

from utils.targetwise_artfid_summary import write_targetwise_artfid_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill aggregate_targetwise_artfid.json from a retained summary.json matrix."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to the evaluator summary.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to aggregate_targetwise_artfid.json next to summary.json.",
    )
    args = parser.parse_args()

    out_path = write_targetwise_artfid_summary(args.summary_json, args.output)
    if out_path is None:
        print("No ArtFID entries were found in matrix_breakdown; nothing written.")
        return 1
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
