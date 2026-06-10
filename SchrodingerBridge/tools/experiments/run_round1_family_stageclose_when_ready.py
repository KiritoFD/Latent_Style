from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
BESTFEW_PIPELINE = SCRIPT_DIR / "run_round1_family_bestfew_pipeline.py"
EXTERNAL_VLM_PACKET = SCRIPT_DIR / "run_round1_family_external_vlm_packet.py"
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config  # noqa: E402
from round1_paths import infer_round1_family_id, round1_fast_local_root, round1_localreview_root  # noqa: E402


def _read_converged(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("converged"))


def _run(cmd: list[str]) -> int:
    print("[run_round1_family_stageclose_when_ready] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _default_paths(config_arg: str, *, fast_local_root: str, review_local_root: str) -> tuple[Path, Path, str, str]:
    cfg_path = Path(config_arg)
    if not cfg_path.is_absolute():
        cfg_path = (WORKSPACE / cfg_path).resolve()
    cfg = load_config(cfg_path)
    run_name = str((cfg.get("ablation") or {}).get("name", cfg_path.stem)).strip() or cfg_path.stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=cfg_path.stem)
    if family_id is None:
        family_id = cfg_path.stem
    if fast_local_root:
        fast_root = Path(fast_local_root).resolve()
    else:
        fast_root = round1_fast_local_root(family_id=family_id, run_name=run_name)
    if review_local_root:
        review_root = Path(review_local_root).resolve()
    else:
        review_root = round1_localreview_root(family_id=family_id, run_name=run_name)
    return fast_root, review_root, family_id, run_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for a round-1 fast curve to converge, then run bestfew local review plus external-baseline VLM packet.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-local-root", default="")
    parser.add_argument("--review-local-root", default="")
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_local")
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--wait-poll-seconds", type=int, default=180)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--use-remote-rerun", action="store_true")
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-introstyle", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--introstyle-batch-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-batch-size", type=int, default=4)
    parser.add_argument("--introstyle-ensemble-size", type=int, default=1)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-runs", nargs="+", default=["Seedream_repaired750", "SaMAM_2250"])
    parser.add_argument("--family-label-prefix", required=True)
    parser.add_argument("--family-method", default="LBM")
    parser.add_argument("--vlm-output-dir", default="")
    parser.add_argument("--vlm-epochs", nargs="*", default=[])
    parser.add_argument("--vlm-reason-contains", nargs="*", default=["best_transfer_lpips", "latest"])
    parser.add_argument("--vlm-limit", type=int, default=205)
    parser.add_argument("--vlm-model", default="xopqwen36v35b")
    parser.add_argument("--vlm-timeout", type=int, default=60)
    parser.add_argument("--vlm-sleep-seconds", type=float, default=0.3)
    parser.add_argument("--vlm-resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    fast_root, review_root, family_id, run_name = _default_paths(
        args.config,
        fast_local_root=str(args.fast_local_root),
        review_local_root=str(args.review_local_root),
    )
    fast_eval_subdir = str(args.fast_eval_subdir).strip() or "full_eval_fast_local"
    review_eval_subdir = str(args.review_eval_subdir).strip() or "full_eval_fresh_localreview"
    convergence_json = fast_root / fast_eval_subdir / "round1_convergence.json"

    while not bool(args.skip_wait):
        converged = _read_converged(convergence_json)
        print(
            json.dumps(
                {
                    "family_id": family_id,
                    "run_name": run_name,
                    "convergence_json": str(convergence_json),
                    "converged": converged,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if converged:
            break
        time.sleep(max(1, int(args.wait_poll_seconds)))

    bestfew_cmd = [
        sys.executable,
        str(BESTFEW_PIPELINE),
        "--config",
        str(args.config),
        "--fast-local-root",
        str(fast_root),
        "--fast-eval-subdir",
        fast_eval_subdir,
        "--review-eval-subdir",
        review_eval_subdir,
        "--review-local-root",
        str(review_root),
        "--introstyle-batch-size",
        str(max(1, int(args.introstyle_batch_size))),
        "--introstyle-bank-batch-size",
        str(max(1, int(args.introstyle_bank_batch_size))),
        "--introstyle-ensemble-size",
        str(max(1, int(args.introstyle_ensemble_size))),
    ]
    if bool(args.use_remote_rerun):
        bestfew_cmd.append("--use-remote-rerun")
    if bool(args.skip_rerun):
        bestfew_cmd.append("--skip-rerun")
    if bool(args.skip_pull):
        bestfew_cmd.append("--skip-pull")
    if bool(args.skip_introstyle):
        bestfew_cmd.append("--skip-introstyle")
    if bool(args.skip_dino):
        bestfew_cmd.append("--skip-dino")
    rc = _run(bestfew_cmd)
    if rc != 0:
        return rc

    handoff_csv = review_root / f"{review_eval_subdir}_bestfew_handoff.csv"
    vlm_output_dir = (
        Path(args.vlm_output_dir).resolve()
        if str(args.vlm_output_dir).strip()
        else (review_root.parent / f"{family_id}_external_vlm_stageclose")
    )
    vlm_cmd = [
        sys.executable,
        str(EXTERNAL_VLM_PACKET),
        "--handoff-csv",
        str(handoff_csv),
        "--baseline-manifest",
        str(Path(args.baseline_manifest).resolve()),
        "--baseline-runs",
        *[str(x) for x in args.baseline_runs],
        "--output-dir",
        str(vlm_output_dir),
        "--family-label-prefix",
        str(args.family_label_prefix),
        "--family-method",
        str(args.family_method),
        "--limit",
        str(max(0, int(args.vlm_limit))),
        "--model",
        str(args.vlm_model),
        "--timeout",
        str(max(1, int(args.vlm_timeout))),
        "--sleep-seconds",
        str(float(args.vlm_sleep_seconds)),
    ]
    if args.vlm_epochs:
        vlm_cmd.extend(["--epochs", *[str(x) for x in args.vlm_epochs]])
    if args.vlm_reason_contains:
        vlm_cmd.extend(["--reason-contains", *[str(x) for x in args.vlm_reason_contains]])
    if bool(args.vlm_resume):
        vlm_cmd.append("--resume")
    return _run(vlm_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
