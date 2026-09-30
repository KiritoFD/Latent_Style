from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_TEST_DIR = Path(r"F:\wikiart_distinct5_samam_512_classview_real\test")
DEFAULT_CACHE_DIR = WORKSPACE / "eval_cache"
DEFAULT_CLIP_CACHE_DIR = DEFAULT_CACHE_DIR / "hf"
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from local_gpu_lock import run_with_local_gpu_lock
from round1_paths import infer_round1_family_id


def _run(cmd: list[str]) -> int:
    print("[watch_local_round1_family_fast_eval] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    proc = subprocess.run(cmd, check=False, cwd=str(SB_ROOT), env=env)
    return int(proc.returncode)


def _run_locked(cmd: list[str], *, owner: str) -> int:
    print("[watch_local_round1_family_fast_eval] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return run_with_local_gpu_lock(cmd, owner=owner, cwd=str(SB_ROOT), env=env)


def _list_remote_epochs(remote_run_dir: str, *, host: str, port: int, wsl_distro: str) -> list[str]:
    remote_py = (
        "from pathlib import Path\n"
        "import sys\n"
        "run = Path(sys.argv[1])\n"
        "print('\\n'.join(x.name for x in sorted(run.glob('epoch_*.pt'))))\n"
    )
    cmd = [
        "ssh",
        "-p",
        str(int(port)),
        host,
        "wsl",
        "-d",
        str(wsl_distro),
        "python3",
        "-",
        str(remote_run_dir),
    ]
    proc = subprocess.run(cmd, input=remote_py, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"failed listing remote epochs under {remote_run_dir}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _summary_curve_row(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis") or {}
    full = analysis.get("all_pairs_overview") or {}
    transfer = analysis.get("style_transfer_ability") or {}
    timings = payload.get("timings_sec") or {}
    return {
        "epoch": summary_path.parent.name,
        "full_clip_style": full.get("clip_style"),
        "full_content_lpips": full.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "wall_total_seconds": timings.get("wall_total"),
        "summary_path": str(summary_path),
    }


def _write_curve_csv(output_root: Path) -> Path | None:
    rows = []
    for summary_path in sorted(output_root.glob("epoch_*/summary.json")):
        rows.append(_summary_curve_row(summary_path))
    if not rows:
        return None
    out_path = output_root / "clip_lpips_curve.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "full_clip_style",
                "full_content_lpips",
                "transfer_clip_style",
                "transfer_content_lpips",
                "wall_total_seconds",
                "summary_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _refresh_sidecars(
    *,
    curve_csv: Path | None,
    config_arg: str,
    local_root: Path,
    output_subdir: str,
    patience: int,
    family_id: str | None,
) -> int:
    if curve_csv is None:
        return 0
    rc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "report_round1_convergence.py"),
            "--curve-csv",
            str(curve_csv),
            "--patience",
            str(int(patience)),
        ]
    )
    if rc != 0:
        return rc
    rc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "refresh_round1_family_bestfew.py"),
            "--config",
            str(config_arg),
            "--fast-local-root",
            str(local_root),
            "--fast-eval-subdir",
            str(output_subdir),
        ]
    )
    if rc != 0:
        return rc
    if family_id:
        rc = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "update_round1_family_status_docs.py"),
                "--family-id",
                str(family_id),
            ]
        )
        if rc != 0:
            return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll remote checkpoints and run local CLIP-S/LPIPS fast eval on every new retained checkpoint.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--clip-hf-cache-dir", type=Path, default=DEFAULT_CLIP_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--output-subdir", default="full_eval_fast_local")
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config((WORKSPACE / Path(args.config)).resolve())
    run_name = str((cfg.get("ablation") or {}).get("name", Path(args.config).stem)).strip() or Path(args.config).stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=Path(args.config).stem)
    remote_run_dir = str((cfg.get("checkpoint") or {}).get("save_dir", "")).strip().replace("./", "/mnt/i/Github/Latent_Style/")
    local_root = Path(args.local_root).resolve()
    local_ckpt_root = local_root / "checkpoints"
    local_eval_root = local_root / str(args.output_subdir)
    local_root.mkdir(parents=True, exist_ok=True)
    local_ckpt_root.mkdir(parents=True, exist_ok=True)
    local_eval_root.mkdir(parents=True, exist_ok=True)

    cycles = 0
    while True:
        refreshed_this_cycle = False
        epochs = _list_remote_epochs(remote_run_dir, host=str(args.host), port=int(args.port), wsl_distro=str(args.wsl_distro))
        for epoch_name in epochs:
            epoch_tag = Path(epoch_name).name
            epoch_stem = Path(epoch_tag).stem
            local_ckpt = local_ckpt_root / epoch_tag
            if not local_ckpt.exists():
                remote_ckpt = f"{remote_run_dir.rstrip('/')}/{epoch_tag}"
                rc = _run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "pull_remote_checkpoint_file.py"),
                        "--host",
                        str(args.host),
                        "--port",
                        str(int(args.port)),
                        "--remote-file",
                        remote_ckpt,
                        "--local-file",
                        str(local_ckpt),
                    ]
                )
                if rc != 0:
                        return rc
            summary_json = local_eval_root / epoch_stem / "summary.json"
            if summary_json.is_file():
                continue
            rc = _run_locked(
                [
                    sys.executable,
                    str(SB_ROOT / "src" / "utils" / "run_evaluation.py"),
                    "--checkpoint",
                    str(local_ckpt),
                    "--output",
                    str(local_eval_root / epoch_stem),
                    "--test_dir",
                    str(Path(args.test_dir).resolve()),
                    "--cache_dir",
                    str(Path(args.cache_dir).resolve()),
                    "--clip_hf_cache_dir",
                    str(Path(args.clip_hf_cache_dir).resolve()),
                    "--batch_size",
                    str(int(args.batch_size)),
                    "--vae_decode_batch_size",
                    str(int(args.vae_decode_batch_size)),
                    "--target_chunk_size",
                    str(int(args.target_chunk_size)),
                    "--eval_only_lpips_clip_style",
                    "--no-save_generated_images",
                    "--no-save_summary_grid",
                ],
                owner=f"watch_local_round1_family_fast_eval:{run_name}",
            )
            if rc != 0:
                return rc
            curve_csv = _write_curve_csv(local_eval_root)
            rc = _refresh_sidecars(
                curve_csv=curve_csv,
                config_arg=str(args.config),
                local_root=local_root,
                output_subdir=str(args.output_subdir),
                patience=int(args.patience),
                family_id=family_id,
            )
            if rc != 0:
                return rc
            refreshed_this_cycle = True

        curve_csv = _write_curve_csv(local_eval_root)
        if curve_csv is not None and not refreshed_this_cycle:
            convergence_json = local_eval_root / "round1_convergence.json"
            bestfew_csv = local_root / f"{str(args.output_subdir)}_bestfew_handoff.csv"
            needs_refresh = (not convergence_json.exists()) or (not bestfew_csv.exists())
            if needs_refresh:
                rc = _refresh_sidecars(
                    curve_csv=curve_csv,
                    config_arg=str(args.config),
                    local_root=local_root,
                    output_subdir=str(args.output_subdir),
                    patience=int(args.patience),
                    family_id=family_id,
                )
                if rc != 0:
                    return rc

        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
