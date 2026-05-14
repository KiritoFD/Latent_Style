from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
ABLATION_ROOT = ROOT / "ablation_destructive_7epoch"


def deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "D0_full_correct_7ep",
        "label": "Full control from corrected config",
        "purpose": "7-epoch control using S-add__K-1_C-0_W-20_Col-0/config.json without model/loss changes.",
        "patch": {},
    },
    {
        "id": "D1_no_terminal_swd",
        "label": "w/o terminal SWD",
        "purpose": "Destructive removal of endpoint style-distribution matching.",
        "patch": {"bridge": {"terminal_swd_weight": 0.0}},
    },
    {
        "id": "D2_no_kinetic",
        "label": "w/o kinetic",
        "purpose": "Destructive removal of flow regularization/content-stability pressure.",
        "patch": {"bridge": {"w_kinetic": 0.0}},
    },
    {
        "id": "D3_no_swd_no_kinetic",
        "label": "w/o SWD and kinetic",
        "purpose": "Strong negative control: remove both style distribution endpoint and trajectory regularization.",
        "patch": {"bridge": {"terminal_swd_weight": 0.0, "w_kinetic": 0.0}},
    },
    {
        "id": "D4_conv_body_no_global_attn",
        "label": "conv body, no global attention",
        "purpose": "Destructive architecture ablation replacing the global-attention body with convolutional blocks.",
        "patch": {"model": {"body_block_type": "conv"}},
    },
    {
        "id": "D5_disable_skip_routing",
        "label": "disable routed skip path",
        "purpose": "Destructive removal of the routed skip pathway that carries clean structure.",
        "patch": {"model": {"skip_routing_mode": "none"}},
    },
    {
        "id": "D6_disable_spatial_prior",
        "label": "disable spatial style prior",
        "purpose": "Destructive removal of the spatial prior used by the style-conditioned model.",
        "patch": {"model": {"ablation_disable_spatial_prior": True}},
    },
    {
        "id": "D7_no_residual_path",
        "label": "no residual path",
        "purpose": "Destructive model ablation disabling the learned residual update path.",
        "patch": {"model": {"ablation_no_residual": True}},
    },
    {
        "id": "D8_strong_color_loss",
        "label": "strong color loss",
        "purpose": "Strong negative control for naive color matching that previously harmed content.",
        "patch": {"bridge": {"w_color": 15.0}},
    },
    {
        "id": "D9_l2_ot_cost",
        "label": "L2 matching cost",
        "purpose": "Replace SWD-based matching cost with global latent L2 cost.",
        "patch": {"bridge": {"ot_cost_mode": "l2"}},
    },
    {
        "id": "D10_micro_hf_swd_trap",
        "label": "micro high-frequency SWD",
        "purpose": "Stress test: force SWD toward tiny high-frequency patches to expose grain/noise tendencies.",
        "patch": {"bridge": {"swd_use_high_freq": True, "swd_patch_sizes": [1, 3], "terminal_swd_weight": 20.0}},
    },
    {
        "id": "D11_single_terminal_step",
        "label": "single terminal step",
        "purpose": "Collapse endpoint matching from four terminal steps to one to test endpoint optimization strength.",
        "patch": {"bridge": {"terminal_num_steps": 1}},
    },
]


def load_base_config() -> dict[str, Any]:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def prepare_config(
    variant: dict[str, Any],
    *,
    batch_size: int | None,
    num_epochs: int,
    eval_batch_size: int | None,
    save_every_epoch: bool,
) -> Path:
    cfg = copy.deepcopy(load_base_config())
    deep_update(cfg, variant["patch"])
    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = int(num_epochs)
    cfg["training"]["save_interval"] = 1 if save_every_epoch else int(num_epochs)
    cfg["training"]["use_tqdm"] = True
    if batch_size is not None:
        cfg["training"]["batch_size"] = int(batch_size)
    if eval_batch_size is not None:
        cfg["training"]["full_eval_batch_size"] = int(eval_batch_size)
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_dir"] = str((ABLATION_ROOT / variant["id"]).resolve())
    cfg.setdefault("ablation", {})
    cfg["ablation"].update(
        {
            "id": variant["id"],
            "label": variant["label"],
            "purpose": variant["purpose"],
            "base_config": str(BASE_CONFIG.resolve()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    config_dir = ABLATION_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{variant['id']}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    with (log_path.open("a", encoding="utf-8") if log_path else subprocess.DEVNULL) as log:
        if log_path:
            log.write("\n$ " + " ".join(cmd) + "\n")
            log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def write_registry(rows: list[dict[str, Any]]) -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ABLATION_ROOT / "destructive_ablation_7epoch_registry.csv"
    md_path = ABLATION_ROOT / "destructive_ablation_7epoch_registry.md"
    keys = [
        "id",
        "label",
        "purpose",
        "config",
        "save_dir",
        "status",
        "train_sec",
        "eval_status",
        "eval_sec",
        "checkpoint",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in rows])
    lines = [
        "# Destructive 7-Epoch Ablation Registry",
        "",
        f"Base config: `{BASE_CONFIG}`",
        "",
        "| ID | Label | Train | Train sec | Eval | Eval sec | Purpose |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('id', '')}` | {row.get('label', '')} | {row.get('status', '')} | "
            f"{row.get('train_sec', '')} | {row.get('eval_status', '')} | {row.get('eval_sec', '')} | "
            f"{row.get('purpose', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_variants(names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return VARIANTS
    wanted = set(names)
    out = [v for v in VARIANTS if v["id"] in wanted]
    missing = sorted(wanted - {v["id"] for v in out})
    if missing:
        raise SystemExit(f"Unknown variant id(s): {', '.join(missing)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run destructive 7-epoch SchrodingerBridge ablations serially.")
    parser.add_argument("--only", nargs="*", default=[], help="Optional variant IDs to run.")
    parser.add_argument("--batch_size", type=int, default=0, help="0 keeps the base config batch_size.")
    parser.add_argument("--eval_batch_size", type=int, default=0, help="0 keeps the base config full_eval_batch_size.")
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Write configs/registry only; do not train.")
    args = parser.parse_args()

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in selected_variants(args.only):
        cfg_path = prepare_config(
            variant,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            num_epochs=args.num_epochs,
            eval_batch_size=args.eval_batch_size if args.eval_batch_size > 0 else None,
            save_every_epoch=args.save_every_epoch,
        )
        save_dir = ABLATION_ROOT / variant["id"]
        ckpt = save_dir / f"epoch_{args.num_epochs:04d}.pt"
        log_path = save_dir / "run_ablation_7epoch.log"
        row: dict[str, Any] = {
            "id": variant["id"],
            "label": variant["label"],
            "purpose": variant["purpose"],
            "config": str(cfg_path),
            "save_dir": str(save_dir),
            "checkpoint": str(ckpt),
            "status": "pending",
            "train_sec": "",
            "eval_status": "pending",
            "eval_sec": "",
        }
        print(f"\n== {variant['id']} | {variant['label']} ==")
        print(f"config: {cfg_path}")
        if args.dry_run or args.prepare_only:
            row["status"] = "prepared" if args.prepare_only else "dry_run"
            row["eval_status"] = "not_started"
            rows.append(row)
            write_registry(rows)
            continue
        if args.skip_existing and ckpt.exists():
            row["status"] = "skipped_existing"
        else:
            t0 = time.perf_counter()
            rc = run_cmd([sys.executable, "run.py", "--config", str(cfg_path)], cwd=ROOT, log_path=log_path)
            row["train_sec"] = f"{time.perf_counter() - t0:.3f}"
            row["status"] = "ok" if rc == 0 and ckpt.exists() else f"failed:{rc}"
        if not args.no_eval and ckpt.exists():
            t0 = time.perf_counter()
            out_dir = save_dir / "full_eval" / f"epoch_{args.num_epochs:04d}"
            rc = run_cmd(
                [
                    sys.executable,
                    "run_evaluation.py",
                    str(ckpt),
                    "--output",
                    str(out_dir),
                    "--batch_size",
                    str(args.eval_batch_size if args.eval_batch_size > 0 else load_base_config()["training"].get("full_eval_batch_size", 20)),
                    "--force_regen",
                ],
                cwd=ROOT,
                log_path=log_path,
            )
            row["eval_sec"] = f"{time.perf_counter() - t0:.3f}"
            row["eval_status"] = "ok" if rc == 0 and (out_dir / "summary.json").exists() else f"failed:{rc}"
        elif args.no_eval:
            row["eval_status"] = "not_requested"
        rows.append(row)
        write_registry(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
