from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_BASE_CONFIG = ROOT / "exp" / "diffeomorphic_tangent_sweep" / "t00_ws0p03_g6_nl0" / "config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "orthogonal_budget36"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "orthogonal_budget36"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    model: dict[str, Any] = field(default_factory=dict)
    bridge: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _tangent_base(*, zero_init: bool, normal_leak: float) -> dict[str, Any]:
    return {
        "use_diffeomorphic_stroke": True,
        "zero_init_output_head": bool(zero_init),
        "diffeomorphic_head_mode": "standard",
        "diffeomorphic_color_strength": 0.85,
        "diffeomorphic_warp_strength": 0.03,
        "diffeomorphic_texture_gate_strength": 6.0,
        "diffeomorphic_normal_leak": float(normal_leak),
        "diffeomorphic_color_lowpass_kernel": 1,
        "diffeomorphic_color_edge_gamma": 0.0,
        "diffeomorphic_amp_strength": 0.5,
        "diffeomorphic_factorized_enable_color": True,
        "diffeomorphic_factorized_enable_amp": True,
        "diffeomorphic_joint_bilateral_kernel": 1,
        "diffeomorphic_joint_bilateral_range_sigma": 0.5,
        "diffeomorphic_divergence_free_warp": False,
        "diffeomorphic_metric_mask_gamma": 0.0,
        "diffeomorphic_metric_mask_smooth_kernel": 3,
        "diffeomorphic_metric_mask_use_z0": False,
        "latent_canvas_strength": 0.0,
        "latent_canvas_edge_gamma": 4.0,
        "latent_canvas_highpass_kernel": 5,
        "pre_integrate_moment_match": False,
        "pre_integrate_moment_blend": 1.0,
        "output_moment_match": False,
        "output_moment_match_train_only": False,
        "dynamic_style_operator_head": False,
        "semantic_self_topology_gate": False,
    }


def _t00(zero_init: bool = False) -> dict[str, Any]:
    return _tangent_base(zero_init=zero_init, normal_leak=0.0)


def _t01(zero_init: bool = False) -> dict[str, Any]:
    return _tangent_base(zero_init=zero_init, normal_leak=0.05)


def _amp_from(base: dict[str, Any], *, color: bool = True, amp_strength: float = 0.5) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "diffeomorphic_head_mode": "factorized_amp",
            "diffeomorphic_color_lowpass_kernel": 5,
            "diffeomorphic_amp_strength": float(amp_strength),
            "diffeomorphic_factorized_enable_color": bool(color),
            "diffeomorphic_factorized_enable_amp": True,
        }
    )
    return payload


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        name: str,
        family: str,
        *,
        model: dict[str, Any] | None = None,
        bridge: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        idx = len(candidates)
        candidates.append(
            Candidate(
                name=f"{idx:02d}_{name}",
                family=family,
                model=model or _t00(),
                bridge=bridge or {},
                data=data or {},
                training=training or {},
                notes=notes,
            )
        )

    zt00 = _t00(zero_init=True)
    zt01 = _t01(zero_init=True)

    # Anchors and long-tail controls.
    add("ctrl_t00", "anchors", model=_t00(), notes="Canonical t00 short-run control.")
    add("ctrl_t01", "anchors", model=_t01(), notes="Canonical t01 short-run control.")
    add("ctrl_t00_zero", "anchors", model=zt00, notes="t00 plus zero-init short-run control.")
    add("ctrl_t01_zero", "anchors", model=zt01, notes="t01 plus zero-init short-run control.")
    add(
        "tail_t00_zero_12ep",
        "anchors",
        model=zt00,
        training={"num_epochs": 12, "eval_epochs": [8, 10, 12]},
        notes="Long-tail test: does t00+zero-init recover style by epoch 12?",
    )
    add(
        "tail_t01_zero_12ep",
        "anchors",
        model=zt01,
        training={"num_epochs": 12, "eval_epochs": [8, 10, 12]},
        notes="Long-tail test: does t01+zero-init recover style by epoch 12?",
    )

    # Family A: zero-init local retuning.
    add("retune_t00_zero_c090", "retune_local_geometry", model={**zt00, "diffeomorphic_color_strength": 0.90})
    add("retune_t00_zero_ws005", "retune_local_geometry", model={**zt00, "diffeomorphic_warp_strength": 0.05})
    add("retune_t00_zero_nl005", "retune_local_geometry", model={**zt00, "diffeomorphic_normal_leak": 0.05})
    add("retune_t01_zero_ws005", "retune_local_geometry", model={**zt01, "diffeomorphic_warp_strength": 0.05})

    # Family B: stochastic interpolant / style-noise bridge.
    add("bridge_t00_zero_hf003", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.03, "bridge_noise_mode": "style_highfreq", "bridge_style_noise_kernel": 5})
    add("bridge_t00_zero_hf005", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.05, "bridge_noise_mode": "style_highfreq", "bridge_style_noise_kernel": 5})
    add("bridge_t00_zero_hf007", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.07, "bridge_noise_mode": "style_highfreq", "bridge_style_noise_kernel": 5})
    add("bridge_t00_zero_flat003", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.03, "bridge_noise_mode": "style_highfreq_flat", "bridge_style_noise_kernel": 5, "bridge_style_noise_flat_gamma": 4.0})
    add("bridge_t00_zero_flat005", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.05, "bridge_noise_mode": "style_highfreq_flat", "bridge_style_noise_kernel": 5, "bridge_style_noise_flat_gamma": 4.0})
    add("bridge_t00_zero_flat007", "style_noise_bridge", model=zt00, bridge={"bridge_sigma": 0.07, "bridge_noise_mode": "style_highfreq_flat", "bridge_style_noise_kernel": 5, "bridge_style_noise_flat_gamma": 4.0})
    add("bridge_t01_zero_hf005", "style_noise_bridge", model=zt01, bridge={"bridge_sigma": 0.05, "bridge_noise_mode": "style_highfreq", "bridge_style_noise_kernel": 5})
    add("bridge_t01_zero_flat005", "style_noise_bridge", model=zt01, bridge={"bridge_sigma": 0.05, "bridge_noise_mode": "style_highfreq_flat", "bridge_style_noise_kernel": 5, "bridge_style_noise_flat_gamma": 4.0})

    # Family C: cheap semantic coupling proxies inside latent space.
    add("couple_t00_zero_low5", "coupling_proxy", model=zt00, bridge={"coupling_feature_mode": "lowfreq", "coupling_lowfreq_kernel": 5})
    add("couple_t00_zero_low9", "coupling_proxy", model=zt00, bridge={"coupling_feature_mode": "lowfreq", "coupling_lowfreq_kernel": 9})
    add("couple_t00_zero_led5", "coupling_proxy", model=zt00, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 5, "coupling_edge_weight": 0.25})
    add("couple_t00_zero_led9", "coupling_proxy", model=zt00, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 9, "coupling_edge_weight": 0.25})
    add("couple_t00_zero_led9h", "coupling_proxy", model=zt00, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 9, "coupling_edge_weight": 0.5})
    add("couple_t01_zero_low9", "coupling_proxy", model=zt01, bridge={"coupling_feature_mode": "lowfreq", "coupling_lowfreq_kernel": 9})
    add("couple_t01_zero_led9", "coupling_proxy", model=zt01, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 9, "coupling_edge_weight": 0.25})
    add("couple_t01_zero_led9h", "coupling_proxy", model=zt01, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 9, "coupling_edge_weight": 0.5})

    # Family D: offline DINO-oracle pairing.
    pairing_top8 = {"pairing_cache_path": "eval_cache/offline_pairing/dinov2_small_train_pairing_top8.pt", "pairing_cache_topk": 8}
    pairing_top4 = {"pairing_cache_path": "eval_cache/offline_pairing/dinov2_small_train_pairing_top4.pt", "pairing_cache_topk": 4}
    add("dino_t00_zero_top8", "offline_dino_pairing", model=zt00, data=pairing_top8)
    add("dino_t00_zero_top4", "offline_dino_pairing", model=zt00, data=pairing_top4)
    add("dino_t00_zero_top8_led", "offline_dino_pairing", model=zt00, data=pairing_top8, bridge={"coupling_feature_mode": "lowfreq_edge", "coupling_lowfreq_kernel": 9, "coupling_edge_weight": 0.25})
    add("dino_t01_zero_top8", "offline_dino_pairing", model=zt01, data=pairing_top8)

    # Family E: spectral-orthogonal SWD.
    add("spec_t00_zero_bal", "spectral_swd", model=zt00, bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 1.0, "spectral_swd_high_weight": 1.0, "spectral_swd_low_kernel": 5})
    add("spec_t00_zero_low2", "spectral_swd", model=zt00, bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 2.0, "spectral_swd_high_weight": 0.7, "spectral_swd_low_kernel": 7})
    add("spec_t01_zero_bal", "spectral_swd", model=zt01, bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 1.0, "spectral_swd_high_weight": 1.0, "spectral_swd_low_kernel": 5})

    # Family F: explicit head budget.
    add("tax_t00_zero_soft", "head_tax", model=zt00, bridge={"w_head_color_tv": 0.02, "w_head_color_energy": 0.002})
    add("tax_t00_zero_std", "head_tax", model=zt00, bridge={"w_head_color_tv": 0.03, "w_head_color_energy": 0.003})
    add("tax_t00_zero_curl", "head_tax", model=zt00, bridge={"w_head_color_tv": 0.03, "w_head_color_energy": 0.003, "w_warp_curl_reward": 0.002})

    assert len(candidates) == 36, f"expected 36 candidates, got {len(candidates)}"

    # Execution order is intentionally decoupled from candidate naming so we can
    # reprioritize unfinished runs without invalidating existing run directories.
    priority = {
        # Pairing-first block: offline DINO pairing before latent proxy coupling.
        "26_dino_t00_zero_top8": 0,
        "27_dino_t00_zero_top4": 1,
        "28_dino_t00_zero_top8_led": 2,
        "29_dino_t01_zero_top8": 3,
        "18_couple_t00_zero_low5": 4,
        "19_couple_t00_zero_low9": 5,
        "20_couple_t00_zero_led5": 6,
        "21_couple_t00_zero_led9": 7,
        "22_couple_t00_zero_led9h": 8,
        "23_couple_t01_zero_low9": 9,
        "24_couple_t01_zero_led9": 10,
        "25_couple_t01_zero_led9h": 11,
        # Remaining bridge variants.
        "14_bridge_t00_zero_flat005": 12,
        "15_bridge_t00_zero_flat007": 13,
        "16_bridge_t01_zero_hf005": 14,
        "17_bridge_t01_zero_flat005": 15,
        # Loss and head-budget families last.
        "30_spec_t00_zero_bal": 16,
        "31_spec_t00_zero_low2": 17,
        "32_spec_t01_zero_bal": 18,
        "33_tax_t00_zero_soft": 19,
        "34_tax_t00_zero_std": 20,
        "35_tax_t00_zero_curl": 21,
    }
    return sorted(candidates, key=lambda c: (priority.get(c.name, -1), c.name))


def _config_payload(candidate: Candidate, *, output_root: Path, default_train_epochs: int) -> dict[str, Any]:
    payload = _load_json(DEFAULT_BASE_CONFIG)
    payload["model"].update(candidate.model)
    payload["bridge"].update(candidate.bridge)
    payload["data"].update(candidate.data)
    payload["training"]["seed"] = 42
    payload["training"]["batch_size"] = int(os.environ.get("LANCET_BATCH_SIZE", "160"))
    payload["training"]["full_eval_batch_size"] = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", "8"))
    payload["training"]["num_epochs"] = int(candidate.training.get("num_epochs", default_train_epochs))
    payload["training"]["num_workers"] = 0
    payload["training"]["persistent_workers"] = False
    payload["training"]["save_interval"] = 1
    payload["training"]["resume_checkpoint"] = ""
    payload["checkpoint"]["save_dir"] = "./" + (output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    payload["ablation"]["name"] = candidate.name
    payload["ablation"]["stage"] = "orthogonal_budget36"
    payload["ablation"]["axis"] = candidate.family
    payload["ablation"]["notes"] = candidate.notes or "Orthogonal budget-36 run."
    return payload


def _load_summary_metrics(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {"clip_style_all": None, "content_lpips_all": None, "clip_content_all": None, "clip_dir_all": None}
    payload = _load_json(summary_path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style_all": float(overview.get("clip_style")) if overview.get("clip_style") is not None else None,
        "content_lpips_all": float(overview.get("content_lpips")) if overview.get("content_lpips") is not None else None,
        "clip_content_all": float(overview.get("clip_content")) if overview.get("clip_content") is not None else None,
        "clip_dir_all": float(overview.get("clip_dir")) if overview.get("clip_dir") is not None else None,
    }


def _score(row: dict[str, Any]) -> float:
    style = float(row.get("clip_style_all") or 0.0)
    content = float(row.get("clip_content_all") or 0.0)
    lpips = float(row.get("content_lpips_all") or 1.0)
    return style + 0.35 * content - 0.25 * lpips


def _eval_one(ckpt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_batch_size = os.environ.get("LANCET_EVAL_BATCH_SIZE", "8")
    _run(
        [
            sys.executable,
            "src/utils/run_evaluation.py",
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(out_dir),
            "--batch_size",
            str(eval_batch_size),
            "--eval_lpips_chunk_size",
            "2",
        ],
        cwd=ROOT,
    )
    summary = out_dir / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"missing eval summary: {summary}")
    return summary


def _write_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "name",
        "family",
        "score",
        "best_epoch",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "clip_dir_all",
        "status",
        "run_dir",
    ]
    ranked = sorted(rows, key=_score, reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            payload = dict(row)
            payload["score"] = _score(payload)
            writer.writerow({key: payload.get(key) for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Orthogonal 36-run screening budget from t00/t01 with zero-init-aware long-tail controls.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--train-epochs", type=int, default=6)
    parser.add_argument("--eval-epochs", type=str, default="4,6")
    parser.add_argument("--max-total", type=int, default=36)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    candidates = build_candidates()[: max(1, int(args.max_total))]
    default_eval_epochs = tuple(int(x) for x in args.eval_epochs.split(",") if x.strip())
    frontier_path = args.output_root / "orthogonal_budget36_frontier.csv"
    ledger_path = args.output_root / "orthogonal_budget36_ledger.jsonl"
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        config_path = args.config_root / f"{candidate.name}.json"
        run_dir = args.output_root / candidate.name
        payload = _config_payload(candidate, output_root=args.output_root, default_train_epochs=args.train_epochs)
        _write_json(config_path, payload)

        train_epochs = int(candidate.training.get("num_epochs", args.train_epochs))
        eval_epochs = tuple(int(x) for x in candidate.training.get("eval_epochs", list(default_eval_epochs)))

        if args.dry_run:
            print(f"[dry-run] {candidate.name} epochs={train_epochs} eval={list(eval_epochs)} -> {config_path}", flush=True)
            continue

        status = "ok"
        eval_rows: list[dict[str, Any]] = []
        try:
            final_ckpt = run_dir / f"epoch_{train_epochs:04d}.pt"
            if args.force_train or not final_ckpt.exists():
                _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
            for epoch in eval_epochs:
                epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
                if not epoch_ckpt.exists():
                    continue
                eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
                summary = eval_dir / "summary.json"
                if args.force_eval or not summary.exists():
                    summary = _eval_one(epoch_ckpt, eval_dir)
                metrics = _load_summary_metrics(summary)
                eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})
        except Exception as exc:
            status = f"failed: {exc}"

        best = max(eval_rows, key=_score, default={})
        row = {
            "name": candidate.name,
            "family": candidate.family,
            "best_epoch": best.get("epoch"),
            "clip_style_all": best.get("clip_style_all"),
            "content_lpips_all": best.get("content_lpips_all"),
            "clip_content_all": best.get("clip_content_all"),
            "clip_dir_all": best.get("clip_dir_all"),
            "status": status,
            "run_dir": run_dir.as_posix(),
        }
        rows.append(row)
        _append_jsonl(
            ledger_path,
            {
                "candidate": asdict(candidate),
                "train_epochs": train_epochs,
                "eval_epochs": list(eval_epochs),
                "result": row,
                "eval_rows": eval_rows,
            },
        )
        _write_frontier(rows, frontier_path)
        print(f"[result] {candidate.name} status={status} best={best}", flush=True)

    if args.dry_run:
        print(f"[dry-run] wrote {len(candidates)} configs under {args.config_root}", flush=True)
    else:
        print(f"[done] {frontier_path}", flush=True)
        print(f"[done] {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
