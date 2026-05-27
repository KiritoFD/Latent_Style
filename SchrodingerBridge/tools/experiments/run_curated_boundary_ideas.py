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

from config_schema import load_config  # noqa: E402


DEFAULT_BASE_CONFIG = ROOT / "configs" / "diffeomorphic_stroke_tangent_local.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "curated_boundary_ideas"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "curated_boundary_ideas"


@dataclass(frozen=True)
class IdeaCandidate:
    name: str
    idea: str
    model: dict[str, Any] = field(default_factory=dict)
    bridge: dict[str, Any] = field(default_factory=dict)
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


def _t00() -> dict[str, Any]:
    return {
        "use_diffeomorphic_stroke": True,
        "diffeomorphic_head_mode": "standard",
        "diffeomorphic_color_strength": 0.85,
        "diffeomorphic_warp_strength": 0.03,
        "diffeomorphic_texture_gate_strength": 6.0,
        "diffeomorphic_normal_leak": 0.0,
        "diffeomorphic_color_lowpass_kernel": 1,
        "diffeomorphic_color_edge_gamma": 0.0,
        "semantic_self_topology_gate": False,
    }


def _t01() -> dict[str, Any]:
    payload = _t00()
    payload["diffeomorphic_normal_leak"] = 0.05
    return payload


def _amp(color: bool = True, amp_strength: float = 0.5) -> dict[str, Any]:
    payload = _t00()
    payload.update(
        {
            "diffeomorphic_head_mode": "factorized_amp",
            "diffeomorphic_color_lowpass_kernel": 5,
            "diffeomorphic_amp_strength": amp_strength,
            "diffeomorphic_factorized_enable_color": bool(color),
            "diffeomorphic_factorized_enable_amp": True,
        }
    )
    return payload


def build_candidates() -> list[IdeaCandidate]:
    c: list[IdeaCandidate] = []

    def add(name: str, idea: str, *, model: dict[str, Any] | None = None, bridge: dict[str, Any] | None = None, notes: str = "") -> None:
        idx = len(c)
        c.append(IdeaCandidate(name=f"{idx:02d}_{name}", idea=idea, model=model or _t00(), bridge=bridge or {}, notes=notes))

    # 1. Frequency-orthogonal terminal SWD.
    add("spectral_swd_bal", "spectral_orthogonal_swd", bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 1.0, "spectral_swd_high_weight": 1.0, "spectral_swd_low_kernel": 5})
    add("spectral_swd_low2", "spectral_orthogonal_swd", bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 2.0, "spectral_swd_high_weight": 0.7, "spectral_swd_low_kernel": 7})
    add("spectral_swd_high2", "spectral_orthogonal_swd", bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 0.7, "spectral_swd_high_weight": 2.0, "spectral_swd_low_kernel": 5})
    add("spectral_swd_amphead", "spectral_orthogonal_swd", model=_amp(True), bridge={"terminal_swd_mode": "spectral_orthogonal", "spectral_swd_low_weight": 1.0, "spectral_swd_high_weight": 1.5, "spectral_swd_low_kernel": 5})

    # 2. Semantic quotient SWD surrogate.
    add("quotient_bins3", "semantic_quotient_swd", bridge={"terminal_swd_mode": "semantic_quotient", "semantic_quotient_bins": 3})
    add("quotient_bins4", "semantic_quotient_swd", bridge={"terminal_swd_mode": "semantic_quotient", "semantic_quotient_bins": 4})
    add("quotient_bins6", "semantic_quotient_swd", bridge={"terminal_swd_mode": "semantic_quotient", "semantic_quotient_bins": 6})
    add("quotient_t01", "semantic_quotient_swd", model=_t01(), bridge={"terminal_swd_mode": "semantic_quotient", "semantic_quotient_bins": 4})

    # 3. Fourier phase lock.
    add("phase_lock_001", "fourier_phase_lock", bridge={"w_fourier_phase_lock": 0.01})
    add("phase_lock_003", "fourier_phase_lock", bridge={"w_fourier_phase_lock": 0.03})
    add("phase_lock_006", "fourier_phase_lock", bridge={"w_fourier_phase_lock": 0.06})
    add("phase_lock_amp", "fourier_phase_lock", model=_amp(True), bridge={"w_fourier_phase_lock": 0.03})

    # 4. Asymmetric kinetic/head taxation.
    add("tax_color_tv", "asymmetric_head_tax", bridge={"w_head_color_tv": 0.03, "w_head_color_energy": 0.003})
    add("tax_color_tv_strong", "asymmetric_head_tax", bridge={"w_head_color_tv": 0.08, "w_head_color_energy": 0.006})
    add("tax_curl_subsidy", "asymmetric_head_tax", bridge={"w_head_color_tv": 0.03, "w_warp_curl_reward": 0.002})
    add("tax_amphead", "asymmetric_head_tax", model=_amp(True), bridge={"w_head_color_tv": 0.03, "w_head_amp_energy": 0.003, "w_warp_curl_reward": 0.002})

    # 5. Cahn-Hilliard / phase separation and impasto-like energy floors.
    add("phase_sep_low", "cahn_hilliard_phase", bridge={"w_phase_separation": 0.008, "phase_gradient_weight": 0.02})
    add("phase_sep_energy", "cahn_hilliard_phase", bridge={"w_phase_separation": 0.01, "phase_gradient_weight": 0.03, "w_style_energy_floor": 0.03, "style_energy_floor_ratio": 0.6})
    add("phase_sep_retinex", "cahn_hilliard_phase", bridge={"w_phase_separation": 0.01, "phase_gradient_weight": 0.03, "retinex_target_blend": 0.5})
    add("phase_sep_amp", "cahn_hilliard_phase", model=_amp(True), bridge={"w_phase_separation": 0.01, "phase_gradient_weight": 0.03, "w_spectral_amplitude": 0.04, "spectral_amplitude_channels": 4})

    # 6. Null-space high-frequency canvas.
    add("canvas_002", "flat_highfreq_canvas", model={**_amp(True), "latent_canvas_strength": 0.02, "latent_canvas_edge_gamma": 4.0})
    add("canvas_005", "flat_highfreq_canvas", model={**_amp(True), "latent_canvas_strength": 0.05, "latent_canvas_edge_gamma": 4.0})
    add("canvas_008", "flat_highfreq_canvas", model={**_amp(True), "latent_canvas_strength": 0.08, "latent_canvas_edge_gamma": 6.0})
    add("canvas_coloroff", "flat_highfreq_canvas", model={**_amp(False), "latent_canvas_strength": 0.05, "latent_canvas_edge_gamma": 4.0})

    # 7. Joint-bilateral guided additive color.
    add("bilateral_k3", "joint_bilateral_color", model={**_t00(), "diffeomorphic_joint_bilateral_kernel": 3, "diffeomorphic_joint_bilateral_range_sigma": 0.35})
    add("bilateral_k5", "joint_bilateral_color", model={**_t00(), "diffeomorphic_joint_bilateral_kernel": 5, "diffeomorphic_joint_bilateral_range_sigma": 0.45})
    add("bilateral_lp3", "joint_bilateral_color", model={**_t00(), "diffeomorphic_color_lowpass_kernel": 3, "diffeomorphic_joint_bilateral_kernel": 5, "diffeomorphic_joint_bilateral_range_sigma": 0.45})
    add("bilateral_amp", "joint_bilateral_color", model={**_amp(True), "diffeomorphic_joint_bilateral_kernel": 5, "diffeomorphic_joint_bilateral_range_sigma": 0.45})

    # 8. Divergence-free Lagrangian warp variants.
    add("divfree_std", "divergence_free_warp", model={**_t00(), "diffeomorphic_divergence_free_warp": True})
    add("divfree_t01", "divergence_free_warp", model={**_t01(), "diffeomorphic_divergence_free_warp": True})
    add("divfree_amp", "divergence_free_warp", model={**_amp(True), "diffeomorphic_divergence_free_warp": True})
    add("divfree_pde", "divergence_free_warp", model={**_t00(), "diffeomorphic_divergence_free_warp": True}, bridge={"w_anisotropic_kinetic": 0.05, "w_stokes_viscous": 0.08})

    # 9. Existing PDE/Retinex knobs recombined as independent ideas.
    add("pde_aniso", "pde_budget", bridge={"w_anisotropic_kinetic": 0.08, "anisotropic_normal_weight": 25.0, "anisotropic_tangent_weight": 0.25})
    add("pde_stokes", "pde_budget", bridge={"w_stokes_viscous": 0.20})
    add("retinex_spec", "retinex_spectral", bridge={"retinex_target_blend": 1.0, "w_spectral_amplitude": 0.05, "spectral_amplitude_channels": 4})
    add("retinex_anchor", "retinex_spectral", bridge={"retinex_target_blend": 0.5, "w_content_anchor": 0.02, "w_edge_anchor": 0.04})

    # 10. My additions: budget normalization proxies via moment matching and terminal horizon pressure.
    add("pre_moment_025", "moment_budget_proxy", model={**_t00(), "pre_integrate_moment_match": True, "pre_integrate_moment_blend": 0.25})
    add("output_moment_train", "moment_budget_proxy", model={**_t00(), "output_moment_match": True, "output_moment_match_train_only": True})
    add("term_steps8", "terminal_horizon_pressure", bridge={"terminal_num_steps": 8})
    add("kinetic_high", "terminal_horizon_pressure", bridge={"w_kinetic": 1.5})

    return c


def _config_payload(candidate: IdeaCandidate, *, base_config: Path, output_root: Path) -> dict[str, Any]:
    base_ref = Path("..") / ".." / Path(base_config).resolve().relative_to(ROOT)
    save_dir = "./" + Path(output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    batch_size = int(os.environ.get("LANCET_BATCH_SIZE", "64"))
    eval_batch_size = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", "8"))
    return {
        "_base": base_ref.as_posix(),
        "model": candidate.model,
        "bridge": candidate.bridge,
        "training": {
            "seed": 42,
            "batch_size": batch_size,
            "full_eval_batch_size": eval_batch_size,
            "num_epochs": 6,
            "num_workers": 0,
            "persistent_workers": False,
            "save_interval": 1,
            "resume_checkpoint": "",
        },
        "checkpoint": {"save_dir": save_dir},
        "ablation": {
            "name": candidate.name,
            "stage": "curated_boundary_ideas",
            "axis": candidate.idea,
            "notes": candidate.notes or "Curated one-idea boundary probe; each idea gets at most 3-4 runs.",
        },
    }


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


def _train_one(config_path: Path) -> Path:
    _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
    config = load_config(config_path)
    save_dir = Path(config["checkpoint"]["save_dir"])
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    ckpt = save_dir.resolve() / f"epoch_{int(config['training']['num_epochs']):04d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint after training: {ckpt}")
    return ckpt


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
    fields = ["name", "idea", "score", "best_epoch", "clip_style_all", "content_lpips_all", "clip_content_all", "clip_dir_all", "status", "run_dir"]
    ranked = sorted(rows, key=_score, reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            payload = dict(row)
            payload["score"] = _score(payload)
            writer.writerow({key: payload.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Curated 40-run idea sweep for LANCET content frontier.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-epochs", type=str, default="4,6")
    parser.add_argument("--max-total", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    eval_epochs = tuple(int(part) for part in args.eval_epochs.split(",") if part.strip())
    candidates = build_candidates()[: max(0, args.max_total)]
    rows: list[dict[str, Any]] = []
    ledger_path = args.output_root / "curated_boundary_ideas_ledger.jsonl"
    frontier_path = args.output_root / "curated_boundary_ideas_frontier.csv"

    for candidate in candidates:
        config_path = args.config_root / f"{candidate.name}.json"
        run_dir = (args.output_root / candidate.name).resolve()
        ckpt = run_dir / "epoch_0006.pt"
        print(f"\n=== {candidate.name} [{candidate.idea}] ===", flush=True)
        if not config_path.exists() or args.force_train or args.force_eval:
            _write_json(config_path, _config_payload(candidate, base_config=args.base_config, output_root=args.output_root))
        if args.dry_run:
            print(f"[dry-run] wrote {config_path}", flush=True)
            continue

        status = "ok"
        eval_rows: list[dict[str, Any]] = []
        try:
            if not ckpt.exists() or args.force_train:
                ckpt = _train_one(config_path)
            else:
                print(f"[skip] checkpoint exists: {ckpt}", flush=True)
            for epoch in eval_epochs:
                epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
                eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
                summary = eval_dir / "summary.json"
                if summary.exists() and not args.force_eval:
                    eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **_load_summary_metrics(summary)})
                    continue
                if not epoch_ckpt.exists():
                    print(f"[warn] missing {epoch_ckpt}", flush=True)
                    continue
                summary = _eval_one(epoch_ckpt, eval_dir)
                eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **_load_summary_metrics(summary)})
        except Exception as exc:
            status = f"failed: {exc}"
            print(f"[error] {status}", flush=True)

        best = max(eval_rows, key=_score, default={})
        row = {
            "name": candidate.name,
            "idea": candidate.idea,
            "best_epoch": best.get("epoch"),
            "clip_style_all": best.get("clip_style_all"),
            "content_lpips_all": best.get("content_lpips_all"),
            "clip_content_all": best.get("clip_content_all"),
            "clip_dir_all": best.get("clip_dir_all"),
            "status": status,
            "run_dir": run_dir.as_posix(),
        }
        rows.append(row)
        _append_jsonl(ledger_path, {"candidate": asdict(candidate), "status": status, "result": row, "eval_rows": eval_rows})
        _write_frontier(rows, frontier_path)
        print(
            f"[best] epoch={row.get('best_epoch')} style={row.get('clip_style_all')} "
            f"lpips={row.get('content_lpips_all')} content={row.get('clip_content_all')} score={_score(row):.4f}",
            flush=True,
        )

    _write_frontier(rows, frontier_path)
    print(f"\n[done] {frontier_path}", flush=True)
    print(f"[done] {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
