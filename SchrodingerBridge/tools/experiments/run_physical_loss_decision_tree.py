from __future__ import annotations

import argparse
import csv
import itertools
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
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "physical_loss_tree"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "physical_loss_tree"
DEFAULT_EVAL_EPOCHS = (4, 6, 8)


def _detect_gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / (1024.0**3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            text=True,
        ).strip().splitlines()
        return float(out[0].strip()) / 1024.0 if out else None
    except Exception:
        return None


def _tiered_batch_sizes() -> tuple[int, int]:
    mem_gb = _detect_gpu_memory_gb()
    if mem_gb is None:
        return 96, 10
    if mem_gb <= 9.0:
        return 96, 10
    return 192, 20


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class PhysicalCandidate:
    name: str
    phase: str
    base: str
    model: dict[str, Any]
    bridge: dict[str, Any]
    notes: str
    num_epochs: int = 8
    seed: int = 42
    families: tuple[str, ...] = field(default_factory=tuple)


BASES: dict[str, dict[str, float]] = {
    "t00": {
        "diffeomorphic_warp_strength": 0.03,
        "diffeomorphic_texture_gate_strength": 6.0,
        "diffeomorphic_normal_leak": 0.0,
        "diffeomorphic_color_strength": 0.85,
    },
    "t01": {
        "diffeomorphic_warp_strength": 0.03,
        "diffeomorphic_texture_gate_strength": 6.0,
        "diffeomorphic_normal_leak": 0.05,
        "diffeomorphic_color_strength": 0.85,
    },
}


LOSS_ATOMS: dict[str, dict[str, Any]] = {
    "imp_low": {
        "family": "impasto",
        "bridge": {"w_impasto_divergence": 0.15, "impasto_energy_ratio": 0.75},
        "notes": "weak divergence-to-high-frequency impasto coupling",
    },
    "imp_high": {
        "family": "impasto",
        "bridge": {"w_impasto_divergence": 0.35, "impasto_energy_ratio": 0.95},
        "notes": "strong divergence-to-high-frequency impasto coupling",
    },
    "grad_low": {
        "family": "gradient_style",
        "bridge": {"w_gradient_anchored_style": 0.15, "gradient_style_gamma": 4.0, "gradient_style_edge_weight": 0.15},
        "notes": "weak low-gradient style energy with edge preservation",
    },
    "grad_high": {
        "family": "gradient_style",
        "bridge": {"w_gradient_anchored_style": 0.35, "gradient_style_gamma": 6.0, "gradient_style_edge_weight": 0.25},
        "notes": "strong low-gradient style energy with edge preservation",
    },
    "curl_low": {
        "family": "curl",
        "bridge": {"w_curl_style_field": 0.05, "curl_style_smooth_weight": 0.05, "curl_style_lowgrad_gamma": 4.0},
        "notes": "weak target orientation-curl field matching",
    },
    "curl_high": {
        "family": "curl",
        "bridge": {"w_curl_style_field": 0.12, "curl_style_smooth_weight": 0.08, "curl_style_lowgrad_gamma": 6.0},
        "notes": "strong target orientation-curl field matching",
    },
    "ssm_low": {
        "family": "self_similarity",
        "bridge": {"w_self_similarity_content": 0.08, "self_similarity_pool_size": 8},
        "notes": "weak latent self-similarity content topology guard",
    },
    "ssm_high": {
        "family": "self_similarity",
        "bridge": {"w_self_similarity_content": 0.18, "self_similarity_pool_size": 8},
        "notes": "strong latent self-similarity content topology guard",
    },
}


def _merge_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] = max(float(merged[key]), float(value))
            else:
                merged[key] = value
    return merged


def _candidate_from_atoms(
    *,
    phase: str,
    base_name: str,
    atom_names: tuple[str, ...],
    idx: int,
    model_override: dict[str, Any] | None = None,
    notes: str | None = None,
) -> PhysicalCandidate:
    atoms = [LOSS_ATOMS[name] for name in atom_names]
    families = tuple(dict.fromkeys(str(atom["family"]) for atom in atoms))
    bridge = _merge_dicts([dict(atom["bridge"]) for atom in atoms])
    model = {"use_diffeomorphic_stroke": True, **BASES[base_name], **(model_override or {})}
    atom_slug = "__".join(atom_names)
    name = f"{phase}_{idx:02d}_{base_name}_{atom_slug}"
    return PhysicalCandidate(
        name=name,
        phase=phase,
        base=base_name,
        model=model,
        bridge=bridge,
        families=families,
        notes=notes or "; ".join(str(atom["notes"]) for atom in atoms),
    )


def build_phase1() -> list[PhysicalCandidate]:
    candidates: list[PhysicalCandidate] = []
    idx = 0
    for base_name in ("t00", "t01"):
        for atom_name in LOSS_ATOMS:
            candidates.append(_candidate_from_atoms(phase="p1", base_name=base_name, atom_names=(atom_name,), idx=idx))
            idx += 1
    return candidates


def _score(row: dict[str, Any]) -> float:
    style = float(row.get("clip_style_all") or 0.0)
    content = float(row.get("clip_content_all") or 0.0)
    dino = float(row.get("dino_structure") or 0.05)
    lpips = float(row.get("content_lpips_all") or 1.0)
    return style + 0.20 * content - 1.5 * max(0.0, dino - 0.026) - 0.20 * max(0.0, lpips - 0.52)


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if (row.get("clip_style_all") is not None)
        and float(row.get("clip_content_all") or 0.0) >= 0.745
        and float(row.get("dino_structure") or 0.05) <= 0.0285
    ]
    return eligible or [row for row in rows if row.get("clip_style_all") is not None]


def build_phase2(phase1_rows: list[dict[str, Any]]) -> list[PhysicalCandidate]:
    ranked = sorted(_eligible(phase1_rows), key=_score, reverse=True)
    atoms: list[str] = []
    seen_families: set[str] = set()
    for row in ranked:
        atom = str(row.get("atoms") or "").split("+")[0]
        family = str(row.get("families") or "").split("+")[0]
        if atom and family not in seen_families:
            atoms.append(atom)
            seen_families.add(family)
        if len(atoms) >= 4:
            break
    if len(atoms) < 2:
        atoms = ["imp_low", "grad_low", "curl_low", "ssm_low"]
    pairs = list(itertools.combinations(atoms, 2))
    candidates: list[PhysicalCandidate] = []
    idx = 0
    for base_name in ("t00", "t01"):
        for pair in pairs:
            candidates.append(_candidate_from_atoms(phase="p2", base_name=base_name, atom_names=pair, idx=idx))
            idx += 1
    return candidates[:16]


def build_phase3(phase2_rows: list[dict[str, Any]]) -> list[PhysicalCandidate]:
    ranked = sorted(_eligible(phase2_rows), key=_score, reverse=True)[:4]
    if not ranked:
        ranked = sorted(phase2_rows, key=_score, reverse=True)[:4]
    candidates: list[PhysicalCandidate] = []
    idx = 0
    for row in ranked:
        atoms = tuple(part for part in str(row.get("atoms") or "").split("+") if part)
        base = str(row.get("base") or "t00")
        for warp, leak in ((0.045, 0.0), (0.055, 0.0), (0.055, 0.05), (0.07, 0.0)):
            candidates.append(
                _candidate_from_atoms(
                    phase="p3",
                    base_name=base if base in BASES else "t00",
                    atom_names=atoms or ("imp_low", "grad_low"),
                    idx=idx,
                    model_override={
                        "diffeomorphic_warp_strength": warp,
                        "diffeomorphic_normal_leak": leak,
                    },
                    notes=f"aggressive warp release around phase2 winner; warp={warp}, leak={leak}",
                )
            )
            idx += 1
    return candidates[:16]


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
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _config_payload(candidate: PhysicalCandidate, *, base_config: Path, output_root: Path) -> dict[str, Any]:
    default_batch, default_eval_batch = _tiered_batch_sizes()
    batch_size = int(os.environ.get("LANCET_BATCH_SIZE", str(default_batch)))
    eval_batch_size = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", str(default_eval_batch)))
    save_dir = "./" + Path(output_root / candidate.name).resolve().relative_to(ROOT).as_posix()
    return {
        "_base": (Path("..") / ".." / Path(base_config).resolve().relative_to(ROOT)).as_posix(),
        "model": candidate.model,
        "bridge": candidate.bridge,
        "training": {
            "seed": candidate.seed,
            "batch_size": batch_size,
            "full_eval_batch_size": eval_batch_size,
            "num_epochs": candidate.num_epochs,
            "num_workers": 0,
            "persistent_workers": False,
            "save_interval": 1,
            "resume_checkpoint": "",
        },
        "checkpoint": {"save_dir": save_dir},
        "ablation": {
            "name": candidate.name,
            "stage": "physical_loss_tree",
            "phase": candidate.phase,
            "axis": "loss_principle_exploration",
            "families": list(candidate.families),
            "notes": candidate.notes,
        },
    }


def _summary_metrics(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {
            "clip_style_all": None,
            "content_lpips_all": None,
            "clip_content_all": None,
            "cmmd": None,
            "dino_structure": None,
            "gram_micro": None,
            "gram_macro": None,
        }
    overview = ((_load_json(summary_path).get("analysis") or {}).get("all_pairs_overview") or {})
    keys = {
        "clip_style_all": "clip_style",
        "content_lpips_all": "content_lpips",
        "clip_content_all": "clip_content",
        "cmmd": "cmmd",
        "dino_structure": "dino_structure",
        "gram_micro": "gram_micro",
        "gram_macro": "gram_macro",
    }
    return {out: float(overview[src]) if overview.get(src) is not None else None for out, src in keys.items()}


def _train_one(config_path: Path) -> Path:
    _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT)
    cfg = load_config(config_path)
    save_dir = Path(cfg["checkpoint"]["save_dir"]).resolve()
    ckpt = save_dir / f"epoch_{int(cfg['training']['num_epochs']):04d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    return ckpt


def _eval_one(ckpt: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, default_eval = _tiered_batch_sizes()
    eval_batch = int(os.environ.get("LANCET_EVAL_BATCH_SIZE", str(default_eval)))
    _run(
        [
            sys.executable,
            "src/utils/run_evaluation.py",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_dir),
            "--batch_size",
            str(eval_batch),
            "--eval_enable_modern_metrics",
        ],
        cwd=ROOT,
    )
    summary = out_dir / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"missing eval summary: {summary}")
    return summary


def _candidate_row(candidate: PhysicalCandidate, config_path: Path, run_dir: Path, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(eval_rows, key=lambda row: float(row.get("clip_style_all") or -9999.0), default={})
    atom_names = tuple(part for part in candidate.name.split("_", 3)[-1].split("__") if part)
    return {
        "name": candidate.name,
        "phase": candidate.phase,
        "base": candidate.base,
        "atoms": "+".join(atom_names),
        "families": "+".join(candidate.families),
        "best_epoch": best.get("epoch"),
        "clip_style_all": best.get("clip_style_all"),
        "content_lpips_all": best.get("content_lpips_all"),
        "clip_content_all": best.get("clip_content_all"),
        "cmmd": best.get("cmmd"),
        "dino_structure": best.get("dino_structure"),
        "gram_micro": best.get("gram_micro"),
        "gram_macro": best.get("gram_macro"),
        "score": _score(best) if best else None,
        "config": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "best_summary": best.get("summary"),
    }


def _write_frontier(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "phase",
        "base",
        "atoms",
        "families",
        "best_epoch",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "cmmd",
        "dino_structure",
        "gram_micro",
        "gram_macro",
        "score",
        "run_dir",
        "best_summary",
    ]
    ranked = sorted(rows, key=lambda row: float(row.get("score") or -9999.0), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in fields})


def _load_existing_rows(ledger_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not ledger_path.exists():
        return rows
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = payload.get("result") or {}
            name = row.get("name")
            if name and row.get("best_epoch") is not None:
                rows[str(name)] = row
    return rows


def _run_candidate(
    candidate: PhysicalCandidate,
    *,
    args: argparse.Namespace,
    eval_epochs: tuple[int, ...],
) -> dict[str, Any]:
    if not args.force_train and not args.force_eval and candidate.name in args.completed_rows:
        row = dict(args.completed_rows[candidate.name])
        print(f"\n=== {candidate.name} ===", flush=True)
        print(f"[resume-skip] completed; style={row.get('clip_style_all')} score={row.get('score')}", flush=True)
        return row

    config_path = args.config_root / f"{candidate.name}.json"
    run_dir = (args.output_root / candidate.name).resolve()
    ckpt = run_dir / f"epoch_{candidate.num_epochs:04d}.pt"
    if not config_path.exists() or args.force_train or args.force_eval:
        _write_json(config_path, _config_payload(candidate, base_config=args.base_config, output_root=args.output_root))
    print(f"\n=== {candidate.name} ===", flush=True)
    print(f"[phase] {candidate.phase} [families] {candidate.families} [bridge] {candidate.bridge}", flush=True)
    if not ckpt.exists() or args.force_train:
        ckpt = _train_one(config_path)
    else:
        print(f"[skip] checkpoint exists: {ckpt}", flush=True)
    eval_rows: list[dict[str, Any]] = []
    for epoch in eval_epochs:
        summary = run_dir / "full_eval" / f"epoch_{epoch:04d}" / "summary.json"
        if summary.exists() and not args.force_eval:
            metrics = _summary_metrics(summary)
            eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **metrics})
            print(f"[skip] eval exists: {summary}", flush=True)
            continue
        epoch_ckpt = run_dir / f"epoch_{epoch:04d}.pt"
        if not epoch_ckpt.exists():
            print(f"[warn] missing checkpoint for epoch {epoch}: {epoch_ckpt}", flush=True)
            continue
        summary = _eval_one(epoch_ckpt, summary.parent)
        eval_rows.append({"epoch": epoch, "summary": summary.as_posix(), **_summary_metrics(summary)})
    row = _candidate_row(candidate, config_path, run_dir, eval_rows)
    _append_jsonl(args.ledger_path, {"candidate": asdict(candidate), "result": row, "eval_rows": eval_rows})
    print(
        f"[best] epoch={row.get('best_epoch')} style={row.get('clip_style_all')} "
        f"lpips={row.get('content_lpips_all')} content={row.get('clip_content_all')} "
        f"dino={row.get('dino_structure')} score={row.get('score')}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical-loss decision tree for 256px diffeomorphic stroke.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-total", type=int, default=64)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    args.ledger_path = args.output_root / "physical_loss_tree_ledger.jsonl"
    frontier_path = args.output_root / "physical_loss_tree_frontier.csv"
    args.completed_rows = _load_existing_rows(args.ledger_path)
    eval_epochs = tuple(int(part) for part in args.eval_epochs.split(",") if part.strip())
    mem_gb = _detect_gpu_memory_gb()
    train_batch, eval_batch = _tiered_batch_sizes()
    print(
        f"[gpu-tier] memory_gb={mem_gb if mem_gb is not None else 'unknown'} "
        f"default_train_batch={train_batch} default_eval_batch={eval_batch} "
        f"env_train={os.environ.get('LANCET_BATCH_SIZE') or ''} env_eval={os.environ.get('LANCET_EVAL_BATCH_SIZE') or ''}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = list(args.completed_rows.values())
    p1_candidates = build_phase1()
    p1_rows = [row for row in all_rows if row.get("phase") == "p1"]
    phase_specs: list[tuple[int, list[PhysicalCandidate]]] = [(1, p1_candidates)]

    while phase_specs:
        phase_idx, candidates = phase_specs.pop(0)
        remaining = max(0, int(args.max_total) - len({row.get("name") for row in all_rows}))
        if remaining <= 0:
            break
        print(f"\n[phase {phase_idx}] candidates={len(candidates[:remaining])}", flush=True)
        phase_rows = []
        for candidate in candidates[:remaining]:
            row = _run_candidate(candidate, args=args, eval_epochs=eval_epochs)
            all_rows = [old for old in all_rows if old.get("name") != row.get("name")]
            all_rows.append(row)
            phase_rows.append(row)
            if row.get("phase") == "p1":
                p1_rows = [old for old in p1_rows if old.get("name") != row.get("name")]
                p1_rows.append(row)
            _write_frontier(all_rows, frontier_path)
        if phase_idx == 1:
            p1_complete = len({row.get("name") for row in p1_rows}) >= len(p1_candidates)
            if p1_complete:
                phase_specs.append((2, build_phase2(p1_rows)))
            else:
                print("[pause] phase 1 incomplete; rerun will resume and then continue automatically.", flush=True)
        elif phase_idx == 2:
            p2_rows = [row for row in all_rows if row.get("phase") == "p2"]
            phase_specs.append((3, build_phase3(p2_rows or phase_rows)))

    _write_frontier(all_rows, frontier_path)
    print(f"\n[done] {frontier_path}", flush=True)
    print(f"[done] {args.ledger_path}", flush=True)


if __name__ == "__main__":
    main()
