from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE616_AUTO_PATH = ROOT / "tools" / "experiments" / "phase616_auto.py"
AUDIT_PATH = ROOT / "tools" / "audit_phase618_run_validity.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PHASE616_AUTO = _load_module(PHASE616_AUTO_PATH, "phase616_auto_backfill")
AUDIT = _load_module(AUDIT_PATH, "audit_phase618_run_validity_backfill")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_entry_with_score(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if "best_transfer_clip_style" in out and "style" not in out:
        out["style"] = out.get("best_transfer_clip_style")
    if "best_transfer_content_lpips" in out and "lpips" not in out:
        out["lpips"] = out.get("best_transfer_content_lpips")
    if "best_objective_gap" in out and "gap" not in out:
        out["gap"] = out.get("best_objective_gap")
    if "style" not in out and "transfer_clip_style" in out:
        out["style"] = out.get("transfer_clip_style")
    if "lpips" not in out and "transfer_content_lpips" in out:
        out["lpips"] = out.get("transfer_content_lpips")
    if "gap" not in out and "objective_gap" in out:
        out["gap"] = out.get("objective_gap")
    return out


def _entry_from_discovered_run(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    curve_path = run_dir / "full_eval_transfer" / "clip_lpips_curve.csv"
    logs_dir = run_dir / "logs"
    if not config_path.is_file() and not curve_path.is_file() and not logs_dir.is_dir():
        return None
    if config_path.is_file():
        entry = PHASE616_AUTO._reuse_existing_entry(run_dir, name=run_dir.name, eval_subdir="full_eval_transfer")
    else:
        best_point = PHASE616_AUTO._best_curve_point(run_dir, eval_subdir="full_eval_transfer")
        if best_point is None:
            return None
        style = float(best_point["style"])
        lpips = float(best_point["lpips"])
        entry = {
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "config_path": "",
            "selected_batch_size": 0,
            "probe_summary_path": str(run_dir / "_probe" / "probe_summary.json"),
            "run_summary_path": str(run_dir / "auto_run_summary.json"),
            "transfer_clip_style": style,
            "transfer_content_lpips": lpips,
            "objective_gap": PHASE616_AUTO._objective_gap(style, lpips),
            "best_epoch": str(best_point["epoch"]),
            "best_epoch_int": int(best_point["epoch_int"]),
            "best_transfer_clip_style": style,
            "best_transfer_content_lpips": lpips,
            "best_objective_gap": float(best_point["gap"]),
            "reused_existing": True,
        }
    return _to_entry_with_score(entry)


def _discover_stage_runs(stage_root: Path) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for child in sorted(stage_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        entry = _entry_from_discovered_run(child)
        if entry is not None:
            discovered.append(entry)
    return discovered


def _audit_entry(entry: dict[str, Any], *, stage_root: Path) -> dict[str, Any]:
    run_dir_text = str(entry.get("logical_run_dir") or entry.get("run_dir") or "").strip()
    config_path_text = str(entry.get("config_path") or "").strip()
    name = str(entry.get("name") or "").strip()
    if run_dir_text:
        run_dir = Path(run_dir_text)
        if not run_dir.is_absolute():
            run_dir = stage_root / run_dir
        if run_dir.is_dir():
            result = AUDIT.audit_phase618_run_validity(run_dir=run_dir)
            return PHASE616_AUTO._summarize_validity_result(result, audit_path=Path(""))
    if config_path_text:
        config_path = Path(config_path_text)
        if not config_path.is_absolute():
            config_path = stage_root / config_path
        if config_path.is_file():
            result = AUDIT.audit_phase618_run_validity(config_path=config_path, variant_name=name or config_path.stem)
            return PHASE616_AUTO._summarize_validity_result(result, audit_path=Path(""))
    return {
        "artifact_status": "unknown",
        "effect_contract": "unknown",
        "suite": "",
        "scientific_reading": "",
        "trust_level": "",
        "recommended_action": "missing run_dir/config_path; unable to audit",
        "issue_codes": ["missing_artifact_path"],
        "issue_count": 1,
        "repaired_lowrank_base": False,
        "audit_path": "",
    }


def backfill_stage_root(stage_root: Path) -> dict[str, Any]:
    summary_path = stage_root / "stage_summary.json"
    manifest_path = stage_root / "stage_manifest.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing stage summary: {summary_path}")
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {"runs": []}

    summary_runs = [dict(item) for item in (summary.get("runs") or []) if isinstance(item, dict)]
    manifest_runs = [dict(item) for item in (manifest.get("runs") or []) if isinstance(item, dict)]
    manifest_by_name = {str(item.get("name") or ""): item for item in manifest_runs}
    discovered_runs = _discover_stage_runs(stage_root)
    discovered_by_name = {str(item.get("name") or ""): item for item in discovered_runs}
    summary_run_names = {str(item.get("name") or "") for item in summary_runs}

    updated_runs: list[dict[str, Any]] = []
    updated_manifest_runs: list[dict[str, Any]] = []
    for run in summary_runs + [item for item in discovered_runs if str(item.get("name") or "") not in summary_run_names]:
        name = str(run.get("name") or "")
        merged = dict(discovered_by_name.get(name) or {})
        merged.update(dict(manifest_by_name.get(name) or {}))
        merged.update(run)
        if not merged.get("validity_audit"):
            merged["validity_audit"] = _audit_entry(merged, stage_root=stage_root)
        merged = _to_entry_with_score(merged)
        updated_runs.append(dict(merged))
        updated_manifest_runs.append(dict(merged))

    summary["runs"] = updated_runs
    manifest["runs"] = updated_manifest_runs

    if isinstance(summary.get("best"), dict):
        best = dict(summary["best"])
        best_name = str(best.get("name") or "")
        matched = next((run for run in updated_runs if str(run.get("name") or "") == best_name), None)
        if matched is not None:
            for key in ("style", "lpips", "gap", "validity_audit"):
                if key in matched and key not in best:
                    best[key] = matched[key]
        summary["best"] = best

    PHASE616_AUTO._attach_close_result_diagnosis(summary)
    _save_json(summary_path, summary)
    _save_json(manifest_path, manifest)
    return {
        "stage_root": str(stage_root),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "run_count": len(updated_runs),
        "close_result_diagnosis": summary.get("close_result_diagnosis"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill validity_audit and close_result_diagnosis into an existing phase618 stage summary/manifest.")
    parser.add_argument("--stage-root", required=True)
    args = parser.parse_args()
    payload = backfill_stage_root(Path(args.stage_root))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
