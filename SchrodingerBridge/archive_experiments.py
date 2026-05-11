from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT / "exp"
RUNS_ROOT = EXP_ROOT / "runs"
CONFIGS_ROOT = EXP_ROOT / "configs"
SCRIPTS_ARCHIVE_ROOT = EXP_ROOT / "scripts" / "tools_archive"
MANIFEST_PATH = EXP_ROOT / "archive_manifest.json"
BLACKDOT_CSV = ROOT / "docs" / "experiments" / "blackdot_mitigation_runs.csv"

KEEP_ROOT_DIRS = {
    "datasets",
    "docs",
    "exp",
    "src",
    "__pycache__",
    "full_dimensional_orthogonal_sweep_20",
}
KEEP_ROOT_FILES = {
    "archive_experiments.py",
    "archive_blackdot_experiments.py",
    "build_experiment_registry.py",
    "gen_orth_12.py",
    "config.json",
    "g0_blackdot_m2_damped20_inplace.json",
    "inference.py",
    "README.md",
    "run.py",
    "run_evaluation.py",
    "src.zip",
}
LEGACY_TOOL_SCRIPTS = {
    "debug_black_artifacts.py",
    "diagnose_full40_step.py",
    "generate_arch_stress_test_60.py",
    "generate_high_tension_phase_space_sweep.py",
    "generate_orthogonal_phase_space_sweep_60.py",
}


@dataclass
class MoveRecord:
    category: str
    name: str
    destination: str


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))


def _save_manifest(entries: list[dict]) -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_manifest(records: list[MoveRecord]) -> None:
    if not records:
        return
    entries = _load_manifest()
    entries.extend({"category": r.category, "name": r.name, "destination": r.destination} for r in records)
    _save_manifest(entries)


def _backfill_experiments_root_manifest() -> list[MoveRecord]:
    target_root = CONFIGS_ROOT / "experiments_root"
    if not target_root.exists():
        return []
    existing_destinations = {
        str(item.get("destination", ""))
        for item in _load_manifest()
    }
    records: list[MoveRecord] = []
    for child in sorted(target_root.iterdir(), key=lambda p: p.name.lower()):
        destination = str(child)
        if destination in existing_destinations:
            continue
        records.append(MoveRecord(category="configs", name=child.name, destination=destination))
    return records


def _move_path(src: Path, dest: Path) -> MoveRecord | None:
    if not src.exists():
        return None
    if dest.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    category = "runs" if dest.is_relative_to(RUNS_ROOT) else "configs" if dest.is_relative_to(CONFIGS_ROOT) else "scripts_archive"
    return MoveRecord(category=category, name=src.name, destination=str(dest))


def _archive_experiments_tree() -> list[MoveRecord]:
    records: list[MoveRecord] = []
    experiments_dir = ROOT / "experiments"
    if not experiments_dir.exists():
        return records

    target_root = CONFIGS_ROOT / "experiments_root"
    for child in sorted(experiments_dir.iterdir(), key=lambda p: p.name.lower()):
        record = _move_path(child, target_root / child.name)
        if record:
            records.append(record)

    try:
        if experiments_dir.exists() and not any(experiments_dir.iterdir()):
            experiments_dir.rmdir()
    except OSError:
        pass
    return records


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").exists()
        or (path / "summary.json").exists()
        or (path / "full_eval").exists()
        or any(path.glob("epoch_*.pt"))
    )


def _archive_root_run_dirs() -> list[MoveRecord]:
    records: list[MoveRecord] = []
    for child in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if child.name in KEEP_ROOT_DIRS or not child.is_dir():
            continue
        if _looks_like_run_dir(child) and child.name.startswith("o"):
            record = _move_path(child, RUNS_ROOT / child.name)
            if record:
                records.append(record)
    return records


def _archive_legacy_root_scripts() -> list[MoveRecord]:
    records: list[MoveRecord] = []
    for child in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file():
            continue
        if child.name in KEEP_ROOT_FILES:
            continue
        if child.suffix.lower() == ".bat" or child.name in LEGACY_TOOL_SCRIPTS:
            record = _move_path(child, SCRIPTS_ARCHIVE_ROOT / child.name)
            if record:
                records.append(record)
    return records


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPTS_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    records: list[MoveRecord] = []
    records.extend(_archive_root_run_dirs())
    records.extend(_archive_experiments_tree())
    records.extend(_archive_legacy_root_scripts())
    records.extend(_backfill_experiments_root_manifest())
    _append_manifest(records)

    _run([sys.executable, str(ROOT / "archive_blackdot_experiments.py"), "--csv", str(BLACKDOT_CSV), "--discover-root", str(RUNS_ROOT)])
    _run([sys.executable, str(ROOT / "build_experiment_registry.py")])

    print("Archived records:")
    for record in records:
        print(f"- [{record.category}] {record.name} -> {record.destination}")
    print(BLACKDOT_CSV)


if __name__ == "__main__":
    main()
