from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "tools" / "experiments"


def test_phase618_remote_launchers_sync_and_verify_validity_audit() -> None:
    launcher_names = [
        "launch_phase618_ot_rerun_remote.sh",
        "launch_phase618_pipeline_remote.sh",
        "launch_phase618_plain_path_distill_remote.sh",
        "launch_phase618_plain_path_distill_remote_when_idle.sh",
        "launch_phase618_style_sweep_remote.sh",
    ]
    sync_line = "--sync-path SchrodingerBridge/tools/audit_phase618_run_validity.py"
    verify_line = "--verify-python-file SchrodingerBridge/tools/audit_phase618_run_validity.py"

    for name in launcher_names:
        text = (EXPERIMENTS / name).read_text(encoding="utf-8")
        assert sync_line in text, name
        assert verify_line in text, name


def test_phase618_pipeline_defaults_to_stable_batch_sizes() -> None:
    launcher_names = [
        "launch_phase618_ot_rerun_remote.sh",
        "launch_phase618_pipeline_remote.sh",
        "launch_phase618_plain_path_distill_remote.sh",
        "launch_phase618_plain_path_distill_remote_when_idle.sh",
        "launch_phase618_style_sweep_remote.sh",
    ]
    pipeline_text = (EXPERIMENTS / "run_phase618_pipeline.sh").read_text(encoding="utf-8")
    plain_distill_text = (EXPERIMENTS / "run_phase618_plain_path_distill.sh").read_text(encoding="utf-8")
    ceiling_line = "--max-runtime-memory-mib 11570"
    guard_line = "--runtime-guard-max-memory-mib 11570"

    for name in launcher_names:
        text = (EXPERIMENTS / name).read_text(encoding="utf-8")
        assert ceiling_line in text, name
        assert guard_line in text, name

    ot_rerun_text = (EXPERIMENTS / "run_phase618_ot_rerun.sh").read_text(encoding="utf-8")

    assert 'OT_FIXED_BATCH_SIZE="${OT_FIXED_BATCH_SIZE:-16}"' in pipeline_text
    assert 'PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE="${PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE:-20}"' in pipeline_text
    assert 'STYLE_SWEEP_FIXED_BATCH_SIZE="${STYLE_SWEEP_FIXED_BATCH_SIZE:-20}"' in pipeline_text
    assert 'OT_FIXED_BATCH_SIZE="${OT_FIXED_BATCH_SIZE:-16}"' in ot_rerun_text
    assert 'PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE="${PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE:-20}"' in plain_distill_text
