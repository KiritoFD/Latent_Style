from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "experiments" / "build_phase618_remote_run_audit.py"
SPEC = importlib.util.spec_from_file_location("phase618_remote_run_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_flatten_stage_and_run_rows_capture_remote_audit_contract() -> None:
    payload = {
        "stage_roots": [
            {
                "stage_root": "/remote/ot",
                "exists": True,
                "stage_summary_present": True,
                "stage_manifest_present": True,
                "backfill": {"status": "ok"},
                "close_result_diagnosis": {"status": "separated", "interpretation": "not_close_yet"},
                "best": {"name": "h0_vertical_fm", "style": 0.6620, "lpips": 0.3341, "gap": 0.1121},
                "runs": [
                    {
                        "name": "h0_vertical_fm",
                        "run_dir": "/remote/ot/h0_vertical_fm",
                        "best_epoch": "epoch_0002",
                        "best_epoch_int": 2,
                        "style": 0.6620,
                        "lpips": 0.3341,
                        "gap": 0.1121,
                        "validity_audit": {
                            "artifact_status": "valid",
                            "effect_contract": "training_real_eval_inert",
                            "suite": "stage1_h0_h6_repaired_lowrank",
                            "trust_level": "medium",
                            "scientific_reading": "old OT family is training-real but plain-eval inert",
                            "recommended_action": "use only post-repair evidence",
                            "issue_codes": [],
                        },
                    }
                ],
                "child_dirs": ["h0_vertical_fm"],
            },
            {
                "stage_root": "/remote/plain_path_distill",
                "exists": False,
                "stage_summary_present": False,
                "stage_manifest_present": False,
                "backfill": {},
                "close_result_diagnosis": {},
                "best": {},
                "runs": [],
                "child_dirs": [],
            },
        ]
    }

    stage_rows = AUDIT._flatten_stage_rows(payload)
    run_rows = AUDIT._flatten_run_rows(payload)

    assert len(stage_rows) == 2
    assert stage_rows[0]["close_status"] == "separated"
    assert stage_rows[0]["best_name"] == "h0_vertical_fm"
    assert stage_rows[1]["exists"] is False

    assert len(run_rows) == 1
    assert run_rows[0]["artifact_status"] == "valid"
    assert run_rows[0]["effect_contract"] == "training_real_eval_inert"
    assert run_rows[0]["suite"] == "stage1_h0_h6_repaired_lowrank"


def test_build_readme_mentions_missing_plain_path_distill_stage() -> None:
    payload = {
        "stage_roots": [
            {
                "stage_root": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto",
                "exists": True,
                "stage_summary_present": True,
                "stage_manifest_present": True,
                "backfill": {"status": "ok"},
                "close_result_diagnosis": {"status": "separated", "interpretation": "not_close_yet"},
                "best": {"name": "h0_vertical_fm", "style": 0.6620, "lpips": 0.3341, "gap": 0.1121},
                "runs": [],
                "child_dirs": [],
            },
            {
                "stage_root": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_plain_path_distill_auto",
                "exists": False,
                "stage_summary_present": False,
                "stage_manifest_present": False,
                "backfill": {},
                "close_result_diagnosis": {},
                "best": {},
                "runs": [],
                "child_dirs": [],
            },
        ]
    }
    stage_rows = AUDIT._flatten_stage_rows(payload)
    run_rows = AUDIT._flatten_run_rows(payload)
    readme = AUDIT._build_readme(payload=payload, stage_rows=stage_rows, run_rows=run_rows)

    assert "plain-path distill stage root exists: `False`" in readme
    assert "remote OT rerun stage currently has `0` discovered runs" in readme


def test_build_readme_surfaces_flagged_runs() -> None:
    payload = {"stage_roots": []}
    stage_rows = []
    run_rows = [
        {
            "stage_root": "/remote/ot",
            "name": "h5_topogate_attention",
            "style": 0.0,
            "lpips": 1.0,
            "gap": 1.44,
            "artifact_status": "stale",
            "effect_contract": "training_real_eval_inert",
            "issue_codes": "topogate_descriptor_metric_missing",
            "recommended_action": "rerun after multiblock logging fix",
        }
    ]
    readme = AUDIT._build_readme(payload=payload, stage_rows=stage_rows, run_rows=run_rows)

    assert "## Flagged runs" in readme
    assert "h5_topogate_attention" in readme
    assert "topogate_descriptor_metric_missing" in readme


def test_remote_audit_script_merges_unsummarized_child_dirs() -> None:
    script = AUDIT._remote_audit_script(
        remote_worktree="/mnt/i/Github/Latent_Style/SchrodingerBridge",
        stage_roots=["/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto"],
    )

    assert "collect_run_entry" in script
    assert "seen_run_dirs = set()" in script
    assert "if str(child) in seen_run_dirs or child.name in seen_names" in script
