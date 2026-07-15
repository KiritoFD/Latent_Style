from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKFILL_PATH = ROOT / "tools" / "experiments" / "backfill_phase618_stage_summary.py"
BACKFILL_SPEC = importlib.util.spec_from_file_location("phase618_stage_summary_backfill", BACKFILL_PATH)
assert BACKFILL_SPEC is not None and BACKFILL_SPEC.loader is not None
BACKFILL = importlib.util.module_from_spec(BACKFILL_SPEC)
BACKFILL_SPEC.loader.exec_module(BACKFILL)


def test_backfill_stage_root_adds_validity_and_close_result_diagnosis(tmp_path: Path) -> None:
    base_cfg = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "2026-06-18-stage1-lowrank-rerun-audit"
            / "baseline_h1_lowrank_config.json"
        ).read_text(encoding="utf-8")
    )
    stage_root = tmp_path / "stage3_style"
    stage_root.mkdir(parents=True)

    runs = []
    for name, blend, style, lpips, gap in (
        ("r11_linear_blend_0p00", 0.0, 0.6680, 0.3010, 0.0730),
        ("r12_linear_blend_0p30", 0.3, 0.6500, 0.3600, 0.1500),
    ):
        run_dir = stage_root / name
        run_dir.mkdir(parents=True)
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("model", {})["semantic_self_topology_blend"] = blend
        (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        eval_dir = run_dir / "full_eval_transfer"
        eval_dir.mkdir(parents=True)
        if name == "r12_linear_blend_0p30":
            curve = (
                "epoch,epoch_int,checkpoint,timestamp,transfer_clip_style,transfer_content_lpips\n"
                "epoch_0001,1,epoch_0001.pt,2026-06-18 00:00:00,0.6672,0.3040\n"
                "epoch_0002,2,epoch_0002.pt,2026-06-18 00:05:00,0.6500,0.3600\n"
            )
            (eval_dir / "clip_lpips_curve.csv").write_text(curve, encoding="utf-8")
        else:
            curve = (
                "epoch,epoch_int,checkpoint,timestamp,transfer_clip_style,transfer_content_lpips\n"
                "epoch_0001,1,epoch_0001.pt,2026-06-18 00:00:00,0.6680,0.3010\n"
            )
            (eval_dir / "clip_lpips_curve.csv").write_text(curve, encoding="utf-8")
        runs.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "config_path": str(run_dir / "config.json"),
                "transfer_clip_style": style,
                "transfer_content_lpips": lpips,
                "objective_gap": gap,
            }
        )

    summary = {
        "stage": "stage3_style",
        "stage_root": str(stage_root),
        "runs": runs,
        "best": {
            "name": "r11_linear_blend_0p00",
            "run_dir": str(stage_root / "r11_linear_blend_0p00"),
            "style": 0.6680,
            "lpips": 0.3010,
            "gap": 0.0730,
        },
        "plan": {"stage": "stage3_style", "runs": [r["name"] for r in runs], "purpose": "test"},
    }
    manifest = {"runs": runs}
    (stage_root / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage_root / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = BACKFILL.backfill_stage_root(stage_root)

    assert payload["run_count"] == 2
    assert payload["close_result_diagnosis"]["status"] == "close_cluster"
    assert payload["close_result_diagnosis"]["interpretation"] == "runtime_real_but_weak"

    summary_after = json.loads((stage_root / "stage_summary.json").read_text(encoding="utf-8"))
    manifest_after = json.loads((stage_root / "stage_manifest.json").read_text(encoding="utf-8"))
    assert summary_after["runs"][0]["validity_audit"]["artifact_status"] == "valid"
    assert summary_after["runs"][0]["validity_audit"]["effect_contract"] == "runtime_and_training_real"
    assert manifest_after["runs"][1]["validity_audit"]["artifact_status"] == "valid"
    assert summary_after["runs"][1]["best_transfer_clip_style"] == 0.6672
    assert summary_after["runs"][1]["best_transfer_content_lpips"] == 0.3040
    assert summary_after["close_result_diagnosis"]["interpretation"] == "runtime_real_but_weak"


def test_backfill_discovers_run_missing_from_summary_and_manifest(tmp_path: Path) -> None:
    base_cfg = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "2026-06-18-stage1-lowrank-rerun-audit"
            / "baseline_h1_lowrank_config.json"
        ).read_text(encoding="utf-8")
    )
    stage_root = tmp_path / "stage1_auto"
    stage_root.mkdir(parents=True)

    h0_dir = stage_root / "h0_vertical_fm"
    h0_dir.mkdir(parents=True)
    (h0_dir / "config.json").write_text(json.dumps(base_cfg), encoding="utf-8")
    (h0_dir / "full_eval_transfer").mkdir(parents=True)
    (h0_dir / "full_eval_transfer" / "clip_lpips_curve.csv").write_text(
        "epoch,epoch_int,checkpoint,timestamp,transfer_clip_style,transfer_content_lpips\n"
        "epoch_0001,1,epoch_0001.pt,2026-06-18 00:00:00,0.6620,0.3341\n",
        encoding="utf-8",
    )

    h2_dir = stage_root / "h2_euclidean_ot"
    h2_dir.mkdir(parents=True)
    cfg_h2 = json.loads(json.dumps(base_cfg))
    cfg_h2.setdefault("bridge", {})["coupling_cost_composition"] = "appearance_only"
    (h2_dir / "config.json").write_text(json.dumps(cfg_h2), encoding="utf-8")
    (h2_dir / "full_eval_transfer").mkdir(parents=True)
    (h2_dir / "full_eval_transfer" / "clip_lpips_curve.csv").write_text(
        "epoch,epoch_int,checkpoint,timestamp,transfer_clip_style,transfer_content_lpips\n"
        "epoch_0001,1,epoch_0001.pt,2026-06-18 00:00:00,0.6697,0.3859\n"
        "epoch_0002,2,epoch_0002.pt,2026-06-18 00:05:00,0.6607,0.3256\n"
        "epoch_0003,3,epoch_0003.pt,2026-06-18 00:10:00,0.6604,0.3342\n",
        encoding="utf-8",
    )

    summary = {
        "stage": "stage1",
        "stage_root": str(stage_root),
        "runs": [
            {
                "name": "h0_vertical_fm",
                "run_dir": str(h0_dir),
                "config_path": str(h0_dir / "config.json"),
                "transfer_clip_style": 0.6620,
                "transfer_content_lpips": 0.3341,
                "objective_gap": 0.1121,
            }
        ],
        "best": {
            "name": "h0_vertical_fm",
            "run_dir": str(h0_dir),
            "style": 0.6620,
            "lpips": 0.3341,
            "gap": 0.1121,
        },
        "plan": {"stage": "stage1", "runs": ["h0_vertical_fm", "h2_euclidean_ot"], "purpose": "test"},
    }
    manifest = {"runs": summary["runs"]}
    (stage_root / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage_root / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = BACKFILL.backfill_stage_root(stage_root)

    assert payload["run_count"] == 2
    assert payload["close_result_diagnosis"]["status"] == "close_cluster"
    summary_after = json.loads((stage_root / "stage_summary.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in summary_after["runs"]]
    assert names == ["h0_vertical_fm", "h2_euclidean_ot"]
    h2_row = next(row for row in summary_after["runs"] if row["name"] == "h2_euclidean_ot")
    assert h2_row["best_transfer_clip_style"] == 0.6607
    assert h2_row["best_transfer_content_lpips"] == 0.3256
