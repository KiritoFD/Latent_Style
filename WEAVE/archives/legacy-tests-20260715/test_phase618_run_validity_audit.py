from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "audit_phase618_run_validity.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_phase618_run_validity", AUDIT_PATH)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def test_old_base_style_sweep_is_confounded() -> None:
    result = AUDIT.audit_phase618_run_validity(
        config_path=ROOT / "docs" / "experiments" / "2026-06-18-remote-h1-e18-diagnosis" / "remote_config.json",
        variant_spec_path=ROOT / "docs" / "experiments" / "2026-06-18-style-sweep-base-audit" / "style_sweep_variant_spec.json",
        variant_name="r8_linear_code_map_lowrank_both",
    )

    assert result["artifact_status"] == "confounded"
    assert result["suite"] == "stage3_style_r1_r10_old_base"
    assert result["effect_contract"] == "runtime_real"
    assert any(issue["code"] == "old_base_style_sweep_confounded" for issue in result["issues"])


def test_plain_path_distill_is_training_only_by_design() -> None:
    result = AUDIT.audit_phase618_run_validity(
        config_path=ROOT / "docs" / "experiments" / "2026-06-18-stage1-lowrank-rerun-audit" / "baseline_h1_lowrank_config.json",
        variant_spec_path=ROOT / "docs" / "experiments" / "2026-06-18-stage1-lowrank-distill-contract-probe" / "variant_spec.json",
        variant_name="h1_plain_path_distill_0p50",
    )

    assert result["artifact_status"] == "valid"
    assert result["suite"] == "plain_path_distill_lowrank"
    assert result["effect_contract"] == "training_only_by_design"
    assert result["repaired_lowrank_base"] is True


def test_repaired_bold_blend_variant_is_runtime_and_training_real() -> None:
    result = AUDIT.audit_phase618_run_validity(
        config_path=ROOT / "docs" / "experiments" / "2026-06-18-stage1-lowrank-rerun-audit" / "baseline_h1_lowrank_config.json",
        variant_spec_path=ROOT / "docs" / "experiments" / "2026-06-18-bold-eval-graph-preflight" / "variant_spec.json",
        variant_name="r11_linear_blend_0p00",
    )

    assert result["artifact_status"] == "valid"
    assert result["suite"] == "bold_r11_r16_repaired_lowrank"
    assert result["effect_contract"] == "runtime_and_training_real"
    assert result["matrix_verdict"]["runtime_strength_bucket"] == "weak_runtime_lever"


def test_topogate_run_without_descriptor_metric_is_stale(tmp_path: Path) -> None:
    run_dir = tmp_path / "h5_topogate_attention"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    config = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
            "semantic_self_topology_gate": True,
            "semantic_self_topology_blend": 1.0,
            "num_res_blocks": 4,
        },
        "bridge": {
            "coupling_structure_cost_mode": "topogate_attention_gw",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with (logs_dir / "training_20260618_000000.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss_total"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "loss_total": 0.5})

    result = AUDIT.audit_phase618_run_validity(run_dir=run_dir)

    assert result["artifact_status"] == "stale"
    assert any(issue["code"] == "topogate_descriptor_metric_missing" for issue in result["issues"])


def test_disabled_topology_gate_with_nonzero_blend_is_suspect(tmp_path: Path) -> None:
    run_dir = tmp_path / "r11_linear_blend_0p30"
    run_dir.mkdir(parents=True)
    config = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
            "semantic_self_topology_gate": False,
            "semantic_self_topology_blend": 0.3,
        },
        "bridge": {
            "coupling_structure_cost_mode": "self_affinity_gw",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = AUDIT.audit_phase618_run_validity(run_dir=run_dir)

    assert result["artifact_status"] == "suspect"
    assert any(issue["code"] == "topology_blend_gate_disabled" for issue in result["issues"])


def test_zero_init_style_injection_probe_is_marked_suspect() -> None:
    result = AUDIT.audit_phase618_run_validity(
        config_path=ROOT / "docs" / "experiments" / "2026-06-18-stage1-lowrank-rerun-audit" / "baseline_h1_lowrank_config.json",
        variant_spec_path=ROOT / "docs" / "experiments" / "2026-06-18-style-injection-live-init-probe" / "variant_spec.json",
        variant_name="z1_body_mixed_zero_init",
    )

    assert result["suite"] == "style_injection_live_init_probe"
    assert result["artifact_status"] == "suspect"
    assert result["matrix_verdict"]["runtime_strength_bucket"] == "exact_noop"
    assert any(issue["code"] == "style_injection_zero_init_exact_noop" for issue in result["issues"])


def test_live_init_style_injection_probe_is_runtime_real() -> None:
    result = AUDIT.audit_phase618_run_validity(
        config_path=ROOT / "docs" / "experiments" / "2026-06-18-stage1-lowrank-rerun-audit" / "baseline_h1_lowrank_config.json",
        variant_spec_path=ROOT / "docs" / "experiments" / "2026-06-18-style-injection-live-init-probe" / "variant_spec.json",
        variant_name="z2_body_mixed_live_init",
    )

    assert result["suite"] == "style_injection_live_init_probe"
    assert result["artifact_status"] == "valid"
    assert result["effect_contract"] == "runtime_real"
    assert result["matrix_verdict"]["runtime_strength_bucket"] == "moderate_runtime_lever"


def test_run_without_unified_convergence_packet_surfaces_info_issue(tmp_path: Path) -> None:
    run_dir = tmp_path / "h0_vertical_fm"
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "full_eval_transfer"
    logs_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    config = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
            "semantic_self_topology_gate": True,
            "semantic_self_topology_blend": 1.0,
        },
        "bridge": {
            "coupling_structure_cost_mode": "self_affinity_gw",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with (logs_dir / "training_20260618_000000.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss_total"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "loss_total": 0.5})
    (eval_dir / "round2_convergence.json").write_text(
        json.dumps(
            {
                "row_count": 6,
                "best_epoch": "epoch_0001",
                "newest_epoch": "epoch_0006",
                "since_best": 5,
                "since_last_pareto": 4,
                "converged": False,
            }
        ),
        encoding="utf-8",
    )

    result = AUDIT.audit_phase618_run_validity(run_dir=run_dir)

    assert result["artifact_status"] == "valid"
    assert any(issue["code"] == "convergence_stop_contract_split" for issue in result["issues"])
