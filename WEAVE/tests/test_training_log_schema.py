import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.training import append_training_log, initialize_training_log  # noqa: E402


def test_training_log_keeps_internal_dynamics_probes(tmp_path):
    log_file = tmp_path / "training.csv"
    initialize_training_log(log_file)
    append_training_log(
        log_file,
        {
            "loss_fm_spectral_ll": 0.12,
            "loss_fm_spectral_lh": 0.34,
            "loss_fm_spectral_hl": 0.56,
            "internal_probe_active": 1.0,
            "internal_probe_gate_mean": 0.175,
            "internal_probe_gate_delta": 0.0037,
            "internal_probe_shared_ll_hf_grad_ratio": 0.889,
            "internal_probe_route_shared_hf_grad_ratio": 0.251,
            "internal_probe_route_hf_head_grad_ratio": 0.359,
            "internal_probe_transition": 1.0,
            "internal_probe_transition_epoch": 4.0,
            "internal_probe_stop_requested": 1.0,
        },
        epoch=4,
    )

    row = list(csv.DictReader(log_file.open("r", encoding="utf-8", newline="")))[0]
    assert row["loss_fm_spectral_ll"] == "0.12"
    assert row["loss_fm_spectral_lh"] == "0.34"
    assert row["loss_fm_spectral_hl"] == "0.56"
    assert row["internal_probe_gate_mean"] == "0.175"
    assert row["internal_probe_shared_ll_hf_grad_ratio"] == "0.889"
    assert row["internal_probe_transition_epoch"] == "4.0"
    assert row["internal_probe_stop_requested"] == "1.0"


def test_training_log_keeps_style_delta_observability(tmp_path):
    log_file = tmp_path / "training.csv"
    initialize_training_log(log_file)
    append_training_log(
        log_file,
        {
            "loss": 1.0,
            "style_delta_basis_active": 1.0,
            "style_delta_basis_rank": 4.0,
            "style_delta_basis_abs": 0.12,
            "style_delta_weight_abs": 0.34,
            "style_delta_side_abs": 0.056,
            "style_delta_side_rms": 0.078,
            "style_delta_scale": 0.15,
        },
        epoch=3,
    )

    rows = list(csv.DictReader(log_file.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == "3"
    assert row["style_delta_basis_active"] == "1.0"
    assert row["style_delta_basis_rank"] == "4.0"
    assert row["style_delta_basis_abs"] == "0.12"
    assert row["style_delta_weight_abs"] == "0.34"
    assert row["style_delta_side_abs"] == "0.056"
    assert row["style_delta_side_rms"] == "0.078"
    assert row["style_delta_scale"] == "0.15"


def test_training_log_keeps_transport_stats_and_bridge_noise_observability(tmp_path):
    log_file = tmp_path / "training.csv"
    initialize_training_log(log_file)
    append_training_log(
        log_file,
        {
            "loss": 2.0,
            "transport_stats_active": 1.0,
            "transport_stats_bank_loaded": 1.0,
            "transport_stats_mode_terminal_affine": 1.0,
            "transport_stats_mode_normalized_solver": 0.0,
            "transport_stats_source_mean_abs": 0.41,
            "transport_stats_source_std_mean": 0.68,
            "transport_stats_target_mean_abs": 0.19,
            "transport_stats_target_std_mean": 0.82,
            "transport_stats_mean_delta": 0.39,
            "transport_stats_std_delta": 0.15,
            "transport_stats_valid_styles": 5.0,
            "transport_stats_missing_bank": 0.0,
            "training_bridge_noise_projection_active": 1.0,
            "training_bridge_noise_projection_mode_pure_vertical_flow": 1.0,
            "training_bridge_noise_projection_kernel": 5.0,
            "training_bridge_noise_projection_preserve_rms": 1.0,
            "training_bridge_noise_projection_pre_rms": 0.25,
            "training_bridge_noise_projection_post_rms": 0.24,
            "training_bridge_noise_projection_low_rms": 0.03,
            "training_bridge_noise_projection_high_rms": 0.21,
        },
        epoch=4,
    )

    rows = list(csv.DictReader(log_file.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == "4"
    assert row["transport_stats_active"] == "1.0"
    assert row["transport_stats_bank_loaded"] == "1.0"
    assert row["transport_stats_mode_terminal_affine"] == "1.0"
    assert row["transport_stats_mode_normalized_solver"] == "0.0"
    assert row["transport_stats_source_mean_abs"] == "0.41"
    assert row["transport_stats_source_std_mean"] == "0.68"
    assert row["transport_stats_target_mean_abs"] == "0.19"
    assert row["transport_stats_target_std_mean"] == "0.82"
    assert row["transport_stats_mean_delta"] == "0.39"
    assert row["transport_stats_std_delta"] == "0.15"
    assert row["transport_stats_valid_styles"] == "5.0"
    assert row["transport_stats_missing_bank"] == "0.0"
    assert row["training_bridge_noise_projection_active"] == "1.0"
    assert row["training_bridge_noise_projection_mode_pure_vertical_flow"] == "1.0"
    assert row["training_bridge_noise_projection_kernel"] == "5.0"
    assert row["training_bridge_noise_projection_preserve_rms"] == "1.0"
    assert row["training_bridge_noise_projection_pre_rms"] == "0.25"
    assert row["training_bridge_noise_projection_post_rms"] == "0.24"
    assert row["training_bridge_noise_projection_low_rms"] == "0.03"
    assert row["training_bridge_noise_projection_high_rms"] == "0.21"


def test_training_log_keeps_topogate_ot_probes(tmp_path):
    log_file = tmp_path / "training.csv"
    initialize_training_log(log_file)
    append_training_log(
        log_file,
        {
            "loss": 3.0,
            "ot_topogate_probe_active": 1.0,
            "ot_topogate_descriptor_blocks": 4.0,
            "ot_topogate_complexity_cost_mean": 0.42,
            "ot_topogate_complexity_cost_var": 0.08,
            "ot_topogate_complexity_term_var": 0.22,
            "ot_topogate_content_complexity_mean": 0.61,
            "ot_topogate_target_complexity_mean": 0.57,
            "ot_latent_affinity_cost_mean": 1.23,
            "ot_latent_affinity_cost_var": 0.34,
            "ot_latent_affinity_term_var": 0.05,
            "ot_total_cost_matrix_var": 0.18,
        },
        epoch=5,
    )

    rows = list(csv.DictReader(log_file.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == "5"
    assert row["ot_topogate_probe_active"] == "1.0"
    assert row["ot_topogate_descriptor_blocks"] == "4.0"
    assert row["ot_topogate_complexity_cost_mean"] == "0.42"
    assert row["ot_topogate_complexity_cost_var"] == "0.08"
    assert row["ot_topogate_complexity_term_var"] == "0.22"
    assert row["ot_topogate_content_complexity_mean"] == "0.61"
    assert row["ot_topogate_target_complexity_mean"] == "0.57"
    assert row["ot_latent_affinity_cost_mean"] == "1.23"
    assert row["ot_latent_affinity_cost_var"] == "0.34"
    assert row["ot_latent_affinity_term_var"] == "0.05"
    assert row["ot_total_cost_matrix_var"] == "0.18"
