import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.training import append_training_log, initialize_training_log  # noqa: E402


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
