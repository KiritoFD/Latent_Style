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
