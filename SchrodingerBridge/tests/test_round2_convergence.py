from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "experiments" / "report_round2_convergence.py"
SPEC = importlib.util.spec_from_file_location("report_round2_convergence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ROUND2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND2)


def test_build_convergence_payload_exposes_objective_gap_stop_signal() -> None:
    rows = [
        {"epoch": "epoch_0001", "transfer_clip_style": "0.6700", "transfer_content_lpips": "0.3800", "all_pairs_clip_style": "0.6700", "all_pairs_content_lpips": "0.3800"},
        {"epoch": "epoch_0002", "transfer_clip_style": "0.6620", "transfer_content_lpips": "0.3340", "all_pairs_clip_style": "0.6620", "all_pairs_content_lpips": "0.3340"},
        {"epoch": "epoch_0003", "transfer_clip_style": "0.6610", "transfer_content_lpips": "0.3360", "all_pairs_clip_style": "0.6610", "all_pairs_content_lpips": "0.3360"},
        {"epoch": "epoch_0004", "transfer_clip_style": "0.6605", "transfer_content_lpips": "0.3500", "all_pairs_clip_style": "0.6605", "all_pairs_content_lpips": "0.3500"},
        {"epoch": "epoch_0005", "transfer_clip_style": "0.6617", "transfer_content_lpips": "0.3440", "all_pairs_clip_style": "0.6617", "all_pairs_content_lpips": "0.3440"},
        {"epoch": "epoch_0006", "transfer_clip_style": "0.6611", "transfer_content_lpips": "0.3614", "all_pairs_clip_style": "0.6611", "all_pairs_content_lpips": "0.3614"},
    ]

    payload = ROUND2.build_convergence_payload(
        rows,
        curve_path=Path("/tmp/clip_lpips_curve.csv"),
        patience=4,
        min_epochs=4,
        flat_tail_window=4,
        flat_eps_style=0.005,
        flat_eps_lpips=0.018,
        objective_style_target=0.74,
        objective_lpips_target=0.30,
    )

    assert payload["converged"] is False
    assert payload["objective_best_epoch"] == "epoch_0002"
    assert payload["objective_epochs_since_best"] == 4
    assert payload["objective_patience_converged"] is True
    assert payload["stop_ready"] is True
    assert payload["stop_reason"] == "objective_gap_patience"
