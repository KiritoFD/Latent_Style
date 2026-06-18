from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "tools" / "experiments" / "collect_round2_eval_curve.py"
SPEC = importlib.util.spec_from_file_location("collect_round2_eval_curve", COLLECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)

BACKFILL_PATH = ROOT / "tools" / "experiments" / "backfill_eval_clip_schema.py"
BACKFILL_SPEC = importlib.util.spec_from_file_location("backfill_eval_clip_schema", BACKFILL_PATH)
assert BACKFILL_SPEC is not None and BACKFILL_SPEC.loader is not None
BACKFILL = importlib.util.module_from_spec(BACKFILL_SPEC)
BACKFILL_SPEC.loader.exec_module(BACKFILL)

RUN_EVAL_PATH = ROOT / "src" / "utils" / "run_evaluation.py"
RUN_EVAL_SPEC = importlib.util.spec_from_file_location("run_evaluation", RUN_EVAL_PATH)
assert RUN_EVAL_SPEC is not None and RUN_EVAL_SPEC.loader is not None
RUN_EVAL = importlib.util.module_from_spec(RUN_EVAL_SPEC)
RUN_EVAL_SPEC.loader.exec_module(RUN_EVAL)


def test_eval_curve_keeps_clip_lpips_clipt_and_delta_idt(tmp_path: Path) -> None:
    eval_dir = tmp_path / "run" / "full_eval" / "epoch_0001"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": "epoch_0001.pt",
                "timestamp": "2026-06-19 00:00:00",
                "analysis": {
                    "style_transfer_ability": {
                        "clip_style": 0.73,
                        "clip_s_delta_idt": -0.04,
                        "clip_t": 0.21,
                        "content_lpips": 0.36,
                    },
                    "all_pairs_overview": {
                        "clip_style": 0.75,
                        "clip_s_delta_idt": -0.02,
                        "clip_t": 0.24,
                        "content_lpips": 0.31,
                    },
                    "identity_reconstruction": {
                        "clip_style": 0.77,
                        "clip_t": 0.25,
                        "content_lpips": 0.08,
                    },
                },
                "idt_baselines": {
                    "clip_style_global": 0.77,
                    "clip_t_global": 0.25,
                },
                "timings_sec": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = COLLECTOR._scan_rows(tmp_path / "run", eval_subdir="full_eval")
    out_csv = tmp_path / "curve.csv"
    COLLECTOR._write_csv(out_csv, rows)

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["transfer_clip_style"] == "0.73"
    assert row["transfer_content_lpips"] == "0.36"
    assert row["transfer_clip_t"] == "0.21"
    assert row["transfer_clip_s_delta_idt"] == "-0.04"
    assert row["all_pairs_clip_s_delta_idt"] == "-0.02"


def test_backfill_metrics_drops_repeated_clipt_idt_and_adds_clips_delta(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "run" / "full_eval" / "epoch_0001"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "metrics.csv").write_text(
        "src_style,tgt_style,src_image,gen_image,content_lpips,clip_dir,clip_style,clip_t,clip_t_idt,clip_t_delta_idt,clip_content,clip_image_vector\n"
        "Rococo,Rococo,a.jpg,a_to_Rococo.png,0.1,0.0,0.80,0.30,0.30,0.0,0.7,0.1 0.2\n"
        "Ukiyo_e,Rococo,b.jpg,b_to_Rococo.png,0.4,0.0,0.72,0.22,0.30,-0.08,0.6,0.3 0.4\n",
        encoding="utf-8",
    )
    (epoch_dir / "summary.json").write_text(
        json.dumps(
            {
                "analysis": {
                    "all_pairs_overview": {},
                    "style_transfer_ability": {},
                    "identity_reconstruction": {},
                },
                "matrix_breakdown": {},
                "metrics_note": {},
            }
        ),
        encoding="utf-8",
    )

    BACKFILL.backfill_eval_root(tmp_path / "run" / "full_eval")

    with (epoch_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (epoch_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        fieldnames = next(csv.reader(handle))

    assert "clip_t_idt" not in fieldnames
    assert "clip_t_delta_idt" not in fieldnames
    assert "clip_s_delta_idt" in fieldnames
    assert abs(float(rows[1]["clip_s_delta_idt"]) - (0.72 - BACKFILL.OLD_IDT_CLIP_STYLE)) < 1e-8

    summary = json.loads((epoch_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["idt_baselines"]["clip_style_global"] == BACKFILL.OLD_IDT_CLIP_STYLE
    assert summary["idt_baselines"]["clip_style_delta_mode"] == "fixed_old_idt"
    assert summary["idt_baselines"]["clip_style_eval_identity_by_target_style"]["Rococo"] == 0.8
    assert summary["analysis"]["style_transfer_ability"]["clip_style"] == 0.72


def test_620_eval_target_dino_bank_loads_one_patch_sequence_per_style(tmp_path: Path) -> None:
    cache = tmp_path / "dino.pt"
    torch.save(
        {
            "rows": [
                {"style": "Rococo", "stem": "r0_latent_ema"},
                {"style": "Ukiyo_e", "stem": "u0_latent_ema"},
                {"style": "Rococo", "stem": "r1_latent_ema"},
            ],
            "cls_embeddings": torch.arange(12, dtype=torch.float32).view(3, 4),
            "patch_embeddings": torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4),
        },
        cache,
    )

    bank = RUN_EVAL._load_eval_target_dino_bank(str(cache), ["Rococo", "Ukiyo_e"])

    assert sorted(bank) == [0, 1]
    assert bank[0]["cls"].shape == (4,)
    assert bank[0]["patches"].shape == (2, 4)
    assert torch.equal(bank[0]["cls"], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(bank[1]["cls"], torch.tensor([4.0, 5.0, 6.0, 7.0]))
