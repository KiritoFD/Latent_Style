"""Fill currently missing timing records with measured or derived values."""
from __future__ import annotations

import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
SB_RUN = WORKSPACE_ROOT / "SchrodingerBridge" / "S-add__K-1_C-0_W-20_Col-0"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ours_train_metrics() -> dict[str, float]:
    log = SB_RUN / "logs" / "training_20260510_185442.csv"
    rows = list(csv.reader(log.open("r", encoding="utf-8-sig", newline="")))
    data = [r for r in rows[1:] if r and int(r[0]) <= 7]
    # This training log version writes the final four columns only:
    # compute_time_sec, epoch_time_sec, samples_seen, samples_per_sec
    total7 = sum(float(r[-3]) for r in data)
    avg7 = total7 / max(len(data), 1)
    return {
        "epochs_measured": len(data),
        "train_total_to_epoch7_sec": round(total7, 3),
        "train_avg_epoch_sec": round(avg7, 3),
        "samples_per_sec_epoch7": round(float(data[-1][-1]), 3),
    }


def main() -> int:
    ours = ours_train_metrics()
    samst_probe = read_json(RUN511_ROOT / "outputs" / "samst_timing_probe" / "summary.json")
    samst_train = samst_probe["runs"][0]
    samst_epoch1 = float(samst_train["elapsed_sec"])
    samst_epochs_target = 30
    styleid_strict = read_json(RUN511_ROOT / "outputs" / "styleid_750_strict" / "summary.json")
    styleid_photo = float(styleid_strict["runs"][0]["per_target"][0]["elapsed_sec"])
    payload = {
        "ours_epoch7_train": ours,
        "ours_epoch7_infer": {
            "generated_images": 750,
            "measured_elapsed_sec": 85.414,
            "sec_per_image": round(85.414 / 750.0, 6),
            "source": str((SB_RUN / "full_eval_timing_epoch7" / "summary.json").resolve()),
        },
        "samst_train_extrapolated": {
            "profile": "4g",
            "styles_measured": 5,
            "epochs_measured": 1,
            "epoch1_elapsed_sec_total": round(samst_epoch1, 3),
            "target_epochs": samst_epochs_target,
            "estimated_total_train_sec": round(samst_epoch1 * samst_epochs_target, 3),
            "note": "Measured 1 epoch across 5 styles with batch_size=1, train_images_per_style=16, then extrapolated linearly.",
        },
        "styleid_infer_estimated": {
            "training_free": True,
            "measured_photo_150_sec": round(styleid_photo, 3),
            "estimated_full_750_sec": round(styleid_photo * 5.0, 3),
            "estimated_sec_per_image": round((styleid_photo * 5.0) / 750.0, 6),
            "note": "Strict run actually generated photo target and reused/copied others; full 750 estimate assumes comparable cost across 5 targets.",
        },
    }
    docs_dir = RUN511_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "timing_filled.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Timing Filled Report",
        "",
        "This report supplements `timing_summary.md` with the currently missing values.",
        "",
        "## Ours",
        "",
        f"- Train to `epoch_0007`: `{payload['ours_epoch7_train']['train_total_to_epoch7_sec']:.3f}s` actual, from training log.",
        f"- Avg epoch time over epochs 1-7: `{payload['ours_epoch7_train']['train_avg_epoch_sec']:.3f}s`.",
        f"- Inference (`epoch_0007`, generation-only, 750 images): `{payload['ours_epoch7_infer']['measured_elapsed_sec']:.3f}s` actual.",
        f"- Inference sec/image: `{payload['ours_epoch7_infer']['sec_per_image']:.6f}`.",
        "",
        "## SaMST",
        "",
        f"- Train probe: `1` epoch across `5` styles took `{payload['samst_train_extrapolated']['epoch1_elapsed_sec_total']:.3f}s` actual.",
        f"- Extrapolated full train (`30` epochs, profile `4g`): `{payload['samst_train_extrapolated']['estimated_total_train_sec']:.3f}s`.",
        "- Strict 750 inference actual: `39.826s`, or `0.053101s/image`.",
        "",
        "## StyleID",
        "",
        "- Training is not needed; method is training-free.",
        f"- Measured actual generation for `photo` target (`150` images): `{payload['styleid_infer_estimated']['measured_photo_150_sec']:.3f}s`.",
        f"- Estimated fair full `750` inference: `{payload['styleid_infer_estimated']['estimated_full_750_sec']:.3f}s`.",
        f"- Estimated sec/image: `{payload['styleid_infer_estimated']['estimated_sec_per_image']:.6f}`.",
        "",
        "## Notes",
        "",
        "- `Ours` train time is taken from the existing training CSV log, not re-trained.",
        "- `Ours` inference time was freshly measured in generation-only mode for `epoch_0007`.",
        "- `SaMST` train time was measured for one epoch and extrapolated linearly, per your requested policy.",
        "- `StyleID` full-750 time is still an estimate derived from the actually measured `photo` target runtime.",
    ]
    md_out = docs_dir / "timing_filled_report.md"
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out)
    print(md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
