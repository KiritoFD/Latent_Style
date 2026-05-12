"""Build Ours (SB) vs SaMST timing comparison."""
import csv
import json
from pathlib import Path

WORKSPACE = Path(r"G:\GitHub\Latent_Style")
SB_RUN = WORKSPACE / "SchrodingerBridge" / "S-add__K-1_C-0_W-20_Col-0"
RUN511 = WORKSPACE / "run_511"

# --- SB training ---
log = SB_RUN / "logs" / "training_20260510_185442.csv"
rows = list(csv.reader(log.open("r", encoding="utf-8-sig", newline="")))
data = [r for r in rows[1:] if r and int(r[0]) <= 7]
total7 = sum(float(r[-3]) for r in data)
avg7 = total7 / max(len(data), 1)

# --- SB inference ---
sb_infer_json = SB_RUN / "full_eval_timing_epoch7" / "summary.json"
sb_infer = json.loads(sb_infer_json.read_text(encoding="utf-8"))

# --- SaMST ---
samst = json.loads((RUN511 / "outputs" / "samst_timing_probe" / "summary.json").read_text(encoding="utf-8"))
samst_ep1 = samst["runs"][0]["elapsed_sec"]

# --- StyleID ---
sid = json.loads((RUN511 / "outputs" / "styleid_750_strict" / "summary.json").read_text(encoding="utf-8"))
sid_photo = sid["runs"][0]["per_target"][0]["elapsed_sec"]

# Print comparison
print("=" * 60)
print("Ours (SB) vs SaMST Timing Comparison")
print("=" * 60)

print(f"\n--- Training ---")
print(f"Ours (SB):  {len(data)} epochs measured, total = {total7:.1f}s ({total7/60:.1f} min)")
print(f"            avg per epoch = {avg7:.1f}s")
print(f"            samples/sec   = {float(data[-1][-1]):.1f}")
print(f"SaMST:      1 epoch probe  = {samst_ep1:.1f}s")
print(f"            5 styles, 16 imgs/style, bs=1")
print(f"            extrapolated 30 epochs = {samst_ep1*30:.1f}s ({samst_ep1*30/60:.1f} min)")

print(f"\n--- Inference (750 images) ---")
print(f"Ours (SB):  85.4s  (0.114 s/img)")
print(f"SaMST:      39.8s  (0.053 s/img)")
print(f"StyleID:    ~{sid_photo*5:.0f}s estimated (0.804 s/img)")

print(f"\n--- Summary Table ---")
print(f"{'Method':<12} {'Train(s)':<12} {'Train(min)':<12} {'Infer(s)':<12} {'Infer/img':<12}")
print(f"{'Ours (SB)':<12} {total7:<12.1f} {total7/60:<12.1f} {'85.4':<12} {'0.114':<12}")
print(f"{'SaMST':<12} {samst_ep1*30:<12.1f} {samst_ep1*30/60:<12.1f} {'39.8':<12} {'0.053':<12}")
print(f"{'StyleID':<12} {'N/A':<12} {'N/A':<12} {f'{sid_photo*5:.0f}':<12} {'0.804':<12}")

# Save JSON
payload = {
    "ours_sb": {
        "train_epochs_measured": len(data),
        "train_total_sec": round(total7, 3),
        "train_avg_epoch_sec": round(avg7, 3),
        "train_samples_per_sec": round(float(data[-1][-1]), 3),
        "infer_750_sec": 85.414,
        "infer_sec_per_image": round(85.414 / 750, 6),
    },
    "samst": {
        "train_epoch1_sec": round(samst_ep1, 3),
        "train_extrapolated_30_epochs_sec": round(samst_ep1 * 30, 3),
        "train_profile": "5 styles, 16 imgs/style, bs=1",
        "infer_750_sec": 39.826,
        "infer_sec_per_image": round(39.826 / 750, 6),
    },
    "styleid": {
        "training_free": True,
        "infer_photo_150_sec": round(sid_photo, 3),
        "infer_est_750_sec": round(sid_photo * 5, 3),
        "infer_sec_per_image": round((sid_photo * 5) / 750, 6),
    },
}
out = RUN511 / "timing_comparison_ours_vs_samst.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\nSaved: {out}")
