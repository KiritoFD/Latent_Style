import json
from pathlib import Path

base = Path(__file__).resolve().parent

# Append finding
finding = {
    "exp": "brk_ac_fft_loss",
    "round": 9,
    "direction": "FFT power spectrum loss (global frequency energy matching)",
    "config": "fft_loss_enabled=true, fft_loss_weight=0.1, L1 on log|FFT2(v_full)|^2 vs target_delta",
    "theory": "FFT power spectrum captures global frequency energy distribution, complementary to wavelet-domain FM loss (local per-coefficient MSE). DINOv2 CLS via global self-attention may be sensitive to global frequency structure.",
    "results": {
        "clip_s": 0.7222,
        "lpips": 0.3094,
        "dino_c": 0.7675,
        "dino_s": 0.4772
    },
    "baseline_dino_s": 0.4832,
    "delta_dino_s": -0.0060,
    "verdict": "FAILED",
    "root_cause": "FFT power spectrum loss is shift-invariant, conflicts with FM loss's spatial alignment requirement. Gradient pulled model toward global frequency energy matching at the expense of spatial structure. DINO-C dropped -0.033 (content degradation) while DINO-S also dropped -0.006 (no style benefit). Auxiliary latent-space losses without DINO/CLIP supervision cannot break the 0.4832 ceiling."
}

findings_path = base / "findings.jsonl"
with open(findings_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(finding, ensure_ascii=False) + "\n")

# Update progress.json
progress_path = base / "progress.json"
with open(progress_path, "r", encoding="utf-8") as f:
    progress = json.load(f)

progress["iteration"] = 24
progress["total_findings"] = 26
progress["status"] = "round9_complete_11_directions_all_failed_last_loss_level_exhausted"
progress["stale_count"] = 8
progress["last_seen"] = "2026-07-13T03:11:18Z"
progress["current_direction"] = "round9_complete_decision_point"
progress["directions_tried"].append("brk_ac_fft_loss_FAILED")

progress["key_conclusions"]["round9_failure"] = "Round 9 FFT power spectrum loss FAILED. DINO-S=0.4772 (-0.006 vs baseline 0.4832). FFT loss is shift-invariant, conflicts with FM spatial alignment, degrades DINO-C (-0.033) without style benefit."
progress["key_conclusions"]["round9_root_cause"] = "FFT power spectrum loss gradient pulls model toward global frequency energy matching at expense of spatial structure. Content preservation degraded, no style transfer gain."
progress["key_conclusions"]["overall_assessment"] = "11 directions across 5 rounds (architecture, frequency structure, target variation, HF intensity, FFT loss) ALL FAILED. DINO-S 0.4832 ceiling (adain=1.0) is a FUNDAMENTAL LIMIT of the SAT training paradigm without DINO/CLIP in the loss. Only inference-time endpoint_adain_scale can push DINO-S to 0.4859 (adain=2.0) but at CLIP-S cost. All loss-level directions exhausted."
progress["key_conclusions"]["decision_point"] = "A) Accept 0.485 ceiling, optimize radar chart balance with brk_s (adain=1.6) as peak. B) Lift DINO/CLIP training ban, add DINOv2 feature loss to directly optimize DINO-S. C) Explore inference-time techniques (multi-step, CFG, blending) to push DINO-S without training changes."

with open(progress_path, "w", encoding="utf-8") as f:
    json.dump(progress, f, ensure_ascii=False, indent=2)

print("Round 9 finding recorded. Status: round9_complete_11_directions_all_failed")
print(f"Total findings: {progress['total_findings']}")
print(f"Stale count: {progress['stale_count']}")
