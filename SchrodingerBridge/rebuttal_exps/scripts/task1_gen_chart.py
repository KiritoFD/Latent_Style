"""Generate generalization comparison chart from task1 results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen\exp\rebuttal\task1_generalization")
results_path = OUTPUT_DIR / "task1_generalization_results.json"
results = json.loads(results_path.read_text(encoding="utf-8"))

# Filter to only OK results with valid metrics
valid = [r for r in results if r.get("status") == "OK" and r.get("mean_dino_s") is not None]
failed = [r for r in results if r.get("status") != "OK"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = [r["label"] for r in valid]
dino_s = [r["mean_dino_s"] for r in valid]
lpips = [r["mean_lpips"] for r in valid]

# Short labels for display
short_labels = {
    "baseline_sd15_1level": "SD1.5 VAE\n1-level Haar\n(baseline)",
    "sd15_2level_haar": "SD1.5 VAE\n2-level Haar",
    "sdxl_1level": "SDXL VAE\n1-level Haar",
    "taesd_1level": "TAESD VAE\n1-level Haar",
}
display_labels = [short_labels.get(l, l) for l in labels]

colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

# DINO-S chart
ax1 = axes[0]
bars1 = ax1.bar(range(len(display_labels)), dino_s, color=colors[:len(display_labels)])
ax1.set_xticks(range(len(display_labels)))
ax1.set_xticklabels(display_labels, fontsize=9)
ax1.set_ylabel("DINO-S (content preservation)", fontsize=11)
ax1.set_title("DINO-S across VAE / Wavelet configs", fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=dino_s[0], color="green", linestyle="--", alpha=0.5, label=f"baseline={dino_s[0]:.3f}")
for bar, val in zip(bars1, dino_s):
    delta = val - dino_s[0]
    delta_str = f"{val:.3f}" if bar == bars1[0] else f"{val:.3f}\n({delta:+.3f})"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             delta_str, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.legend(fontsize=8)

# LPIPS chart
ax2 = axes[1]
bars2 = ax2.bar(range(len(display_labels)), lpips, color=colors[:len(display_labels)])
ax2.set_xticks(range(len(display_labels)))
ax2.set_xticklabels(display_labels, fontsize=9)
ax2.set_ylabel("LPIPS (lower = better)", fontsize=11)
ax2.set_title("LPIPS across VAE / Wavelet configs", fontsize=12)
ax2.set_ylim(0, max(lpips) * 1.2)
ax2.axhline(y=lpips[0], color="green", linestyle="--", alpha=0.5, label=f"baseline={lpips[0]:.3f}")
for bar, val in zip(bars2, lpips):
    delta = val - lpips[0]
    delta_str = f"{val:.3f}" if bar == bars2[0] else f"{val:.3f}\n({delta:+.3f})"
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             delta_str, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.legend(fontsize=8)

# Add FLUX failure note
if failed:
    note = "FLUX VAE: load failed (gated repo, needs HF token)\n16-channel latent requires channel adaptation"
    fig.text(0.5, -0.02, note, ha='center', fontsize=9, style='italic', color='red')

plt.tight_layout()
chart_path = OUTPUT_DIR / "task1_generalization_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved: {chart_path}")
