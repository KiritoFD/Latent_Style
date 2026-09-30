"""Baseline convergence dashboard for AAAI 2027 paper.

Generates a comprehensive dashboard visualizing all 12 baseline methods:
1. CLIP-S vs LPIPS scatter (SDEdit series connected as parameter sweep)
2. Delta_idt ranking bar chart
3. SaMam training convergence curve (8 intermediate checkpoints, wikiart5)
4. Training time vs CLIP-S (training-required methods)
5. Art FID bar chart with training time annotations
6. Inference-only methods bubble chart

Design principles (per user requirements):
- Clear visual separation for IDT point
- SDEdit series connected by lines to show convergence trajectory
- All intermediate training points evaluated and plotted to ensure convergence
- Art FID bar chart with training time written inside bars
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Load unified results
RESULTS_PATH = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\unified_results.json")
OUTPUT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

# Method metadata: (display_name, category, color, marker)
METHOD_META = {
    "identity":      ("Identity",       "baseline",         "#2c2c2c", "X"),
    "adain":         ("AdaIN",          "classical-inf",    "#1f77b4", "s"),
    "wct_vgg19":     ("WCT (VGG19)",    "classical-inf",    "#17becf", "D"),
    "sdturbo":       ("SD-Turbo",       "diffusion-inf",    "#9467bd", "v"),
    "sdedit_str0.10":("SDEdit s=0.10", "diffusion-inf-sweep", "#ff7f0e", "o"),
    "sdedit_str0.20":("SDEdit s=0.20", "diffusion-inf-sweep", "#ff7f0e", "o"),
    "sdedit_str0.35":("SDEdit s=0.35", "diffusion-inf-sweep", "#ff7f0e", "o"),
    "sdedit_str0.40":("SDEdit s=0.40", "diffusion-inf-sweep", "#ff7f0e", "o"),
    "styleid":       ("StyleID",        "diffusion-inf",    "#d62728", "^"),
    "cut":           ("CUT",            "gan-train",        "#2ca02c", "P"),
    "samst":         ("SaMST",          "mamba-train",      "#8c564b", "p"),
    "samam":         ("SaMam",          "mamba-train",      "#e377c2", "h"),
}

# Training times (min) on distinct5_512
TRAIN_TIMES = {
    "samst": 39.5,
    "samam": 209.2,
    "cut": 322.6,
    "wct_vgg19": 0.0,
}

# ============================================================
# SaMam 512 scratch convergence data (wikiart5 dataset)
# 8 intermediate checkpoints, step 1k-10k
# Source: samam_wsl_mamba_512_scratch_clean_silent_b1_20k/formal_eval_750/
# Training rate: ~1.767 sec/step (from 7k-10k segment WALL_SECONDS=5302.35)
# ============================================================
SAMAM_CURVE = {
    "steps":      [1000,  3000,  5000,   6000,   7000,   8000,   9000,   10000],
    "clip_style": [0.7255, 0.7869, 0.7912, 0.7881, 0.7848, 0.7879, 0.7868, 0.7851],
    "lpips":      [0.5560, 0.3430, 0.2833, 0.2646, 0.2461, 0.1906, 0.1661, 0.1643],
    "art_fid":    [None,  None,  289.13, None,   None,   None,   None,   254.39],
    # Estimated training time (min): step * 1.767 / 60
    "train_min":  [17.7,  53.0,  88.4,   147.3,  176.7,  206.2,  235.6,  294.5],
}
# Note: train_min corrected - step 5000 = 5000*1.767/60 = 147.25 min
SAMAM_CURVE["train_min"] = [s * 1.767 / 60 for s in SAMAM_CURVE["steps"]]

# Create dashboard: 2x3 grid
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# ============ Plot 1: CLIP-S vs LPIPS scatter ============
ax1 = fig.add_subplot(gs[0, 0])

# Plot SDEdit sweep as connected line first
sdedit_methods = ["sdedit_str0.10", "sdedit_str0.20", "sdedit_str0.35", "sdedit_str0.40"]
sdedit_x = [results[m]["content_lpips"] for m in sdedit_methods]
sdedit_y = [results[m]["clip_style"] for m in sdedit_methods]
ax1.plot(sdedit_x, sdedit_y, '-', color='#ff7f0e', alpha=0.6, linewidth=2, zorder=2)
for i, m in enumerate(sdedit_methods):
    ax1.annotate(f"s={[0.10,0.20,0.35,0.40][i]}", (sdedit_x[i], sdedit_y[i]),
                 textcoords="offset points", xytext=(8, 4), fontsize=7, color='#ff7f0e')

# Plot all methods as scatter
for method, data in results.items():
    name, cat, color, marker = METHOD_META.get(method, (method, "other", "#7f7f7f", "o"))
    lpips = data["content_lpips"]
    clip_s = data["clip_style"]
    size = 250 if method == "identity" else 120
    edgecolor = "black" if method == "identity" else color
    linewidth = 2.0 if method == "identity" else 0.8
    ax1.scatter(lpips, clip_s, c=color, marker=marker, s=size,
                edgecolors=edgecolor, linewidths=linewidth, label=name, zorder=3)
    # Annotate (offset to avoid overlap)
    offset_x, offset_y = 6, 4
    if method == "identity":
        offset_x, offset_y = -30, -12
    elif method == "sdturbo":
        offset_x, offset_y = 8, -10
    elif method == "adain":
        offset_x, offset_y = 8, -8
    ax1.annotate(name, (lpips, clip_s), textcoords="offset points",
                 xytext=(offset_x, offset_y), fontsize=7.5, alpha=0.9)

# Identity reference lines
idt_clip = results["identity"]["clip_style"]
ax1.axhline(y=idt_clip, color='#2c2c2c', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(x=0.0, color='#2c2c2c', linestyle='--', alpha=0.3, linewidth=1)

ax1.set_xlabel("Content LPIPS (lower = better preservation)", fontsize=10)
ax1.set_ylabel("CLIP-S Style (higher = more style)", fontsize=10)
ax1.set_title("(a) Style-Content Trade-off (12 baselines, distinct5_512)", fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-0.02, 0.82)
ax1.set_ylim(0.60, 0.84)

# ============ Plot 2: Delta_idt ranking ============
ax2 = fig.add_subplot(gs[0, 1])

methods_sorted = sorted(results.keys(), key=lambda m: results[m]["clip_s_delta_idt"], reverse=True)
deltas = [results[m]["clip_s_delta_idt"] for m in methods_sorted]
names_sorted = [METHOD_META.get(m, (m, "", "#7f7f7f", ""))[0] for m in methods_sorted]
colors_sorted = [METHOD_META.get(m, (m, "", "#7f7f7f", ""))[2] for m in methods_sorted]

bars = ax2.barh(range(len(methods_sorted)), deltas, color=colors_sorted, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(methods_sorted)))
ax2.set_yticklabels(names_sorted, fontsize=8)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.axvline(x=results["identity"]["clip_s_delta_idt"], color='#2c2c2c', linestyle='--', alpha=0.5, label='Identity Δ')
ax2.set_xlabel("Δ_idt (CLIP-S gain over Identity)", fontsize=10)
ax2.set_title("(b) Style Transfer Strength Ranking", fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='x')
ax2.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, deltas)):
    x_pos = val + (0.003 if val >= 0 else -0.003)
    ha = 'left' if val >= 0 else 'right'
    ax2.text(x_pos, i, f"{val:+.4f}", va='center', ha=ha, fontsize=7.5)

# ============ Plot 3: SaMam convergence curve (NEW) ============
ax3 = fig.add_subplot(gs[0, 2])

steps = SAMAM_CURVE["steps"]
clip_s = SAMAM_CURVE["clip_style"]
lpips = SAMAM_CURVE["lpips"]

# Plot CLIP-S on left Y-axis
color_clip = '#e377c2'
ax3.plot(steps, clip_s, 'o-', color=color_clip, linewidth=2, markersize=7, label='CLIP-S Style', zorder=3)
ax3.set_xlabel("Training Step", fontsize=10)
ax3.set_ylabel("CLIP-S Style", fontsize=10, color=color_clip)
ax3.tick_params(axis='y', labelcolor=color_clip)
ax3.set_ylim(0.70, 0.80)

# Plot LPIPS on right Y-axis
ax3_r = ax3.twinx()
color_lpips = '#1f77b4'
ax3_r.plot(steps, lpips, 's--', color=color_lpips, linewidth=2, markersize=6, label='Content LPIPS', zorder=3)
ax3_r.set_ylabel("Content LPIPS", fontsize=10, color=color_lpips)
ax3_r.tick_params(axis='y', labelcolor=color_lpips)
ax3_r.set_ylim(0.10, 0.60)

# Mark convergence zone (after step 5000, CLIP-S stabilizes)
ax3.axvspan(5000, 10000, alpha=0.08, color='green', label='Convergence zone')
ax3.axvline(x=5000, color='green', linestyle=':', alpha=0.5, linewidth=1)

# Annotate Art FID at step 5000 and 10000
for i, step in enumerate(steps):
    art_fid = SAMAM_CURVE["art_fid"][i]
    if art_fid is not None:
        ax3.annotate(f'ArtFID={art_fid:.1f}', (step, clip_s[i]),
                     textcoords="offset points", xytext=(8, 12),
                     fontsize=7.5, color='#2c2c2c', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

# Convergence annotation
ax3.annotate('CLIP-S converged\n(Δ<0.01 after 5k)', xy=(5000, 0.7912),
             xytext=(1500, 0.745), fontsize=8, color='green',
             arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))

ax3.set_title("(c) SaMam Training Convergence (wikiart5, 8 checkpoints)",
              fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.set_xlim(0, 11000)

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

# ============ Plot 4: Training time vs CLIP-S ============
ax4 = fig.add_subplot(gs[1, 0])

train_methods = ["samst", "samam", "cut"]
train_names = [METHOD_META[m][0] for m in train_methods]
train_colors = [METHOD_META[m][2] for m in train_methods]
train_markers = [METHOD_META[m][3] for m in train_methods]
train_times = [TRAIN_TIMES[m] for m in train_methods]
train_clips = [results[m]["clip_style"] for m in train_methods]
train_lpips = [results[m]["content_lpips"] for m in train_methods]

for i, m in enumerate(train_methods):
    ax4.scatter(train_times[i], train_clips[i], c=train_colors[i], marker=train_markers[i],
                s=200, edgecolors='black', linewidth=0.8, zorder=3)
    ax4.annotate(f"{train_names[i]}\n(LPIPS={train_lpips[i]:.3f})",
                 (train_times[i], train_clips[i]),
                 textcoords="offset points", xytext=(10, -5), fontsize=8)

# Add identity reference
ax4.axhline(y=results["identity"]["clip_style"], color='#2c2c2c', linestyle='--', alpha=0.4, label='Identity CLIP-S')
ax4.axhline(y=results["adain"]["clip_style"], color='#1f77b4', linestyle=':', alpha=0.4, label='AdaIN CLIP-S')

ax4.set_xlabel("Training Time (min)", fontsize=10)
ax4.set_ylabel("CLIP-S Style", fontsize=10)
ax4.set_title("(d) Training Cost vs Style Transfer (distinct5_512)", fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.2)
ax4.legend(fontsize=8, loc='lower right')
ax4.set_xlim(0, 360)
ax4.set_ylim(0.60, 0.74)

# ============ Plot 5: Art FID bar chart with training time (NEW) ============
ax5 = fig.add_subplot(gs[1, 1])

# SaMam Art FID at 2 evaluated checkpoints (wikiart5)
artfid_steps = [5000, 10000]
artfid_values = [289.13, 254.39]
artfid_train_min = [s * 1.767 / 60 for s in artfid_steps]  # [147.3, 294.5]
artfid_labels = [f"SaMam\nstep {s}" for s in artfid_steps]
bar_colors = ['#e377c2', '#c51b8a']

bars = ax5.bar(range(len(artfid_steps)), artfid_values, color=bar_colors,
               edgecolor='black', linewidth=0.8, width=0.55, zorder=3)

# Write training time INSIDE the bars
for i, (bar, val, t_min) in enumerate(zip(bars, artfid_values, artfid_train_min)):
    # Art FID value on top
    ax5.text(bar.get_x() + bar.get_width()/2, val + 3, f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Training time INSIDE the bar
    ax5.text(bar.get_x() + bar.get_width()/2, val/2, f'train\n{t_min:.1f} min',
             ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Add improvement annotation
improvement = (artfid_values[0] - artfid_values[1]) / artfid_values[0] * 100
ax5.annotate('', xy=(1, artfid_values[1] + 2), xytext=(1, artfid_values[0] - 2),
             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax5.text(1.3, (artfid_values[0] + artfid_values[1])/2,
         f'↓{improvement:.1f}%\n(better)', fontsize=8, color='red', va='center')

ax5.set_xticks(range(len(artfid_steps)))
ax5.set_xticklabels(artfid_labels, fontsize=9)
ax5.set_ylabel("Art FID (lower = better)", fontsize=10)
ax5.set_title("(e) Art FID vs Training Time (SaMam, wikiart5)", fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.2, axis='y')
ax5.set_ylim(0, max(artfid_values) * 1.25)

# Note about SaMST/CUT
ax5.text(0.5, 0.95, 'Note: SaMST/CUT Art FID = N/A\n(needs --eval_enable_art_fid re-run)',
         transform=ax5.transAxes, fontsize=7, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============ Plot 6: Inference-only methods comparison ============
ax6 = fig.add_subplot(gs[1, 2])

inf_methods = ["adain", "wct_vgg19", "sdturbo", "sdedit_str0.10", "sdedit_str0.20",
               "sdedit_str0.35", "sdedit_str0.40", "styleid"]
inf_names = [METHOD_META[m][0] for m in inf_methods]
inf_clips = [results[m]["clip_style"] for m in inf_methods]
inf_lpips = [results[m]["content_lpips"] for m in inf_methods]
inf_colors = [METHOD_META[m][2] for m in inf_methods]
inf_markers = [METHOD_META[m][3] for m in inf_methods]

# Bubble chart: x=CLIP-S, y=LPIPS, size=1/LPIPS (bigger=better content)
sizes = [np.exp(-l*2) * 800 + 80 for l in inf_lpips]
for i, m in enumerate(inf_methods):
    ax6.scatter(inf_clips[i], inf_lpips[i], c=inf_colors[i], marker=inf_markers[i],
                s=sizes[i], alpha=0.7, edgecolors='black', linewidth=0.6, zorder=3)
    ax6.annotate(inf_names[i], (inf_clips[i], inf_lpips[i]),
                 textcoords="offset points", xytext=(8, 4), fontsize=7.5)

# Identity reference
ax6.axhline(y=0.0, color='#2c2c2c', linestyle='--', alpha=0.3)
ax6.axvline(x=results["identity"]["clip_style"], color='#2c2c2c', linestyle='--', alpha=0.3)

ax6.set_xlabel("CLIP-S Style (higher = more style)", fontsize=10)
ax6.set_ylabel("Content LPIPS (lower = better)", fontsize=10)
ax6.set_title("(f) Inference-only Methods (bubble ∝ content preservation)", fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.2)
ax6.set_xlim(0.65, 0.84)
ax6.set_ylim(-0.02, 0.82)

# Legend for categories
cat_patches = [
    mpatches.Patch(color='#1f77b4', label='Classical (AdaIN, WCT)'),
    mpatches.Patch(color='#ff7f0e', label='Diffusion sweep (SDEdit)'),
    mpatches.Patch(color='#d62728', label='Diffusion (StyleID, SD-Turbo)'),
    mpatches.Patch(color='#2ca02c', label='GAN (CUT)'),
    mpatches.Patch(color='#e377c2', label='Mamba (SaMam, SaMST)'),
    mpatches.Patch(color='#2c2c2c', label='Identity baseline'),
]
fig.legend(handles=cat_patches, loc='lower center', ncol=6, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='gray')

fig.suptitle("Baseline Convergence Dashboard — distinct5_512 (12 methods) + SaMam Convergence (wikiart5)",
             fontsize=14, fontweight='bold', y=0.995)

output_path = OUTPUT_DIR / "baseline_convergence_dashboard.png"
plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
print(f"Dashboard saved: {output_path}")
plt.close()
