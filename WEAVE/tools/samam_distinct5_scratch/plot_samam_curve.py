"""Plot SaMam HF CLIP curve (81 checkpoints) for inspection."""
import csv
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).parent / "curve_metrics_hf.csv"
OUT_DIR = Path(__file__).parent

# Baseline horizontal references (from unified_results.json)
REFS = {
    "Identity":    (0.6933, 0.0000, "#888888", "--"),
    "AdaIN":       (0.6679, 0.7425, "#aa6666", ":"),
    "WCT":         (0.7063, 0.6348, "#cc8800", ":"),
    "CUT":         (0.7137, 0.3743, "#cc0066", "--"),
    "SDEdit@0.40": (0.7934, 0.4826, "#0066cc", ":"),
    "StyleID":     (0.8223, 0.5523, "#0000cc", "--"),
    "SeeDream":    (0.7198, 0.4767, "#00cccc", "--"),
    "SaMST":       (0.6183, 0.7490, "#9933cc", ":"),
}


def parse_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            img_dir = r["image_dir"]
            # extract step from path: step_000250 or last
            m = re.search(r"step_(\d+)", img_dir)
            if m:
                step = int(m.group(1))
            elif "/last/" in img_dir or img_dir.rstrip("/").endswith("/last"):
                step = 21000  # plot 'last' beyond 20000
            else:
                continue
            rows.append({
                "step": step,
                "clip_style": float(r["clip_style"]),
                "lpips": float(r["content_lpips"]),
                "clip_content": float(r["clip_content"]),
            })
    rows.sort(key=lambda x: x["step"])
    return rows


def main():
    rows = parse_rows()
    print(f"Parsed {len(rows)} rows")
    steps = [r["step"] for r in rows]
    cs = [r["clip_style"] for r in rows]
    lp = [r["lpips"] for r in rows]
    cc = [r["clip_content"] for r in rows]

    # Stats
    cs_max = max(cs)
    cs_max_step = steps[cs.index(cs_max)]
    lp_min = min(lp)
    lp_min_step = steps[lp.index(lp_min)]
    print(f"CLIP-S max = {cs_max:.4f} @ step {cs_max_step}")
    print(f"LPIPS min  = {lp_min:.4f} @ step {lp_min_step}")
    print(f"Final step {steps[-2]}: CLIP-S={cs[-2]:.4f}, LPIPS={lp[-2]:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # ---- Panel 1: CLIP-S (style) ----
    ax = axes[0]
    ax.plot(steps, cs, "-o", color="#cc0000", lw=2, ms=4, label="SaMam (HF CLIP)")
    ax.axhline(y=cs_max, color="#cc0000", ls=":", alpha=0.4,
               label=f"SaMam max={cs_max:.4f} @ {cs_max_step}")
    for name, (c, l, col, ls) in REFS.items():
        ax.axhline(y=c, color=col, ls=ls, alpha=0.55, label=f"{name}={c:.4f}")
    ax.set_ylabel("CLIP-S (style) ↑", fontsize=12)
    ax.set_title("SaMam HF CLIP convergence curve (81 ckpts, step 250-20000)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_ylim(0.50, 0.85)

    # ---- Panel 2: LPIPS ----
    ax = axes[1]
    ax.plot(steps, lp, "-s", color="#0066cc", lw=2, ms=4, label="SaMam LPIPS")
    ax.axhline(y=lp_min, color="#0066cc", ls=":", alpha=0.4,
               label=f"SaMam min={lp_min:.4f} @ {lp_min_step}")
    for name, (c, l, col, ls) in REFS.items():
        ax.axhline(y=l, color=col, ls=ls, alpha=0.55, label=f"{name}={l:.4f}")
    ax.set_ylabel("LPIPS ↓", fontsize=12)
    ax.set_title("SaMam LPIPS (content preservation) — lower is better", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0.0, 0.9)

    # ---- Panel 3: CLIP content ----
    ax = axes[2]
    ax.plot(steps, cc, "-^", color="#008800", lw=2, ms=4, label="SaMam CLIP-content")
    ax.axhline(y=1.0, color="#888888", ls=":", alpha=0.4, label="perfect=1.0")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("CLIP-content ↑", fontsize=12)
    ax.set_title("SaMam CLIP-content (content fidelity vs. source)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.55, 1.02)

    fig.tight_layout()
    out = OUT_DIR / "samam_hf_curve.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also plot zoomed CLIP-S only with all reference lines, more legible
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(steps, cs, "-o", color="#cc0000", lw=2.2, ms=5, label="SaMam (HF CLIP)")
    ax2.scatter([cs_max_step], [cs_max], color="#cc0000", s=180, zorder=5,
                edgecolors="black", linewidths=2,
                label=f"SaMam BEST={cs_max:.4f} @ {cs_max_step}")
    ax2.scatter([steps[-2]], [cs[-2]], color="#ff6600", s=180, zorder=5, marker="*",
                edgecolors="black", linewidths=2,
                label=f"SaMam FINAL={cs[-2]:.4f} @ {steps[-2]}")
    for name, (c, l, col, ls) in REFS.items():
        ax2.axhline(y=c, color=col, ls=ls, alpha=0.7, lw=1.6,
                    label=f"{name} = {c:.4f}")
    ax2.set_xlabel("Training step", fontsize=12)
    ax2.set_ylabel("CLIP-S (style)", fontsize=12)
    ax2.set_title("SaMam CLIP-S vs all baselines — never exceeds 0.60, well below Identity (0.6933)",
                  fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center right", fontsize=9, ncol=1)
    ax2.set_ylim(0.50, 0.85)
    fig2.tight_layout()
    out2 = OUT_DIR / "samam_clip_style_vs_baselines.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
