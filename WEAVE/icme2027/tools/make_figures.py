"""Generate the data figures of the ICME paper from icme2027/data/.

    python tools/make_figures.py

Writes to figures/:
  fig_sandwich.pdf     Fig. 1  (data/main_table.csv, data/artfid_d5.csv)
  fig_spectral.pdf     Fig. 2  (data/probe_frequency.csv, data/probe_separability.csv)
  fig_qualitative.pdf  Fig. 4  (figures/fig_teaser_comparison.png, re-laid out with close-ups)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

RED = "#C8102E"
GREEN = "#2E8B57"
FAIL = "#C0392B"
BOARDS = ["D5-512", "P2A-256", "R5-WikiArt"]
FAMILY = {
    "SD-Turbo": "diffusion", "StyleAligned": "diffusion", "Z-STAR": "diffusion",
    "StyleShot": "diffusion", "StyleID": "diffusion",
    "CUT": "learned", "SaMST": "learned", "SaMam": "learned", "StyTR-2": "learned",
    "AesPA-Net": "learned", "Seedream 4.5": "api", "Latent-WCT": "analytic", "WEAVE": "ours",
}
MARK = {
    "diffusion": ("s", "#2B7A4B", "training-free diffusion"),
    "learned": ("o", "#2C5D9E", "learned"),
    "api": ("D", "#8A5A00", "commercial API"),
    "analytic": ("^", "#6B6B6B", "analytic"),
    "ours": ("*", RED, "WEAVE (ours)"),
}


def load_main():
    with open(DATA / "main_table.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = ("dino_s", "clip_s", "lpips", "dino_c")
    return {(r["method"], r["board"]): {k: float(r[k]) for k in keys} for r in rows}


def failures(t, m, b):
    """'' if m passes every sandwich test on board b; otherwise 'S', 'C', or 'SC'."""
    r, idt, tgt = t[(m, b)], t[("Identity", b)], t[("Target Style", b)]
    style = r["dino_s"] <= idt["dino_s"] or r["clip_s"] <= idt["clip_s"]
    content = r["lpips"] >= tgt["lpips"] or r["dino_c"] <= tgt["dino_c"]
    return ("S" if style else "") + ("C" if content else "")


# Label placement for the D5 scatter: (dx, dy, ha) in points, or ("data", x, y, ha) for a
# leader line to an explicit position; tuned to avoid overlaps.
OFFSETS = {
    "SD-Turbo": (-3, 6, "right"), "StyleAligned": (0, -7, "center"), "Z-STAR": (4, -6, "left"),
    "StyleShot": (5, 1, "left"), "StyleID": (5, 2, "left"), "CUT": ("data", 0.585, 0.432, "right"),
    "SaMST": (5, 0, "left"), "SaMam": (5, -5, "left"), "StyTR-2": (-5, -1, "right"),
    "AesPA-Net": (-5, 3, "right"), "Seedream 4.5": ("data", 0.43, 0.452, "right"),
    "Latent-WCT": (-5, 0, "right"), "WEAVE": (0, 7, "center"),
}


def panel_scatter(ax, t):
    b = "D5-512"
    idt, tgt = t[("Identity", b)], t[("Target Style", b)]
    x_tgt, y_idt = 1 - tgt["lpips"], idt["dino_s"]
    xmax, ymin, ymax = 1.06, 0.25, 0.72
    ax.add_patch(patches.Rectangle((x_tgt, y_idt), xmax - x_tgt, ymax - y_idt,
                                   facecolor="#EAF4EC", edgecolor="none", zorder=0))
    ax.axhline(y_idt, color="#4A4A4A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(x_tgt, color="#8A4F3D", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, y_idt + 0.006, "IDT style", fontsize=6.3, color="#4A4A4A", va="bottom")
    ax.text(x_tgt - 0.012, ymin + 0.01, "TGT content", fontsize=6.3, color="#8A4F3D",
            rotation=90, va="bottom", ha="right")
    ax.text(0.99, ymax - 0.012, "valid region", fontsize=6.3, color=GREEN, ha="right", va="top",
            style="italic")
    ax.text(x_tgt + 0.015, ymax - 0.006, r"TGT above (D-S 1.0) $\uparrow$", fontsize=5.8,
            color="#8A4F3D", va="top")
    ax.scatter([1.0], [y_idt], marker="D", s=16, color="black", zorder=5, clip_on=False)
    ax.annotate("IDT", (1.0, y_idt), xytext=(4, -1), textcoords="offset points",
                ha="left", va="center", fontsize=6.3)
    for m, fam in FAMILY.items():
        r = t[(m, b)]
        x, y = 1 - r["lpips"], r["dino_s"]
        marker, color, _ = MARK[fam]
        valid = failures(t, m, b) == ""
        size = 95 if fam == "ours" else 20
        ax.scatter([x], [y], marker=marker, s=size, zorder=6 if fam == "ours" else 4,
                   facecolor=color if valid else "white", edgecolor=color,
                   linewidth=0.5 if fam == "ours" else 0.9)
        spec = OFFSETS[m]
        name = "WEAVE" if m == "WEAVE" else m.replace(" 4.5", "")
        kw = dict(va="center", fontsize=6.6 if fam == "ours" else 6.0,
                  color=RED if fam == "ours" else "#222222",
                  fontweight="bold" if fam == "ours" else "normal")
        if spec[0] == "data":
            ax.annotate(name, (x, y), xytext=spec[1:3], textcoords="data", ha=spec[3],
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777",
                                        shrinkA=1, shrinkB=2.5), **kw)
        else:
            ax.annotate(name, (x, y), xytext=spec[:2], textcoords="offset points", ha=spec[2], **kw)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"content fidelity, 1 $-$ LPIPS $\rightarrow$")
    ax.set_ylabel(r"style, DINO-S $\rightarrow$")
    ax.set_title("(a) D5-512: where methods land", loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=MARK[k][0], color=MARK[k][1], markersize=4,
                          markerfacecolor=MARK[k][1], label=MARK[k][2])
               for k in ("diffusion", "learned", "api", "analytic")]
    handles.append(plt.Line2D([], [], ls="", marker="o", color="#555555", markersize=4,
                              markerfacecolor="white", label="hollow: fails a D5 test"))
    ax.legend(handles=handles, loc="lower right", fontsize=5.6, frameon=False,
              handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)


def panel_audit(ax, t):
    fails = {m: [failures(t, m, b) for b in BOARDS] for m in FAMILY}
    order = sorted(FAMILY, key=lambda m: (-sum(f == "" for f in fails[m]), m != "WEAVE", m.lower()))
    n = len(order)
    for i, m in enumerate(order):
        y = n - 1 - i
        if m == "WEAVE":
            ax.add_patch(patches.Rectangle((-0.55, y - 0.45), 3.4, 0.9, facecolor="#FBE9EB",
                                           edgecolor="none", zorder=0))
        for j, f in enumerate(fails[m]):
            if f == "":
                ax.scatter(j, y, s=16, marker="o", color=GREEN, zorder=3)
            else:
                ax.scatter(j, y, s=18, marker="x", color=FAIL, linewidths=1.0, zorder=3)
                ax.text(j + 0.2, y, f, fontsize=5.2, color=FAIL, va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([m.replace(" 4.5", "") for m in reversed(order)], fontsize=6.0)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "WEAVE":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["D5", "P2A", "R5"], fontsize=6.3)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.55, 2.85)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    baselines = [m for m in FAMILY if m != "WEAVE"]
    n_fail = sum(any(f != "" for f in fails[m]) for m in baselines)
    ax.set_title(f"(b) {n_fail} of {len(baselines)} baselines fail", loc="left", pad=12)
    ax.text(1.15, -1.35, "S: style not above IDT    C: content beyond TGT",
            fontsize=5.4, color="#444444", ha="center", va="top")


def panel_artfid(ax):
    with open(DATA / "artfid_d5.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    colors = {"IDT": "#333333", "WEAVE": RED, "SaMam": "#2C5D9E", "Seedream 4.5": "#8A5A00",
              "TGT": "#9A9A9A"}
    names = [r["method"] for r in rows]
    vals = [float(r["artfid"]) for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs, vals, width=0.68, color=[colors[nm] for nm in names], zorder=2)
    for x, r, v in zip(xs, rows, vals):
        if r["ci_low"]:
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
            top = hi
        else:
            top = v
        ax.text(x, top + 12, f"{v:.0f}", ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["IDT", "WEAVE", "SaMam", "Seedream$^{*}$", "TGT"], fontsize=6.0,
                       rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 760)
    ax.set_ylabel(r"ArtFID $\leftarrow$")
    ax.set_title("(c) ArtFID ranks the copy first", loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)


def fig_sandwich():
    t = load_main()
    fig = plt.figure(figsize=(7.16, 2.0))
    gs = GridSpec(1, 3, width_ratios=[2.9, 1.55, 2.0], wspace=0.42, figure=fig)
    panel_scatter(fig.add_subplot(gs[0]), t)
    panel_audit(fig.add_subplot(gs[1]), t)
    panel_artfid(fig.add_subplot(gs[2]))
    fig.savefig(FIG / "fig_sandwich.pdf")
    plt.close(fig)


def fig_spectral(alpha=0.3):
    bands = ["LL", "LH", "HL", "HH"]
    with open(DATA / "probe_frequency.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    energy = {b: np.mean([float(r[f"energy_{b}"]) for r in rows]) for b in bands}
    total = sum(energy.values())
    share = [100 * energy[b] / total for b in bands]
    se = [100 * np.std([float(r[f"share_{b}"]) for r in rows], ddof=1) / np.sqrt(len(rows))
          for b in bands]
    # Bound from Proposition 1: anchored LL energy <= alpha^2 * direct LL energy.
    hf = energy["LH"] + energy["HL"] + energy["HH"]
    bound = 100 * alpha ** 2 * energy["LL"] / (alpha ** 2 * energy["LL"] + hf)
    with open(DATA / "probe_separability.csv", encoding="utf-8") as f:
        sep = {r["band"]: (float(r["between_within_ratio"]), float(r["ratio_sem"]))
               for r in csv.DictReader(f)}
    colors = ["#2C5D9E", "#E6A532", "#E6A532", "#C8102E"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.3), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    xs = np.arange(4)
    ax.bar(xs, share, yerr=se, color=colors, width=0.66, error_kw=dict(lw=0.6, capsize=1.5))
    for x, v in zip(xs, share):
        ax.text(x, v + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.0)
    ax.plot([-0.33, 0.33], [bound, bound], color="white", lw=1.1, ls=(0, (2, 1.2)))
    ax.annotate(rf"anchored: $\leq${bound:.0f}", xy=(0.33, bound), xytext=(0.8, 46),
                fontsize=5.8, arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333"))
    ax.set_xticks(xs)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 84)
    ax.set_ylabel("gradient energy (%)")
    ax.set_title("(a) who drives the update", loc="left", pad=3, fontsize=7.5)
    ax = axes[1]
    vals = [sep[b][0] for b in bands]
    ax.bar(xs, vals, yerr=[sep[b][1] for b in bands], color=colors, width=0.66,
           error_kw=dict(lw=0.6, capsize=1.5))
    for x, v, b in zip(xs, vals, bands):
        ax.text(x + 0.05, v + sep[b][1] + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("style separability")
    ax.set_title("(b) who carries style", loc="left", pad=3, fontsize=7.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig_spectral.pdf")
    plt.close(fig)
    return share, bound


def fig_qualitative():
    src = Image.open(FIG / "fig_teaser_comparison.png").convert("RGB")
    width = src.size[0] / 8
    titles = ["IDT", "AdaIN", "StyleID", "CUT", "SD-Turbo", "SaMam", "Seedream 4.5", "WEAVE (ours)"]
    crops = {t: src.crop((int(i * width) + 6, 32, int((i + 1) * width) - 6, 266))
             for i, t in enumerate(titles)}
    keep = ["IDT", "StyleID", "CUT", "SD-Turbo", "SaMam", "Seedream 4.5", "WEAVE (ours)"]
    box = (60, 60, 140, 140)
    fig, axes = plt.subplots(1, 9, figsize=(7.16, 0.98),
                             gridspec_kw={"wspace": 0.04, "width_ratios": [1] * 7 + [1.0, 1.0]})
    for ax, name in zip(axes, keep):
        ax.imshow(crops[name])
        if name in ("IDT", "WEAVE (ours)"):
            ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, ec="white", lw=0.8, ls=(0, (2, 1))))
        ax.set_title(name, fontsize=6.8, pad=2, color=RED if "WEAVE" in name else "black",
                     fontweight="bold" if "WEAVE" in name else "normal")
    for ax, name in zip(axes[7:], ["IDT", "WEAVE (ours)"]):
        ax.imshow(crops[name].crop(box).resize((240, 240), Image.LANCZOS))
        ax.set_title(("IDT" if name == "IDT" else "WEAVE") + r" $\times$3", fontsize=6.8, pad=2,
                     color=RED if "WEAVE" in name else "black")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#999999")
    for s in axes[6].spines.values():
        s.set_color(RED)
        s.set_linewidth(1.0)
    fig.savefig(FIG / "fig_qualitative.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_sandwich()
    share, bound = fig_spectral()
    fig_qualitative()
    print("band shares (%):", [round(s, 1) for s in share], "| LL bound with anchored endpoint:",
          round(bound, 1))
