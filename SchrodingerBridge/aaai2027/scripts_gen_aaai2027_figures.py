from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


COLORS = {
    "content": "#DCE8F6",
    "style": "#F7E7B2",
    "model": "#FBE8B5",
    "loss": "#ECECEC",
    "transport": "#E5D8F1",
    "output": "#F7DCDC",
    "line": "#263547",
    "train": "#8A6A12",
    "muted": "#6E6E6E",
}


def _box(ax, xy, wh, text, fc, ec="#2F2F2F", fs=9, weight="normal", text_color="#222222"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=text_color,
        linespacing=1.15,
    )
    return patch


def _arrow(ax, start, end, color=None, style="-", lw=1.5, rad=0.0, label=None, fs=8):
    color = color or COLORS["line"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        ax.text(x, y + 0.035, label, ha="center", va="center", fontsize=fs, color=color)


def _grid(ax, xy, wh, rows=3, cols=5, colors=None, alpha=1.0):
    x, y = xy
    w, h = wh
    colors = colors or ["#DDE9F6", "#AFC6E4", "#6D91C2", "#F0F3FA"]
    for r in range(rows):
        for c in range(cols):
            idx = (r * 3 + c * 5 + r + c) % len(colors)
            ax.add_patch(
                FancyBboxPatch(
                    (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                    w / cols * 0.92,
                    h / rows * 0.86,
                    boxstyle="round,pad=0.001,rounding_size=0.002",
                    linewidth=0.35,
                    edgecolor="#506070",
                    facecolor=colors[idx],
                    alpha=alpha,
                )
            )


def make_framework() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Latent Bridge Matching: Tokenizer-Controlled Latent Transport with SA-SWD",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Outer panels.
    _box(ax, (0.01, 0.085), (0.635, 0.86), "", "#FFFFFF", ec="#1E2D3C")
    _box(ax, (0.66, 0.085), (0.33, 0.86), "", "#FFFFFF", ec="#1E2D3C")
    ax.text(0.327, 0.925, "(a) Inference and training graph", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.825, 0.925, "(b) Terminal matching and tokenizer probes", ha="center", fontsize=11, fontweight="bold")

    # Left inputs.
    _box(ax, (0.03, 0.68), (0.075, 0.19), "Target\nstyle id\ns", COLORS["style"], fs=8.2)
    _grid(ax, (0.042, 0.71), (0.050, 0.045), rows=2, cols=5, colors=["#F5D98F", "#A8C5E6", "#8B6BB4", "#E9B3B3"])
    _box(ax, (0.03, 0.40), (0.075, 0.20), "Content\nimage\nx", COLORS["content"], fs=8.2)
    _grid(ax, (0.044, 0.435), (0.047, 0.065), rows=4, cols=4, colors=["#C8D7E8", "#7195B6", "#DCE7D2", "#7FA064"])

    # Style conditioning lane.
    _box(ax, (0.14, 0.775), (0.085, 0.09), "1\nTokenizer\nT_phi(s)", COLORS["style"], fs=7.8)
    _box(ax, (0.245, 0.775), (0.095, 0.09), "2\nprototype\np_s", COLORS["style"], fs=7.8)
    _box(ax, (0.36, 0.775), (0.105, 0.09), "3\nshared atoms\nalpha_s A", COLORS["style"], fs=7.8)
    _grid(ax, (0.386, 0.785), (0.053, 0.028), rows=2, cols=4, colors=["#F4D97D", "#85A8D8", "#C78396", "#91C49A"])
    _box(ax, (0.485, 0.775), (0.090, 0.09), "4\nspatial prior\nP_s", COLORS["style"], fs=7.8)
    _box(ax, (0.595, 0.775), (0.040, 0.09), "5\nc_s", COLORS["style"], fs=7.8)
    for a, b in [((0.105, 0.775), (0.14, 0.820)), ((0.225, 0.820), (0.245, 0.820)), ((0.340, 0.820), (0.360, 0.820)), ((0.465, 0.820), (0.485, 0.820)), ((0.575, 0.820), (0.595, 0.820))]:
        _arrow(ax, a, b, color=COLORS["train"])

    # Content and bridge lane.
    _box(ax, (0.14, 0.505), (0.085, 0.105), "1\nVAE encode\nz_0", COLORS["content"], fs=7.8)
    _box(ax, (0.14, 0.385), (0.085, 0.085), "2\nlatent cache\ntrain only", "#F7E2A0", fs=7.8)
    _box(ax, (0.245, 0.445), (0.095, 0.14), "3\nmini-batch OT\nSWD cost\n+ Sinkhorn", "#FFF0C8", fs=7.5)
    _box(ax, (0.36, 0.445), (0.105, 0.14), "4\nstochastic bridge\n(z_0, z_1) -> z_t", "#FFF0C8", fs=7.5)
    _box(ax, (0.485, 0.445), (0.090, 0.14), "5\ntarget\nvelocity\nu_t", "#FFF0C8", fs=7.5)
    for a, b in [((0.105, 0.50), (0.14, 0.552)), ((0.225, 0.552), (0.245, 0.515)), ((0.340, 0.515), (0.36, 0.515)), ((0.465, 0.515), (0.485, 0.515))]:
        _arrow(ax, a, b, color=COLORS["train"])

    # Transport core.
    _box(ax, (0.215, 0.205), (0.340, 0.205), "", "#EFF6FF", ec="#1E5A91")
    ax.text(0.385, 0.382, "LANCET latent vector-field renderer", ha="center", fontsize=9.2, color="#1E5A91", fontweight="bold")
    _box(ax, (0.230, 0.285), (0.048, 0.070), "bridge\nstate\nz_t", "#DCE8F6", fs=7.2)
    _box(ax, (0.293, 0.285), (0.120, 0.070), "semantic trunk\nSA -> SA -> ...", "#DCE8F6", fs=7.2)
    _box(ax, (0.428, 0.285), (0.048, 0.070), "32x32\nlift", "#DCE8F6", fs=7.2)
    _box(ax, (0.491, 0.285), (0.042, 0.070), "skip\nfuse", "#DCE8F6", fs=7.2)
    _box(ax, (0.300, 0.225), (0.205, 0.042), "style/time modulation + routed prior", "#EFE2FA", ec="#7B52A1", fs=7.4)
    for a, b in [((0.225, 0.552), (0.230, 0.330)), ((0.615, 0.775), (0.385, 0.350)), ((0.530, 0.775), (0.385, 0.350)), ((0.530, 0.445), (0.385, 0.350))]:
        _arrow(ax, a, b, color=COLORS["train"], style="--", rad=-0.1)
    _arrow(ax, (0.278, 0.320), (0.293, 0.320))
    _arrow(ax, (0.413, 0.320), (0.428, 0.320))
    _arrow(ax, (0.476, 0.320), (0.491, 0.320))

    # Decode lane.
    _box(ax, (0.575, 0.235), (0.052, 0.155), "Decode\nK-step\nEuler\n+\nVAE", "#E9F6E1", ec="#4B7C45", fs=7.3)
    _arrow(ax, (0.533, 0.320), (0.575, 0.320))
    _box(ax, (0.575, 0.145), (0.052, 0.060), "output\nI_hat", COLORS["output"], ec="#A94E66", fs=7.5)
    _arrow(ax, (0.601, 0.235), (0.601, 0.205), color="#4B7C45")

    # Losses.
    _box(ax, (0.075, 0.120), (0.115, 0.060), "Flow matching\nL_FM", COLORS["loss"], fs=7.8)
    _box(ax, (0.265, 0.120), (0.115, 0.060), "SA-SWD\nL_sem-swd", COLORS["loss"], fs=7.8)
    _box(ax, (0.445, 0.120), (0.110, 0.060), "Kinetic\nL_kin", COLORS["loss"], fs=7.8)
    _arrow(ax, (0.175, 0.385), (0.132, 0.180), color=COLORS["train"], style="--")
    _arrow(ax, (0.385, 0.225), (0.322, 0.180), color=COLORS["train"], style="--")
    _arrow(ax, (0.465, 0.285), (0.500, 0.180), color=COLORS["train"], style="--")

    # Right panel: SA-SWD.
    _box(ax, (0.675, 0.832), (0.120, 0.034), "1) Semantic-Aligned SWD", "#123A63", ec="#123A63", fs=8.0, weight="bold", text_color="#FFFFFF")
    _box(ax, (0.675, 0.545), (0.305, 0.275), "", "#F8FBFF", ec="#1E5A91")
    ax.text(0.728, 0.790, "generated endpoint\npatches", ha="center", fontsize=7.5)
    ax.text(0.825, 0.790, "semantic\nbins", ha="center", fontsize=7.5)
    ax.text(0.925, 0.790, "target-style\npatches", ha="center", fontsize=7.5)
    _grid(ax, (0.696, 0.632), (0.065, 0.10), rows=4, cols=5, colors=["#EDF3FA", "#B7CAE5", "#7E9CC7"])
    _grid(ax, (0.895, 0.632), (0.065, 0.10), rows=4, cols=5, colors=["#F4F0FA", "#B7A4D8", "#7B5CAE"])
    for i, (yy, cc) in enumerate([(0.725, "#80A9D6"), (0.675, "#95C79A"), (0.625, "#D6869C")]):
        _box(ax, (0.795, yy - 0.018), (0.070, 0.034), f"bin {i+1}", "#F8FBFF", ec=cc, fs=6.9)
        _arrow(ax, (0.761, 0.682), (0.795, yy), color=cc, rad=0.15 * (i - 1))
        _arrow(ax, (0.865, yy), (0.895, 0.682), color=cc, rad=0.15 * (1 - i))
    _box(ax, (0.705, 0.565), (0.245, 0.045), "sorted projections match distributions inside each bin", "#EFF6FF", ec="#5B86B9", fs=7.7)

    # Right panel: tokenizer diagnostics.
    _box(ax, (0.675, 0.477), (0.165, 0.034), "2) Executable representation probes", "#123A63", ec="#123A63", fs=8.0, weight="bold", text_color="#FFFFFF")
    _box(ax, (0.675, 0.175), (0.305, 0.285), "", "#FFFFFF", ec="#1E5A91")
    _box(ax, (0.692, 0.327), (0.072, 0.073), "token code\nrank 3.986\ncos 0.015", "#DDEFE4", fs=7.1)
    _box(ax, (0.795, 0.327), (0.074, 0.073), "executed\nresidual\ncos 0.725", "#DDEFE4", fs=7.1)
    _box(ax, (0.900, 0.327), (0.052, 0.073), "source\nstyle\nANOVA", "#DDEFE4", fs=7.1)
    _arrow(ax, (0.764, 0.363), (0.795, 0.363), color="#315A42")
    _arrow(ax, (0.869, 0.363), (0.900, 0.363), color="#315A42")
    _box(ax, (0.692, 0.220), (0.128, 0.066), "finding:\nraw codes separate", "#F5F5F5", fs=7.3)
    _box(ax, (0.850, 0.220), (0.128, 0.066), "next design:\ncarrier + risk gate", "#F5F5F5", fs=7.3)
    _arrow(ax, (0.820, 0.253), (0.850, 0.253), color="#315A42")

    # Legend.
    legend_y = 0.038
    for x, label, color in [
        (0.04, "conditioning / tokenizer", COLORS["style"]),
        (0.20, "OT / bridge target", "#FFF0C8"),
        (0.36, "transport core", "#DCE8F6"),
        (0.51, "decode / output", "#E9F6E1"),
        (0.65, "loss / diagnostics", COLORS["loss"]),
    ]:
        _box(ax, (x, legend_y - 0.012), (0.025, 0.024), "", color, fs=1)
        ax.text(x + 0.032, legend_y, label, va="center", fontsize=7.6)
    ax.plot([0.86, 0.90], [legend_y, legend_y], color=COLORS["line"], lw=1.6)
    ax.text(0.905, legend_y, "inference", va="center", fontsize=7.6)
    ax.plot([0.94, 0.975], [legend_y, legend_y], color=COLORS["train"], lw=1.6, ls="--")
    ax.text(0.978, legend_y, "training", va="center", fontsize=7.6, ha="left")

    fig.tight_layout(pad=0.2)
    fig.savefig(ROOT / "framework_figure.pdf")
    fig.savefig(ROOT / "framework_figure.png", dpi=220)
    plt.close(fig)


def _read_distinct5():
    path = REPO / "SchrodingerBridge" / "docs" / "experiments" / "distinct5_512_20260602" / "tables" / "clip_style_vs_1lpips_points.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def make_eval_landscape() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.20, wspace=0.32)

    # Panel A: historical strict-750.
    hist = [
        ("Ours e7", 0.7161, 0.4514, "#D9543F", "o"),
        ("SaMST", 0.7194, 0.4664, "#7A4FA2", "o"),
        ("S2WAT", 0.7139, 0.5263, "#F0A22E", "o"),
        ("StyleID", 0.7597, 0.7497, "#3A98D4", "o"),
        ("AdaIN", 0.7130, 0.6298, "#3EBF76", "o"),
    ]
    ax = axes[0]
    idt_hist = 0.6593609295606613
    ax.axhline(idt_hist, color="#9A9A9A", lw=1.2, ls="--", zorder=1)
    ax.text(0.205, idt_hist + 0.004, "idt", fontsize=7.5, color="#6A6A6A")
    for name, style, lpips, color, marker in hist:
        ax.scatter(1 - lpips, style, s=58, color=color, edgecolor="black", linewidth=0.7, marker=marker, zorder=3)
        dx, dy = (0.007, 0.004)
        if name == "StyleID":
            dx, dy = (-0.03, 0.006)
        if name == "Ours e7":
            dx, dy = (0.004, -0.012)
        ax.text(1 - lpips + dx, style + dy, name, fontsize=8)
    ax.set_title("(a) Historical strict-750", fontsize=10, fontweight="bold")
    ax.set_xlabel("1 - LPIPS")
    ax.set_ylabel("CLIP-style")
    ax.set_xlim(0.20, 0.60)
    ax.set_ylim(0.62, 0.775)
    ax.grid(True, alpha=0.25)

    # Panel B: WikiArt512/SaMAM convergence reference.
    idt_w512 = 0.7815262297789255
    samam512 = [
        (1000, 0.725534, 0.555994),
        (3000, 0.786911, 0.342996),
        (5000, 0.791244, 0.283292),
        (6000, 0.788131, 0.264603),
        (7000, 0.784850, 0.246103),
        (8000, 0.787916, 0.190641),
        (9000, 0.786826, 0.166118),
        (10000, 0.785089, 0.164336),
    ]
    samam256 = [
        (5000, 0.684885, 0.534389),
        (10000, 0.687492, 0.473146),
        (14000, 0.696867, 0.436278),
        (17000, 0.695625, 0.419127),
        (20000, 0.694062, 0.409598),
        (25000, 0.693823, 0.393958),
    ]
    samst512 = [
        (5, 0.773973307609558, 0.6123956792533333),
        (10, 0.7745031952063244, 0.6108974266133333),
        (15, 0.7767400147120159, 0.6088616661733334),
    ]
    ax = axes[1]
    ax.axhline(idt_w512, color="#9A9A9A", lw=1.2, ls="--", zorder=1)
    ax.text(0.445, idt_w512 + 0.004, "idt", fontsize=7.5, color="#6A6A6A")
    for data, color, label in [(samam512, "#2B78B8", "SaMAM-512"), (samam256, "#7C7C7C", "SaMAM-256")]:
        xs = [1 - lp for _, _, lp in data]
        ys = [cs for _, cs, _ in data]
        ax.plot(xs, ys, color=color, lw=1.6, marker="o", ms=4, label=label)
        for step, cs, lp in data:
            if step in {5000, 10000, 14000, 25000}:
                off = {
                    5000: (0.004, -0.006),
                    10000: (0.006, -0.004),
                    14000: (0.004, 0.005),
                    25000: (0.006, -0.002),
                }.get(step, (0.004, 0.002))
                ax.text(1 - lp + off[0], cs + off[1], f"{step//1000}k", fontsize=7, color=color)
    xs = [1 - lp for _, _, lp in samst512]
    ys = [cs for _, cs, _ in samst512]
    ax.plot(xs, ys, color="#7A4FA2", lw=1.4, marker="s", ms=4, label="SaMST-512")
    ax.text(xs[-1] + 0.006, ys[-1] - 0.004, "e15", fontsize=7, color="#7A4FA2")
    ax.scatter(1 - 0.355038, 0.792298, s=88, color="#D9543F", marker="*", edgecolor="black", linewidth=0.7, zorder=5, label="LBM-512")
    ax.text(1 - 0.355038 + 0.006, 0.792298 + 0.004, "LBM", fontsize=8, color="#A13327")
    ax.set_title("(b) WikiArt512 convergence", fontsize=10, fontweight="bold")
    ax.set_xlabel("1 - LPIPS")
    ax.set_xlim(0.36, 0.86)
    ax.set_ylim(0.66, 0.81)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # Panel C: Distinct5-512.
    rows = _read_distinct5()
    ax = axes[2]
    idt_d5 = 0.6801251527865727
    ax.axhline(idt_d5, color="#9A9A9A", lw=1.2, ls="--", zorder=1)
    ax.text(0.375, idt_d5 + 0.004, "idt", fontsize=7.5, color="#6A6A6A")
    samam_rows = [r for r in rows if r["family"] == "SaMAM"]
    lancet_rows = [r for r in rows if r["family"] == "LANCET" and r["label"] in {"F e1", "H e1", "H e2", "K e1"}]
    sx = [float(r["one_minus_lpips"]) for r in samam_rows]
    sy = [float(r["clip_style"]) for r in samam_rows]
    ax.plot(sx, sy, color="#777777", marker="o", ms=3.5, lw=1.5, label="SaMAM")
    for r in samam_rows:
        step = int(r["step_or_epoch"])
        if step in {250, 1000, 2000}:
            ax.text(float(r["one_minus_lpips"]) + 0.004, float(r["clip_style"]) + 0.003, str(step), fontsize=7, color="#555555")
    colors = {"F e1": "#2FA36B", "H e1": "#2B78B8", "H e2": "#5C66C8", "K e1": "#D9543F"}
    label_offsets = {
        "F e1": (0.012, -0.010),
        "H e1": (0.012, 0.004),
        "H e2": (-0.038, 0.004),
        "K e1": (0.006, 0.008),
    }
    for r in lancet_rows:
        ax.scatter(float(r["one_minus_lpips"]), float(r["clip_style"]), s=66, marker="D", color=colors[r["label"]], edgecolor="black", linewidth=0.7, zorder=4)
        off = label_offsets.get(r["label"], (0.004, 0.003))
        ax.text(float(r["one_minus_lpips"]) + off[0], float(r["clip_style"]) + off[1], r["label"], fontsize=8)
    ax.scatter(1 - 0.6255497488, 0.7247245136102042, s=78, marker="X", color="#7A4FA2", edgecolor="black", linewidth=0.7, zorder=5, label="SaMST-512 e15")
    ax.text(1 - 0.6255497488 + 0.006, 0.7247245136102042 + 0.004, "SaMST", fontsize=8, color="#7A4FA2")
    ax.set_title("(c) Distinct5-512 stress", fontsize=10, fontweight="bold")
    ax.set_xlabel("1 - LPIPS")
    ax.set_xlim(0.37, 0.73)
    ax.set_ylim(0.53, 0.74)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Evaluation landscape: style strength versus content preservation", fontsize=12, fontweight="bold")
    fig.savefig(ROOT / "fig_eval_landscape.pdf")
    fig.savefig(ROOT / "fig_eval_landscape.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_framework()
    make_eval_landscape()
