from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def _read_distinct5_points() -> list[dict[str, str]]:
    path = REPO / "SchrodingerBridge" / "docs" / "experiments" / "distinct5_512_20260602" / "tables" / "clip_style_vs_1lpips_points.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_latent_convergence() -> list[dict[str, str]]:
    path = REPO / "SchrodingerBridge" / "docs" / "experiments" / "2026-06-07-distinct5_latent_baseline_convergence.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_inmortal_results() -> dict[str, dict[str, str]]:
    path = REPO / "SchrodingerBridge" / "docs" / "experiments" / "aaai2027_inmortal_results_master.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["experiment"]: row for row in rows}


def make_eval_landscape_with_latent() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharey=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.20, wspace=0.34)

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

    # Panel C: Distinct5-512 + latent baselines + current inmortal frontier.
    rows = _read_distinct5_points()
    latent_rows = _read_latent_convergence()
    inmortal = _read_inmortal_results()
    ax = axes[2]
    idt_d5 = 0.6801251527865727
    ax.axhline(idt_d5, color="#9A9A9A", lw=1.2, ls="--", zorder=1)
    ax.text(0.165, idt_d5 + 0.004, "idt", fontsize=7.5, color="#6A6A6A")

    samam_rows = [r for r in rows if r["family"] == "SaMAM"]
    sx = [float(r["one_minus_lpips"]) for r in samam_rows]
    sy = [float(r["clip_style"]) for r in samam_rows]
    ax.plot(sx, sy, color="#777777", marker="o", ms=3.5, lw=1.5, label="SaMAM")
    for r in samam_rows:
        step = int(r["step_or_epoch"])
        if step in {250, 1000, 2000}:
            ax.text(float(r["one_minus_lpips"]) + 0.004, float(r["clip_style"]) + 0.003, str(step), fontsize=7, color="#555555")

    lancet_rows = [r for r in rows if r["family"] == "LANCET" and r["label"] in {"F e1", "H e1", "H e2", "K e1"}]
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

    # Latent SaMAM convergence curve.
    samam_latent = [
        ("20", 0.6297, 0.7823),
        ("110", 0.6388, 0.7042),
        ("300", 0.6223, 0.5650),
        ("600", 0.6541, 0.5468),
        ("1000", 0.6667, 0.2744),
        ("1200", 0.6550, 0.1739),
        ("1300", 0.6533, 0.2198),
        ("1500", 0.6547, 0.1635),
    ]
    lat_samam_x = [1 - lp for _, _, lp in samam_latent]
    lat_samam_y = [cs for _, cs, _ in samam_latent]
    ax.plot(lat_samam_x, lat_samam_y, color="#0F766E", lw=1.6, marker="o", ms=3.5, label="SaMAM-latent")
    for label, cs, lp in samam_latent:
        if label in {"20", "1000", "1500"}:
            ax.text(1 - lp + 0.006, cs + 0.003, label, fontsize=7, color="#0F766E")

    # Latent SaMST convergence/bad plateau.
    samst_latent = [
        ("300", 0.6893, 0.8382),
        ("950", 0.6944, 0.8409),
        ("1050", 0.6820, 0.8318),
    ]
    lat_samst_x = [1 - lp for _, _, lp in samst_latent]
    lat_samst_y = [cs for _, cs, _ in samst_latent]
    ax.plot(lat_samst_x, lat_samst_y, color="#A855F7", lw=1.4, marker="P", ms=4.2, ls="--", label="SaMST-latent")
    ax.text(lat_samst_x[-1] + 0.006, lat_samst_y[-1] - 0.006, "1050", fontsize=7, color="#A855F7")

    # Current inmortal frontier points.
    bal = inmortal["inmortal_xpred_kmanifold_pattn_stokes_from_pattn_seed42_b16"]
    raw = inmortal["inmortal_xpred_kmanifold_pattn_stokes002_from_pattn_seed42_b16"]
    bal_x = 1 - float(bal["transfer_content_lpips"])
    bal_y = float(bal["transfer_clip_style"])
    raw_x = 1 - float(raw["transfer_content_lpips"])
    raw_y = float(raw["transfer_clip_style"])
    ax.scatter(bal_x, bal_y, s=84, marker="*", color="#C2410C", edgecolor="black", linewidth=0.7, zorder=6, label="LBM bal")
    ax.text(bal_x + 0.008, bal_y - 0.006, "LBM bal", fontsize=8, color="#9A3412")
    ax.scatter(raw_x, raw_y, s=72, marker="*", color="#1D4ED8", edgecolor="black", linewidth=0.7, zorder=6, label="LBM style")
    ax.text(raw_x + 0.008, raw_y + 0.004, "LBM style", fontsize=8, color="#1D4ED8")

    ax.set_title("(c) Distinct5-512 + latent baselines", fontsize=10, fontweight="bold")
    ax.set_xlabel("1 - LPIPS")
    ax.set_xlim(0.15, 0.86)
    ax.set_ylim(0.53, 0.75)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    fig.suptitle("Evaluation landscape: style strength versus content preservation", fontsize=12, fontweight="bold")
    fig.savefig(ROOT / "fig_eval_landscape_with_latent.pdf")
    fig.savefig(ROOT / "fig_eval_landscape_with_latent.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_eval_landscape_with_latent()
