from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "fig_data" / "method_probes.json"
SWD_DATA = HERE / "fig_data" / "swd_loss_separability.json"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frequency(payload: dict) -> None:
    frequency = payload["frequency_probe"]
    bands = frequency["bands"]
    shares = np.asarray(frequency["fm_gradient_energy_share"], dtype=float) * 100.0
    route_shares = np.asarray(
        [[row[f"share_{band}"] for band in bands] for row in frequency["route_rows"]], dtype=float
    ) * 100.0
    route_sem = route_shares.std(axis=0, ddof=1) / np.sqrt(route_shares.shape[0])
    colors = ["#4C78A8", "#F2CF5B", "#ECA82C", "#D97706"]

    fig, ax = plt.subplots(figsize=(3.22, 1.92))
    x = np.arange(len(bands))
    bars = ax.bar(x, shares, yerr=route_sem, color=colors, width=0.68, capsize=2.0, edgecolor="white")
    ax.set_xticks(x, bands)
    ax.set_ylabel("FM gradient-energy share (%)")
    ax.set_ylim(0, 78)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.8, f"{value:.1f}", ha="center", fontsize=7.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.20, top=0.96)
    finish(fig, "fig_probe_frequency")


def plot_style(payload: dict) -> None:
    rows = payload["style_separability_probe"]["rows"]
    bands = [row["band"] for row in rows]
    ratios = np.asarray([row["between_within_ratio"] for row in rows])
    sem = np.asarray([row["ratio_sem"] for row in rows])
    colors = ["#4C78A8", "#F2CF5B", "#ECA82C", "#D97706"]

    fig, ax = plt.subplots(figsize=(3.22, 1.92))
    x = np.arange(len(bands))
    bars = ax.bar(x, ratios, yerr=sem, color=colors, width=0.68, capsize=2.0, edgecolor="white")
    ax.set_xticks(x, bands)
    ax.set_ylabel("Style between/within variance")
    ax.set_ylim(0, max(0.66, float((ratios + sem).max()) * 1.12))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=7.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.20, top=0.96)
    finish(fig, "fig_probe_style")


def plot_combined(payload: dict) -> None:
    frequency = payload["frequency_probe"]
    style_rows = payload["style_separability_probe"]["rows"]
    bands = frequency["bands"]
    transport = np.asarray(frequency["fm_gradient_energy_share"], dtype=float) * 100.0
    route_shares = np.asarray(
        [[row[f"share_{band}"] for band in bands] for row in frequency["route_rows"]], dtype=float
    ) * 100.0
    transport_sem = route_shares.std(axis=0, ddof=1) / np.sqrt(route_shares.shape[0])
    style_ratio = np.asarray([row["between_within_ratio"] for row in style_rows])
    style_sem = np.asarray([row["ratio_sem"] for row in style_rows])
    colors = ["#4C78A8", "#F2CF5B", "#ECA82C", "#D97706"]

    fig, axes = plt.subplots(1, 2, figsize=(3.32, 1.48), gridspec_kw={"wspace": 0.58})
    x = np.arange(len(bands))
    axes[0].bar(x, transport, yerr=transport_sem, color=colors, width=0.7, capsize=1.5, edgecolor="white")
    axes[0].set_xticks(x, bands)
    axes[0].set_ylabel("FM energy (%)", labelpad=1)
    axes[0].set_ylim(0, 78)
    axes[0].set_title("(a) Transport", loc="left", fontsize=7.8, fontweight="bold")

    axes[1].bar(x, style_ratio, yerr=style_sem, color=colors, width=0.7, capsize=1.5, edgecolor="white")
    axes[1].set_xticks(x, bands)
    axes[1].set_ylabel("Style B/W", labelpad=1)
    axes[1].set_ylim(0, 0.72)
    axes[1].set_title("(b) Style statistics", loc="left", fontsize=7.8, fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.8, pad=1)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.22, top=0.86)
    finish(fig, "fig_probe_combined")


def plot_swd_separation(payload: dict) -> None:
    rows = payload["rows"]
    labels = ["Pixel\nMSE", "Latent\nMSE", "Latent\nSWD", "VGG\nGram"]
    values = [np.asarray(row["values"], dtype=float) for row in rows]
    means = np.asarray([value.mean() for value in values])
    sem = np.asarray([value.std(ddof=1) / np.sqrt(len(value)) for value in values])
    colors = ["#9CA3AF", "#6B8EAD", "#D97706", "#E9B949"]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(2.72, 1.45))
    ax.bar(x, means, yerr=sem, color=colors, width=0.68, capsize=1.7, edgecolor="white")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.65)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$D_{inter}/D_{intra}-1$", labelpad=1)
    ax.set_ylim(0.0, 0.34)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5, pad=1)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.28, top=0.96)
    finish(fig, "fig_probe_swd_separation")


def main() -> None:
    configure()
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    plot_frequency(payload)
    plot_style(payload)
    plot_combined(payload)
    plot_swd_separation(json.loads(SWD_DATA.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
