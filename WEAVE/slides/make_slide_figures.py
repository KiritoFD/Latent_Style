"""Slide-native figures for the WEAVE defense deck (Chinese labels, HarmonyOS Sans SC).

    python make_slide_figures.py

Every figure is drawn at the exact size it occupies on the slide, so font sizes are true
point sizes on screen. All numbers come from the paper's source-of-record files:
  ../icme2027/data/main_table.csv, artfid_d5.csv, probe_frequency.csv, probe_separability.csv
  ../../SchrodingerBridge/experiments/rebuttal_20260716/expA_seed7/per_epoch_metrics.csv
  ../docs/model_probe/target_hf_delta_eval_summary.json   (texture-route probe)
Sensitivity sweeps are transcribed from SchrodingerBridge/rebuttal_exps/docs/
stability_experiments_summary.md (same values as supplement Table S4).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ICME = HERE.parent / "icme2027"
REPO = HERE.parents[1]
OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ICME / "tools"))
import make_figures as mf  # noqa: E402  (load_main, failures, FAMILY, BOARDS)

FONT = "HarmonyOS Sans SC"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": FONT,
    "mathtext.bf": FONT,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

BLUE, NAVY, RED, GREEN = "#0C49B7", "#0E2841", "#C8102E", "#2E8B57"
GRAY, LGRAY, ORANGE = "#6B7280", "#D1D5DB", "#E08A1E"
FAM_ZH = {
    "diffusion": ("s", "#2B7A4B", "扩散编辑器"),
    "learned": ("o", "#2C5D9E", "学习型模型"),
    "api": ("D", "#8A5A00", "商用 API"),
    "analytic": ("^", "#6B6B6B", "解析方法"),
    "ours": ("*", RED, "WEAVE（本文）"),
}
NAME_ZH = {"StyTR-2": "StyTr²", "Seedream 4.5": "Seedream 4.5"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", transparent=False, facecolor="white")
    plt.close(fig)


def disp(m):
    return NAME_ZH.get(m, m)


# ---------------------------------------------------------------- sandwich (D5)
SCATTER_OFF = {
    "StyleAligned": (0, -12, "center"), "Z-STAR": (8, -10, "left"), "StyleShot": (9, 2, "left"),
    "StyleID": (9, 3, "left"), "CUT": ("data", 0.585, 0.444, "right"), "SaMST": (9, 0, "left"),
    "SaMam": (9, -8, "left"), "StyTR-2": (-9, -2, "right"), "AesPA-Net": (-9, 5, "right"),
    "Seedream 4.5": ("data", 0.40, 0.452, "right"), "Latent-WCT": (-9, 0, "right"),
    "WEAVE": (0, 13, "center"),
}


def fig_sandwich(t):
    b = "D5-512"
    idt, tgt = t[("Identity", b)], t[("Target Style", b)]
    x_tgt, y_idt = 1 - tgt["lpips"], idt["dino_s"]
    xmax, ymin, ymax = 1.06, 0.24, 0.72
    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    ax.add_patch(patches.Rectangle((x_tgt, y_idt), xmax - x_tgt, ymax - y_idt, facecolor="#E7F3EA",
                                   edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((x_tgt, ymin), xmax - x_tgt, y_idt - ymin, facecolor="#F3F4F6",
                                   edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, ymin), x_tgt, ymax - ymin, facecolor="#FBEFEC",
                                   edgecolor="none", zorder=0))
    ax.axhline(y_idt, color="#374151", lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.axvline(x_tgt, color="#8A4F3D", lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.text(0.02, y_idt + 0.008, "IDT 风格线", fontsize=13, color="#374151", va="bottom")
    ax.text(x_tgt + 0.015, ymax - 0.008, "TGT 内容线", fontsize=13, color="#8A4F3D", va="top")
    ax.text(1.04, ymax - 0.008, "有效区域", fontsize=15, color=GREEN, ha="right", va="top",
            fontweight="bold")
    ax.text(1.04, y_idt - 0.012, "恒等捷径", fontsize=14, color="#4B5563", ha="right", va="top",
            fontweight="bold")
    ax.text(0.02, ymin + 0.008, "内容崩塌", fontsize=14, color="#9A3B2A", ha="left", va="bottom",
            fontweight="bold")
    ax.scatter([1.0], [y_idt], marker="D", s=46, color="black", zorder=5, clip_on=False)
    ax.annotate("IDT", (1.0, y_idt), xytext=(7, -1), textcoords="offset points", ha="left",
                va="center", fontsize=13)
    for m, fam in mf.FAMILY.items():
        r = t[(m, b)]
        x, y = 1 - r["lpips"], r["dino_s"]
        marker, color, _ = FAM_ZH[fam]
        valid = mf.failures(t, m, b) == ""
        ax.scatter([x], [y], marker=marker, s=420 if fam == "ours" else 70,
                   facecolor=color if valid else "white", edgecolor=color,
                   linewidth=0.8 if fam == "ours" else 1.6, zorder=7 if fam == "ours" else 4)
        spec = SCATTER_OFF[m]
        kw = dict(va="center", fontsize=15 if fam == "ours" else 12.5,
                  color=RED if fam == "ours" else "#1F2937",
                  fontweight="bold" if fam == "ours" else "normal")
        if spec[0] == "data":
            ax.annotate(disp(m), (x, y), xytext=spec[1:3], textcoords="data", ha=spec[3],
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#9CA3AF", shrinkA=2, shrinkB=4),
                        **kw)
        else:
            ax.annotate(disp(m), (x, y), xytext=spec[:2], textcoords="offset points", ha=spec[2], **kw)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("内容保持  1 − LPIPS  →")
    ax.set_ylabel("风格相似度  DINO-S  →")
    handles = [plt.Line2D([], [], ls="", marker=FAM_ZH[k][0], color=FAM_ZH[k][1], markersize=8,
                          markerfacecolor=FAM_ZH[k][1], label=FAM_ZH[k][2])
               for k in ("diffusion", "learned", "api", "analytic")]
    handles.append(plt.Line2D([], [], ls="", marker="o", color="#6B7280", markersize=8,
                              markerfacecolor="white", label="空心：未通过"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=11.5,
              frameon=False, handletextpad=0.3, labelspacing=0.35, columnspacing=0.9, borderaxespad=0.3)
    save(fig, "sandwich_d5")


# ---------------------------------------------------------------- audit matrix
def fig_audit(t):
    fails = {m: [mf.failures(t, m, b) for b in mf.BOARDS] for m in mf.FAMILY}
    order = sorted(mf.FAMILY, key=lambda m: (-sum(f == "" for f in fails[m]), m != "WEAVE", m.lower()))
    n = len(order)
    fig, ax = plt.subplots(figsize=(4.9, 5.1))
    for i, m in enumerate(order):
        y = n - 1 - i
        if m == "WEAVE":
            ax.add_patch(patches.Rectangle((-0.5, y - 0.45), 3.6, 0.9, facecolor="#FBE9EB",
                                           edgecolor="none", zorder=0))
        for j, f in enumerate(fails[m]):
            if f == "":
                ax.scatter(j, y, s=170, marker="o", color=GREEN, zorder=3)
            else:
                ax.scatter(j, y, s=150, marker="X", color=RED, zorder=3, linewidths=0)
                ax.text(j + 0.2, y, f, fontsize=12, color=RED, va="center", fontweight="bold")
    n_valid = sum(all(f == "" for f in fails[m]) for m in mf.FAMILY if m != "WEAVE")
    ax.axhline(n - 1 - (n_valid + 1) + 0.5, color="#9CA3AF", lw=1.0, ls=(0, (3, 3)))
    ax.set_yticks(range(n))
    ax.set_yticklabels([disp(m) for m in reversed(order)], fontsize=13)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "WEAVE":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["D5", "P2A", "R5"], fontsize=14)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "audit")


# ---------------------------------------------------------------- ArtFID
def fig_artfid():
    with open(ICME / "data" / "artfid_d5.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    colors = {"IDT": "#374151", "WEAVE": RED, "SaMam": "#2C5D9E", "Seedream 4.5": "#8A5A00", "TGT": "#9CA3AF"}
    labels = {"IDT": "原样复制\n(IDT)", "WEAVE": "WEAVE", "SaMam": "SaMam", "Seedream 4.5": "Seedream*",
              "TGT": "参考图\n(TGT)"}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    xs = np.arange(len(rows))
    vals = [float(r["artfid"]) for r in rows]
    ax.bar(xs, vals, width=0.66, color=[colors[r["method"]] for r in rows], zorder=2)
    for x, r, v in zip(xs, rows, vals):
        top = v
        if r["ci_low"]:
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="black", elinewidth=1.2,
                        capsize=4, capthick=1.2, zorder=3)
            top = hi
        ax.text(x, top + 14, f"{v:.0f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold" if r["method"] in ("IDT", "WEAVE") else "normal")
    ax.annotate("第 1 名", xy=(0.12, 268), xytext=(0.7, 470), fontsize=14, color="#374151",
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"))
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[r["method"]] for r in rows], fontsize=12.5)
    ax.set_ylim(0, 760)
    ax.set_ylabel("ArtFID（越低越好）")
    ax.grid(axis="y", lw=0.6, color="#E5E7EB", zorder=0)
    save(fig, "artfid")


# ---------------------------------------------------------------- spectral diagnosis
def spectral_data(alpha=0.3):
    bands = ["LL", "LH", "HL", "HH"]
    with open(ICME / "data" / "probe_frequency.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    energy = {b: np.mean([float(r[f"energy_{b}"]) for r in rows]) for b in bands}
    total = sum(energy.values())
    share = [100 * energy[b] / total for b in bands]
    se = [100 * np.std([float(r[f"share_{b}"]) for r in rows], ddof=1) / np.sqrt(len(rows)) for b in bands]
    hf = energy["LH"] + energy["HL"] + energy["HH"]
    bound = 100 * alpha ** 2 * energy["LL"] / (alpha ** 2 * energy["LL"] + hf)
    with open(ICME / "data" / "probe_separability.csv", encoding="utf-8") as f:
        sep = {r["band"]: (float(r["between_within_ratio"]), float(r["ratio_sem"])) for r in csv.DictReader(f)}
    return bands, share, se, bound, sep


def fig_spectral():
    bands, share, se, bound, sep = spectral_data()
    colors = ["#2C5D9E", ORANGE, ORANGE, RED]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    xs = np.arange(4)
    ax.bar(xs, share, yerr=se, color=colors, width=0.64, error_kw=dict(lw=1.0, capsize=3))
    for x, v in zip(xs, share):
        ax.text(x, v + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=13.5,
                fontweight="bold" if x == 0 else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels(bands, fontsize=14)
    ax.set_ylim(0, 84)
    ax.set_ylabel("梯度能量占比（%）")
    ax.set_title("谁在主导梯度", fontsize=15, pad=8, fontweight="bold", color=NAVY)
    ax = axes[1]
    vals = [sep[b][0] for b in bands]
    ax.bar(xs, vals, yerr=[sep[b][1] for b in bands], color=colors, width=0.64,
           error_kw=dict(lw=1.0, capsize=3))
    for x, v, b in zip(xs, vals, bands):
        ax.text(x + 0.06, v + sep[b][1] + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=13.5,
                fontweight="bold" if b in ("LL", "HH") else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels(bands, fontsize=14)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("风格可分性（组间 / 组内方差）")
    ax.set_title("谁携带风格", fontsize=15, pad=8, fontweight="bold", color=NAVY)
    save(fig, "spectral")
    return share, bound


def fig_llshare(share, bound):
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ll = [share[0], bound]
    hf = [100 - share[0], 100 - bound]
    xs = np.arange(2)
    ax.bar(xs, ll, width=0.56, color="#2C5D9E", label="LL（结构）")
    ax.bar(xs, hf, width=0.56, bottom=ll, color=ORANGE, label="高频（纹理）")
    ax.text(0, ll[0] / 2, f"{ll[0]:.1f}%", ha="center", va="center", color="white", fontsize=17,
            fontweight="bold")
    ax.text(1, ll[1] / 2, f"≤{ll[1]:.0f}%", ha="center", va="center", color="white", fontsize=15,
            fontweight="bold")
    ax.text(0, ll[0] + hf[0] / 2, f"{hf[0]:.1f}%", ha="center", va="center", color="white", fontsize=14)
    ax.text(1, ll[1] + hf[1] / 2, f"≥{hf[1]:.0f}%", ha="center", va="center", color="white", fontsize=15,
            fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["直接目标", "源锚定端点"], fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_ylabel("梯度能量占比（%）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=12.5,
              handlelength=1.2, columnspacing=1.0)
    save(fig, "llshare")


# ---------------------------------------------------------------- texture route probe
def fig_route():
    rows = json.load(open(HERE.parent / "docs" / "model_probe" / "target_hf_delta_eval_summary.json"))
    pick = {r["file"]: r for r in rows}
    spatial = pick["target_hf_spatial_ft6_epoch0006_adain15_dino_summary.json"]
    pooled = pick["target_hf_subband_epoch0006_adain15_dino_summary.json"]
    fig, ax = plt.subplots(figsize=(4.5, 3.7))
    groups = [("DINO-S（风格）", "dino_s"), ("DINO-C（内容）", "dino_c")]
    w = 0.34
    for k, (lab, key) in enumerate(groups):
        for j, (r, color, name) in enumerate([(spatial, "#9CA3AF", "空间高频图"), (pooled, BLUE, "池化纹理码")]):
            x = k + (j - 0.5) * (w + 0.04)
            ax.bar(x, r[key], width=w, color=color, label=name if k == 0 else None)
            ax.text(x, r[key] + 0.015, f"{r[key]:.2f}", ha="center", va="bottom", fontsize=13,
                    fontweight="bold" if key == "dino_c" else "normal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=12.5)
    save(fig, "route")


# ---------------------------------------------------------------- stopping curve
def fig_stop():
    path = REPO / "SchrodingerBridge" / "experiments" / "rebuttal_20260716" / "expA_seed7" / "per_epoch_metrics.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    ds = [float(r["dino_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    ax.plot(ep, ds, color=NAVY, lw=2.0, marker="o", ms=5.5, zorder=3)
    stop = 4
    ax.axvline(stop, color=RED, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.scatter([stop], [ds[stop - 1]], s=160, color=RED, zorder=4)
    ax.annotate("内部规则在此停止\n= DINO-S 峰值", xy=(stop, ds[stop - 1]), xytext=(7.2, 0.4907),
                fontsize=13.5, color=RED, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.3, color=RED))
    ax.set_xlabel("训练 epoch（种子 7，训练满 15 个 epoch）")
    ax.set_ylabel("DINO-S")
    ax.set_xticks([1, 4, 8, 12, 15])
    ax.set_ylim(0.4825, 0.4925)
    ax.grid(axis="y", lw=0.6, color="#E5E7EB")
    save(fig, "stop_curve")


# ---------------------------------------------------------------- three boards
CODES = {
    "StyleAligned": "SA", "Z-STAR": "ZS", "StyleShot": "SS", "StyleID": "SI", "CUT": "CU",
    "SaMST": "ST", "SaMam": "SM", "StyTR-2": "T2", "AesPA-Net": "AP", "Seedream 4.5": "SD",
    "Latent-WCT": "LW", "WEAVE": "WEAVE",
}
CODE_OFF = {
    "D5-512": {"CU": (-5, -9), "SM": (5, -9), "ST": (5, -4), "AP": (-5, 6), "SI": (5, 3), "T2": (-5, -8)},
    "P2A-256": {"WEAVE": (6, -11), "ZS": (0, 11), "SD": (5, 8), "SM": (7, -1), "ST": (-5, -9),
                "CU": (-5, 5), "AP": (-5, -6), "T2": (5, -1)},
    "R5-WikiArt": {"SI": (-5, 8), "AP": (5, 6), "ZS": (5, -8), "T2": (-6, 0), "SD": (5, -9),
                   "SM": (5, -9), "CU": (-6, -11), "ST": (5, -1)},
}


def fig_boards(t):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), gridspec_kw={"wspace": 0.2})
    for ax, b in zip(axes, mf.BOARDS):
        idt, tgt = t[("Identity", b)], t[("Target Style", b)]
        x_tgt, y_idt = 1 - tgt["lpips"], idt["dino_s"]
        ymin, ymax = 0.22, 0.72
        ax.add_patch(patches.Rectangle((x_tgt, y_idt), 1.06 - x_tgt, ymax - y_idt, facecolor="#E7F3EA",
                                       edgecolor="none", zorder=0))
        ax.axhline(y_idt, color="#374151", lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.axvline(x_tgt, color="#8A4F3D", lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.scatter([1.0], [y_idt], marker="D", s=36, color="black", zorder=5, clip_on=False)
        for m, fam in mf.FAMILY.items():
            r = t[(m, b)]
            x, y = 1 - r["lpips"], r["dino_s"]
            marker, color, _ = FAM_ZH[fam]
            valid = mf.failures(t, m, b) == ""
            ax.scatter([x], [y], marker=marker, s=300 if fam == "ours" else 46,
                       facecolor=color if valid else "white", edgecolor=color,
                       linewidth=0.8 if fam == "ours" else 1.4, zorder=6 if fam == "ours" else 4)
            code = CODES[m]
            dx, dy = CODE_OFF[b].get(code, (5, 5))
            ax.annotate(code, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right", va="center",
                        fontsize=13 if fam == "ours" else 11.5,
                        color=RED if fam == "ours" else "#1F2937",
                        fontweight="bold" if fam == "ours" else "normal")
        n_valid = sum(mf.failures(t, m, b) == "" for m in mf.FAMILY if m != "WEAVE")
        ax.set_title(f"{b}：{n_valid} / {len(mf.FAMILY) - 1} 个基线有效", fontsize=14, pad=6,
                     loc="left", color=NAVY, fontweight="bold")
        ax.set_xlim(0, 1.06)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("1 − LPIPS →", fontsize=13)
    axes[0].set_ylabel("DINO-S →", fontsize=13)
    save(fig, "boards")


# ---------------------------------------------------------------- cost
def fig_cost():
    # (name, minutes, color, label) — Table I values; WEAVE trains 82.8 s and generates 750 images in 126 s.
    train = [("WEAVE", 1.38, RED, "1.4"), ("SaMST", 39.5, "#2C5D9E", "39.5"), ("CUT", 322.6, "#2C5D9E", "323"),
             ("SaMam", 436.0, "#2C5D9E", "436"), ("StyTr²*", 1440.0, "#2C5D9E", "~1440")]
    infer = [("Latent-WCT", 0.3, "#9CA3AF", "0.3"), ("WEAVE", 2.1, RED, "2.1"), ("CUT", 5, "#2C5D9E", "5"),
             ("SaMST", 10, "#2C5D9E", "10"), ("SaMam", 17.6, "#2C5D9E", "17.6"), ("AesPA-Net", 23, "#2C5D9E", "23"),
             ("StyTr²", 38, "#2C5D9E", "38"), ("StyleID", 63, "#2B7A4B", "63"), ("StyleAligned", 77, "#2B7A4B", "77"),
             ("Z-STAR†", 180, "#2B7A4B", "~180"), ("StyleShot", 306, "#2B7A4B", "306")]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.4), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.25]})
    for ax, data, title, fmt in [
        (axes[0], train, "训练时间（分钟，对数刻度）", None),
        (axes[1], infer, "750 张图推理时间（分钟，对数刻度）", None),
    ]:
        names = [d[0] for d in data][::-1]
        vals = [d[1] for d in data][::-1]
        cols = [d[2] for d in data][::-1]
        ys = np.arange(len(data))
        ax.barh(ys, vals, color=cols, height=0.62)
        labels = [d[3] for d in data][::-1]
        for y, v, n, lab in zip(ys, vals, names, labels):
            ax.text(v * 1.12, y, lab, va="center", fontsize=12,
                    fontweight="bold" if n == "WEAVE" else "normal", color=RED if n == "WEAVE" else "#1F2937")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=12.5)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "WEAVE":
                lab.set_color(RED)
                lab.set_fontweight("bold")
        ax.set_xscale("log")
        ax.set_xlim(0.15 if ax is axes[1] else 0.8, max(vals) * 4)
        ax.set_title(title, fontsize=14, loc="left", color=NAVY, fontweight="bold", pad=6)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.tick_params(axis="y", length=0)
    save(fig, "cost")


# ---------------------------------------------------------------- qualitative
def fig_qual():
    src = Image.open(ICME / "figures" / "fig_teaser_comparison.png").convert("RGB")
    width = src.size[0] / 8
    order = ["IDT", "AdaIN", "StyleID", "CUT", "SD-Turbo", "SaMam", "Seedream 4.5", "WEAVE"]
    crops = {n: src.crop((int(i * width) + 6, 32, int((i + 1) * width) - 6, 266)) for i, n in enumerate(order)}
    show = [("IDT", "原图（IDT）"), ("StyleID", "StyleID"), ("CUT", "CUT"), ("SaMam", "SaMam"),
            ("Seedream 4.5", "Seedream"), ("WEAVE", "WEAVE")]
    box = (60, 60, 140, 140)
    fig, axes = plt.subplots(1, 8, figsize=(12.3, 1.95), gridspec_kw={"wspace": 0.05})
    for ax, (k, title) in zip(axes, show):
        ax.imshow(crops[k])
        if k in ("IDT", "WEAVE"):
            ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                                           ec="white", lw=1.4, ls=(0, (3, 2))))
        ax.set_title(title, fontsize=13, pad=4, color=RED if k == "WEAVE" else "#111827",
                     fontweight="bold" if k == "WEAVE" else "normal")
    for ax, (k, title) in zip(axes[6:], [("IDT", "原图 ×3"), ("WEAVE", "WEAVE ×3")]):
        ax.imshow(crops[k].crop(box).resize((240, 240), Image.LANCZOS))
        ax.set_title(title, fontsize=13, pad=4, color=RED if k == "WEAVE" else "#111827",
                     fontweight="bold" if k == "WEAVE" else "normal")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_color("#9CA3AF")
    for s in axes[5].spines.values():
        s.set_color(RED)
        s.set_linewidth(2.0)
    save(fig, "qualitative")


# ---------------------------------------------------------------- sensitivity sweeps (appendix)
def fig_sweeps():
    lam = [(0.1, .4856), (0.15, .4857), (0.2, .4856), (0.25, .4853), (0.35, .4850), (0.4, .4847),
           (0.45, .4850), (0.5, .4848)]
    beta = [(1.0, .4831), (1.25, .4838), (1.5, .4844), (2.0, .4920)]
    alpha = [(0.1, .4859), (0.2, .4864), (0.4, .4845), (0.5, .4843)]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.3), gridspec_kw={"wspace": 0.3})
    for ax, data, xl, title in [
        (axes[0], lam, r"LL 损失权重 λ$_{LL}$", "λ$_{LL}$：DINO-S 几乎不变"),
        (axes[1], beta, "逐步 AdaIN 强度 β", "β：1.0–1.5 平稳，2.0 跃升"),
        (axes[2], alpha, "LL 混合系数 α", "α：变化约 0.002"),
    ]:
        xs, ys = zip(*data)
        ax.plot(xs, ys, color=NAVY, lw=2.0, marker="o", ms=6)
        ax.set_ylim(0.480, 0.494)
        ax.set_xlabel(xl, fontsize=13)
        ax.set_title(title, fontsize=13.5, loc="left", color=NAVY, fontweight="bold", pad=6)
        ax.grid(axis="y", lw=0.6, color="#E5E7EB")
    axes[0].set_ylabel("DINO-S")
    save(fig, "sweeps")


def main():
    t = mf.load_main()
    fig_sandwich(t)
    fig_audit(t)
    fig_artfid()
    share, bound = fig_spectral()
    fig_llshare(share, bound)
    fig_route()
    fig_stop()
    fig_boards(t)
    fig_cost()
    fig_qual()
    fig_sweeps()
    print("LL share %.1f%% -> bound %.1f%%" % (share[0], bound))


if __name__ == "__main__":
    main()
