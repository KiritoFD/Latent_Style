"""Radar chart of baselines using the PAPER MAIN TABLE data (Table 1, paper.tex).

Source of truth = aaai2027_v4/paper.tex, Table "tab:main" (lines 330-343).
14 axes = 12 metric axes (4 metrics x 3 datasets) + 2 speed axes:
  CLIP-S(D5,P2A,R5) -> 1-LPIPS(D5,P2A,R5) -> DINO-C(D5,P2A,R5) -> DINO-S(D5,P2A,R5)
  -> Train speed(min) -> Infer speed(750img)
MUSIQ is intentionally omitted. Metric axes come from the paper table; DINO-C / DINO-S
from local DINOv2 inference. Train / Infer speed axes come from the paper main table
(D5 context, single value per method); both are LOG-INVERTED so the FASTEST method sits
at the OUTER edge (value -> 1.0). Metric axes are per-axis normalized by v/max (strongest
per axis -> 1.0, outer; others keep their TRUE ratio, not stretched down to 0). Speed axes
are log-inverted to 0..1 (fastest -> 1.0). Training-free methods leave a GAP on the Train-speed
axis (not drawn, no cross-gap fill). Cells without DINO data are gaps (NaN).

Color & visual hierarchy:
  - BOLD (thick, drawn on top): Ours(WEAVE), Seedream 4.5, Z-STAR, StyleAligned
  - FAINT (thin, low alpha): everything else (classical / secondary baselines)

Prints weakness analysis of WEAVE vs tier-A baselines.
"""
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import io

# Force stdout to UTF-8 for Windows GBK console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Self-contained: paths resolve relative to this script (which lives in
# aaai2027_v4); bundled inputs are read from ./fig_data so the paper folder
# stands alone and can regenerate every figure.
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "radar_baselines_14.png"
DINO_JSON = SCRIPT_DIR / "fig_data" / "dino_main.json"

DATA = {
    # tuple = (CLIP-S, LPIPS(raw, lower=better), DINO-C, DINO-S); MUSIQ dropped.
    # CLIP-S/LPIPS from paper tab:main; DINO-C/DINO-S from local DINOv2 (state/dino/dino_main.json).
    "Identity":       [(0.693, 0.000, 1.000, 0.419), (0.663, 0.000, 1.000, 0.416), (0.731, 0.000, 1.000, 0.433)],
    "SD-Turbo":       [(0.693, 0.003, 0.922, 0.484), (0.674, 0.603, 0.279, 0.341), (0.767, 0.449, 0.538, 0.505)],
    "StyleAligned":   [(0.780, 0.869, 0.239, 0.675), (0.768, 0.786, 0.310, 0.612), (0.824, 0.829, 0.315, 0.649)],
    "Z-STAR":         [(0.784, 0.347, 0.549, 0.449), (0.786, 0.332, 0.552, 0.498), (0.822, 0.384, 0.526, 0.514)],
    "StyleShot":      [(0.787, 0.765, 0.377, 0.563), (0.774, 0.713, 0.335, 0.552), (0.787, 0.795, 0.380, 0.484)],
    "CUT":            [(0.714, 0.374, 0.795, 0.471), (0.754, 0.494, 0.7020, 0.5387), (0.710, 0.620, 0.795, 0.441)],
    "SaMST":          [(0.618, 0.749, 0.145, 0.271), (0.709, 0.398, 0.838, 0.501), (0.667, 0.612, 0.615, 0.461)],
    "SaMam":          [(0.582, 0.243, 0.812, 0.477), (0.677, 0.205, 0.870, 0.505), (0.712, 0.227, 0.925, 0.503)],
    "Seedream 4.5":   [(0.720, 0.477, 0.739, 0.486), (0.752, 0.227, 0.785, 0.517), (0.742, 0.486, 0.725, 0.504)],
    "Ours (WEAVE)":   [(0.7075, 0.2583, 0.8287, 0.4859), (0.6681, 0.3116, 0.8612, 0.4801), (0.7747, 0.2895, 0.7717, 0.5226)],
}
DATASETS = ["D5-512", "P2A-256", "R5-WikiArt"]

# Train / Infer times from paper tab:main (D5 context, single value per method).
# Converted to SECONDS. math.nan = training-free ("free") or closed API ("--"/"API").
TRAIN_SEC = {
    "Identity":       math.nan,        # free
    "SD-Turbo":       math.nan,        # free
    "StyleAligned":   math.nan,        # free
    "Z-STAR":         math.nan,        # free
    "StyleShot":      math.nan,        # free
    "CUT":            322.6 * 60.0,    # 322.6 min
    "SaMST":          39.5  * 60.0,    # 39.5 min
    "SaMam":          436.0 * 60.0,    # 436.0 min
    "Seedream 4.5":   math.nan,        # API
    "Ours (WEAVE)":   3.0  * 60.0,    # 3.0 min (10 epochs)
}
INFER_SEC = {
    "Identity":       math.nan,        # 0 s dropped (training-free -> gap on infer axis)
    "SD-Turbo":       303.0,           # 5.1 m  (0.404 s/img x750, RTX3060 measured)
    "StyleAligned":   4635.0,          # 77 m   (6.18 s/img x750, RTX3060 measured)
    "Z-STAR":         10800.0,         # ~3 h   ESTIMATED (30-step SD1.5 + dual-latent reweight;
                                       #          OOMs at 512^2 on 12GB -> step-count scaling)
    "StyleShot":      18472.0,         # 5.1 h  (24.63 s/img x750, RTX3060 measured)
    "CUT":            300.0,           # 5 m    (not re-measured on 3060; prior value)
    "SaMST":          10.0 * 60.0,     # 10 m   (not re-measured on 3060; prior value)
    "SaMam":          17.6 * 60.0,     # 17.6 m (not re-measured on 3060; prior value)
    "Seedream 4.5":   math.nan,        # --
    "Ours (WEAVE)":   50.0,            # 50 s (b16 optimized, RTX3060) per paper tab:main
}

# radar method name -> local results/<ds>/<dir> name (for DINO lookup)
METHOD_DIR = {
    "Identity": "identity",
    "SD-Turbo": "sdturbo", "StyleAligned": "stylealigned",
    "Z-STAR": "zstar", "StyleShot": "styleshot", "CUT": "cut", "SaMST": "samst",
    "SaMam": "samam", "Seedream 4.5": "seedream", "Ours (WEAVE)": "weave",
}

# Load DINO (may be partial)
DINO = {}
if DINO_JSON.exists():
    try:
        DINO = json.loads(DINO_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print("WARN: could not read DINO json:", e)

# Three-tier visual hierarchy (plot line + legend text):
#   Tier-1 (thickest): Ours (WEAVE)
#   Tier-2 (secondary): Seedream 4.5, Z-STAR
#   Tier-3 (tertiary):  StyleAligned, SaMam
# Everything else is FAINT (thin line, low alpha) so the key competitors stand out.
TIER1 = {
    "Ours (WEAVE)": ("#D62728", 4.2, 0.24, 1.00, 14),
}
TIER2 = {
    "Seedream 4.5": ("#1565C0", 3.4, 0.22, 0.96, 13),
    "Z-STAR":       ("#2CA02C", 3.4, 0.20, 0.96, 12),
}
TIER3 = {
    "StyleAligned": ("#FF7F0E", 2.7, 0.20, 0.92, 11),
    "SaMam":        ("#BCBD22", 2.7, 0.18, 0.92, 11),
}
FAINT = {
    "StyleShot":    ("#9467BD", 1.3, 0.05, 0.45, 4),
    "CUT":          ("#E377C2", 1.3, 0.05, 0.45, 4),
    "SaMST":        ("#17BECF", 1.2, 0.04, 0.40, 3),
    "SD-Turbo":     ("#AEC7E8", 1.1, 0.03, 0.35, 3),
    "Identity":     ("#000000", 1.9, 0.09, 0.75, 3),
}
TIERS = {}
TIERS.update(FAINT)
TIERS.update(TIER3)
TIERS.update(TIER2)
TIERS.update(TIER1)

# BOLD set used by the weakness analysis (all emphasized competitors).
BOLD = {}
BOLD.update(TIER1); BOLD.update(TIER2); BOLD.update(TIER3)

# Legend tier of each method (drives per-entry font weight / color in the legend).
TIER_OF = {}
TIER_OF.update({k: "t1" for k in TIER1})
TIER_OF.update({k: "t2" for k in TIER2})
TIER_OF.update({k: "t3" for k in TIER3})

# Identity is drawn as a distinctive secondary reference: black, dashed, thin.
LINESTYLE = {"Identity": "--"}

LEGEND_ORDER = ["Ours (WEAVE)", "Seedream 4.5", "Z-STAR", "StyleAligned", "SaMam",
                "IP-Adapter", "StyleShot", "CUT", "SaMST", "SD-Turbo", "Identity"]

# Year / venue for each method (best-effort; AMiner token unavailable -> from public record).
META = {
    "Identity":       ("ref",        "reference"),
    "SD-Turbo":       ("2023",       "Sauer et al."),
    "IP-Adapter":     ("2023",       "Ye et al."),
    "StyleAligned":   ("CVPR 2024",  "Hertz et al."),
    "Z-STAR":         ("CVPR 2024",  "Deng et al."),
    "StyleShot":      ("2024",       "Gao et al."),
    "CUT":            ("ECCV 2020",  "Park et al."),
    "SaMST":          ("ACCV 2024",  "Liu et al."),
    "SaMam":          ("CVPR 2025",  "Liu et al."),
    "Seedream 4.5":   ("2025",       "ByteDance"),
    "Ours (WEAVE)":   ("AAAI 2027",  "this paper"),
}

# Metric groups (5 metrics x 3 datasets = 15 axes). kind -> how to compute.
METRICS = [
    ("CLIP-S$\\uparrow$",   "clip"),
    ("1$-$LPIPS$\\uparrow$", "lpips"),
    ("DINO-C$\\uparrow$",    "dino_content"),
    ("DINO-S$\\uparrow$",    "dino_style"),
]

# Per-metric dataset display order. CLIP-S: swap D5<->R5 so the R5 axis comes first.
# Default follows DATASETS; only "clip" overrides it.
DS_ORDER = {"clip": [2, 1, 0]}   # -> R5-WikiArt, P2A-256, D5-512
def _ds_idx(kind):
    return DS_ORDER.get(kind, list(range(len(DATASETS))))

# Time axes (2 single axes, no dataset multiplier). Log-inverted: fastest -> outer.
def _time_range(secdict):
    vals = [v for v in secdict.values()
            if isinstance(v, float) and not math.isnan(v)]
    return (min(vals), max(vals))

# Infer BEFORE Train: most training-free methods have an Infer value but Train=NaN, so putting
# Infer adjacent to the metric block lets their polygon connect through Infer, leaving the gap on
# the trailing Train axis instead of breaking the line mid-way.
TIME_AXES = [
    ("Infer speed$\\uparrow$\n750img", INFER_SEC, *_time_range(INFER_SEC)),
    ("Train speed$\\uparrow$\nmin",  TRAIN_SEC, *_time_range(TRAIN_SEC)),
]

def _norm_time(sec, lo, hi):
    """Log-inverted normalization: fastest (smallest sec) -> 1.0 (outer), slowest -> 0.0."""
    if not isinstance(sec, float) or math.isnan(sec):
        return math.nan          # training-free / API -> gap
    if sec <= 0:
        return 1.0               # instantaneous (e.g. Identity 0 s) -> outermost
    return 1.0 - (math.log(sec + 1.0) - math.log(lo + 1.0)) / \
               (math.log(hi + 1.0) - math.log(lo + 1.0))

AXIS_LABELS = [f"{gname}\n{DATASETS[d]}" for gname, kind in METRICS for d in _ds_idx(kind)] + \
              [tlab for tlab, _, _, _ in TIME_AXES]
N_METRIC_AXES = len(METRICS) * len(DATASETS)   # 12 metric axes; remaining are log speed axes


def _dino_val(method, kind, d_idx):
    ds = DATASETS[d_idx]
    dirname = METHOD_DIR.get(method)
    if dirname is None:
        return math.nan
    cell = DINO.get(f"{ds}|{dirname}")
    if not cell:
        return math.nan
    v = cell.get(kind)
    return float(v) if isinstance(v, (int, float)) else math.nan


def raw_axes(method):
    """Raw (un-normalized) values: metrics as-is, speed axes log-inverted to 0..1.
    DATA tuple = (CLIP-S, LPIPS(raw), DINO-C, DINO-S); MUSIQ dropped."""
    triples = DATA[method]
    out = []
    for _, kind in METRICS:
        for d in _ds_idx(kind):
            if kind == "clip":
                out.append(triples[d][0])
            elif kind == "lpips":
                out.append(1.0 - triples[d][1])
            elif kind == "dino_content":
                out.append(triples[d][2])
            elif kind == "dino_style":
                out.append(triples[d][3])
    for _, secdict, lo, hi in TIME_AXES:
        out.append(_norm_time(secdict.get(method), lo, hi))
    return out


# Per-axis v/v_max for the METRIC axes: the strongest method per axis -> 1.0 (outer
# edge); the others keep their TRUE proportional ratio (v / v_max), i.e. we do NOT stretch
# the minimum down to 0 (that would fabricate "CLIP-S=1.0" while crushing weak methods to
# the center). Speed axes are already log-inverted to 0..1 (fastest -> 1.0) -> left as-is.
METRIC_VMAX = []
for i in range(N_METRIC_AXES):
    col = [raw_axes(m)[i] for m in DATA]
    vals = [v for v in col if isinstance(v, float) and not math.isnan(v)]
    METRIC_VMAX.append(max(vals))


def to_axes(method):
    raw = raw_axes(method)
    out = []
    for i, v in enumerate(raw):
        if isinstance(v, float) and not math.isnan(v):
            if i < N_METRIC_AXES:
                out.append(v / METRIC_VMAX[i] if METRIC_VMAX[i] > 0 else 0.5)
            else:
                out.append(v)   # speed axis already 0..1 (log-inverted)
        else:
            out.append(math.nan)
    return out


def contiguous_runs(mask):
    """Maximal runs of True in a boolean list (no wrap-around)."""
    n = len(mask); runs = []; i = 0
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            runs.append((s, i - 1))
        else:
            i += 1
    return runs


N = len(AXIS_LABELS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Wider figure; the polar axes is squeezed into a LEFT-anchored, horizontally
# squashed box so its right-side axis labels (CLIP-S, speed axes) clear the
# dedicated legend/caption column on the right.
fig, ax = plt.subplots(figsize=(18, 11), subplot_kw=dict(polar=True))
fig.subplots_adjust(left=0.03, right=0.44, top=0.94, bottom=0.10)

for name in DATA:
    vals = to_axes(name)
    c, lw, af, al, zo = TIERS[name]
    mask = [not (isinstance(v, float) and math.isnan(v)) for v in vals]
    runs = contiguous_runs(mask)
    labeled = False
    for (s, e) in runs:
        # Fan-to-center fill: no cross-gap chord, no shadow drawn across missing axes.
        if e - s + 1 >= 2:
            th = [angles[s]] + angles[s:e+1] + [angles[e]]
            rv = [0.0] + vals[s:e+1] + [0.0]
            ax.fill(th, rv, color=c, alpha=af, zorder=zo)
        seg_a = angles[s:e+1]
        seg_v = vals[s:e+1]
        if s == 0 and e == N - 1:           # full circle -> close the loop
            seg_a = seg_a + [seg_a[0]]
            seg_v = seg_v + [seg_v[0]]
        ax.plot(seg_a, seg_v, color=c, linewidth=lw, alpha=al, zorder=zo,
                linestyle=LINESTYLE.get(name, "-"),
                label=(name if not labeled else None))
        labeled = True
    va = [angles[i] for i in range(N) if mask[i]]
    vv = [vals[i] for i in range(N) if mask[i]]
    if va:
        ax.scatter(va, vv, color=c, s=16, alpha=min(1.0, al + 0.25),
                   zorder=zo + 1, edgecolors="none")

# ---- axis labels drawn OUTSIDE the circle so they never overlap the spokes ----
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])                       # labels drawn manually, radially outside
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=11, color="grey")
ax.grid(True, alpha=0.35)
ax.tick_params(axis="y", labelsize=11, pad=6)

_LABEL_R = 1.085   # radius (in data units); ~half the previous 0.17 gap from the outer ring
for ang, lab in zip(angles[:-1], AXIS_LABELS):
    deg = np.degrees(ang) % 360
    if 45 <= deg < 135:
        ha, va = "center", "bottom"
    elif 135 <= deg < 225:
        ha, va = "right", "center"
    elif 225 <= deg < 315:
        ha, va = "center", "top"
    else:
        ha, va = "left", "center"
    ax.text(ang, _LABEL_R, lab, ha=ha, va=va,
            fontsize=13, fontweight="bold", color="#1a1a1a")

# ----- legend on the RIGHT, larger, single column, with year/venue -----
handles, labels = ax.get_legend_handles_labels()
ordered_hl = []
for n in LEGEND_ORDER:
    for h, l in zip(handles, labels):
        if l == n:
            ordered_hl.append((h, n))
            break
oh, onames = list(zip(*ordered_hl)) if ordered_hl else ([], [])
olabels = [f"{n}  ({META.get(n, ('?',))[0]})" for n in onames]

# Legend sits in the right margin (dedicated column, clear of the squashed radar);
# three-tier font weight mirrors the plot hierarchy.
_LEG_X = 0.58
leg = fig.legend(oh, olabels, loc="upper left", bbox_to_anchor=(_LEG_X, 0.94),
                 fontsize=13, frameon=False, handlelength=3.2,
                 handletextpad=0.7, labelspacing=0.5,
                 title="Methods  (emphasis tier)",
                 title_fontsize=12.5)
leg.get_title().set_fontweight("bold")
# make legend line samples a touch thicker / more visible
for ln in leg.get_lines():
    ln.set_linewidth(2.8)
# per-entry font weight / color by tier (T1 black > T2 bold > T3 bold+grey > faint normal)
for txt, n in zip(leg.get_texts(), onames):
    t = TIER_OF.get(n, "faint")
    if t == "t1":
        txt.set_fontweight("black"); txt.set_color("black"); txt.set_fontsize(14)
    elif t == "t2":
        txt.set_fontweight("bold"); txt.set_color("black")
    elif t == "t3":
        txt.set_fontweight("bold"); txt.set_color("#555555")
    else:
        txt.set_fontweight("normal"); txt.set_color("black")

# ----- in-figure caption: placed in the RIGHT margin, aligned to the legend width -----
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_w_px = fig.get_size_inches()[0] * fig.dpi
fig_h_px = fig.get_size_inches()[1] * fig.dpi
# derive the caption wrap-width and x-anchor from the legend's own extent
_leg_ext = leg.get_window_extent(renderer)
_CAP_W_PX = _leg_ext.width * 0.99
_leg_x0_frac = _leg_ext.x0 / fig_w_px
_leg_y0_frac = _leg_ext.y0 / fig_h_px


def _wrap_to_px(text, fs, max_px):
    """Wrap `text` so every line is <= max_px display pixels (figure-dpi)."""
    probe = fig.text(0, 0, "", fontsize=fs)
    out, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        probe.set_text(trial)
        if probe.get_window_extent(renderer).width <= max_px:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    probe.remove()
    return "\n".join(out)


_CAPTION = (
    "Axes use per-metric v/max normalization, so the outer ring is the strongest observed method "
    "and every other radius keeps its true ratio. DINO-S is deliberately shown because it is more "
    "discriminative for real style movement than CLIP-S alone: Identity preserves content but stays "
    "near the inner region on DINO-S, exposing no-op transfer. Speed is log-normalized and inverted "
    "(faster -> outer ring), so WEAVE's real wall-clock advantage is larger than the visual gap suggests. "
    "Across datasets, WEAVE keeps content preservation competitive with or above strong baselines while "
    "remaining the only method that is both high-quality and ultra-efficient."
)
_CAPTION_FS = 15.0
_CAPTION_WRAPPED = _wrap_to_px(_CAPTION, _CAPTION_FS, _CAP_W_PX)
# Caption sits in the RIGHT margin, directly below the legend and aligned to its width.
fig.text(_leg_x0_frac, _leg_y0_frac - 0.028, _CAPTION_WRAPPED,
         ha="left", va="top",
         fontsize=_CAPTION_FS, color="#242424", linespacing=1.3,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#D8D8D8", linewidth=0.4, alpha=0.9))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("WROTE %s" % OUT)
print("DINO cells available: %d" % len(DINO))


# ======== WEAVE weakness analysis ============================================
OURS = "Ours (WEAVE)"
ours_flat = raw_axes(OURS)    # weakness analysis uses RAW values, not the v/v_max figure
# axis order (incl. per-metric dataset swap) -> (metric_index, dataset_index)
axis_specs = [(mi, d) for mi, (g, kind) in enumerate(METRICS) for d in _ds_idx(kind)]
axis_names = [f"{METRICS[mi][0]}({DATASETS[d]})" for mi, d in axis_specs]
metric_per_axis = [METRICS[mi][0] for mi, d in axis_specs]
ax_ds = [DATASETS[d] for mi, d in axis_specs]

print("\n" + "=" * 76)
print("WEAVE weakness analysis (per axis, vs tier-A baselines)")
print("=" * 76)
for i, aname in enumerate(axis_names):
    ov = ours_flat[i]
    if isinstance(ov, float) and math.isnan(ov):
        print(f"\n  [{i+1}] {aname:<20} WEAVE=NaN (no DINO data)")
        continue
    worse = []
    for name in BOLD:
        if name == OURS:
            continue
        tv = raw_axes(name)[i]
        if isinstance(tv, float) and math.isnan(tv):
            continue
        gap = ov - tv
        if gap < -0.02:
            worse.append((gap, name, metric_per_axis[i], ax_ds[i], ov, tv))
    worse.sort(key=lambda x: x[0])
    print(f"\n  [{i+1}] {aname:<20} WEAVE={ov:.3f}")
    if not worse:
        print("         [OK] no meaningful deficit vs tier-A baselines")
    else:
        for gap, nm, rn, ds_val, ro, rt in worse:
            print(f"         [!] loses to {nm:<14} by {abs(gap):.3f}  ({rn}={rt:.3f} vs ours {ro:.3f})")

print("\n--- Summary ---")
for m_idx, metric in enumerate(["CLIP-S", "LPIPS", "DINO-C", "DINO-S"]):
    losses_per_ds = []
    for d_idx, ds in enumerate(DATASETS):
        aidx = next(i for i, (mi, d) in enumerate(axis_specs) if mi == m_idx and d == d_idx)
        count = sum(1 for nm in BOLD if nm != OURS
                    and not (isinstance(raw_axes(nm)[aidx], float) and math.isnan(raw_axes(nm)[aidx]))
                    and to_axes(nm)[aidx] > ours_flat[aidx] + 0.02)
        if count:
            losses_per_ds.append(f"{ds}(-{count})")
    if losses_per_ds:
        print(f"  * {metric}: weak on {', '.join(losses_per_ds)}")

print("\nNote: LPIPS column in paper uses down-arrow (lower is better). "
      "On radar we plot 1-LPIPS so higher = always better. DINO-C/DINO-S higher = better.\n"
      "Metric axes are per-axis normalized by v/max (strongest per axis = 1.0, outer; others\n"
      "keep their true ratio, not stretched to 0). The weakness analysis above uses RAW values.\n"
      "Train/Infer speed axes are log-inverted (1 - (log(t+1)-log(lo+1))/(log(hi+1)-log(lo+1))): "
      "fastest=1.0 (outer). Missing (training-free / API) cells are gaps; no fill crosses them.")
