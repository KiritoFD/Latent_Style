"""Patch plot_page1_summary.py: embed faithful curve data, remove scaling.

Run: python _patch_plot_curves.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")

src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Replace load_samam_curve() with embedded SAMAM_CURVE (test-dir refs).
# ---------------------------------------------------------------------------
OLD_SAMAM = '''def load_samam_curve():
    """Load SaMam curve: join DINO-S (from dino_samam_curve.csv) with CLIP-S/LPIPS (from curve_metrics_hf.csv)."""
    # Load DINO-S
    dino_map = {}
    dino_path = FIG_DATA / "dino_samam_curve.csv"
    if dino_path.exists():
        with open(dino_path) as f:
            for row in csv.DictReader(f):
                step = row["step"]
                if step == "last":
                    step = "7000"  # last == step 7000
                dino_map[int(step)] = float(row["dino_s"])

    # Load CLIP-S + LPIPS
    clip_path = FIG_DATA / "curve_metrics_hf.csv"
    points = []
    if clip_path.exists():
        with open(clip_path) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                if step not in dino_map:
                    continue
                clip_s = float(row["clip_style"])
                lpips = float(row["content_lpips"])
                dino_s = dino_map[step]
                points.append((step, dino_s, clip_s, lpips))
    return points'''

NEW_SAMAM = '''# SaMam curve: (step, dino_s, clip_s, lpips).
# DINO-S re-evaluated with test-dir style refs (same protocol as main table).
# CLIP-S/LPIPS from curve_metrics_hf.csv (step 20000 matches main table exactly).
# Faithful data — no scaling; later points may be worse than earlier (user-confirmed).
SAMAM_CURVE = [
    (250,   0.297740, 0.520778, 0.844077),
    (500,   0.222840, 0.524102, 0.628049),
    (1000,  0.365621, 0.554769, 0.567738),
    (2000,  0.454250, 0.585493, 0.456313),
    (3000,  0.468658, 0.586755, 0.380337),
    (5000,  0.475705, 0.587252, 0.339445),
    (7000,  0.475826, 0.590472, 0.320912),
    (20000, 0.415409, 0.581637, 0.243443),
]


def load_samam_curve():
    """Return SaMam curve trajectory (faithful, no scaling)."""
    return list(SAMAM_CURVE)'''

assert OLD_SAMAM in src, "SaMam block not found"
src = src.replace(OLD_SAMAM, NEW_SAMAM)

# ---------------------------------------------------------------------------
# 2. Replace load_samst_curve() with embedded SAMST_CURVE (test-dir refs).
# ---------------------------------------------------------------------------
OLD_SAMST = '''def load_samst_curve():
    """Load SaMST 3-point curve. DINO-S estimated via ratio."""
    path = FIG_DATA / "samst_clip_lpips_curve.csv"
    if not path.exists():
        # Try local
        path = Path("G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_20260610_172206/eval_bundle/clip_lpips_curve.csv")
    if not path.exists():
        return []

    # SaMST scatter point: DINO-S=0.2710, CLIP-S=0.6183
    # Curve CLIP-S ~0.688, scale to match scatter: 0.6183/0.688 = 0.8986
    SAMST_DINO_R = 0.2710 / 0.6183  # 0.4383
    CLIP_SCALE = 0.6183 / 0.6889  # scale curve CLIP-S to scatter final
    LPIPS_SCALE = 0.7490 / 0.6203  # scale curve LPIPS to scatter final

    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch_num"])
            clip_s = float(row["transfer_clip_style"]) * CLIP_SCALE
            lpips = float(row["transfer_content_lpips"]) * LPIPS_SCALE
            dino_s = clip_s * SAMST_DINO_R  # estimate
            points.append((ep, dino_s, clip_s, lpips))
    return points'''

NEW_SAMST = '''# SaMST curve: (epoch, dino_s, clip_s, lpips).
# DINO-S re-evaluated with test-dir style refs (same protocol as main table).
# CLIP-S/LPIPS from README (e5/e15 plateau points; e10 interpolated).
# Faithful data — no scaling.
SAMST_CURVE = [
    (5,  0.441664, 0.7276, 0.6271),
    (10, 0.438931, 0.7262, 0.6263),
    (15, 0.440354, 0.7247, 0.6255),
]


def load_samst_curve():
    """Return SaMST curve trajectory (faithful, no scaling)."""
    return list(SAMST_CURVE)'''

assert OLD_SAMST in src, "SaMST block not found"
src = src.replace(OLD_SAMST, NEW_SAMST)

# ---------------------------------------------------------------------------
# 3. Remove scaling logic from plot_curve_trajectory().
# ---------------------------------------------------------------------------
OLD_SCALE = '''    if len(curve_points) < 2:
        return

    # Scale curve so final point matches scatter final point
    cf = curve_points[-1]
    d_scale = scatter_final["dino_s"] / cf[1] if cf[1] != 0 else 1.0
    c_scale = scatter_final["clip_s"] / cf[2] if cf[2] != 0 else 1.0
    l_scale = scatter_final["lpips"] / cf[3] if cf[3] != 0 else 1.0

    xs, ys = [], []
    for step, dino_s, clip_s, lpips in curve_points:
        sd = dino_s * d_scale
        sc = clip_s * c_scale
        sl = lpips * l_scale
        x = 1.0 - sl
        y = 0.5 * (sd + sc)
        xs.append(x)
        ys.append(y)'''

NEW_SCALE = '''    if len(curve_points) < 2:
        return

    # Faithful plotting — no scaling; use raw metric values directly.
    xs, ys = [], []
    for step, dino_s, clip_s, lpips in curve_points:
        x = 1.0 - lpips
        y = 0.5 * (dino_s + clip_s)
        xs.append(x)
        ys.append(y)'''

assert OLD_SCALE in src, "Scaling block not found"
src = src.replace(OLD_SCALE, NEW_SCALE)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - SAMAM_CURVE embedded (8 points, test-dir refs)")
print("  - SAMST_CURVE embedded (3 points, test-dir refs)")
print("  - plot_curve_trajectory: scaling removed (faithful plot)")
