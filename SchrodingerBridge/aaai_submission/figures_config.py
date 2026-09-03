"""
Configuration and data for all figures.

This module centralizes all data, captions, and configuration parameters
for consistent figure generation across the project.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================================
# Quality Tradeoff Data (LPIPS vs CLIP-Style)
# ============================================================================
QUALITY_TRADEOFF_DATA = {
    'Ours e7':  {'lpips': 0.449, 'clip_s': 0.7165, 'params': 3.9, 'ec': 0.395, 'label': 'Ours e7'},
    'Ours e8':  {'lpips': 0.451, 'clip_s': 0.7158, 'params': 3.9, 'ec': 0.393, 'label': 'Ours e8'},
    'SaMST':    {'lpips': 0.466, 'clip_s': 0.7195, 'params': 6.0, 'ec': 0.384, 'label': 'SaMST'},
    'S2WAT':    {'lpips': 0.526, 'clip_s': 0.714, 'params': 65,  'ec': 0.338},
    'AdaIN':    {'lpips': 0.630, 'clip_s': 0.713, 'params': 5,   'ec': 0.264},
    'StyleID':  {'lpips': 0.750, 'clip_s': 0.760, 'params': 30,  'ec': 0.190},
    'CAST':     {'lpips': 0.726, 'clip_s': 0.665, 'params': 7.0, 'ec': 0.182},
}

# ============================================================================
# Training Efficiency Data
# ============================================================================
TRAIN_EFFICIENCY_DATA = [
    ("Ours", 0.393, 310, 3.9, "#e64b35"),
    ("SaMST", 0.384, 6769, 6.0, "#000000"),
    ("S2WAT", 0.338, 10600, 65, "#00a087"),
    ("AdaIN", 0.264, 9220, 5, "#3c5488"),
]

TRAIN_EFFICIENCY_OFFSETS = {
    "Ours": (-500, 0.008),      # Left and up
    "SaMST": (250, 0.008),      # Right and up
    "S2WAT": (250, -0.010),     # Right and down
    "AdaIN": (250, 0.008),      # Right and up
}

# ============================================================================
# Artifact Diagnostic Data
# ============================================================================
ARTIFACT_DIAGNOSTICS_DATA = {
    "Ours": {
        "mean_sq_grad": 0.0087,
        "high_freq_energy": 0.156,
        "edge_crispness": 0.283,
        "sat_artifact": 0.0034,
    },
    "SaMST": {
        "mean_sq_grad": 0.0091,
        "high_freq_energy": 0.168,
        "edge_crispness": 0.291,
        "sat_artifact": 0.0041,
    },
}

# ============================================================================
# Ablation Study Data
# ============================================================================
ABLATION_DATA = [
    ("D0★", 0.751, 0.393, "#e64b35", "Full model"),
    ("D1", 0.701, 0.418, "#e8b4af", "No SWD"),
    ("D2", 0.627, 0.466, "#ebaeaa", "No kinetic"),
    ("D4", 0.747, 0.391, "#efc4c1", "No skip fuse"),
    ("D5", 0.750, 0.391, "#efc4c1", "No spatial prior"),
    ("D6", 0.747, 0.390, "#efc4c1", "Shallow bottleneck"),
    ("D7", 0.748, 0.389, "#efc4c1", "3 attn heads"),
    ("D8", 0.746, 0.388, "#efc4c1", "No layer norm"),
    ("D9", 0.749, 0.392, "#efc4c1", "RMSNorm"),
    ("D10", 0.750, 0.391, "#efc4c1", "Conv blocks"),
    ("D11", 0.746, 0.387, "#efc4c1", "Fewer dims"),
]

# ============================================================================
# Weight Sweep Data
# ============================================================================
WEIGHT_SWEEP_DATA = {
    "K1": {"name": "K1 (Style-heavy)", "color": "#ffc000"},
    "K2": {"name": "K2 (Balanced)", "color": "#70ad47"},
    "K3": {"name": "K3 (Content-heavy)", "color": "#4472c4"},
}

# ============================================================================
# Figure Captions
# ============================================================================
FIGURE_CAPTIONS = {
    "fig_framework_overview": (
        "Overview of the proposed latent bridge-inspired style transfer framework. "
        "A content image is encoded into a compact latent, transported by a style-conditioned "
        "velocity field, constrained by terminal SWD and kinetic regularization, and decoded into a stylized output."
    ),
    "fig_quality_tradeoff": (
        "Strict-750 style-content trade-off. Ours is close to SaMST in raw CLIP-style while retaining "
        "a better LPIPS-based trade-off than several baselines; StyleID reaches high style but collapses content."
    ),
    "fig_artifact_diagnostics": (
        "Artifact-sensitive diagnostics for Ours and SaMST. Mixed-scale metrics are pairwise normalized for readability; "
        "the raw numeric values are reported in the experimental table."
    ),
    "fig_ablation_pareto": (
        "12-point destructive ablation (7-epoch). D0 full (★) anchors the upper-right cluster. "
        "D1 no SWD trades style for content identity (high invLPIPS). D2 no kinetic boosts style at the cost of severe content degradation. "
        "D4-D11 are architectural variants with marginal difference from D0, confirming robustness."
    ),
    "fig_weight_sweep_summary": (
        "Summary of the 40-run category-weight sweep. K2 balanced default gives the best EC, "
        "whereas K1 balanced default gives the best raw CLIP-style among evaluated checkpoints."
    ),
    "fig_train_efficiency_pareto": (
        "Training-efficiency trade-off (linear scale, training-free methods omitted). Ours achieves the highest EC (0.393) "
        "at the lowest training cost (310s), while SaMST (6769s) and S2WAT (10600s) require significantly more compute for comparable quality."
    ),
}

# ============================================================================
# Color Schemes
# ============================================================================
QUALITY_COLORS = {
    'Ours e7': '#E74C3C', 
    'Ours e8': '#E74C3C', 
    'SaMST': '#F39C12', 
    'S2WAT': '#8E44AD', 
    'AdaIN': '#2ECC71', 
    'StyleID': '#3498DB', 
    'CAST': '#95A5A6'
}

# ============================================================================
# Plot Configuration
# ============================================================================
PLOT_CONFIG = {
    "font_family": "DejaVu Sans",
    "font_size": 10,
    "axes_titlesize": 12,
    "axes_labelsize": 10,
    "legend_fontsize": 8,
    "figure_dpi": 150,
}
