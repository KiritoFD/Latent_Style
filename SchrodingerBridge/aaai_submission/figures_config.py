"""Shared figure metadata for auxiliary AAAI submission plots."""

# The current paper only uses the ablation caption from this module, but the
# plotting defaults remain shared so helper scripts render consistently.

FIGURE_CAPTIONS = {
    "fig_ablation_pareto": (
        "Representative destructive ablations from the historical standard benchmark. "
        "Full LBM anchors the clean frontier; removing SA-SWD preserves near-identity structure while weakening "
        "target-style commitment, removing kinetic regularization increases style at the cost of geometry and "
        "content stability, and strong color loss also degrades the trade-off."
    ),
}

PLOT_CONFIG = {
    "font_family": "serif",
    "font_size": 11,
    "axes_titlesize": 13,
    "axes_labelsize": 11,
    "legend_fontsize": 9,
    "figure_dpi": 300,
}
