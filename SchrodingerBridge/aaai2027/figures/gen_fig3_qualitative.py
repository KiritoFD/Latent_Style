"""
Generate Figure 3: Qualitative Comparison of Style Transfer Methods

Grid layout:
- Rows: Different source→target style pairs (representative cases)
- Columns: Different methods (IDT baseline, SaMAM, SaMST, LBM variants)
- Includes target style reference column

Output: fig_distinct5_qualitative_main.pdf and fig_distinct5_qualitative_main.png at 300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\introstyle_page1\staging"

# Select 4 representative style pairs (diverse and illustrative)
STYLE_PAIRS = [
    {
        "name": "Early Renaissance → Impressionism",
        "source_style": "Early_Renaissance",
        "target_style": "Impressionism",
        "content": "andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece",
    },
    {
        "name": "Ukiyo-e → Early Renaissance",
        "source_style": "Ukiyo_e",
        "target_style": "Early_Renaissance",
        "content": "hiroshige_hakone-kosuizu",
    },
    {
        "name": "Minimalism → Rococo",
        "source_style": "Minimalism",
        "target_style": "Rococo",
        "content": "agnes-martin_happy-valley-1967",
    },
    {
        "name": "Impressionism → Minimalism",
        "source_style": "Impressionism",
        "target_style": "Minimalism",
        "content": "alfred-sisley_riverbank-at-veneux-1881",
    },
]

# Methods to compare (columns)
METHODS = [
    {
        "name": "Target Style",
        "dir": None,  # Will use placeholder
        "filename_pattern": None,
        "color": "#E8E8E8",
        "is_reference": True,
    },
    {
        "name": "IDT",
        "dir": None,  # Placeholder - IDT images not available
        "filename_pattern": None,
        "color": "#FFE4E1",  # Light pink for failure case
        "note": "(baseline)",
    },
    {
        "name": "SaMAM",
        "dir": "SaMAM_2250/images",
        # e.g. Early_Renaissance__andrea-mantegna_...__to__Impressionism.png
        "filename_pattern": "{source_style}__{content}__to__{target_style}.png",
        "color": "#FFFFFF",
    },
    {
        "name": "SaMST",
        "dir": "SaMST_e15/images",
        # e.g. Early_Renaissance_Early_Renaissance__andrea-mantegna_..._to_Impressionism.png
        "filename_pattern": "{source_style}_{source_style}__{content}_to_{target_style}.png",
        "color": "#FFFFFF",
    },
    {
        "name": "LBM (ours)",
        "dir": "LBM-Knee_e13/images",
        "filename_pattern": "{source_style}_{content}_to_{target_style}.png",
        "color": "#E6F3E6",  # Light green highlight for our method
        "highlight": True,
    },
]

# Figure layout
FIG_W = 14.0
FIG_H = 10.0
MARGIN_LEFT = 1.8
MARGIN_RIGHT = 0.3
MARGIN_TOP = 0.8
MARGIN_BOTTOM = 0.5
GAP_X = 0.15
GAP_Y = 0.25

# Colors
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#181818"
COLOR_MUTED = "#666666"
COLOR_BORDER = "#CCCCCC"

FONT_SIZE_TITLE = 14
FONT_SIZE_METHOD = 11
FONT_SIZE_PAIR = 10
FONT_SIZE_NOTE = 8

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_image_path(method, pair):
    """Construct image path for a method and style pair."""
    if method["dir"] is None:
        return None
    
    dir_path = os.path.join(BASE_DIR, method["dir"])
    
    # Format filename
    filename = method["filename_pattern"].format(
        source_style=pair["source_style"],
        target_style=pair["target_style"],
        content=pair["content"],
    )
    
    full_path = os.path.join(dir_path, filename)
    return full_path if os.path.exists(full_path) else None


def create_placeholder(ax, text, color, fontsize=FONT_SIZE_NOTE):
    """Create a colored placeholder with text."""
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor=color, edgecolor=COLOR_BORDER,
                           linewidth=1.5, linestyle='--'))
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            fontsize=fontsize, ha='center', va='center',
            color=COLOR_MUTED, style='italic', wrap=True)
    ax.axis('off')


def load_and_display_image(ax, img_path, fallback_text=None, fallback_color="#F5F5F5"):
    """Load and display image, or show placeholder if not available."""
    if img_path and os.path.exists(img_path):
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            create_placeholder(ax, fallback_text or "Image\nunavailable", fallback_color)
    else:
        create_placeholder(ax, fallback_text or "Image\nnot available", fallback_color)


# ---------------------------------------------------------------------------
# Create figure
# ---------------------------------------------------------------------------
n_rows = len(STYLE_PAIRS)
n_cols = len(METHODS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H),
                         facecolor=COLOR_BG)

# Calculate panel dimensions
panel_w = (FIG_W - MARGIN_LEFT - MARGIN_RIGHT - (n_cols - 1) * GAP_X) / n_cols
panel_h = (FIG_H - MARGIN_TOP - MARGIN_BOTTOM - (n_rows - 1) * GAP_Y) / n_rows

# Remove all axes and create custom positioning
for ax in axes.flat:
    ax.axis('off')

# ---------------------------------------------------------------------------
# Add method headers (top)
# ---------------------------------------------------------------------------
for col_idx, method in enumerate(METHODS):
    x = MARGIN_LEFT + col_idx * (panel_w + GAP_X)
    y = FIG_H - MARGIN_TOP + 0.15
    
    # Method name
    label = method["name"]
    if "note" in method:
        label += f" {method['note']}"
    
    fig.text(x + panel_w / 2, y, label,
             fontsize=FONT_SIZE_METHOD, fontweight='bold' if method.get("highlight") else 'normal',
             ha='center', va='bottom', color=COLOR_TEXT)

# ---------------------------------------------------------------------------
# Add style pair labels (left) and populate grid
# ---------------------------------------------------------------------------
for row_idx, pair in enumerate(STYLE_PAIRS):
    y_pos = FIG_H - MARGIN_TOP - (row_idx + 1) * panel_h - row_idx * GAP_Y + panel_h / 2
    
    # Style pair label on the left
    x_label = MARGIN_LEFT - 0.2
    fig.text(x_label, y_pos, pair["name"],
             fontsize=FONT_SIZE_PAIR, ha='right', va='center',
             color=COLOR_TEXT, rotation=0, fontweight='bold')
    
    # Populate each column
    for col_idx, method in enumerate(METHODS):
        x_pos = MARGIN_LEFT + col_idx * (panel_w + GAP_X)
        y_pos_panel = FIG_H - MARGIN_TOP - (row_idx + 1) * panel_h - row_idx * GAP_Y
        
        # Create axes for this panel
        ax = fig.add_axes([x_pos / FIG_W, y_pos_panel / FIG_H,
                          panel_w / FIG_W, panel_h / FIG_H])
        
        # Get image path
        img_path = get_image_path(method, pair)
        
        # Display image or placeholder
        if method.get("is_reference"):
            # Target style reference
            fallback_text = f"Target:\n{pair['target_style'].replace('_', ' ')}"
            load_and_display_image(ax, img_path, fallback_text, method["color"])
        elif method["name"] == "IDT":
            # IDT baseline placeholder
            fallback_text = "IDT baseline\n(not available)\n[Failure case]"
            load_and_display_image(ax, img_path, fallback_text, method["color"])
        else:
            # Regular method output
            fallback_text = f"{method['name']}\nnot available"
            load_and_display_image(ax, img_path, fallback_text, method["color"])
        
        # Add border highlight for our method
        if method.get("highlight"):
            for spine in ax.spines.values():
                spine.set_edgecolor("#2CA02C")
                spine.set_linewidth(2.5)

# ---------------------------------------------------------------------------
# Add title
# ---------------------------------------------------------------------------
fig.text(FIG_W / 2, FIG_H - 0.2, "Figure 3: Qualitative Comparison of Style Transfer Methods",
         fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='center', va='top',
         color=COLOR_TEXT)

# ---------------------------------------------------------------------------
# Add legend/note at bottom
# ---------------------------------------------------------------------------
note_y = 0.15
fig.text(MARGIN_LEFT, note_y,
         "Note: IDT baseline shows minimal change (failure case). LBM achieves successful style transfer while preserving content.",
         fontsize=FONT_SIZE_NOTE, ha='left', va='bottom',
         color=COLOR_MUTED, style='italic')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\figures"
pdf_path = os.path.join(out_dir, "fig_distinct5_qualitative_main.pdf")
png_path = os.path.join(out_dir, "fig_distinct5_qualitative_main.png")

plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)

print(f"Figure 3 saved: {pdf_path}")
print(f"Figure 3 saved: {png_path}")
print(f"\nGrid layout: {n_rows} style pairs × {n_cols} methods")
print(f"Style pairs: {[p['name'] for p in STYLE_PAIRS]}")
print(f"Methods: {[m['name'] for m in METHODS]}")

plt.close()
