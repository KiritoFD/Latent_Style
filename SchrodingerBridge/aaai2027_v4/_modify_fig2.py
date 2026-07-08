"""
Modify the existing draw.io SVG (framework_sfm_main.svg) by adding overlay groups.
Then render with Chrome headless to PNG.
Uses accurate coordinates extracted from the original SVG geometry.
"""
import os, subprocess
from xml.etree import ElementTree as ET

DIR = os.path.dirname(os.path.abspath(__file__))
SVG = os.path.join(DIR, "framework_sfm_main.svg")
PNG = os.path.join(DIR, "framework_sfm_main.png")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

NS = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

# Read original SVG (restore from backup first)
ORIG = os.path.join(DIR, "framework_sfm_main_ORIG.svg")
with open(ORIG, "r", encoding="utf-8") as f:
    svg_text = f.read()

root = ET.fromstring(svg_text)

# Extend viewBox upward by 90 units to make room for timeline
vb = root.get("viewBox", "0 0 1614 602").split()
root.set("viewBox", f"{vb[0]} -90 {vb[2]} {int(vb[3])+90}")
root.set("height", "692px")

# Helper functions
def make(tag, attrs):
    return ET.Element(f"{{{NS['svg']}}}{tag}", attrs)

def text(x, y, txt, size=14, fill="#1f2937", anchor="start", weight="normal", font="Helvetica"):
    el = make("text", {
        "x": str(x), "y": str(y), "font-family": font, "font-size": str(size),
        "fill": fill, "text-anchor": anchor, "font-weight": weight,
        "dominant-baseline": "middle"
    })
    el.text = txt
    return el

# ── DEFS: arrow markers ──
defs = make("defs", {})
for aid, col in [("arrow-red", "#c53030"), ("arrow-red-small", "#c53030"), ("arrow-green", "#276749"), ("arrow-blue", "#2b6cb0")]:
    m = make("marker", {"id": aid, "markerWidth": "10", "markerHeight": "10",
                        "refX": "9", "refY": "3", "orient": "auto", "markerUnits": "strokeWidth"})
    m.append(make("path", {"d": "M0,0 L0,6 L9,3 z", "fill": col}))
    defs.append(m)
root.insert(0, defs)

g = make("g", {"id": "weave-overlays"})

# ═══════════════════════════════════════════════════════
# 1. TIMELINE COMPARISON (top, y = -85 to -10)
# ═══════════════════════════════════════════════════════
g.append(make("rect", {"x": "30", "y": "-85", "width": "750", "height": "75", "rx": "8",
                       "fill": "#fff5f5", "stroke": "#ffcccc", "stroke-width": "1"}))
g.append(make("rect", {"x": "830", "y": "-85", "width": "750", "height": "75", "rx": "8",
                       "fill": "#f5fff8", "stroke": "#ccffcc", "stroke-width": "1"}))

g.append(text(405, -75, "Prior Arts: Per-step Style Injection", 16, "#c53030", "middle", "bold"))
g.append(text(405, -58, "(variance decay / content collapse)", 12, "#718096", "middle"))
g.append(text(1205, -75, "WEAVE: Endpoint-only Alignment", 16, "#276749", "middle", "bold"))
g.append(text(1205, -58, "(WCT + Semantic SWD only at t=1)", 12, "#718096", "middle"))

# left timeline
l_x0, l_x1, l_y = 70, 740, -30
g.append(make("line", {"x1": str(l_x0), "y1": str(l_y), "x2": str(l_x1), "y2": str(l_y),
                       "stroke": "#c53030", "stroke-width": "3", "marker-end": "url(#arrow-red)"}))
for i in range(6):
    x = l_x0 + (l_x1 - l_x0) * i // 5
    g.append(make("circle", {"cx": str(x), "cy": str(l_y), "r": "6", "fill": "#c53030"}))
    g.append(text(x, l_y - 16, f"t={i/5:.1f}", 10, "#c53030", "middle"))
    g.append(make("line", {"x1": str(x), "y1": str(l_y - 35), "x2": str(x), "y2": str(l_y - 12),
                           "stroke": "#c53030", "stroke-width": "2", "marker-end": "url(#arrow-red-small)"}))
    if i == 2:
        g.append(text(x, l_y - 48, "Style\nInject", 10, "#c53030", "middle"))
g.append(make("line", {"x1": str(l_x1 + 18), "y1": str(l_y - 12), "x2": str(l_x1 + 38), "y2": str(l_y + 12),
                       "stroke": "#c53030", "stroke-width": "3"}))
g.append(make("line", {"x1": str(l_x1 + 18), "y1": str(l_y + 12), "x2": str(l_x1 + 38), "y2": str(l_y - 12),
                       "stroke": "#c53030", "stroke-width": "3"}))

# right timeline
r_x0, r_x1, r_y = 870, 1540, -30
g.append(make("line", {"x1": str(r_x0), "y1": str(r_y), "x2": str(r_x1 - 50), "y2": str(r_y),
                       "stroke": "#276749", "stroke-width": "3"}))
for i in range(1, 6):
    x = r_x0 + (r_x1 - 50 - r_x0) * i // 6
    g.append(make("circle", {"cx": str(x), "cy": str(r_y), "r": "3", "fill": "#a0aec0"}))
    if i % 2 == 0:
        g.append(text(x, r_y + 14, f"t={i/5:.1f}", 10, "#718096", "middle"))
ep_x = r_x1 - 40
g.append(make("rect", {"x": str(ep_x - 55), "y": str(r_y - 25), "width": "110", "height": "50", "rx": "6",
                       "fill": "#f0fff4", "stroke": "#276749", "stroke-width": "2"}))
g.append(text(ep_x, r_y - 12, "t = 1", 14, "#276749", "middle", "bold"))
g.append(text(ep_x, r_y + 6, "Endpoint", 12, "#276749", "middle"))
g.append(text(ep_x, r_y + 20, "WCT + SWD", 11, "#276749", "middle"))
g.append(make("polygon", {"points": f"{ep_x + 55},{r_y} {ep_x + 45},{r_y - 6} {ep_x + 45},{r_y + 6}", "fill": "#276749"}))
g.append(text(r_x1 + 15, r_y, "\u2713", 24, "#276749", "middle", "bold"))

# ═══════════════════════════════════════════════════════
# 2. SUBBAND COLOR HIGHLIGHTS (accurate coords from SVG)
# LL: x=140,y=324,w=30,h=24  (fill #dbeafe, stroke #1e40af)
# LH: x=194,y=324,w=30,h=24  (fill #ecfdf5, stroke #047857)
# HL: x=140,y=362,w=30,h=24  (fill #ecfdf5, stroke #047857)
# HH: x=192,y=362,w=34,h=40  (actually a larger transparent rect)
# ═══════════════════════════════════════════════════════
subbands = [
    ("LL", 140, 324, 30, 24, "#2b6cb0", "Base Locking\n(content / weak \u03bb_LL=0.3)"),
    ("LH", 194, 324, 30, 24, "#dd6b20", "Style HF"),
    ("HL", 140, 362, 30, 24, "#dd6b20", "Style HF"),
    ("HH", 192, 362, 34, 40, "#dd6b20", "Endpoint-only"),
]
for name, x, y, w, h, col, label in subbands:
    pad = 3
    g.append(make("rect", {"x": str(x - pad), "y": str(y - pad), "width": str(w + 2*pad),
                           "height": str(h + 2*pad), "rx": "4", "fill": "none", "stroke": col, "stroke-width": "3"}))
    tag_y = y + h + 16
    tag_w = 90
    g.append(make("rect", {"x": str(x + w/2 - tag_w/2), "y": str(tag_y - 7), "width": str(tag_w),
                           "height": "18", "rx": "3", "fill": col}))
    lines = label.split("\n")
    g.append(text(x + w/2, tag_y + 2, lines[0], 9, "#ffffff", "middle", "bold"))

# ═══════════════════════════════════════════════════════
# 3. STYLE QUERY BLOCK SYMBOL
# Style path from top-right (Style ID ~ x=1106,y=44) to LL (x=155,y=336)
# Place red block marker near LL entry
# ═══════════════════════════════════════════════════════
block_x, block_y = 120, 336
g.append(make("circle", {"cx": str(block_x), "cy": str(block_y), "r": "12",
                         "fill": "#fff5f5", "stroke": "#c53030", "stroke-width": "2"}))
g.append(text(block_x, block_y, "\u2717", 14, "#c53030", "middle", "bold"))
g.append(text(block_x, block_y + 22, "blocked", 10, "#c53030", "middle"))

# ═══════════════════════════════════════════════════════
# 4. SEMANTIC SWD MODULE (next to Fiber WCT at x=1250, y=228, w=118, h=108)
# ═══════════════════════════════════════════════════════
swd_x, swd_y = 1410, 282  # to the right of WCT module
g.append(make("rect", {"x": str(swd_x - 85), "y": str(swd_y - 55), "width": "170", "height": "110", "rx": "8",
                       "fill": "#ebf8ff", "stroke": "#2b6cb0", "stroke-width": "2"}))
g.append(text(swd_x, swd_y - 42, "Semantic Region SWD", 13, "#2b6cb0", "middle", "bold"))
# sub-blocks
for i, txt in enumerate(["K-means\npartition", "Quantile\nmatching"]):
    bx = swd_x - 55 + i * 110
    g.append(make("rect", {"x": str(bx - 42), "y": str(swd_y - 12), "width": "84", "height": "48", "rx": "4",
                           "fill": "#ffffff", "stroke": "#4299e1", "stroke-width": "1.5"}))
    g.append(text(bx, swd_y + 6, txt, 10, "#2b6cb0", "middle"))
# arrow between sub-blocks
g.append(make("line", {"x1": str(swd_x - 15), "y1": str(swd_y + 12), "x2": str(swd_x + 15), "y2": str(swd_y + 12),
                       "stroke": "#2b6cb0", "stroke-width": "2", "marker-end": "url(#arrow-blue)"}))
# connection from WCT to SWD
g.append(make("line", {"x1": str(1250 + 118), "y1": str(228 + 54), "x2": str(swd_x - 85), "y2": str(swd_y),
                       "stroke": "#2b6cb0", "stroke-width": "2", "stroke-dasharray": "5,3"}))

# ═══════════════════════════════════════════════════════
# 5. LEGEND
# ═══════════════════════════════════════════════════════
leg_x, leg_y = 1250, 500
g.append(make("rect", {"x": str(leg_x - 10), "y": str(leg_y - 18), "width": "300", "height": "56", "rx": "6",
                       "fill": "#ffffff", "stroke": "#e2e8f0", "stroke-width": "1"}))
g.append(text(leg_x, leg_y - 8, "Orthogonal Subspaces", 12, "#2d3748", "start", "bold"))
g.append(make("rect", {"x": str(leg_x), "y": str(leg_y + 6), "width": "18", "height": "12", "fill": "#2b6cb0"}))
g.append(text(leg_x + 24, leg_y + 12, "Content / LL (structure)", 11, "#4a5568", "start"))
g.append(make("rect", {"x": str(leg_x + 160), "y": str(leg_y + 6), "width": "18", "height": "12", "fill": "#dd6b20"}))
g.append(text(leg_x + 184, leg_y + 12, "Style / HF (texture)", 11, "#4a5568", "start"))

# ═══════════════════════════════════════════════════════
# 6. COVER OLD NOTE, WRITE NEW ONE
# The original note is at the bottom (approximately y=555-580)
# ═══════════════════════════════════════════════════════
g.append(make("rect", {"x": "20", "y": "545", "width": "1570", "height": "45", "fill": "#ffffff", "stroke": "none"}))
g.append(text(35, 570,
    "LL is weakly supervised (\u03bb_LL = 0.3) for global color stability, while style queries strictly bypass it. "
    "WCT and Semantic Region SWD are applied only at the endpoint t=1.",
    13, "#4a5568", "start", "normal"))

# Insert overlay group
root.append(g)

# Serialize
ET.register_namespace("", NS["svg"])
tree = ET.ElementTree(root)
tree.write(SVG, encoding="utf-8", xml_declaration=True)
print("modified SVG saved to", SVG)

# Render with Chrome
subprocess.run([
    CHROME, "--headless", "--disable-gpu", "--no-sandbox",
    "--screenshot=" + PNG, "--window-size=1800,900",
    "file:///" + SVG.replace("\\", "/")
], check=True)
print("rendered PNG", PNG)
