"""Generate AAAI-style main architecture diagram for Spectral ODE Bridge (v5.1).

Fixes from v5:
- Remove heavy background bands; use clean section labels.
- Add right margin so export doesn't clip Output / Decoder.
- Slightly tighter vertical layout.
- Single consistent background color.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

PAGE_W, PAGE_H = 1200, 760

COLORS = {
    "content": "#E8F4FC",
    "content_dark": "#2E75B6",
    "style": "#FFF2E6",
    "style_dark": "#C65911",
    "spectral": "#E2F0D9",
    "spectral_dark": "#548235",
    "train": "#F2E6F5",
    "train_dark": "#7B4F8E",
    "dead": "#F2F2F2",
    "dead_dark": "#7F7F7F",
    "backbone": "#E1D5E7",
    "backbone_dark": "#5B3A6C",
    "insight": "#FFF8DC",
    "insight_dark": "#B7950B",
    "text": "#1F2937",
    "page_bg": "#FFFFFF",
}

ids = {"node": 0, "edge": 0, "txt": 0}
root = ET.Element("root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def new_id(kind):
    ids[kind] += 1
    return f"{kind}_{ids[kind]}"


def cell(value, x, y, w, h, fill, stroke, font_size=12, bold=False,
         dashed="0", stroke_width=1, align="center", font_color=None):
    font_color = font_color or COLORS["text"]
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;arcSize=5;"
        f"fillColor={fill};strokeColor={stroke};"
        f"fontFamily=Helvetica;fontSize={font_size};fontColor={font_color};"
        f"align={align};verticalAlign=middle;dashed={dashed};strokeWidth={stroke_width};"
    )
    if bold:
        style += "fontStyle=1;"
    c = ET.SubElement(root, "mxCell", {
        "id": new_id("node"),
        "value": value,
        "style": style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(c, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry",
    })
    return c


def txt(value, x, y, w, h, font_size=11, align="left", color=None, bold=False):
    color = color or COLORS["text"]
    style = (
        f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign=middle;whiteSpace=wrap;rounded=0;"
        f"fontFamily=Helvetica;fontSize={font_size};fontColor={color};"
    )
    if bold:
        style += "fontStyle=1;"
    c = ET.SubElement(root, "mxCell", {
        "id": new_id("txt"),
        "value": value,
        "style": style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(c, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry",
    })
    return c


def edge(source, target, dashed="0", color="#666666", label="", waypoints=None):
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        f"html=1;strokeColor={color};dashed={dashed};"
        f"fontFamily=Helvetica;fontSize=10;fontColor={COLORS['text']};"
        f"startArrow=none;startFill=1;endArrow=classic;endFill=1;"
    )
    if label:
        style += "labelBackgroundColor=#FFFFFF;"
    attrs = {
        "id": new_id("edge"),
        "value": label,
        "style": style,
        "edge": "1",
        "parent": "1",
        "source": source,
        "target": target,
    }
    e = ET.SubElement(root, "mxCell", attrs)
    geom = ET.SubElement(e, "mxGeometry", {"relative": "1", "as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for px, py in waypoints:
            ET.SubElement(arr, "Point", {"x": str(px), "y": str(py)})
    return e


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
bg = ET.SubElement(root, "mxCell", {
    "id": "bg",
    "value": "",
    "style": f"rounded=0;whiteSpace=wrap;html=1;fillColor={COLORS['page_bg']};strokeColor=none;",
    "vertex": "1",
    "parent": "1",
})
ET.SubElement(bg, "mxGeometry", {
    "x": "0", "y": "0", "width": str(PAGE_W), "height": str(PAGE_H), "as": "geometry",
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
cell("Core Insight\n"
     "Euclidean FM: one velocity field forces structure & style to move together.\n"
     "Spectral FM: each Haar sub-band gets its own velocity field.",
     35, 20, 560, 60, COLORS["insight"], COLORS["insight_dark"],
     font_size=13, align="left")

# ---------------------------------------------------------------------------
# Style control
# ---------------------------------------------------------------------------
txt("Style control", 35, 100, 140, 18, font_size=13, bold=True, color=COLORS["style_dark"])
cell("Style", 60, 122, 75, 42, COLORS["style"], COLORS["style_dark"], font_size=12)
s_mem = cell("Style Memory\n(learnable)\n5 styles × 256", 155, 116, 120, 55,
             COLORS["style"], COLORS["style_dark"], font_size=11)
s_tok = cell("Style Tokens\nS ∈ R^{256×64}", 295, 116, 120, 55,
             COLORS["style"], COLORS["style_dark"], font_size=11)
edge("node_1", s_mem.get("id"), color=COLORS["style_dark"])
edge(s_mem.get("id"), s_tok.get("id"), color=COLORS["style_dark"])

# ---------------------------------------------------------------------------
# Inference path
# ---------------------------------------------------------------------------
txt("Inference path", 35, 195, 140, 18, font_size=13, bold=True, color=COLORS["content_dark"])

content = cell("Content", 60, 310, 75, 42, COLORS["content"], COLORS["content_dark"], font_size=12)
enc = cell("VAE\nEncoder", 155, 305, 80, 52, COLORS["content"], COLORS["content_dark"], font_size=11)
z0 = cell("z₀", 255, 310, 50, 42, COLORS["content"], COLORS["content_dark"], font_size=16)
edge(content.get("id"), enc.get("id"), color=COLORS["content_dark"])
edge(enc.get("id"), z0.get("id"), color=COLORS["content_dark"])

# Haar DWT
ll = cell("LL", 340, 265, 50, 38, "#D0E8F7", COLORS["content_dark"], font_size=14, bold=True)
lh = cell("LH", 415, 240, 48, 32, COLORS["spectral"], COLORS["spectral_dark"], font_size=13)
hl = cell("HL", 415, 282, 48, 32, COLORS["spectral"], COLORS["spectral_dark"], font_size=13)
hh = cell("HH", 415, 330, 48, 32, COLORS["dead"], COLORS["dead_dark"], font_size=13)
txt("structure\n& tone", 333, 307, 64, 28, font_size=10, align="center")
txt("vertical\ntexture", 410, 213, 58, 26, font_size=10, align="center")
txt("horizontal\ntexture", 410, 316, 58, 26, font_size=10, align="center")
txt("discarded", 475, 333, 60, 22, font_size=10, align="left", color="#C00000")

edge(z0.get("id"), ll.get("id"), color=COLORS["content_dark"])
edge(z0.get("id"), lh.get("id"), color=COLORS["spectral_dark"])
edge(z0.get("id"), hl.get("id"), color=COLORS["spectral_dark"])
edge(z0.get("id"), hh.get("id"), color=COLORS["dead_dark"])

# Stack & Project
stack = cell("Stack &\nProject\n4 bands × C → hidden", 505, 258, 110, 62,
             COLORS["spectral"], COLORS["spectral_dark"], font_size=11)
edge(ll.get("id"), stack.get("id"), color=COLORS["content_dark"])
edge(lh.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(439, 256), (439, 278), (505, 278)])
edge(hl.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(439, 298), (439, 300), (505, 300)])

# Shared Backbone
bb_container = cell("", 645, 240, 140, 115, COLORS["backbone"], COLORS["backbone_dark"],
                    dashed="1", stroke_width=2)
bb = cell("Shared\nBackbone\n(×4 blocks)", 657, 253, 116, 72,
          COLORS["backbone"], COLORS["backbone_dark"], font_size=12, bold=True)
txt("t → time emb", 657, 330, 116, 14, font_size=9, align="center", color="#555555")
edge(stack.get("id"), bb.get("id"), color=COLORS["backbone_dark"])

# One Block inset
block_bg = cell("", 810, 232, 180, 105, "#FBECEA", "#A67C7A", stroke_width=1)
txt("(a) One Block", 815, 238, 100, 15, font_size=12, bold=True)
txt("AdaLN(time) → Self-Attn\n"
    "DWT-Route Cross-Attn\n"
    "ReLU² Attention + tanh gate\n"
    "→ FFN",
    815, 256, 170, 76, font_size=11, align="left")

# Velocity heads
v_ll = cell("v_LL", 645, 420, 52, 36, "#D0E8F7", COLORS["content_dark"], font_size=14)
v_lh = cell("v_LH", 707, 420, 52, 36, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
v_hl = cell("v_HL", 769, 420, 52, 36, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
txt("Per-subband Heads", 645, 401, 176, 15, font_size=11, bold=True, align="center")
edge(bb.get("id"), v_ll.get("id"), color=COLORS["content_dark"],
     waypoints=[(685, 325), (685, 380), (671, 380), (671, 420)])
edge(bb.get("id"), v_lh.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(705, 325), (705, 388), (733, 388), (733, 420)])
edge(bb.get("id"), v_hl.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(725, 325), (725, 380), (795, 380), (795, 420)])

# ODE
ode_int = cell("Spectral ODE Integrator\nh_i ← h_i + v_i · dt",
               645, 485, 190, 52, COLORS["spectral"], COLORS["spectral_dark"], font_size=12)
edge(v_ll.get("id"), ode_int.get("id"), color=COLORS["content_dark"],
     waypoints=[(671, 456), (671, 485)])
edge(v_lh.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(733, 456), (733, 485)])
edge(v_hl.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(795, 456), (795, 485)])

# K steps self-loop
edge(ode_int.get("id"), ode_int.get("id"), dashed="1", color=COLORS["spectral_dark"], label="K steps",
     waypoints=[(835, 511)])

# iDWT + endpoint + output
idwt = cell("iDWT", 645, 560, 62, 36, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
edge(ode_int.get("id"), idwt.get("id"), color=COLORS["spectral_dark"])

endpoint = cell("Endpoint\nAdaIN / WCT", 730, 552, 108, 52,
                COLORS["style"], COLORS["style_dark"], font_size=12, bold=True)
edge(idwt.get("id"), endpoint.get("id"), color=COLORS["spectral_dark"])

zT = cell("z_T", 865, 556, 52, 44, COLORS["content"], COLORS["content_dark"], font_size=16)
dec = cell("VAE\nDecoder", 940, 551, 72, 54, COLORS["content"], COLORS["content_dark"], font_size=11)
out = cell("Output", 1035, 556, 65, 44, COLORS["content"], COLORS["content_dark"], font_size=12)
edge(endpoint.get("id"), zT.get("id"), color=COLORS["content_dark"])
edge(zT.get("id"), dec.get("id"), color=COLORS["content_dark"])
edge(dec.get("id"), out.get("id"), color=COLORS["content_dark"])

# Style dashed edges
edge(s_tok.get("id"), bb.get("id"), dashed="1", color=COLORS["style_dark"],
     waypoints=[(355, 144), (355, 315), (657, 315)])
edge(s_tok.get("id"), endpoint.get("id"), dashed="1", color=COLORS["style_dark"],
     waypoints=[(355, 144), (355, 578), (730, 578)])

# ---------------------------------------------------------------------------
# Training supervision
# ---------------------------------------------------------------------------
txt("Training supervision", 35, 535, 180, 18, font_size=13, bold=True, color=COLORS["train_dark"])

txt("x_t = (1 − t)·z₀ + t·z_target", 60, 560, 260, 16, font_size=12)
xt_box = cell("x_t", 60, 582, 60, 38, COLORS["train"], COLORS["train_dark"], font_size=16)
dwt = cell("DWT", 140, 582, 55, 38, COLORS["train"], COLORS["train_dark"], font_size=13)
pred = cell("Predict\nv_LL, v_LH, v_HL", 220, 575, 110, 52,
            COLORS["train"], COLORS["train_dark"], font_size=11)
tgt = cell("Target\nΔ_i = DWT(z_t − z₀)_i", 355, 575, 130, 52,
           COLORS["train"], COLORS["train_dark"], font_size=11)
loss = cell("L = w_LL MSE(v_LL, Δ_LL) + w_LH MSE(v_LH, Δ_LH) + w_HL MSE(v_HL, Δ_HL)",
            510, 578, 320, 46, COLORS["train"], COLORS["train_dark"], font_size=11)
edge(xt_box.get("id"), dwt.get("id"), color=COLORS["train_dark"])
edge(dwt.get("id"), pred.get("id"), color=COLORS["train_dark"])
edge(pred.get("id"), loss.get("id"), color=COLORS["train_dark"])
edge(tgt.get("id"), loss.get("id"), color=COLORS["train_dark"])

# Feedback edges
edge(xt_box.get("id"), z0.get("id"), dashed="1", color=COLORS["train_dark"],
     waypoints=[(90, 582), (90, 331), (255, 331)])
edge(loss.get("id"), bb.get("id"), dashed="1", color=COLORS["train_dark"],
     waypoints=[(670, 578), (670, 390), (705, 390), (705, 325)])

# ---------------------------------------------------------------------------
# Legend and caption
# ---------------------------------------------------------------------------
txt("Legend:", 60, 665, 55, 16, font_size=12, bold=True)
cell("", 110, 668, 20, 12, COLORS["content"], COLORS["content_dark"], font_size=8)
txt("content", 135, 665, 55, 16, font_size=10)
cell("", 200, 668, 20, 12, COLORS["style"], COLORS["style_dark"], font_size=8)
txt("style", 225, 665, 40, 16, font_size=10)
cell("", 275, 668, 20, 12, COLORS["spectral"], COLORS["spectral_dark"], font_size=8)
txt("spectral", 300, 665, 55, 16, font_size=10)
cell("", 365, 668, 20, 12, COLORS["train"], COLORS["train_dark"], font_size=8)
txt("training", 390, 665, 55, 16, font_size=10)
txt("— inference    · · · training", 465, 665, 150, 16, font_size=10)

txt("Figure 2. Overview of Spectral ODE Bridge. The content latent is decomposed into Haar sub-bands; "
    "LL preserves structure and tone, LH/HL carry oriented texture and edge information, and HH is discarded. "
    "A shared backbone learns a cross-band representation, while three independent velocity heads predict "
    "per-subband velocities. The ODE is integrated in the spectral domain, and style statistics are injected "
    "only at the endpoint via AdaIN/WCT.",
    35, 695, 1130, 50, font_size=11, align="left")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
mxfile = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "modified": "2026-07-03T00:00:00.000Z",
    "agent": "custom-python-generator-v5-1",
    "etag": "aaai-arch-v5-1",
    "version": "24.0.0",
    "type": "device",
})
diagram = ET.SubElement(mxfile, "diagram", {"name": "Page-1", "id": "aaai-arch-page-v5-1"})
graph_model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1434", "dy": "780", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1",
    "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
    "math": "0", "shadow": "0",
})
graph_model.append(root)

ET.indent(mxfile, space="")
out_path = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v5_1.drawio")
out_path.write_bytes(ET.tostring(mxfile, encoding="utf-8", xml_declaration=True))
print(f"Saved {out_path}")
