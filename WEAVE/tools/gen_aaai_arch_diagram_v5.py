"""Generate AAAI-style main architecture diagram for Spectral ODE Bridge (v5).

Redesign goals:
- Clean white background, print-friendly palette.
- No embedded images; use schematic boxes with clear labels.
- Horizontal main flow with compact grouping.
- Fix double-escaped ampersands and inconsistent fonts.
- Better backbone block inset and dead-head marker.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

PAGE_W, PAGE_H = 1200, 780
MARGIN = 30

# Print-friendly palette (high contrast, low saturation)
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
    "white": "#FFFFFF",
    "page_bg": "#FAFAFA",
}

ids = {"node": 0, "edge": 0, "txt": 0, "band": 0}
root = ET.Element("root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def new_id(kind):
    ids[kind] += 1
    return f"{kind}_{ids[kind]}"


def cell(value, x, y, w, h, fill, stroke, font_size=12, bold=False,
         dashed="0", stroke_width=1, shape="rounded=1", align="center",
         parent="1", html=True, font_color=None):
    font_color = font_color or COLORS["text"]
    style = (
        f"{shape};whiteSpace=wrap;html={1 if html else 0};"
        f"fillColor={fill};strokeColor={stroke};"
        f"fontFamily=Helvetica;fontSize={font_size};fontColor={font_color};"
        f"align={align};verticalAlign=middle;dashed={dashed};strokeWidth={stroke_width};"
    )
    if bold:
        style += "fontStyle=1;"
    if shape == "rounded=1":
        style += "arcSize=5;"
    c = ET.SubElement(root, "mxCell", {
        "id": new_id("node"),
        "value": value,
        "style": style,
        "vertex": "1",
        "parent": parent,
    })
    ET.SubElement(c, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry",
    })
    return c


def txt(value, x, y, w, h, font_size=11, align="left", color=None, bold=False, parent="1"):
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
        "parent": parent,
    })
    ET.SubElement(c, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry",
    })
    return c


def edge(source, target, dashed="0", color="#666666", label="",
         waypoints=None, start="none", end="classic"):
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        f"html=1;strokeColor={color};dashed={dashed};"
        f"fontFamily=Helvetica;fontSize=10;fontColor={COLORS['text']};"
        f"startArrow={start};startFill=1;endArrow={end};endFill=1;"
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
# Header / core insight
# ---------------------------------------------------------------------------
cell("Core Insight\n"
     "Euclidean FM: one velocity field forces structure & style to move together.\n"
     "Spectral FM: each Haar sub-band gets its own velocity field.",
     30, 20, 540, 65, COLORS["insight"], COLORS["insight_dark"],
     font_size=13, align="left")

# ---------------------------------------------------------------------------
# Style control band
# ---------------------------------------------------------------------------
cell("Style Control", 30, 105, 1140, 85, "#FFFDF5", "#E0D5C0",
     font_size=13, bold=True, align="left", stroke_width=0)
cell("Style", 60, 125, 80, 45, COLORS["style"], COLORS["style_dark"], font_size=12)
s_mem = cell("Style Memory\n(learnable)\n5 styles × 256", 165, 120, 125, 55,
             COLORS["style"], COLORS["style_dark"], font_size=11)
s_tok = cell("Style Tokens\nS ∈ R^{256×64}", 315, 120, 125, 55,
             COLORS["style"], COLORS["style_dark"], font_size=11)
edge("node_1", s_mem.get("id"), color=COLORS["style_dark"])
edge(s_mem.get("id"), s_tok.get("id"), color=COLORS["style_dark"])

# ---------------------------------------------------------------------------
# Main inference path band
# ---------------------------------------------------------------------------
cell("Main Inference Path", 30, 205, 1140, 340, "#F5FAFF", "#D0D8E0",
     font_size=13, bold=True, align="left", stroke_width=0)

# Content side
content = cell("Content", 60, 325, 80, 45, COLORS["content"], COLORS["content_dark"], font_size=12)
enc = cell("VAE\nEncoder", 160, 320, 85, 55, COLORS["content"], COLORS["content_dark"], font_size=11)
z0 = cell("z₀", 270, 325, 55, 45, COLORS["content"], COLORS["content_dark"], font_size=16)
edge(content.get("id"), enc.get("id"), color=COLORS["content_dark"])
edge(enc.get("id"), z0.get("id"), color=COLORS["content_dark"])

# Haar DWT decomposition
ll = cell("LL", 360, 275, 55, 40, "#D0E8F7", COLORS["content_dark"], font_size=14, bold=True)
lh = cell("LH", 440, 250, 50, 35, COLORS["spectral"], COLORS["spectral_dark"], font_size=13)
hl = cell("HL", 440, 295, 50, 35, COLORS["spectral"], COLORS["spectral_dark"], font_size=13)
hh = cell("HH", 440, 345, 50, 35, COLORS["dead"], COLORS["dead_dark"], font_size=13)
txt("structure\n& tone", 355, 320, 65, 30, font_size=10, align="center")
txt("vertical\ntexture", 435, 220, 60, 28, font_size=10, align="center")
txt("horizontal\ntexture", 435, 333, 60, 28, font_size=10, align="center")
txt("discarded", 500, 350, 65, 25, font_size=10, align="left", color="#C00000")

edge(z0.get("id"), ll.get("id"), color=COLORS["content_dark"])
edge(z0.get("id"), lh.get("id"), color=COLORS["spectral_dark"])
edge(z0.get("id"), hl.get("id"), color=COLORS["spectral_dark"])
edge(z0.get("id"), hh.get("id"), color=COLORS["dead_dark"])

# Stack & Project
stack = cell("Stack &\nProject\n4 bands × C → hidden", 535, 270, 115, 65,
             COLORS["spectral"], COLORS["spectral_dark"], font_size=11)
edge(ll.get("id"), stack.get("id"), color=COLORS["content_dark"])
edge(lh.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(465, 267), (465, 290), (535, 290)])
edge(hl.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(465, 312), (465, 315), (535, 315)])

# Shared Backbone
bb_container = cell("", 680, 255, 150, 120, COLORS["backbone"], COLORS["backbone_dark"],
                    dashed="1", stroke_width=2)
bb = cell("Shared\nBackbone\n(×4 blocks)", 695, 270, 120, 75,
          COLORS["backbone"], COLORS["backbone_dark"], font_size=12, bold=True)
txt("t → time emb", 695, 350, 120, 15, font_size=9, align="center", color="#555555")
edge(stack.get("id"), bb.get("id"), color=COLORS["backbone_dark"])

# One Block inset
block_bg = cell("", 855, 245, 190, 110, "#FBECEA", "#A67C7A",
                stroke_width=1)
txt("(a) One Block", 860, 250, 100, 16, font_size=12, bold=True)
txt("AdaLN(time) → Self-Attn\n"
    "DWT-Route Cross-Attn\n"
    "ReLU² Attention + tanh gate\n"
    "→ FFN",
    860, 270, 180, 80, font_size=11, align="left")

# Per-subband velocity heads
v_ll = cell("v_LL", 690, 425, 55, 38, "#D0E8F7", COLORS["content_dark"], font_size=14)
v_lh = cell("v_LH", 755, 425, 55, 38, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
v_hl = cell("v_HL", 820, 425, 55, 38, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
txt("Per-subband Heads", 690, 405, 185, 16, font_size=11, bold=True, align="center")
edge(bb.get("id"), v_ll.get("id"), color=COLORS["content_dark"],
     waypoints=[(725, 345), (725, 400), (717, 400), (717, 425)])
edge(bb.get("id"), v_lh.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(745, 345), (745, 410), (782, 410), (782, 425)])
edge(bb.get("id"), v_hl.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(765, 345), (765, 400), (847, 400), (847, 425)])

# Spectral ODE integrator
ode_int = cell("Spectral ODE Integrator\nh_i ← h_i + v_i · dt",
               680, 490, 195, 55, COLORS["spectral"], COLORS["spectral_dark"], font_size=12)
edge(v_ll.get("id"), ode_int.get("id"), color=COLORS["content_dark"],
     waypoints=[(717, 463), (717, 490)])
edge(v_lh.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(782, 463), (782, 490)])
edge(v_hl.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
     waypoints=[(847, 463), (847, 490)])

# K steps self-loop
edge(ode_int.get("id"), ode_int.get("id"), dashed="1", color=COLORS["spectral_dark"], label="K steps",
     waypoints=[(875, 517)])

# iDWT and endpoint
idwt = cell("iDWT", 680, 570, 65, 38, COLORS["spectral"], COLORS["spectral_dark"], font_size=14)
edge(ode_int.get("id"), idwt.get("id"), color=COLORS["spectral_dark"])

endpoint = cell("Endpoint\nAdaIN / WCT", 775, 562, 115, 55,
                COLORS["style"], COLORS["style_dark"], font_size=12, bold=True)
edge(idwt.get("id"), endpoint.get("id"), color=COLORS["spectral_dark"])

zT = cell("z_T", 925, 567, 55, 45, COLORS["content"], COLORS["content_dark"], font_size=16)
dec = cell("VAE\nDecoder", 1010, 562, 75, 55, COLORS["content"], COLORS["content_dark"], font_size=11)
out = cell("Output", 1115, 567, 70, 45, COLORS["content"], COLORS["content_dark"], font_size=12)
edge(endpoint.get("id"), zT.get("id"), color=COLORS["content_dark"])
edge(zT.get("id"), dec.get("id"), color=COLORS["content_dark"])
edge(dec.get("id"), out.get("id"), color=COLORS["content_dark"])

# Style dashed edges
edge(s_tok.get("id"), bb.get("id"), dashed="1", color=COLORS["style_dark"],
     waypoints=[(377, 148), (377, 330), (695, 330)])
edge(s_tok.get("id"), endpoint.get("id"), dashed="1", color=COLORS["style_dark"],
     waypoints=[(377, 148), (377, 590), (775, 590)])

# ---------------------------------------------------------------------------
# Training supervision band
# ---------------------------------------------------------------------------
cell("Training Supervision", 30, 565, 1140, 110, "#FCF5FF", "#D8D0E0",
     font_size=13, bold=True, align="left", stroke_width=0)

txt("x_t = (1 − t)·z₀ + t·z_target", 60, 585, 280, 18, font_size=13)
xt_box = cell("x_t", 60, 610, 65, 40, COLORS["train"], COLORS["train_dark"], font_size=16)
dwt = cell("DWT", 150, 610, 60, 40, COLORS["train"], COLORS["train_dark"], font_size=13)
pred = cell("Predict\nv_LL, v_LH, v_HL", 240, 603, 120, 55,
            COLORS["train"], COLORS["train_dark"], font_size=11)
tgt = cell("Target\nΔ_i = DWT(z_t − z₀)_i", 390, 603, 140, 55,
           COLORS["train"], COLORS["train_dark"], font_size=11)
loss = cell("L = w_LL MSE(v_LL, Δ_LL) + w_LH MSE(v_LH, Δ_LH) + w_HL MSE(v_HL, Δ_HL)",
            560, 605, 340, 50, COLORS["train"], COLORS["train_dark"], font_size=11)
edge(xt_box.get("id"), dwt.get("id"), color=COLORS["train_dark"])
edge(dwt.get("id"), pred.get("id"), color=COLORS["train_dark"])
edge(pred.get("id"), loss.get("id"), color=COLORS["train_dark"])
edge(tgt.get("id"), loss.get("id"), color=COLORS["train_dark"])

# Training → inference dashed feedback edges
edge(xt_box.get("id"), z0.get("id"), dashed="1", color=COLORS["train_dark"],
     waypoints=[(92, 610), (92, 347), (270, 347)])
edge(loss.get("id"), bb.get("id"), dashed="1", color=COLORS["train_dark"],
     waypoints=[(730, 605), (730, 400), (755, 400), (755, 345)])

# ---------------------------------------------------------------------------
# Legend and caption
# ---------------------------------------------------------------------------
txt("Legend:", 60, 705, 55, 18, font_size=12, bold=True)
cell("", 115, 708, 22, 13, COLORS["content"], COLORS["content_dark"], font_size=8)
txt("content", 142, 705, 60, 18, font_size=10)
cell("", 215, 708, 22, 13, COLORS["style"], COLORS["style_dark"], font_size=8)
txt("style", 242, 705, 45, 18, font_size=10)
cell("", 300, 708, 22, 13, COLORS["spectral"], COLORS["spectral_dark"], font_size=8)
txt("spectral", 327, 705, 60, 18, font_size=10)
cell("", 400, 708, 22, 13, COLORS["train"], COLORS["train_dark"], font_size=8)
txt("training", 427, 705, 60, 18, font_size=10)
txt("— inference    · · · training", 510, 705, 160, 18, font_size=10)

txt("Figure 2. Overview of Spectral ODE Bridge. The content latent is decomposed into Haar sub-bands; "
    "LL preserves structure and tone, LH/HL carry oriented texture and edge information, and HH is discarded. "
    "A shared backbone learns a cross-band representation, while three independent velocity heads predict "
    "per-subband velocities. The ODE is integrated in the spectral domain, and style statistics are injected "
    "only at the endpoint via AdaIN/WCT.",
    30, 735, 1140, 35, font_size=11, align="left")

# ---------------------------------------------------------------------------
# Assemble mxfile
# ---------------------------------------------------------------------------
mxfile = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "modified": "2026-07-03T00:00:00.000Z",
    "agent": "custom-python-generator-v5",
    "etag": "aaai-arch-v5",
    "version": "24.0.0",
    "type": "device",
})
diagram = ET.SubElement(mxfile, "diagram", {"name": "Page-1", "id": "aaai-arch-page-v5"})
graph_model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1434", "dy": "780", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1",
    "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
    "math": "0", "shadow": "0",
})
graph_model.append(root)

ET.indent(mxfile, space="")
xml_bytes = ET.tostring(mxfile, encoding="utf-8", xml_declaration=True)
out_path = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v5.drawio")
out_path.write_bytes(xml_bytes)
print(f"Saved {out_path}")
