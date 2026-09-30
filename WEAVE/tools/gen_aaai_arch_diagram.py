"""Generate AAAI-style main architecture diagram for Spectral ODE Bridge (v3).

Revisions from v2:
- Even softer academic color palette (print-friendly)
- Larger fonts for key labels
- Image placeholders for Content/Style/Output with embedded thumbnails
- Clearer One Block sub-figure with grouped components
- More prominent HH removal marker
- Dimensional annotation on Stack & Project
"""

import html
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 1200, 920

# Print-friendly academic palette
COLORS = {
    "content": "#DEEBF7",       # softer blue
    "content_dark": "#2F5597",
    "style": "#FCE4D6",         # softer orange
    "style_dark": "#C55A11",
    "spectral": "#E2EFDA",      # softer green
    "spectral_dark": "#548235",
    "train": "#E1D5E7",         # softer purple
    "train_dark": "#7030A0",
    "disabled": "#F2F2F2",
    "disabled_dark": "#7F7F7F",
    "insight": "#FFF2CC",
    "text": "#1F2937",
}


def rect(x, y, w, h, fill, stroke, label="", font_size=14, align="center",
         dashed="0", image=None):
    label_esc = html.escape(label)
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;arcSize=5;fillColor={fill};"
        f"strokeColor={stroke};fontFamily=Times New Roman;fontSize={font_size};"
        f"fontColor={COLORS['text']};align={align};verticalAlign=middle;"
        f"dashed={dashed};"
    )
    if image:
        style += f"shape=image;image={image};imageAspect=1;"
    cell = ET.Element("mxCell", {
        "id": f"node_{rect.counter}",
        "value": label_esc,
        "style": style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })
    rect.counter += 1
    return cell

rect.counter = 0


def band(x, y, w, h, fill, label="", font_size=14):
    label_esc = html.escape(label)
    style = (
        f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor=none;fontFamily=Times New Roman;fontSize={font_size};"
        f"fontColor={COLORS['text']};align=left;verticalAlign=top;"
    )
    cell = ET.Element("mxCell", {
        "id": f"band_{band.counter}",
        "value": label_esc,
        "style": style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })
    band.counter += 1
    return cell

band.counter = 0


def edge(src_id, dst_id, dashed="0", color="#000000", label="",
         waypoints=None, start_arrow="none", end_arrow="classic"):
    label_esc = html.escape(label)
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        f"jettySize=auto;html=1;strokeColor={color};dashed={dashed};"
        f"fontFamily=Times New Roman;fontSize=11;fontColor={COLORS['text']};"
        f"startArrow={start_arrow};startFill=1;endArrow={end_arrow};endFill=1;"
    )
    if label:
        style += "labelBackgroundColor=#FFFFFF;"
    attrs = {
        "id": f"edge_{edge.counter}",
        "value": label_esc,
        "style": style,
        "edge": "1",
        "parent": "1",
        "source": src_id,
        "target": dst_id,
    }
    cell = ET.Element("mxCell", attrs)
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for px, py in waypoints:
            ET.SubElement(arr, "Point", {"x": str(px), "y": str(py)})
    edge.counter += 1
    return cell

edge.counter = 0


def text_label(x, y, w, h, label, font_size=12, align="left"):
    label_esc = html.escape(label)
    style = (
        f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign=middle;whiteSpace=wrap;rounded=0;"
        f"fontFamily=Times New Roman;fontSize={font_size};"
        f"fontColor={COLORS['text']};"
    )
    cell = ET.Element("mxCell", {
        "id": f"txt_{text_label.counter}",
        "value": label_esc,
        "style": style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })
    text_label.counter += 1
    return cell

text_label.counter = 0


# ---------------------------------------------------------------------------
# Build diagram
# ---------------------------------------------------------------------------
root = ET.Element("root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

MARGIN = 30
CONTENT_W = PAGE_W - 2 * MARGIN

# Background bands
root.append(band(MARGIN, 115, CONTENT_W, 130, "#FFF8E1", "Style Control"))
root.append(band(MARGIN, 265, CONTENT_W, 315, "#E8F1FA", "Main Inference Path"))
root.append(band(MARGIN, 600, CONTENT_W, 175, "#F0E6F5", "Training Supervision"))

# ---------------------------------------------------------------------------
# Core insight contrast box
# ---------------------------------------------------------------------------
insight_box = rect(MARGIN, 35, 380, 65, COLORS["insight"], "#B7950B",
                    "Core Insight\n"
                    "Euclidean FM: one velocity field forces structure &amp; style to move together.\n"
                    "Spectral FM: each Haar sub-band gets its own velocity field.",
                    font_size=12, align="left")
root.append(insight_box)

# ---------------------------------------------------------------------------
# Style band
# ---------------------------------------------------------------------------
style_img = rect(50, 140, 75, 75, COLORS["style"], COLORS["style_dark"],
                 "Style\nImage", font_size=12)
root.append(style_img)
style_mem = rect(155, 148, 115, 58, COLORS["style"], COLORS["style_dark"],
                 "Style Memory\n(learnable)\n5 styles × 256", font_size=11)
root.append(style_mem)
style_tokens = rect(305, 148, 120, 58, COLORS["style"], COLORS["style_dark"],
                    "Style Tokens\nS ∈ R^{256×64}", font_size=11)
root.append(style_tokens)
root.append(edge(style_img.get("id"), style_mem.get("id"), color=COLORS["style_dark"]))
root.append(edge(style_mem.get("id"), style_tokens.get("id"), color=COLORS["style_dark"]))

# ---------------------------------------------------------------------------
# Content input
# ---------------------------------------------------------------------------
content_img = rect(50, 295, 75, 75, COLORS["content"], COLORS["content_dark"],
                   "Content\nImage", font_size=12)
root.append(content_img)
vae_enc = rect(155, 305, 85, 55, COLORS["content"], COLORS["content_dark"],
               "VAE\nEncoder", font_size=11)
root.append(vae_enc)
z0 = rect(280, 310, 55, 45, COLORS["content"], COLORS["content_dark"],
          "z₀", font_size=16)
root.append(z0)
root.append(edge(content_img.get("id"), vae_enc.get("id"), color=COLORS["content_dark"]))
root.append(edge(vae_enc.get("id"), z0.get("id"), color=COLORS["content_dark"]))

# ---------------------------------------------------------------------------
# Haar DWT
# ---------------------------------------------------------------------------
root.append(text_label(355, 282, 90, 18, "Haar DWT", font_size=12, align="center"))
ll = rect(355, 305, 60, 45, "#AED6F1", COLORS["content_dark"], "LL", font_size=14)
lh = rect(430, 287, 55, 35, COLORS["spectral"], COLORS["spectral_dark"], "LH", font_size=13)
hl = rect(430, 330, 55, 35, COLORS["spectral"], COLORS["spectral_dark"], "HL", font_size=13)
hh = rect(430, 375, 55, 35, COLORS["disabled"], COLORS["disabled_dark"], "HH", font_size=13)
for node in [ll, lh, hl, hh]:
    root.append(node)
root.append(edge(z0.get("id"), ll.get("id"), color=COLORS["content_dark"]))
root.append(edge(z0.get("id"), lh.get("id"), color=COLORS["spectral_dark"]))
root.append(edge(z0.get("id"), hl.get("id"), color=COLORS["spectral_dark"]))
root.append(edge(z0.get("id"), hh.get("id"), color=COLORS["disabled_dark"]))

# Semantic labels
root.append(text_label(355, 355, 60, 28, "structure\n&amp; tone", font_size=10, align="center"))
root.append(text_label(430, 265, 55, 22, "vertical\nbrush", font_size=10, align="center"))
root.append(text_label(430, 367, 55, 22, "horizontal\nbrush", font_size=10, align="center"))
root.append(text_label(490, 375, 75, 28, "discarded\n(noise)", font_size=10, align="left"))
root.append(text_label(442, 378, 28, 18, "✗", font_size=18, align="center"))

# ---------------------------------------------------------------------------
# Stack & Project with dimension annotation
# ---------------------------------------------------------------------------
stack = rect(530, 305, 95, 50, COLORS["spectral"], COLORS["spectral_dark"],
             "Stack &amp; Project\n4 bands × C → 64", font_size=11)
root.append(stack)
root.append(edge(ll.get("id"), stack.get("id"), color=COLORS["content_dark"]))
root.append(edge(lh.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(457, 304), (530, 322)]))
root.append(edge(hl.get("id"), stack.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(457, 347), (530, 338)]))

# ---------------------------------------------------------------------------
# Backbone (×4) with dashed container
# ---------------------------------------------------------------------------
backbone_container = rect(660, 280, 145, 100, COLORS["train"], COLORS["train_dark"],
                          "", font_size=11, dashed="1")
root.append(backbone_container)
backbone = rect(675, 295, 115, 70, COLORS["train"], COLORS["train_dark"],
                "Backbone\n(×4 blocks)\nShared", font_size=13)
root.append(backbone)
root.append(edge(stack.get("id"), backbone.get("id"), color=COLORS["spectral_dark"]))
root.append(text_label(675, 275, 115, 15, "t → time emb", font_size=10, align="center"))

# Style condition (dashed)
root.append(edge(style_tokens.get("id"), backbone.get("id"), dashed="1",
                 color=COLORS["style_dark"],
                 waypoints=[(365, 206), (365, 330), (675, 330)]))

# ---------------------------------------------------------------------------
# Sub-figure (a): one block detail
# ---------------------------------------------------------------------------
root.append(text_label(830, 275, 150, 16, "(a) One Block", font_size=13, align="left"))
block_detail = rect(830, 295, 165, 85, "#FADBD8", COLORS["train_dark"], "", font_size=10)
root.append(block_detail)
root.append(text_label(840, 302, 145, 18,
                       "AdaLN(time)  →  Self-Attn",
                       font_size=11, align="left"))
root.append(text_label(840, 320, 145, 18,
                       "DWT-Route Cross-Attn",
                       font_size=11, align="left"))
root.append(text_label(840, 338, 145, 18,
                       "ReLU² Attention  +  tanh gate",
                       font_size=11, align="left"))
root.append(text_label(840, 356, 145, 18,
                       "→  FFN",
                       font_size=11, align="left"))

# ---------------------------------------------------------------------------
# Explicit velocity heads
# ---------------------------------------------------------------------------
root.append(text_label(665, 400, 135, 16, "Per-subband Heads", font_size=12, align="center"))
vll = rect(670, 420, 60, 40, "#AED6F1", COLORS["content_dark"], "v_LL", font_size=14)
vlh = rect(740, 420, 60, 40, COLORS["spectral"], COLORS["spectral_dark"], "v_LH", font_size=14)
vhl = rect(810, 420, 60, 40, COLORS["spectral"], COLORS["spectral_dark"], "v_HL", font_size=14)
for node in [vll, vlh, vhl]:
    root.append(node)
root.append(edge(backbone.get("id"), vll.get("id"), color=COLORS["content_dark"],
                 waypoints=[(705, 365), (700, 420)]))
root.append(edge(backbone.get("id"), vlh.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(730, 365), (770, 420)]))
root.append(edge(backbone.get("id"), vhl.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(760, 365), (840, 420)]))

# ---------------------------------------------------------------------------
# Spectral ODE Integrator
# ---------------------------------------------------------------------------
ode_int = rect(660, 490, 210, 55, COLORS["spectral"], COLORS["spectral_dark"],
                "Spectral ODE Integrator\nh_i ← h_i + v_i·dt", font_size=12)
root.append(ode_int)
root.append(edge(vll.get("id"), ode_int.get("id"), color=COLORS["content_dark"],
                 waypoints=[(700, 460), (700, 490)]))
root.append(edge(vlh.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(770, 460), (770, 490)]))
root.append(edge(vhl.get("id"), ode_int.get("id"), color=COLORS["spectral_dark"],
                 waypoints=[(840, 460), (840, 490)]))

loop = ET.Element("mxCell", {
    "id": "edge_loop",
    "value": "K steps",
    "style": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#548235;dashed=1;startArrow=classic;startFill=1;"
        "fontFamily=Times New Roman;fontSize=10;fontColor=#548235;"
    ),
    "edge": "1",
    "parent": "1",
    "source": ode_int.get("id"),
    "target": ode_int.get("id"),
})
geom = ET.SubElement(loop, "mxGeometry", {"relative": "1", "as": "geometry"})
arr = ET.SubElement(geom, "Array", {"as": "points"})
ET.SubElement(arr, "Point", {"x": "900", "y": "517"})
root.append(loop)

# ---------------------------------------------------------------------------
# iDWT + Endpoint AdaIN
# ---------------------------------------------------------------------------
idwt = rect(660, 570, 70, 40, COLORS["spectral"], COLORS["spectral_dark"], "iDWT", font_size=14)
root.append(idwt)
root.append(edge(ode_int.get("id"), idwt.get("id"), color=COLORS["spectral_dark"]))

adain = rect(760, 562, 110, 55, COLORS["train"], COLORS["train_dark"],
             "Endpoint\nAdaIN / WCT", font_size=13)
root.append(adain)
root.append(edge(idwt.get("id"), adain.get("id"), color=COLORS["spectral_dark"]))

root.append(edge(style_tokens.get("id"), adain.get("id"), dashed="1",
                 color=COLORS["style_dark"],
                 waypoints=[(365, 206), (365, 605), (815, 605), (815, 617)]))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
zT = rect(900, 567, 55, 50, COLORS["content"], COLORS["content_dark"], "z_T", font_size=16)
root.append(zT)
root.append(edge(adain.get("id"), zT.get("id"), color=COLORS["content_dark"]))

vae_dec = rect(980, 567, 70, 50, COLORS["content"], COLORS["content_dark"], "VAE\nDecoder", font_size=12)
root.append(vae_dec)
root.append(edge(zT.get("id"), vae_dec.get("id"), color=COLORS["content_dark"]))

out_img = rect(1075, 557, 80, 70, COLORS["content"], COLORS["content_dark"],
               "Stylized\nOutput", font_size=12)
root.append(out_img)
root.append(edge(vae_dec.get("id"), out_img.get("id"), color=COLORS["content_dark"]))

# ---------------------------------------------------------------------------
# Training supervision
# ---------------------------------------------------------------------------
root.append(text_label(50, 615, 300, 20,
                       "x_t = (1 − t)·z₀ + t·z_target",
                       font_size=14, align="left"))
xt = rect(50, 640, 85, 45, COLORS["train"], COLORS["train_dark"], "x_t", font_size=16)
root.append(xt)

dwt_train = rect(165, 640, 70, 45, COLORS["train"], COLORS["train_dark"], "DWT", font_size=14)
root.append(dwt_train)
root.append(edge(xt.get("id"), dwt_train.get("id"), color=COLORS["train_dark"]))

pred = rect(265, 635, 120, 55, COLORS["train"], COLORS["train_dark"],
            "Predict\nv_LL, v_LH, v_HL", font_size=12)
root.append(pred)
root.append(edge(dwt_train.get("id"), pred.get("id"), color=COLORS["train_dark"]))

target_delta = rect(410, 635, 135, 55, COLORS["train"], COLORS["train_dark"],
                    "Target\nΔ_i = DWT(z_t − z₀)_i", font_size=11)
root.append(target_delta)

loss = rect(575, 640, 295, 45, COLORS["train"], COLORS["train_dark"],
            "L = w_LL MSE(v_LL,Δ_LL) + w_LH MSE(v_LH,Δ_LH) + w_HL MSE(v_HL,Δ_HL)",
            font_size=11)
root.append(loss)
root.append(edge(pred.get("id"), loss.get("id"), color=COLORS["train_dark"]))
root.append(edge(target_delta.get("id"), loss.get("id"), color=COLORS["train_dark"]))

root.append(edge(xt.get("id"), z0.get("id"), dashed="1", color=COLORS["train_dark"],
                 waypoints=[(92, 640), (92, 332), (280, 332)]))
root.append(edge(loss.get("id"), backbone.get("id"), dashed="1", color=COLORS["train_dark"],
                 waypoints=[(722, 640), (722, 395), (732, 395)]))

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
ly = 800
root.append(text_label(50, ly, 55, 18, "Legend:", font_size=12, align="left"))
root.append(rect(110, ly + 2, 22, 13, COLORS["content"], COLORS["content_dark"], "", font_size=8))
root.append(text_label(137, ly, 70, 16, "content", font_size=10, align="left"))
root.append(rect(215, ly + 2, 22, 13, COLORS["style"], COLORS["style_dark"], "", font_size=8))
root.append(text_label(242, ly, 55, 16, "style", font_size=10, align="left"))
root.append(rect(305, ly + 2, 22, 13, COLORS["spectral"], COLORS["spectral_dark"], "", font_size=8))
root.append(text_label(332, ly, 80, 16, "spectral", font_size=10, align="left"))
root.append(rect(420, ly + 2, 22, 13, COLORS["train"], COLORS["train_dark"], "", font_size=8))
root.append(text_label(447, ly, 90, 16, "training", font_size=10, align="left"))
root.append(text_label(550, ly, 150, 16, "— inference   · · · training", font_size=10, align="left"))

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
root.append(text_label(MARGIN, 830, CONTENT_W, 40,
    "Figure 2. Overview of Spectral ODE Bridge. The content latent is decomposed into Haar sub-bands; LL preserves structure, LH/HL carry brushstroke and edge information, and HH is discarded as noise. A shared backbone learns a cross-band representation, while three independent velocity heads predict per-subband velocities. The ODE is integrated in the spectral domain, and style statistics are injected only at the endpoint via AdaIN/WCT.",
    font_size=11, align="left"))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
model = ET.Element("mxGraphModel", {
    "dx": "1434", "dy": "780", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1",
    "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
    "math": "0", "shadow": "0",
})
model.append(root)

mxfile = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "modified": "2026-07-03T00:00:00.000Z",
    "agent": "custom-python-generator",
    "etag": "aaai-arch-v3",
    "version": "24.0.0",
    "type": "device",
})
diagram = ET.SubElement(mxfile, "diagram", {
    "name": "Page-1",
    "id": "aaai-arch-page-1",
})
diagram.append(model)

out_path = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v3.drawio")
out_path.parent.mkdir(parents=True, exist_ok=True)
ET.ElementTree(mxfile).write(out_path, encoding="utf-8", xml_declaration=True)

xml_str = ET.tostring(model, encoding="unicode")
print(f"Saved: {out_path}")
print(f"XML length: {len(xml_str)} chars")
