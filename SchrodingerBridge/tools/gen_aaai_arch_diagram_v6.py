"""Generate AAAI-style main architecture diagram for Spectral ODE Bridge (v6).

Reference style: SaMam (low-saturation Morandi palette, orthogonal routing,
line jumps, LaTeX math, big grouping boxes, explicit ODE loop).
"""

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageFilter

PAGE_W, PAGE_H = 1650, 900

COLORS = {
    "ll": "#DAE8FC",
    "ll_dark": "#6C8EBF",
    "mid": "#D5E8D4",
    "mid_dark": "#82B366",
    "backbone": "#E1D5E7",
    "backbone_dark": "#9673A6",
    "style": "#FFF2E6",
    "style_dark": "#C65911",
    "style_line": "#7B1FA2",
    "dead": "#F5F5F5",
    "dead_dark": "#666666",
    "red_x": "#C62828",
    "train": "#FFEBEE",
    "train_dark": "#B71C1C",
    "bypass": "#FBE9E7",
    "bypass_dark": "#C62828",
    "lane_style": "#FFF9E6",
    "lane_infer": "#F4F9FF",
    "lane_train": "#FFF5F5",
    "text": "#1F2937",
    "white": "#FFFFFF",
}

THUMB_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/thumbs")
OUT_PATH = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v6.drawio")


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def make_micro(path: Path, kind: str) -> str:
    """Generate a micro thumbnail for LL (blurred) or LH/HL (edges)."""
    img = Image.open(path).convert("RGB")
    if kind == "ll":
        out = img.filter(ImageFilter.GaussianBlur(radius=2.5)).convert("P", palette=Image.ADAPTIVE)
    else:
        out = img.convert("L").filter(ImageFilter.FIND_EDGES)
        out = out.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_data_uri(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


ids = {"node": 0, "edge": 0, "txt": 0}
root = ET.Element("root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def new_id(kind):
    ids[kind] += 1
    return f"{kind}_{ids[kind]}"


def style_string(base: str, extra: str = "") -> str:
    s = base
    if extra:
        s += extra
    return s


def rect(value, x, y, w, h, fill, stroke, font_size=12, bold=False,
         dashed="0", stroke_width=1, align="center", font_color=None, rounded=1):
    font_color = font_color or COLORS["text"]
    style = (
        f"rounded={rounded};whiteSpace=wrap;html=1;arcSize=4;"
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


def image_node(label, x, y, w, h, b64, font_size=11):
    uri = b64_data_uri(b64)
    style = (
        f"shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
        f"labelBackgroundColor=none;imageAspect=0;aspect=fixed;html=1;"
        f"image={uri};fontFamily=Helvetica;fontSize={font_size};fontColor={COLORS['text']};"
    )
    c = ET.SubElement(root, "mxCell", {
        "id": new_id("node"),
        "value": label,
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


def edge(source, target, color="#666666", dashed="0", stroke_width=1, label="",
         waypoints=None, exit_x=None, exit_y=None, entry_x=None, entry_y=None,
         jump=True, arrow="classic"):
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        f"html=1;strokeColor={color};dashed={dashed};strokeWidth={stroke_width};"
        f"fontFamily=Helvetica;fontSize=10;fontColor={COLORS['text']};"
        f"startArrow=none;startFill=1;endArrow={arrow};endFill=1;"
    )
    if label:
        style += "labelBackgroundColor=#FFFFFF;"
    if jump:
        style += "jumpStyle=arc;jumpSize=8;"
    if exit_x is not None:
        style += f"exitX={exit_x};exitY={exit_y};"
    if entry_x is not None:
        style += f"entryX={entry_x};entryY={entry_y};"
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
# Load / generate thumbnails
# ---------------------------------------------------------------------------
content_b64 = encode_image(THUMB_DIR / "content_thumb.png")
style_b64 = encode_image(THUMB_DIR / "style_thumb.png")
output_b64 = encode_image(THUMB_DIR / "output_thumb.png")
ll_micro_b64 = make_micro(THUMB_DIR / "content_thumb.png", "ll")
mid_micro_b64 = make_micro(THUMB_DIR / "content_thumb.png", "mid")

# ---------------------------------------------------------------------------
# Background grouping lanes
# ---------------------------------------------------------------------------
# Style lane
rect("", 20, 40, 1610, 105, COLORS["lane_style"], "none", font_size=1)
# Inference lane
rect("", 20, 160, 1610, 465, COLORS["lane_infer"], "none", font_size=1)
# Training lane
rect("", 20, 640, 1610, 150, COLORS["lane_train"], "none", font_size=1)

# ---------------------------------------------------------------------------
# Section labels
# ---------------------------------------------------------------------------
txt("Style pathway", 35, 50, 140, 20, font_size=14, bold=True, color=COLORS["style_dark"])
txt("Main inference path", 35, 170, 180, 20, font_size=14, bold=True, color=COLORS["ll_dark"])
txt("Training objective", 35, 650, 180, 20, font_size=14, bold=True, color=COLORS["train_dark"])

# ---------------------------------------------------------------------------
# Style pathway
# ---------------------------------------------------------------------------
style_img = image_node("$I_s$", 60, 58, 70, 70, style_b64, font_size=12)
s_mem = rect("Style Memory\n(learnable)\n5 styles × 256", 155, 62, 145, 62,
             COLORS["style"], COLORS["style_dark"], font_size=11)
s_tok = rect("$S\\in\\mathbb{R}^{256\\times64}$", 325, 62, 150, 62,
             COLORS["style"], COLORS["style_dark"], font_size=13)
edge(style_img.get("id"), s_mem.get("id"), color=COLORS["style_dark"], exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(s_mem.get("id"), s_tok.get("id"), color=COLORS["style_dark"], exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# ---------------------------------------------------------------------------
# Content input
# ---------------------------------------------------------------------------
content_img = image_node("$I_c$", 60, 300, 80, 80, content_b64, font_size=12)
enc = rect("VAE\nEncoder", 170, 315, 90, 55,
           COLORS["backbone"], COLORS["backbone_dark"], font_size=12)
z0 = rect("$z_0$", 290, 320, 60, 48,
          COLORS["ll"], COLORS["ll_dark"], font_size=18)
edge(content_img.get("id"), enc.get("id"), color=COLORS["ll_dark"], exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(enc.get("id"), z0.get("id"), color=COLORS["ll_dark"], exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# ---------------------------------------------------------------------------
# DWT split
# ---------------------------------------------------------------------------
dwt_label = rect("Haar DWT", 390, 200, 90, 30,
                 COLORS["backbone"], COLORS["backbone_dark"], font_size=11, bold=True)
edge(z0.get("id"), dwt_label.get("id"), color=COLORS["ll_dark"],
     exit_x=0.5, exit_y=0, entry_x=0.5, entry_y=1,
     waypoints=[(320, 245), (435, 245)])

# LL locked
ll_micro = image_node("", 398, 145, 64, 45, ll_micro_b64)
ll_box = rect("LL", 400, 210, 75, 55,
              COLORS["ll"], COLORS["ll_dark"], font_size=16, bold=True)
ll_lock = rect("🔒  $v_{LL} \\equiv 0$\n(Base Locked)", 490, 215, 150, 45,
               COLORS["ll"], COLORS["ll_dark"], font_size=12, bold=True, align="left")

# Mid-frequency
mid_micro = image_node("", 398, 280, 64, 40, mid_micro_b64)
lh = rect("LH", 400, 340, 65, 45,
          COLORS["mid"], COLORS["mid_dark"], font_size=14, bold=True)
hl = rect("HL", 400, 405, 65, 45,
          COLORS["mid"], COLORS["mid_dark"], font_size=14, bold=True)

# HH discarded
hh = rect("HH", 400, 475, 60, 40,
          COLORS["dead"], COLORS["dead_dark"], font_size=13)
# Big red X over HH
red_x = txt("<font color=\"#C62828\" size=\"28\"><b>✖</b></font>\nDiscarded",
            405, 480, 110, 35, font_size=10, color=COLORS["red_x"], align="left")

# DWT edges from label to subbands
edge(dwt_label.get("id"), ll_box.get("id"), color=COLORS["ll_dark"],
     exit_x=0.25, exit_y=1, entry_x=0.5, entry_y=0)
edge(dwt_label.get("id"), lh.get("id"), color=COLORS["mid_dark"],
     exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0)
edge(dwt_label.get("id"), hl.get("id"), color=COLORS["mid_dark"],
     exit_x=0.75, exit_y=1, entry_x=0.5, entry_y=0)
edge(dwt_label.get("id"), hh.get("id"), color=COLORS["dead_dark"],
     exit_x=1, exit_y=1, entry_x=0.5, entry_y=0)

# ---------------------------------------------------------------------------
# Spectral ODE Integrator (big dashed box)
# ---------------------------------------------------------------------------
ode_ode = rect("", 620, 270, 450, 270,
                COLORS["white"], COLORS["mid_dark"],
                dashed="1", stroke_width=2, font_size=1)
txt("Spectral ODE Integrator  ($t: 0 \\rightarrow 1$)",
    645, 275, 360, 20, font_size=14, bold=True, color=COLORS["mid_dark"], align="left")

# State H_t
h_t = rect("$H_t$", 645, 365, 70, 55,
           COLORS["mid"], COLORS["mid_dark"], font_size=16, bold=True)
edge(lh.get("id"), h_t.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5,
     waypoints=[(520, 362), (520, 392), (645, 392)])
edge(hl.get("id"), h_t.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5,
     waypoints=[(520, 427), (520, 392), (645, 392)])

# Time t
t_box = rect("$t$", 760, 300, 45, 35,
             COLORS["backbone"], COLORS["backbone_dark"], font_size=16)

# Shared backbone
bb = rect("Shared\nBackbone\n(×4 blocks)", 740, 355, 150, 90,
          COLORS["backbone"], COLORS["backbone_dark"], font_size=12, bold=True)
edge(h_t.get("id"), bb.get("id"), color=COLORS["backbone_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(t_box.get("id"), bb.get("id"), color=COLORS["backbone_dark"],
     exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0)

# Velocity heads
v_lh = rect("$v_{LH}$", 925, 350, 70, 42,
            COLORS["mid"], COLORS["mid_dark"], font_size=14, bold=True)
v_hl = rect("$v_{HL}$", 925, 410, 70, 42,
            COLORS["mid"], COLORS["mid_dark"], font_size=14, bold=True)
edge(bb.get("id"), v_lh.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.35, entry_x=0, entry_y=0.5)
edge(bb.get("id"), v_hl.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.65, entry_x=0, entry_y=0.5)

# Update formula
update = rect("$H_{t+\\Delta t} = H_t + v_H \\cdot \\Delta t$",
              720, 475, 245, 45,
              COLORS["mid"], COLORS["mid_dark"], font_size=13)
edge(v_lh.get("id"), update.get("id"), color=COLORS["mid_dark"],
     exit_x=0.5, exit_y=1, entry_x=0.75, entry_y=0,
     waypoints=[(960, 475), (960, 462), (904, 462)])
edge(v_hl.get("id"), update.get("id"), color=COLORS["mid_dark"],
     exit_x=0.5, exit_y=1, entry_x=0.85, entry_y=0,
     waypoints=[(960, 452), (960, 462), (928, 462)])

# ODE loop: from update back to H_t (around bottom of integrator)
edge(update.get("id"), h_t.get("id"), color=COLORS["mid_dark"], dashed="1",
     label="K steps", stroke_width=2,
     exit_x=0, exit_y=0.5, entry_x=0, entry_y=0.5,
     waypoints=[(720, 497), (620, 497), (620, 392), (645, 392)])

# One Block zoom inset
block_inset = rect("", 1100, 285, 160, 120,
                   "#FFF0EE", "#A67C7A", stroke_width=1, font_size=1)
txt("(a) One Block", 1105, 290, 100, 18, font_size=12, bold=True)
txt("AdaLN($t$) → Self-Attn\nDWT-Route X-Attn\nReLU² + tanh gate\n→ FFN",
    1105, 310, 150, 90, font_size=11, align="left")

# ---------------------------------------------------------------------------
# Right: Reconstruction & AdaIN
# ---------------------------------------------------------------------------
# Final high-freq from ODE
h_1 = rect("$\\hat{H}_1$", 1085, 365, 70, 55,
           COLORS["mid"], COLORS["mid_dark"], font_size=16, bold=True)
edge(update.get("id"), h_1.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5,
     waypoints=[(965, 497), (1120, 497), (1120, 392)])

# iDWT
idwt = rect("iDWT", 1185, 350, 75, 55,
            COLORS["backbone"], COLORS["backbone_dark"], font_size=16, bold=True)
edge(h_1.get("id"), idwt.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# LL locked line flying across the top to iDWT
edge(ll_box.get("id"), idwt.get("id"), color=COLORS["ll_dark"], stroke_width=3,
     exit_x=1, exit_y=0.5, entry_x=0.5, entry_y=0,
     waypoints=[(475, 237), (1222, 237)])

# Endpoint AdaIN (diamond-ish highlighted module) - use rounded rectangle with special fill
adain = rect("Endpoint\nAdaIN / WCT", 1295, 345, 120, 65,
             COLORS["style"], COLORS["style_line"], font_size=12, bold=True)
edge(idwt.get("id"), adain.get("id"), color=COLORS["mid_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# VAE Decoder
dec = rect("VAE\nDecoder", 1445, 350, 90, 55,
           COLORS["backbone"], COLORS["backbone_dark"], font_size=12, bold=True)
edge(adain.get("id"), dec.get("id"), color=COLORS["ll_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# Output image
out_img = image_node("$\\hat{x}_{out}$", 1560, 338, 80, 80, output_b64, font_size=12)
edge(dec.get("id"), out_img.get("id"), color=COLORS["ll_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# ---------------------------------------------------------------------------
# Style condition lines (thick purple)
# ---------------------------------------------------------------------------
# S_tokens down to backbone cross-attn
edge(s_tok.get("id"), bb.get("id"), color=COLORS["style_line"], stroke_width=3,
     exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0,
     waypoints=[(400, 124), (400, 320), (815, 320), (815, 355)])

# S_global across top to AdaIN
edge(s_tok.get("id"), adain.get("id"), color=COLORS["style_line"], stroke_width=4,
     label="$S_{global}$", exit_x=1, exit_y=0.5, entry_x=0.5, entry_y=0,
     waypoints=[(475, 93), (1355, 93), (1355, 345)])

# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------
train_box = rect("", 40, 660, 1000, 120,
                 COLORS["train"], COLORS["train_dark"],
                 dashed="0", stroke_width=1, font_size=1)

xt_label = txt("$x_t = (1-t)z_0 + t z_{target}$", 55, 670, 260, 18, font_size=12)
xt = rect("$x_t$", 60, 695, 55, 40,
          COLORS["train"], COLORS["train_dark"], font_size=16)
dwt_t = rect("DWT", 135, 695, 55, 40,
             COLORS["train"], COLORS["train_dark"], font_size=13)
pred = rect("Predict\n$v_{LH}, v_{HL}$", 215, 688, 130, 55,
            COLORS["train"], COLORS["train_dark"], font_size=12)
tgt = rect("Target\n$\\Delta_i = \\text{DWT}(z_t-z_0)_i$", 370, 688, 170, 55,
           COLORS["train"], COLORS["train_dark"], font_size=12)
loss = rect("$\\mathcal{L} = \\omega_{LH}\\|v_{LH}-\\Delta_{LH}\\|_2^2 + \\omega_{HL}\\|v_{HL}-\\Delta_{HL}\\|_2^2$\n($\\omega_{LL}=0$)",
            570, 685, 420, 65,
            COLORS["train"], COLORS["train_dark"], font_size=13)

edge(xt.get("id"), dwt_t.get("id"), color=COLORS["train_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(dwt_t.get("id"), pred.get("id"), color=COLORS["train_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(pred.get("id"), loss.get("id"), color=COLORS["train_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
edge(tgt.get("id"), loss.get("id"), color=COLORS["train_dark"],
     exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)

# Training feedback dashed edges
edge(xt.get("id"), z0.get("id"), color=COLORS["train_dark"], dashed="1",
     exit_x=0.5, exit_y=0, entry_x=0.5, entry_y=1,
     waypoints=[(87, 695), (87, 440), (320, 440), (320, 368)])
edge(loss.get("id"), bb.get("id"), color=COLORS["train_dark"], dashed="1",
     exit_x=0.5, exit_y=0, entry_x=0.5, entry_y=1,
     waypoints=[(780, 685), (780, 550), (815, 550), (815, 445)])

# ---------------------------------------------------------------------------
# Bypassed mechanisms
# ---------------------------------------------------------------------------
bypass = rect("", 1080, 660, 530, 120,
              COLORS["bypass"], COLORS["bypass_dark"],
              dashed="0", stroke_width=1, font_size=1)
txt("Bypassed Mechanisms (Ablated)", 1095, 670, 300, 20,
    font_size=13, bold=True, color=COLORS["bypass_dark"])
txt("<font color=\"#C62828\"><b>✖</b></font>  Euclidean OT Matching\n"
    "<font color=\"#C62828\"><b>✖</b></font>  GroupNorm / Whitening\n"
    "<font color=\"#C62828\"><b>✖</b></font>  Multi-step Style Guidance",
    1095, 695, 500, 75, font_size=12, align="left", color=COLORS["text"])

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_x, legend_y = 40, 815
txt("Legend:", legend_x, legend_y, 55, 18, font_size=12, bold=True)
rect("", legend_x + 60, legend_y + 3, 20, 12, COLORS["ll"], COLORS["ll_dark"], font_size=8)
txt("LL / content", legend_x + 85, legend_y, 80, 18, font_size=10)
rect("", legend_x + 180, legend_y + 3, 20, 12, COLORS["mid"], COLORS["mid_dark"], font_size=8)
txt("LH / HL / spectral", legend_x + 205, legend_y, 110, 18, font_size=10)
rect("", legend_x + 330, legend_y + 3, 20, 12, COLORS["backbone"], COLORS["backbone_dark"], font_size=8)
txt("network", legend_x + 355, legend_y, 50, 18, font_size=10)
rect("", legend_x + 420, legend_y + 3, 20, 12, COLORS["style"], COLORS["style_line"], font_size=8)
txt("style condition", legend_x + 445, legend_y, 95, 18, font_size=10)
txt("— inference    · · · training    — style", legend_x + 560, legend_y, 200, 18, font_size=10)

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
txt("Figure 2. Overview of Spectral ODE Bridge. The content latent is decomposed by Haar DWT; "
    "LL is locked ($v_{LL}\\equiv0$), LH/HL form the spectral state $H_t$ driven by a shared backbone "
    "with per-subband velocity heads, and HH is discarded. The ODE is integrated for $K$ steps; "
    "the locked LL bypasses the ODE and reunites with the final high-frequency estimate at iDWT. "
    "Style is injected only at the endpoint via AdaIN/WCT ($S_{global}$), while $S_{tokens}$ conditions "
    "the backbone cross-attention. Training supervises only LH/HL velocities with $\\omega_{LL}=0$.",
    35, 845, 1580, 45, font_size=11, align="left")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
mxfile = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "modified": "2026-07-03T00:00:00.000Z",
    "agent": "custom-python-generator-v6",
    "etag": "aaai-arch-v6",
    "version": "24.0.0",
    "type": "device",
})
diagram = ET.SubElement(mxfile, "diagram", {"name": "Page-1", "id": "aaai-arch-page-v6"})
graph_model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1650", "dy": "900", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1",
    "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
    "math": "1", "shadow": "0",
})
graph_model.append(root)

ET.indent(mxfile, space="")
OUT_PATH.write_bytes(ET.tostring(mxfile, encoding="utf-8", xml_declaration=True))
print(f"Saved {OUT_PATH}")
