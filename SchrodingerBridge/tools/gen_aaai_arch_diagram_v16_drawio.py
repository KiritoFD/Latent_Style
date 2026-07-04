"""Generate an editable draw.io file that closely matches v16.

User request for this version:
- keep v16 content/layout semantics unchanged
- only make the upper and lower bands a bit taller
- export as an editable .drawio file for manual adjustment
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


OUT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\docs\630\aaai_arch_diagram_v16_staggered_bundle.drawio")

COL = {
    "bg": "#FFFFFF",
    "title": "#111827",
    "text": "#1F2937",
    "style_bg": "#FFFBEB",
    "style_stroke": "#D97706",
    "main_bg": "#EFF6FF",
    "main_stroke": "#2563EB",
    "train_bg": "#FEF2F2",
    "train_stroke": "#DC2626",
    "style_fill": "#FEF3C7",
    "style_dark": "#92400E",
    "content_dark": "#1E40AF",
    "spectral_fill": "#ECFDF5",
    "spectral_dark": "#047857",
    "net_fill": "#F3E8FF",
    "net_dark": "#5B21B6",
    "gray_fill": "#F3F4F6",
    "gray_dark": "#4B5563",
}


def box_style(fill: str, stroke: str, *, font: int = 14, rounded: int = 1, dashed: bool = False, bold: bool = True, sw: float = 1.8) -> str:
    parts = [
        f"rounded={rounded}",
        "whiteSpace=wrap",
        "html=1",
        "arcSize=6",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={sw}",
        "fontFamily=Helvetica",
        f"fontSize={font}",
        f"fontColor={COL['text']}",
        "align=center",
        "verticalAlign=middle",
    ]
    if dashed:
        parts.extend(["dashed=1", "dashPattern=8 6"])
    if bold:
        parts.append("fontStyle=1")
    return ";".join(parts) + ";"


def text_style(*, font: int = 13, color: str | None = None, bold: bool = False, align: str = "center") -> str:
    parts = [
        "text",
        "html=1",
        "strokeColor=none",
        "fillColor=none",
        f"align={align}",
        "verticalAlign=middle",
        "whiteSpace=wrap",
        "rounded=0",
        "fontFamily=Helvetica",
        f"fontSize={font}",
        f"fontColor={color or COL['text']}",
    ]
    if bold:
        parts.append("fontStyle=1")
    return ";".join(parts) + ";"


def edge_style(*, color: str = "#111111", dashed: bool = False, width: float = 2.0) -> str:
    parts = [
        "edgeStyle=orthogonalEdgeStyle",
        "rounded=1",
        "orthogonalLoop=1",
        "jettySize=auto",
        "html=1",
        f"strokeColor={color}",
        f"strokeWidth={width}",
        "startArrow=none",
        "endArrow=classic",
        "endFill=1",
        "jumpStyle=arc",
        "jumpSize=8",
    ]
    if dashed:
        parts.extend(["dashed=1", "dashPattern=8 6"])
    return ";".join(parts) + ";"


class Drawio:
    def __init__(self) -> None:
        modified = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.mxfile = Element(
            "mxfile",
            {"host": "app.diagrams.net", "modified": modified, "agent": "codex-v16-drawio-generator", "version": "24.0.0", "type": "device"},
        )
        diagram = SubElement(self.mxfile, "diagram", {"name": "Page-1", "id": "aaai-arch-page-v16"})
        self.model = SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1700",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": "2100",
                "pageHeight": "860",
                "math": "1",
                "shadow": "0",
            },
        )
        self.root = SubElement(self.model, "root")
        SubElement(self.root, "mxCell", {"id": "0"})
        SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def vertex(self, cell_id: str, value: str, style: str, x: float, y: float, w: float, h: float) -> None:
        cell = SubElement(self.root, "mxCell", {"id": cell_id, "value": value, "style": style, "vertex": "1", "parent": "1"})
        SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def edge(self, cell_id: str, source: str, target: str, style: str, points: list[tuple[float, float]] | None = None) -> None:
        cell = SubElement(
            self.root,
            "mxCell",
            {"id": cell_id, "value": "", "style": style, "edge": "1", "parent": "1", "source": source, "target": target},
        )
        geom = SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            arr = SubElement(geom, "Array", {"as": "points"})
            for x, y in points:
                SubElement(arr, "Point", {"x": str(x), "y": str(y)})

    def save(self, path: Path) -> None:
        path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + tostring(self.mxfile, encoding="unicode"), encoding="utf-8")


def build() -> None:
    d = Drawio()

    # Bands: keep v16 content, but give the middle band noticeably more height.
    d.vertex("band_style", "", box_style(COL["style_bg"], COL["style_stroke"], font=1, sw=2.2), 34, 26, 1980, 92)
    d.vertex("band_main", "", box_style(COL["main_bg"], COL["main_stroke"], font=1, sw=2.4), 34, 136, 1980, 500)
    d.vertex("band_train", "", box_style(COL["train_bg"], COL["train_stroke"], font=1, sw=2.2), 34, 670, 1980, 90)

    d.vertex("title", "Spectral ODE Bridge", text_style(font=28, color=COL["title"], bold=True), 760, 6, 520, 40)

    # Style row.
    d.vertex("style_id", "Style ID&#xa;<i>s</i>", box_style(COL["style_fill"], COL["style_dark"], font=14), 220, 40, 112, 48)
    d.vertex("style_mem", "Style Memory", box_style("#FFFFFF", COL["style_dark"], font=14), 366, 40, 152, 48)
    d.vertex("style_tok", "Style Tokens", box_style("#FFFFFF", COL["style_dark"], font=14), 554, 40, 144, 48)
    d.vertex("style_code", "style code&#xa;<i>c</i><sub>s</sub>", box_style(COL["style_fill"], COL["style_dark"], font=14), 734, 40, 136, 48)
    d.edge("e_style_1", "style_id", "style_mem", edge_style(color=COL["style_dark"], width=1.8))
    d.edge("e_style_2", "style_mem", "style_tok", edge_style(color=COL["style_dark"], width=1.8))
    d.edge("e_style_3", "style_tok", "style_code", edge_style(color=COL["style_dark"], width=1.8))

    # Main trunk.
    d.vertex("content", "Content <i>x</i>", box_style("#FFFFFF", COL["content_dark"], font=14), 78, 312, 104, 82)
    d.vertex("vae_enc", "VAE Encode", box_style("#FFFFFF", COL["content_dark"], font=14), 244, 316, 136, 74)
    d.vertex("latent", "latent <i>z</i><sub>0</sub>", box_style("#FFFFFF", COL["content_dark"], font=14), 438, 320, 112, 66)
    d.vertex("dwt_outer", "", box_style("#FFFFFF", COL["content_dark"], font=1), 606, 304, 124, 98)
    d.vertex("dwt_ll", "LL", box_style(COL["ll_fill"] if "ll_fill" in COL else "#DBEAFE", COL["content_dark"], rounded=0, font=12, sw=1.2), 626, 324, 30, 24)
    d.vertex("dwt_lh", "LH", box_style(COL["spectral_fill"], COL["spectral_dark"], rounded=0, font=12, sw=1.2), 666, 324, 30, 24)
    d.vertex("dwt_hl", "HL", box_style(COL["spectral_fill"], COL["spectral_dark"], rounded=0, font=12, sw=1.2), 626, 356, 30, 24)
    d.vertex("dwt_hh", "HH", box_style(COL["gray_fill"], COL["gray_dark"], rounded=0, font=12, sw=1.2), 666, 356, 30, 24)
    d.vertex("dwt_x", "&#10006;", text_style(font=26, color="#DC2626", bold=True), 664, 352, 34, 34)
    d.vertex("dwt_label", "Haar DWT", text_style(font=15, bold=True), 612, 382, 112, 18)
    d.edge("e_main_1", "content", "vae_enc", edge_style())
    d.edge("e_main_2", "vae_enc", "latent", edge_style())
    d.edge("e_main_3", "latent", "dwt_outer", edge_style())

    # Top base pathway.
    d.vertex("base_note", "base manifold", text_style(font=15, color=COL["content_dark"], bold=True), 706, 186, 158, 18)
    d.vertex("pi_note", "<i>&pi;</i>(<i>z</i><sub>t</sub>) = <i>l</i><sub>t</sub>", text_style(font=12, color=COL["content_dark"]), 716, 204, 138, 16)
    d.vertex("ell", "<i>l</i><sub>t</sub> (LL)", box_style("#DBEAFE", COL["content_dark"], font=14), 692, 220, 122, 44)
    d.vertex("vell0", "<i>w</i><sub>l</sub> = 0", box_style("#DBEAFE", COL["content_dark"], font=13), 860, 220, 96, 44)
    d.edge("e_dwt_ell", "dwt_outer", "ell", edge_style(color=COL["content_dark"], width=2.1), [(748, 242)])
    d.edge("e_ell_lock", "ell", "vell0", edge_style(color=COL["content_dark"], width=2.1))

    # Fiber pathway.
    d.vertex("bundle", "", box_style(COL["spectral_fill"], COL["spectral_dark"], font=1), 664, 292, 144, 118)
    d.vertex("bundle_title", "Fiber Bundle <i>H</i><sub>t</sub>", text_style(font=15, color=COL["spectral_dark"], bold=True), 670, 300, 132, 18)
    d.vertex("h1", "<i>h</i><sub>1,t</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 680, 324, 112, 26)
    d.vertex("h2", "<i>h</i><sub>2,t</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 680, 358, 112, 26)
    d.edge("e_dwt_bundle", "dwt_outer", "bundle", edge_style(color=COL["spectral_dark"], width=2.0), [(748, 360)])

    d.vertex("time", "time <i>t</i>", box_style("#FFFFFF", COL["gray_dark"], font=12), 924, 208, 78, 30)
    d.vertex("backbone", "", box_style(COL["net_fill"], COL["net_dark"], font=1, sw=2.2), 886, 252, 322, 152)
    d.vertex("backbone_title", "Shared Backbone", text_style(font=18, color=COL["net_dark"], bold=True), 906, 262, 282, 20)
    d.vertex("backbone_route", "routed on (<i>h</i><sub>1</sub>, <i>h</i><sub>2</sub>, <i>h</i><sub>3</sub>)", text_style(font=12, color=COL["net_dark"]), 914, 282, 266, 16)
    d.vertex("adaln", "AdaLN(<i>t</i>)", box_style("#FFFFFF", COL["net_dark"], font=12, sw=1.2), 912, 314, 94, 30)
    d.vertex("selfattn", "Self-Attn", box_style("#FFFFFF", COL["net_dark"], font=12, sw=1.2), 1032, 314, 104, 30)
    d.vertex("routed", "Routed X-Attn", box_style("#FFFFFF", COL["net_dark"], font=12, sw=1.2), 912, 360, 124, 34)
    d.vertex("gate", "gate", box_style("#FFFFFF", COL["net_dark"], font=12, sw=1.2), 1060, 360, 76, 34)
    d.vertex("ffn", "FFN + Res", box_style("#FFFFFF", COL["net_dark"], font=12, sw=1.2), 942, 408, 164, 24)
    d.edge("e_bundle_backbone", "bundle", "backbone", edge_style(width=2.0))
    d.edge("e_time_backbone", "time", "backbone", edge_style(color=COL["gray_dark"], width=1.8))

    d.vertex("heads", "", box_style(COL["spectral_fill"], COL["spectral_dark"], font=1), 1284, 296, 104, 104)
    d.vertex("heads_title", "Fiber Heads", text_style(font=15, color=COL["spectral_dark"], bold=True), 1288, 304, 96, 18)
    d.vertex("v1", "<i>v</i><sub>1</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 1302, 332, 68, 26)
    d.vertex("v2", "<i>v</i><sub>2</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 1302, 366, 68, 26)
    d.edge("e_backbone_heads", "backbone", "heads", edge_style(color=COL["spectral_dark"], width=2.0))

    d.vertex("ode", "", box_style("#FFFFFF", COL["spectral_dark"], font=1, sw=2.2), 1438, 252, 262, 152)
    d.vertex("ode_title", "Spectral ODE Integrator", text_style(font=18, color=COL["spectral_dark"], bold=True), 1452, 262, 234, 20)
    d.vertex("ode_sub", "active fibers: (<i>h</i><sub>1</sub>, <i>h</i><sub>2</sub>)", text_style(font=12, color=COL["spectral_dark"]), 1458, 284, 222, 16)
    d.vertex("k1", "<i>k</i><sub>1</sub> = <i>v</i>(<i>H</i><sub>t</sub>, <i>t</i>)", box_style("#F9FFFC", COL["spectral_dark"], font=11, bold=False, sw=1.1), 1462, 320, 96, 24)
    d.vertex("k2", "<i>k</i><sub>2</sub> = <i>v</i>(<i>H</i><sub>t</sub> + &#916;<i>t</i><i>k</i><sub>1</sub>)", box_style("#F9FFFC", COL["spectral_dark"], font=10, bold=False, sw=1.1), 1582, 320, 102, 24)
    d.vertex("heun", "<i>H</i><sub>t+&#916;t</sub> = <i>H</i><sub>t</sub> + (&#916;<i>t</i>/2)(<i>k</i><sub>1</sub> + <i>k</i><sub>2</sub>)", text_style(font=12, color=COL["spectral_dark"]), 1460, 366, 226, 18)
    d.edge("e_heads_ode", "heads", "ode", edge_style(color=COL["spectral_dark"], width=2.0))

    d.vertex("hhat", "", box_style(COL["spectral_fill"], COL["spectral_dark"], font=1), 1742, 296, 98, 104)
    d.vertex("hhat_title", "<i>H&#770;</i><sub>1</sub>", text_style(font=15, color=COL["spectral_dark"], bold=True), 1746, 304, 90, 18)
    d.vertex("hh1", "<i>h&#770;</i><sub>1</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 1758, 332, 66, 26)
    d.vertex("hh2", "<i>h&#770;</i><sub>2</sub>", box_style("#FFFFFF", COL["spectral_dark"], rounded=0, font=12, bold=False, sw=1.1), 1758, 366, 66, 26)
    d.edge("e_ode_hhat", "ode", "hhat", edge_style(width=2.0))

    # h3 endpoint-only lane.
    d.vertex("h3", "<i>h</i><sub>3,t</sub>", box_style(COL["gray_fill"], COL["gray_dark"], font=13, sw=1.6), 678, 446, 94, 32)
    d.vertex("h3_note", "<i>h</i><sub>3</sub>: endpoint-only fiber", text_style(font=12, color=COL["gray_dark"]), 786, 452, 180, 18)
    d.vertex("h3_skip", "<i>h</i><sub>3</sub> skip", box_style(COL["gray_fill"], COL["gray_dark"], font=13, sw=1.6), 1292, 446, 94, 32)
    d.edge("e_dwt_h3", "dwt_outer", "h3", edge_style(color=COL["gray_dark"], dashed=True, width=1.8), [(744, 430)])
    d.edge("e_h3_skip", "h3", "h3_skip", edge_style(color=COL["gray_dark"], dashed=True, width=1.8))

    # Right side endpoint and decode.
    d.vertex("wct", "Fiber WCT&#xa;<i>T</i><sub>1</sub>, <i>T</i><sub>2</sub>, <i>T</i><sub>3</sub>&#xa;<i>l</i> fixed", box_style(COL["style_fill"], COL["style_dark"], font=14, sw=2.0), 1882, 292, 118, 108)
    d.vertex("endpoint_text", "endpoint style injection", text_style(font=12, color=COL["style_dark"]), 1698, 360, 164, 18)
    d.vertex("idwt", "iDWT", box_style("#FFFFFF", COL["content_dark"], font=14, sw=2.0), 2028, 330, 76, 44)
    d.vertex("vae_dec", "VAE Decode", box_style("#FFFFFF", COL["content_dark"], font=13, sw=2.0), 2022, 446, 92, 40)
    d.vertex("output", "output <i>x&#770;</i>", box_style("#FFFFFF", COL["style_dark"], font=14, sw=2.0), 2150, 328, 98, 70)
    d.edge("e_hhat_wct", "hhat", "wct", edge_style(width=2.0))
    d.edge("e_h3_wct", "h3_skip", "wct", edge_style(color=COL["gray_dark"], dashed=True, width=1.8), [(1560, 462), (1560, 384)])
    d.edge("e_wct_idwt", "wct", "idwt", edge_style(width=2.0))
    d.edge("e_idwt_vae", "idwt", "vae_dec", edge_style(width=2.0))
    d.edge("e_vae_out", "vae_dec", "output", edge_style(width=2.0))
    d.edge("e_base_bypass", "vell0", "idwt", edge_style(color=COL["content_dark"], dashed=True, width=1.8), [(1030, 242), (2066, 242), (2066, 330)])
    d.edge("e_style_tok_backbone", "style_tok", "backbone", edge_style(color=COL["style_dark"], dashed=True, width=1.6), [(626, 120), (626, 250)])
    d.edge("e_style_code_wct", "style_code", "wct", edge_style(color=COL["style_dark"], dashed=True, width=1.8), [(804, 120), (804, 352), (1882, 352)])

    # Training strip: same content, shifted lower with the wider middle band.
    d.vertex("mix", "Mix <i>x</i><sub>t</sub>", box_style("#FFFFFF", COL["train_stroke"], font=13, sw=1.8), 796, 696, 86, 32)
    d.vertex("tdwt", "DWT", box_style("#FFFFFF", COL["train_stroke"], font=13, sw=1.8), 910, 696, 62, 32)
    d.vertex("pred", "Predict (<i>v</i><sub>1</sub>, <i>v</i><sub>2</sub>)", box_style("#FFFFFF", COL["train_stroke"], font=13, sw=1.8), 1002, 696, 144, 32)
    d.vertex("target", "Target (<i>u</i><sub>1</sub>, <i>u</i><sub>2</sub>)", box_style("#FFFFFF", COL["train_stroke"], font=13, sw=1.8), 1178, 696, 140, 32)
    d.vertex("loss", "<i>L</i><sub>WD-VF</sub> (<i>w</i><sub>l</sub> = 0, no <i>h</i><sub>3</sub> head)", box_style("#FEE2E2", COL["train_stroke"], font=12, sw=1.8), 1352, 694, 232, 36)
    d.edge("e_train_1", "mix", "tdwt", edge_style(width=1.8))
    d.edge("e_train_2", "tdwt", "pred", edge_style(width=1.8))
    d.edge("e_train_3", "pred", "target", edge_style(width=1.8))
    d.edge("e_train_4", "target", "loss", edge_style(width=1.8))
    d.edge("e_loss_up", "loss", "backbone", edge_style(color=COL["train_stroke"], dashed=True, width=1.8), [(1468, 660), (1048, 660), (1048, 404)])

    d.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
