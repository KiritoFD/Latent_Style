"""Build the WEAVE defense deck (Chinese) from the lab template.

    python make_slide_figures.py && python make_equations.py && python make_deck.py

template.pptx holds three slides of the original mid-term deck (cover, section divider, content
page with the university logo) and its embedded HarmonyOS Sans SC fonts; every slide here is a copy
of one of them. Numbers are read from the paper's source-of-record files, not typed in, except the
few transcribed values noted inline (reference-pool, Other5, SD1.5 plug-in: same as the paper).
Output: WEAVE_defense.pptx, with speaker notes on every slide.
"""
import copy
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.text.text import _Run
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ICME = HERE.parent / "icme2027"
REPO = HERE.parents[1]
RB = REPO / "SchrodingerBridge" / "experiments" / "rebuttal_20260716"
sys.path.insert(0, str(ICME / "tools"))
import make_figures as mf  # noqa: E402  (load_main, failures, BOARDS)

FONT, FONT_MED, FONT_BLACK = "HarmonyOS Sans SC", "HarmonyOS Sans SC Medium", "HarmonyOS Sans SC Black"
BLUE, NAVY, RED, GREEN, ORANGE = "0C49B7", "0E2841", "C8102E", "2E8B57", "D9822B"
TEXT, GRAY, LGRAY, LIGHT, CARD = "1F2937", "4B5563", "9CA3AF", "EAF0FB", "F5F7FB"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SHAPE_TAGS = {qn(t) for t in ("p:sp", "p:grpSp", "p:graphicFrame", "p:cxnSp", "p:pic", "p:contentPart")}
FOOTER = "WEAVE · 打破轻量风格迁移的恒等捷径"
N_TEMPLATE = 3


# ============================================================== low-level helpers
def duplicate(prs, src):
    """Copy a template slide (shapes and their image relationships) to a new slide."""
    new = prs.slides.add_slide(src.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    rid_map = {}
    for rel in src.part.rels.values():
        if rel.reltype.endswith(("/slideLayout", "/notesSlide")):
            continue
        if rel.is_external:
            rid_map[rel.rId] = new.part.relate_to(rel.target_ref, rel.reltype, is_external=True)
        else:
            rid_map[rel.rId] = new.part.relate_to(rel.target_part, rel.reltype)
    for el in src.shapes._spTree.iterchildren():
        if el.tag not in SHAPE_TAGS:
            continue
        new_el = copy.deepcopy(el)
        for node in new_el.iter():
            for attr, val in list(node.attrib.items()):
                if attr.startswith("{%s}" % R_NS) and val in rid_map:
                    node.set(attr, rid_map[val])
        new.shapes._spTree.append(new_el)
    return new


def style_run(run, size=None, bold=None, color=None, font=FONT):
    f = run.font
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color is not None:
        f.color.rgb = RGBColor.from_string(color)
    f.name = font
    rPr = run._r.get_or_add_rPr()
    rPr.set("lang", "zh-CN")
    rPr.set("altLang", "en-US")
    latin = rPr.find(qn("a:latin"))
    prev = latin
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            prev.addnext(el)  # keep schema order: latin, ea, cs
        el.set("typeface", font)
        prev = el


NOBREAK = ["CLIP-S", "DINO-S", "DINO-C", "Qwen-Image", "SA-Flow", "AesPA-Net", "Z-STAR", "Latent-WCT", "SD-Turbo",
           "D5-512", "P2A-256", "R5-WikiArt", "ViT-B/32", "DINOv2-small", "Cycle-NCE", "IDT–TGT", "11–34×", "11–18×",
           "28–1000×", "6.7–8.5×", "39.5–1440", "1.1–2.4B", "5–18", "7–9M", "0.4847–0.4857"]


def nobreak(txt):
    """Keep metric and method names on one line (U+2011 non-breaking hyphen, U+2060 word joiner)."""
    for tok in NOBREAK:
        if tok in txt:
            txt = txt.replace(tok, tok.replace("-", "\u2011").replace("–", "\u2060–\u2060"))
    return txt


TOKEN = re.compile(r"(\*\*.+?\*\*|!!.+?!!|__.+?__|~~.+?~~)")


def segments(txt):
    """Mini markup: **blue bold**, !!red bold!!, __bold__, ~~gray~~."""
    out = []
    for part in TOKEN.split(txt):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], {"bold": True, "color": BLUE}))
        elif part.startswith("!!"):
            out.append((part[2:-2], {"bold": True, "color": RED}))
        elif part.startswith("__"):
            out.append((part[2:-2], {"bold": True}))
        elif part.startswith("~~"):
            out.append((part[2:-2], {"color": LGRAY}))
        else:
            out.append((part, {}))
    return out


ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def fill_paragraph(p, txt, size, color, bold, font):
    for seg, st in segments(txt):
        r = p.add_run()
        r.text = nobreak(seg)
        style_run(r, size=size, bold=st.get("bold", bold), color=st.get("color", color), font=font)


def prep_frame(tf, margin=0.04, anchor="t"):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = ANCHOR[anchor]


def text(slide, x, y, w, h, paras, size=17, color=TEXT, bold=False, font=FONT, align="l", anchor="t",
         spacing=1.15, after=0, margin=0.04):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    prep_frame(tf, margin, anchor)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ALIGN[align]
        p.line_spacing = spacing
        p.space_after = Pt(after)
        fill_paragraph(p, para, size, color, bold, font)
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=TEXT, spacing=1.15, after=7, bullet_color=BLUE,
            indent=0.24):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    prep_frame(tf)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = spacing
        p.space_after = Pt(after)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(indent))))
        pPr.set("indent", str(-int(Inches(indent))))
        clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", bullet_color)
        etree.SubElement(pPr, qn("a:buSzPct")).set("val", "100000")
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        fill_paragraph(p, item, size, color, False, FONT)
    return tb


def box(slide, x, y, w, h, fill=CARD, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def shape_text(s, paras, size=16, color="FFFFFF", bold=False, font=FONT, align="c", anchor="m", margin=0.06):
    tf = s.text_frame
    prep_frame(tf, margin, anchor)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ALIGN[align]
        fill_paragraph(p, para, size, color, bold, font)
    return s


def line(slide, x1, y1, x2, y2, color="DCE3EE", width=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(width)
    return c


def picture(slide, name, x, y, w=None, h=None):
    """Place assets/<name>; give w or h and the other follows the image aspect. Returns (w, h)."""
    path = ASSETS / name
    pw, ph = Image.open(path).size
    if h is None:
        h = w * ph / pw
    elif w is None:
        w = h * pw / ph
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def picture_fit(slide, name, y, max_h, max_w=12.09, x0=0.62):
    """Largest placement of assets/<name> within max_w x max_h, centred horizontally. Returns (w, h)."""
    pw, ph = Image.open(ASSETS / name).size
    w = min(max_w, max_h * pw / ph)
    return picture(slide, name, x0 + (max_w - w) / 2, y, w=w)


EQ_SIZES = json.loads((ASSETS / "equations.json").read_text())


def equation(slide, name, x, y, pt=20):
    """Place a rendered equation so its math is set at about `pt` points (rendered at 10 pt)."""
    w_pt, _ = EQ_SIZES[name]
    return picture(slide, f"{name}.png", x, y, w=w_pt / 72 * pt / 10)


def table(slide, x, y, col_w, rows, row_h=0.32, size=13, cell_style=None):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nr))
    tbl = gf.table
    tbl.horz_banding = False
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j)
            st = {"fill": NAVY if i == 0 else ("FFFFFF" if i % 2 else "F4F6FA"),
                  "color": "FFFFFF" if i == 0 else TEXT, "bold": i == 0, "align": "l" if j == 0 else "c"}
            if cell_style and i > 0:
                st.update(cell_style(i, j, rows[i][j]) or {})
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(st["fill"])
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = ALIGN[st["align"]]
            r = p.add_run()
            r.text = nobreak(str(rows[i][j]))
            style_run(r, size=st.get("size", size), bold=st["bold"], color=st["color"])
    return tbl


def notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = "\n".join(l.strip() for l in txt.strip().splitlines())


def replace_runs(shape, paras):
    """Rewrite a template text shape; paras = [[(text, style), ...], ...]; "\\n" is a line break.
    The first existing run supplies the base look (color, effects)."""
    tf = shape.text_frame
    tmpl = next((copy.deepcopy(r._r) for p in tf.paragraphs for r in p.runs), None)
    first = tf.paragraphs[0]._p
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for child in list(first):
        if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
            first.remove(child)
    for i, runs in enumerate(paras):
        if i == 0:
            p_el = first
        else:
            p_el = copy.deepcopy(first)
            for child in list(p_el):
                if child.tag in (qn("a:r"), qn("a:br")):
                    p_el.remove(child)
            tf._txBody.append(p_el)
        end = p_el.find(qn("a:endParaRPr"))
        for txt, st in runs:
            el = OxmlElement("a:br") if txt == "\n" else (
                copy.deepcopy(tmpl) if tmpl is not None else OxmlElement("a:r"))
            if end is not None:
                end.addprevious(el)
            else:
                p_el.append(el)
            if txt == "\n":
                continue
            if el.find(qn("a:t")) is None:
                el.append(OxmlElement("a:t"))
            el.find(qn("a:t")).text = txt
            style_run(_Run(el, shape.text_frame.paragraphs[i]), size=st.get("size"), bold=st.get("bold"),
                      color=st.get("color"), font=st.get("font", FONT))


def find_all(shapes, name):
    hits = []
    for s in shapes:
        if s.name == name:
            hits.append(s)
        if s.shape_type == 6:
            hits.extend(find_all(s.shapes, name))
    return hits


# ============================================================== data
def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


T = mf.load_main()  # (method, board) -> metrics; SD-Turbo excluded (in_paper=0)
KEYS = ("dino_s", "clip_s", "lpips", "dino_c")
EXPD = json.loads((RB / "expD" / "results.json").read_text())
W0 = EXPD["D0_full"]
ADAIN0 = json.loads((RB / "results" / "d1_adain_corrected.json").read_text())["D1_adain0_corrected"]
ROB = {r["setting"]: r for r in read_csv(HERE.parent / "docs" / "reproduction" / "internal_dynamics_robustness.csv")}
SEED = {k: [W0[k]] + [float(ROB[s][k]) for s in ("seed7", "seed123")] for k in KEYS}
SEED_SD = {k: statistics.stdev(v) for k, v in SEED.items()}
BASELINES = [m for m in dict.fromkeys(m for m, _ in T) if m not in ("Identity", "Target Style", "WEAVE")]
N_BASE = len(BASELINES)
N_FAIL = sum(any(mf.failures(T, m, b) for b in mf.BOARDS) for m in BASELINES)
N_STYLE_FAIL = sum(any("S" in mf.failures(T, m, b) for b in mf.BOARDS) for m in BASELINES)
D5 = "D5-512"
IDT5, W5 = T[("Identity", D5)], T[("WEAVE", D5)]


def epoch_rows(run):
    rows = read_csv(RB / run / "per_epoch_metrics.csv")
    for r in rows:
        for k in KEYS:
            r[k] = float(r[k])
    return rows


def at_epoch(rows, e):
    return next(r for r in rows if int(r["epoch"]) == e)


# ============================================================== deck scaffolding
class Deck:
    def __init__(self):
        self.prs = Presentation(HERE / "template.pptx")
        self.cover_t, self.section_t, self.content_t = list(self.prs.slides)[:N_TEMPLATE]

    def finish(self, out):
        lst = self.prs.slides._sldIdLst
        for sldId in list(lst)[:N_TEMPLATE]:
            self.prs.part.drop_rel(sldId.rId)
            lst.remove(sldId)
        self.prs.save(out)

    def cover(self, lines, en, note, date="日期：2026 年 10 月"):
        s = duplicate(self.prs, self.cover_t)
        runs = []
        for k, (txt, size) in enumerate(lines):
            if k:
                runs.append(("\n", {}))
            runs.append((txt, {"size": size, "font": FONT_BLACK}))
        replace_runs(find_all(s.shapes, "Rectangle 9")[0], [runs, [(en, {"size": 22})]])
        replace_runs(find_all(s.shapes, "TextBox 90")[0], [[(date, {"size": 16, "color": BLUE})]])
        notes(s, note)
        return s

    def section(self, num, title, subtitle, note):
        s = duplicate(self.prs, self.section_t)
        replace_runs(find_all(s.shapes, "文本框 8")[0], [[(num, {"font": "Arial Black"})]])
        t0, t1 = find_all(s.shapes, "标题 4")[:2]
        t0.left, t0.width = Inches(1.2), Inches(10.93)
        replace_runs(t0, [[(title, {"size": 46, "color": "768394", "font": FONT_MED})]])
        t0.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        t1.left, t1.width, t1.top = Inches(1.2), Inches(10.93), Inches(5.92)
        replace_runs(t1, [[(subtitle, {"size": 20, "color": "8C9AAE"})]])
        t1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        notes(s, note)
        return s

    def content(self, title, lead, note):
        s = duplicate(self.prs, self.content_t)
        for shp in list(s.shapes):
            if shp.is_placeholder and shp.placeholder_format.idx == 1:
                shp._element.getparent().remove(shp._element)
        ph = next(p for p in s.placeholders if p.placeholder_format.idx == 0)
        ph.left, ph.top, ph.width, ph.height = Inches(0.62), Inches(0.36), Inches(10.0), Inches(0.66)
        prep_frame(ph.text_frame, 0.0, "m")
        replace_runs(ph, [[(title, {"size": 28, "bold": True, "color": NAVY, "font": FONT_MED})]])
        box(s, 0.40, 0.45, 0.09, 0.48, fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
        if lead:
            text(s, 0.62, 1.08, 12.09, 0.42, lead, size=16, color=GRAY)
        line(s, 0.62, 1.56, 12.71, 1.56)
        line(s, 0.62, 7.0, 12.71, 7.0, color="E5E7EB", width=0.75)
        text(s, 0.62, 7.03, 8.0, 0.3, FOOTER, size=10, color=LGRAY)
        text(s, 11.71, 7.03, 1.0, 0.3, str(len(self.prs.slides) - N_TEMPLATE), size=10, color=LGRAY, align="r")
        notes(s, note)
        return s


def takeaway(s, msg, y=6.22, h=0.62):
    box(s, 0.62, y, 12.09, h, fill=LIGHT, radius=0.12)
    box(s, 0.62, y, 0.1, h, fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.86, y, 11.7, h, msg, size=17, color=NAVY, bold=True, anchor="m")


def stat_card(s, x, y, w, h, big, label, color=BLUE, big_size=34, label_size=14, fill=CARD, outline=None):
    box(s, x, y, w, h, fill=fill, line=outline, radius=0.1)
    text(s, x + 0.15, y + 0.06, w - 0.3, h * 0.5, big, size=big_size, color=color, font=FONT_BLACK, anchor="m")
    text(s, x + 0.15, y + h * 0.54, w - 0.3, h * 0.44, label, size=label_size, color=GRAY, spacing=1.1)


def pill(s, x, y, w, h, label, fill):
    shape_text(box(s, x, y, w, h, fill=fill, radius=0.5), label, size=13, bold=True)


def numbered(s, x, y, n, color=BLUE, d=0.46, size=18):
    shape_text(box(s, x, y, d, d, fill=color, shape=MSO_SHAPE.OVAL), str(n), size=size, bold=True,
               font=FONT_BLACK, margin=0.0)


def callout(s, x, y, w, h, head, body, accent=RED, head_size=15, body_size=13):
    box(s, x, y, w, h, fill=CARD, radius=0.08)
    box(s, x, y, 0.09, h, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    text(s, x + 0.23, y + 0.08, w - 0.32, 0.42, head, size=head_size, color=NAVY, bold=True)
    text(s, x + 0.23, y + 0.5, w - 0.32, h - 0.55, body.split("\n"), size=body_size, color=TEXT, spacing=1.15)


# ============================================================== the deck
def build():
    d = Deck()
    t = T
    pareto_free = "11 个基线中没有一个能同时在一项风格指标和一项内容指标上胜过它"

    # ---------------------------------------------------------------- cover
    d.cover([("腾飞科创项目答辩", 32), ("WEAVE：打破轻量风格迁移的恒等捷径", 40)],
            "Breaking the Identity Shortcut in Lightweight Style Transfer with Wavelets", """
            各位老师好，我是肖阳。今天汇报的题目是《WEAVE：打破轻量风格迁移的恒等捷径》。
            中期之后，我们把方法从 LANCET 演进到基于小波解耦的速度匹配框架 WEAVE，并完成了 ICME 格式的论文全文。
            汇报按“问题—诊断—方法—实验”的顺序展开。""")

    # ---------------------------------------------------------------- agenda
    s = d.content("汇报提纲", "按“发现问题 → 找到病因 → 对症设计 → 充分验证”的顺序展开", """
            汇报分四部分：先说明轻量风格迁移里一个被忽视的问题——恒等捷径；再分析它的成因；
            然后介绍我们的方法 WEAVE；最后是三个基准、十一个基线上的实验和总结。附录里准备了答辩可能问到的问题。""")
    cards = [("01", "问题：恒等捷径", "轻量模型真的在迁移风格吗？\nIDT–TGT 夹逼判据"),
             ("02", "诊断：频谱失衡", "为什么会走捷径？\n低频主导梯度"),
             ("03", "方法：WEAVE", "小波解耦的速度匹配\n命题 1 与四个设计"),
             ("04", "实验与总结", "3 个基准 × 11 个基线\n消融、稳健性与迁移")]
    for i, (num, title, desc) in enumerate(cards):
        x = 0.62 + i * 3.08
        box(s, x, 1.95, 2.85, 3.9, fill=CARD, radius=0.06)
        box(s, x, 1.95, 2.85, 0.1, fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
        text(s, x + 0.25, 2.25, 2.4, 0.9, num, size=44, color=BLUE, font=FONT_BLACK)
        text(s, x + 0.25, 3.25, 2.45, 0.55, title, size=21, color=NAVY, bold=True, font=FONT_MED)
        text(s, x + 0.25, 3.95, 2.45, 1.6, desc.split("\n"), size=15, color=GRAY, spacing=1.3)
    text(s, 0.62, 6.15, 12.09, 0.45, "附录：研究历程 · 答辩问答 · 超参数敏感性 · ArtFID 审计", size=15, color=LGRAY,
         align="c")

    # ---------------------------------------------------------------- overview
    s = d.content("研究概览：一个问题、一个病因、一个方法", "从评测漏洞出发，找到机理，再对症设计方法", f"""
            先用一页概括全部工作。
            问题：轻量模型常常“看似内容保持很好”，其实几乎没改风格。原样复制输入在 D5 上的 CLIP-S 就有 0.693，比 SaMam、SaMST 还高。
            病因：在潜空间速度匹配里，低频 LL 频带占了约七成梯度，但它最不携带风格。
            方法：WEAVE 把结构和风格分到不同的小波频带里处理，只有 104 万参数，单卡训练 1.4 分钟。
            结果：三个基准全部有效；在 512 像素下，{pareto_free}。""")
    cols = [("问题", "恒等捷径", RED, ["原样复制输入的 CLIP-S（**0.693**）高于 SaMam（0.582）、SaMST（0.618）",
                                      f"IDT–TGT 判据下，{N_BASE} 个基线中 **{N_FAIL} 个**出界"]),
            ("病因", "频谱失衡", ORANGE, ["LL 频带承担 **69.5%** 的梯度能量",
                                      "却是风格区分度**最低**的频带（0.12，HH 为 0.56）"]),
            ("方法", "WEAVE", BLUE, ["源锚定端点 + 频带加权速度匹配 + 逐步高频对齐",
                                    "**1.04M** 参数，RTX 3060 训练 **1.4 分钟**"])]
    for i, (tag, head, color, items) in enumerate(cols):
        x = 0.62 + i * 4.1
        box(s, x, 1.82, 3.85, 3.15, fill=CARD, radius=0.06)
        pill(s, x + 0.25, 2.02, 0.95, 0.38, tag, color)
        text(s, x + 1.32, 1.98, 2.4, 0.48, head, size=22, color=color, bold=True, font=FONT_MED)
        bullets(s, x + 0.22, 2.62, 3.45, 2.3, items, size=16, bullet_color=color)
    results = [("3 / 3", "基准全部有效"), ("第 1", "有效方法中的内容保持"),
               ("0 / 11", "512px 下能同时在风格与内容上胜过 WEAVE 的基线"), ("28–1000×", "比学习型基线训练更快")]
    for i, (big, lab) in enumerate(results):
        stat_card(s, 0.62 + i * 3.08, 5.18, 2.85, 1.62, big, lab, color=RED if i == 0 else BLUE, big_size=30,
                  label_size=13, fill="FFFFFF", outline="DCE3EE")

    # ---------------------------------------------------------------- mid-term plan
    s = d.content("对照中期计划的完成情况", "中期提出的五项计划：四项完成、一项部分完成，另有三项新增成果", """
            这一页对照中期答辩时提出的五项计划。
            新 backbone、更高分辨率、更完备的对比、更全面的消融这四项都已完成；其中对比扩展到 11 个基线、3 个基准，还包括商用的 Seedream 4.5。
            “换用不同 VAE”只做了小规模可行性试验，完整评测放到后续计划里。
            此外有三项新增成果：IDT–TGT 评测判据、频谱机理分析与命题 1、以及 ICME 格式的论文全文。""")
    plan = [("尝试 DiT、Flow Matching、薛定谔桥等新 backbone", "已完成", GREEN,
             "系统试验了薛定谔桥；最终采用**流匹配（速度匹配）+ 小波解耦**，即 WEAVE"),
            ("尝试更高分辨率", "已完成", GREEN, "主基准升级到 **512px**（D5、R5），同时保留 256px 的 P2A"),
            ("完成更完备的相关工作对比", "已完成", GREEN, "**11 个基线 × 3 个基准**，含商用 Seedream 4.5；统一 DINO / CLIP / LPIPS 协议"),
            ("实施更全面的消融实验", "已完成", GREEN, "5 组组件消融、3 个种子、3 组超参数扫描、参考池重采样、ArtFID 审计"),
            ("换用不同 VAE 验证泛化性", "部分完成", ORANGE, "SDXL、TAESD VAE 上完成小规模可行性试验；完整评测列入后续计划")]
    text(s, 0.78, 1.72, 4.3, 0.36, "中期计划", size=14, color=LGRAY, bold=True)
    text(s, 5.3, 1.72, 1.1, 0.36, "状态", size=14, color=LGRAY, bold=True, align="c")
    text(s, 6.75, 1.72, 5.9, 0.36, "完成情况", size=14, color=LGRAY, bold=True)
    for i, (item, status, color, detail) in enumerate(plan):
        y = 2.12 + i * 0.66
        box(s, 0.62, y, 12.09, 0.58, fill=CARD if i % 2 == 0 else "FFFFFF", radius=0.1)
        text(s, 0.78, y, 4.4, 0.58, item, size=15, color=NAVY, bold=True, anchor="m")
        pill(s, 5.3, y + 0.11, 1.1, 0.36, status, color)
        text(s, 6.75, y, 5.9, 0.58, detail, size=14, anchor="m")
    y = 2.12 + 5 * 0.66 + 0.06
    box(s, 0.62, y, 12.09, 0.78, fill=LIGHT, radius=0.1)
    pill(s, 0.8, y + 0.2, 0.95, 0.38, "新增", BLUE)
    text(s, 1.95, y, 10.6, 0.78, "① 提出 **IDT–TGT 评测判据**并审计 11 个基线　② 频谱机理分析与**命题 1**　③ 完成 **ICME 格式论文全文**",
         size=15, color=NAVY, anchor="m")

    # ================================================================ 01 problem
    d.section("01", "问题：恒等捷径", "The Identity Shortcut", """
            第一部分：问题。我们首先回答一个看似简单的问题——轻量风格迁移模型，真的在迁移风格吗？""")

    s = d.content("背景：大模型太重，轻量模型够用吗？", "大模型效果强但部署成本高；轻量模型能部署，却缺少“是否真的迁移了风格”的检验", """
            现在效果最强的是扩散编辑器和图像基础模型，但它们非常重：Qwen-Image、FLUX.2 推理要 60 GB 以上显存；
            在我们的 RTX 3060 上，StyleID 处理 750 张图要 63 分钟，StyleShot 要 5 个多小时。
            轻量模型只有几百万参数，可以部署。但问题是：它们的输出真的变成目标风格了吗？现有指标其实回答不了这个问题。""")
    for i, (head, color, items) in enumerate([
        ("大模型路线：效果强，但太重", GRAY,
         ["图像基础模型推理需 **60 GB 以上**显存（Qwen-Image、FLUX.2）",
          "扩散编辑器每张图都要运行**十亿级**去噪网络",
          "单卡 750 张：StyleID **63 分钟**，StyleShot **5.1 小时**"]),
        ("轻量路线：能部署，但……", BLUE,
         ["CUT、SaMST、SaMam：**7–9M** 可训练参数",
          "750 张图推理 **5–18 分钟**，适合消费级设备",
          "!!输出真的变成目标风格了吗？!!"])]):
        x = 0.62 + i * 6.17
        box(s, x, 1.82, 5.92, 3.55, fill=CARD, radius=0.05)
        box(s, x, 1.82, 5.92, 0.62, fill=color, radius=0.05)
        text(s, x + 0.3, 1.82, 5.4, 0.62, head, size=19, color="FFFFFF", bold=True, font=FONT_MED, anchor="m")
        bullets(s, x + 0.25, 2.68, 5.5, 2.6, items, size=15, bullet_color=color, after=12)
    takeaway(s, "核心问题：轻量模型是否真的完成了风格迁移？——现有指标给不出答案", y=5.72, h=0.8)

    s = d.content("恒等捷径：原样复制就能“赢”", "三个常用指标都在奖励“什么都不改”", """
            一个反直觉的现象：把输入原样复制，在指标上就能“赢”。
            D5 上复制输入的 CLIP-S 是 0.693，比两个最新的轻量模型 SaMam 和 SaMST 都高；
            ArtFID 把复制排在所有方法之前——右图最矮的柱子就是原样复制；LPIPS 更是直接认为复制是完美的输出。
            常用指标都在奖励“什么都不改”，评测缺少参照系。我们把这种现象称为“恒等捷径”。""")
    for i, (big, lab) in enumerate([("0.693", "原样复制输入的 CLIP-S（D5）\n高于 SaMam（0.582）和 SaMST（0.618）"),
                                    ("第 1 名", "ArtFID 把“原样复制”排在\n所有方法之前（216.5）"),
                                    ("0", "LPIPS 认为复制是\n“完美”的输出")]):
        y = 1.8 + i * 1.42
        box(s, 0.62, y, 6.1, 1.27, fill=CARD, radius=0.08)
        text(s, 0.82, y, 2.0, 1.27, big, size=34, color=RED, font=FONT_BLACK, anchor="m")
        text(s, 2.95, y, 3.7, 1.27, lab.split("\n"), size=15, anchor="m", spacing=1.2)
    picture(s, "artfid.png", 7.05, 1.72, h=4.3)
    takeaway(s, "常用指标都在奖励“什么都不改”——评测缺少明确的参照系", y=6.18)

    s = d.content("IDT–TGT 夹逼判据：给每个指标一个方向", "用两个“零成本”的参照输出，把有效迁移夹在中间", """
            为此我们提出 IDT–TGT 夹逼判据。它只用两个零成本的参照输出：
            IDT 直接返回原图，代表“完全没迁移”；TGT 直接返回目标风格的参考图，代表“完全丢掉了内容”。
            有效的迁移，风格指标必须高于 IDT，内容指标必须优于 TGT。前一条排除恒等捷径，后一条排除内容崩塌。
            右图是 D5 上的结果：绿色区域是有效区域，空心点表示至少一项没通过。这个判据不需要权重或阈值，适用于任意指标。""")
    box(s, 0.62, 1.8, 5.3, 1.45, fill=CARD, radius=0.08)
    text(s, 0.82, 1.88, 5.0, 1.32, ["**IDT**（Identity）：直接返回原图——零迁移",
                                    "**TGT**（Target）：直接返回目标风格参考图——零内容"],
         size=16, spacing=1.25, after=8, anchor="m")
    text(s, 0.65, 3.42, 5.2, 0.4, "有效迁移需同时满足：", size=16, color=NAVY, bold=True)
    equation(s, "eq_sandwich", 0.75, 3.88, pt=19)
    bullets(s, 0.65, 4.95, 5.3, 1.9, ["风格高于 IDT：排除**恒等捷径**", "内容优于 TGT：排除**内容崩塌**",
                                      "零成本、原始单位、无需权重或阈值"], size=16, after=6)
    picture(s, "sandwich_d5.png", 6.15, 1.68, w=6.56)

    s = d.content(f"审计：{N_BASE} 个基线中 {N_FAIL} 个越出夹逼区间", "三个基准逐一检验：绿点通过，红叉未通过（S 风格不高于 IDT，C 内容不优于 TGT）", f"""
            我们用这个判据审计了 {N_BASE} 个基线在三个基准上的表现，{N_FAIL} 个至少在一个基准上出界。
            SaMam 内容指标很好，但在两个基准上 CLIP-S 低于原样复制，低 LPIPS 是用风格换来的。
            StyleAligned 风格分最高，但在 D5、R5 上离原图比参考图本身还远，属于内容崩塌。
            Latent-WCT 三个基准都过不了风格线，说明只靠小波统计量不够。
            三个轻量学习型基线 CUT、SaMST、SaMam，都在某个基准上不比原样复制更有风格。""")
    picture(s, "audit.png", 0.62, 1.68, h=4.4)
    bullets(s, 6.05, 1.82, 6.66, 4.3, [
        "**SaMam**：两个基准上 CLIP-S 低于 IDT——低 LPIPS 是用风格换来的",
        "**SaMST**（D5）、**StyleID**（P2A）：风格与内容两项同时失败",
        "**StyleAligned**：DINO-S 最高，但在 D5、R5 上离原图比参考图本身还远",
        "**Latent-WCT**：三个基准都过不了风格线——只靠小波统计量不够",
        "三个轻量学习型基线（CUT、SaMST、SaMam）都在某个基准上**不比复制更有风格**"], size=16, after=11)
    takeaway(s, "只有 Z-STAR、StyTr²、AesPA-Net、Seedream 4.5 和 WEAVE 在三个基准上全部有效", y=6.22)

    # ================================================================ 02 diagnosis
    d.section("02", "诊断：频谱失衡", "Why compact models take the shortcut", """
            第二部分：诊断。为什么轻量模型会走恒等捷径？我们认为瓶颈不在模型容量，而在训练信号。""")

    s = d.content("频谱诊断：损失被“最不重要”的频带主导", "在正交 Haar 坐标下测量 5,000 对跨风格潜变量的梯度能量与风格可分性", """
            Haar 是正交变换，损失可以按频带拆开，初始化时每个频带占的梯度能量就等于它占的位移能量。
            在 5000 对跨风格样本上，低频 LL 占 69.5% 的梯度能量；但 LL 的风格可分性最低，只有 0.12；
            最高频的 HH 只占 8.9% 的梯度，却最能区分风格，达到 0.56。
            训练被最不携带风格的频带主导；网络又倾向于先拟合低频，于是收敛到原图附近。
            这个失衡来自回归目标本身，加宽网络没用。WEAVE 用比这些轻量基线少 6.7 到 8.5 倍的参数就走出了捷径。""")
    picture(s, "spectral.png", 0.62, 1.72, w=7.4)
    bullets(s, 8.3, 1.82, 4.41, 4.3, [
        "正交 Haar：损失按频带可分，初始化时**梯度占比 = 位移占比**",
        "LL 承担 **69.5%** 的梯度能量",
        "但 LL 风格可分性**最低（0.12）**；HH 只占 8.9%，却最能区分风格（**0.56**）",
        "网络先拟合低频 → 收敛到原图附近"], size=16, after=10)
    takeaway(s, "瓶颈在训练信号，而非模型容量：WEAVE 以少 6.7–8.5× 的参数走出捷径", y=6.22)

    # ================================================================ 03 method
    d.section("03", "方法：WEAVE", "Wavelet-Decoupled Velocity Matching", """
            第三部分：方法。既然问题出在频带之间的失衡，我们就让结构和风格各走各的频带。""")

    s = d.content("WEAVE 总览：结构与风格各走各的频带", "冻结 SD1.5 VAE，在其潜空间的 Haar 子带上做速度匹配；只训练绿色模块", """
            内容图和参考图经过冻结的 SD1.5 VAE 编码，再做一级 Haar 分解，得到低频 LL 和三个高频子带。
            四个关键设计：一，源锚定端点，让 LL 只做调色；二，频带加权的速度匹配，降低 LL 损失的权重；
            三，方向性纹理码，把参考图的 LH、HL 池化成编码，只调制对应的输出头；四，每个欧拉步之后做高频 AdaIN 对齐，HH 不需要学习。
            整个模型 104 万可训练参数，推理 8 步，单卡训练 82.8 秒。后面三页分别展开。""")
    _, ah = picture(s, "architecture.png", 0.62, 1.7, w=12.09)
    comps = [("源锚定端点", "LL 只做调色"), ("频带加权速度匹配", "LL 损失权重 λ = 0.3"),
             ("方向性纹理码", "LH / HL 分方向注入"), ("逐步高频对齐", "每步 AdaIN，HH 免学习")]
    for i, (h, sub) in enumerate(comps):
        x = 0.62 + i * 3.06
        box(s, x, 5.95, 2.9, 0.88, fill=CARD, radius=0.12)
        numbered(s, x + 0.14, 6.16, i + 1, color=RED, d=0.46, size=17)
        text(s, x + 0.7, 5.98, 2.15, 0.44, h, size=15, color=NAVY, bold=True, anchor="m")
        text(s, x + 0.7, 6.38, 2.15, 0.42, sub, size=12, color=GRAY, anchor="m")

    s = d.content("① 源锚定端点：只允许 LL 做“调色”", "以源图的 LL 为锚点，只从参考图取高频；命题 1 给出 LL 运动的上界", """
            普通做法直接以参考图的潜变量为目标，网络就得学会参考图那与内容无关的布局。
            我们保留源图的 LL 作为锚点，只允许它向参考图的统计量移动一小步，α 取 0.3；高频直接取参考图。
            命题 1：这样要求的 LL 速度只是逐通道的增益和偏置，也就是调色和对比度，不包含参考图的布局；
            它的能量不超过直接目标的 α 平方，即至少去掉 91%。代入实测数据，LL 的梯度占比从 69.5% 降到不超过 17%。""")
    text(s, 0.65, 1.75, 7.6, 0.4, "源锚定端点（α = 0.3）", size=16, color=NAVY, bold=True)
    equation(s, "eq_target", 0.75, 2.2, pt=19)
    box(s, 0.62, 2.95, 7.55, 2.02, fill=LIGHT, radius=0.08)
    text(s, 0.85, 3.02, 7.2, 0.4, "命题 1（LL 运动上界）", size=16, color=BLUE, bold=True)
    equation(s, "eq_bound", 0.9, 3.5, pt=19)
    text(s, 0.85, 4.1, 0.75, 0.62, "证明", size=13, color=GRAY, bold=True, anchor="m")
    equation(s, "eq_proof", 1.55, 4.08, pt=14)
    bullets(s, 0.65, 5.12, 7.55, 1.8, ["LL 只改变逐通道的**增益和偏置**（色调、对比度），不含参考图布局",
                                       "至少去掉直接目标 **91%** 的 LL 能量（α² = 0.09）",
                                       "LL 梯度占比：**69.5% → ≤ 17%**"], size=16, after=8)
    picture(s, "llshare.png", 8.4, 1.75, h=3.9)

    s = d.content("②③ 频带加权匹配 + 方向性纹理码", "端点决定“要什么运动”，权重决定“多大优化压力”；纹理码只调制对应方向的输出头", """
            第二个设计：共享主干加 LL、LH、HL 三个独立输出头，LL 损失权重取 0.3。端点决定“要求什么运动”，权重决定“优化压力多大”。
            第三个设计：把参考图的 LH、HL 频带池化成与坐标无关的编码，只调制对应方向的输出头。
            为什么要池化？右图的探针实验：直接把空间高频图输入网络，DINO-S 几乎一样，但 DINO-C 从 0.80 掉到 0.40——参考图的布局泄漏进来了。
            HH 不设输出头，交给下一页的逐步对齐。""")
    equation(s, "eq_loss", 0.75, 1.82, pt=19)
    text(s, 0.75, 2.35, 7.3, 0.4, "λLL = 0.3；HH 无输出头，交给 ④", size=14, color=GRAY)
    bullets(s, 0.65, 2.95, 7.4, 3.2, ["共享主干（4 个残差块，宽度 64）+ LL / LH / HL **独立输出头**",
                                      "风格记忆（按风格 id）提供粗粒度先验",
                                      "参考图 LH、HL 池化为**与坐标无关、分方向**的纹理码，只调制对应输出头",
                                      "直接输入空间高频图会**泄漏参考布局**：DINO-C 跌到 0.40",
                                      "整个模型 **1,037,087** 个可训练参数"], size=16, after=9)
    _, rh = picture(s, "route.png", 8.35, 1.78, w=4.3)
    text(s, 8.35, 1.78 + rh + 0.04, 4.35, 0.5, "探针：同一基础模型上两种纹理注入方式（D5）", size=12, color=LGRAY, align="c")
    takeaway(s, "池化纹理码：DINO-S 几乎不变，DINO-C 从 0.40 回到 0.80", y=6.22)

    s = d.content("④ 逐步高频对齐 + 无指标早停", "对齐是推理轨迹的一部分；停止时机由内部梯度决定，不看任何图像指标", """
            第四个设计：每个欧拉步之后，对高频残差做一次 AdaIN，β 取 2，向目标统计量过松弛；LL 从不被这一步修改。
            对齐成为轨迹的一部分，而不是事后处理，也顺带补上了 HH 的统计量。
            训练的早停不看图像指标：每个 epoch 用 4 个固定潜变量测共享主干上 LL 与高频梯度的比值 ρ，ρ 骤降且纹理门控增长时就停。
            D5 上第 4 个 epoch 触发，82.8 秒。右图是种子 7 训满 15 个 epoch 的曲线：规则选中的位置恰好是风格分的峰值。""")
    equation(s, "eq_step", 0.75, 1.8, pt=18)
    bullets(s, 0.65, 2.75, 6.6, 1.5, ["每个欧拉步后对高频残差做 AdaIN，β = 2 过松弛",
                                      "LL 从不被这一步修改；对齐成为**轨迹的一部分**，而非后处理"], size=16, after=7)
    box(s, 0.62, 4.32, 6.6, 1.78, fill=LIGHT, radius=0.08)
    text(s, 0.85, 4.38, 6.2, 0.42, "无指标早停", size=16, color=BLUE, bold=True)
    bullets(s, 0.82, 4.82, 6.3, 1.3, ["每个 epoch 测共享主干的 LL / 高频梯度比 ρ",
                                      "ρ 骤降（≤ 0.65 倍）且纹理门控增长 → 停止",
                                      "D5：第 4 epoch 触发（ρ 1.09 → 0.16），**82.8 秒**"], size=15, after=4)
    picture(s, "stop_curve.png", 7.45, 1.8, w=5.26)
    takeaway(s, "训练全程不需要解码图像，也不需要任何外部指标", y=6.22)

    # ================================================================ 04 experiments
    d.section("04", "实验与总结", "Experiments", """
            第四部分：实验与总结。""")

    s = d.content("实验设置", "3 个基准 × 11 个基线 × 4 项指标，全部在单卡 RTX 3060 上完成", """
            三个基准：D5-512 是主基准，5 种差异很大的 WikiArt 风格，512 像素，750 个迁移请求；
            P2A-256 是照片和 4 位画家，256 像素；R5 是随机抽取的 5 个 WikiArt 风格族。
            DINO-S 是主要的风格指标，另有 CLIP-S、DINO-C 和 LPIPS，再加上 IDT–TGT 判据。
            11 个基线：4 个扩散编辑器、5 个学习型模型、商用的 Seedream 4.5，以及解析方法 Latent-WCT。""")
    benches = [("D5-512", "主基准", "5 种 WikiArt 风格：早期文艺复兴、印象派、极简主义、洛可可、浮世绘\n512px · 每风格 3,600 张训练图 · 750 个请求"),
               ("P2A-256", "照片→绘画", "照片、莫奈、梵高、塞尚、宫崎骏\n256px · 同样的 750 请求布局"),
               ("R5-WikiArt", "随机风格族", "立体主义、表现主义、波普、浪漫主义、象征主义\n512px · 随机抽取的 5 个风格族")]
    for i, (name, tag, desc) in enumerate(benches):
        x = 0.62 + i * 4.1
        box(s, x, 1.8, 3.85, 2.45, fill=CARD, radius=0.06)
        text(s, x + 0.25, 1.92, 2.6, 0.5, name, size=21, color=BLUE, bold=True, font=FONT_MED)
        pill(s, x + 2.45, 1.98, 1.2, 0.36, tag, NAVY)
        text(s, x + 0.25, 2.55, 3.4, 1.65, desc.split("\n"), size=14, spacing=1.25, after=6)
    for i, (k, v) in enumerate([
            ("指标", "**DINO-S**（主风格指标，DINOv2-small）· CLIP-S（ViT-B/32）· DINO-C（内容）· LPIPS（AlexNet）· IDT–TGT 判据"),
            ("基线", "扩散编辑器：StyleAligned、Z-STAR、StyleShot、StyleID　学习型：CUT、SaMST、SaMam、StyTr²、AesPA-Net　商用：Seedream 4.5　解析：Latent-WCT"),
            ("实现", "AdamW（lr 2×10⁻⁴，余弦衰减）· bf16 · batch 96 · 8 步欧拉积分 · 训练时间不含一次性 latent 缓存")]):
        y = 4.48 + i * 0.78
        box(s, 0.62, y, 12.09, 0.68, fill="FFFFFF", line="DCE3EE", radius=0.1)
        text(s, 0.82, y, 1.0, 0.68, k, size=16, color=NAVY, bold=True, anchor="m")
        text(s, 1.85, y, 10.7, 0.68, v, size=14, anchor="m")

    # ---------------------------------------------------------------- main table
    s = d.content("主结果：D5-512 完整对比", "红字：未通过 IDT–TGT 判据　粗体：D5 上有效方法中该列最优　末列：三个基准中通过的个数", f"""
            D5 上的完整结果。红字表示没通过夹逼判据：SaMam 和 SaMST 的 CLIP-S 低于原样复制，StyleAligned 的 LPIPS 比参考图还差。
            粗体只在 D5 上有效的方法中比较——无效方法的“最优”没有意义。WEAVE 的 LPIPS 0.260、DINO-C 0.810 都是有效方法中最好的。
            和商用 Seedream 4.5 相比，WEAVE 风格分更高，0.492 对 0.486，LPIPS 只有它的一半左右。
            在 D5 和 R5 上，{pareto_free}，也就是帕累托最优。""")
    order = ["Identity", "Target Style", "StyleAligned", "Z-STAR", "StyleShot", "StyleID", "CUT", "SaMST", "SaMam",
             "StyTR-2", "AesPA-Net", "Seedream 4.5", "Latent-WCT", "WEAVE"]
    name = {"Identity": "IDT（原样复制）", "Target Style": "TGT（参考图）", "StyTR-2": "StyTr²", "WEAVE": "WEAVE（本文）"}
    params = {"StyleAligned": "0", "Z-STAR": "0", "StyleShot": "0", "StyleID": "0", "CUT": "7.0", "SaMST": "8.3",
              "SaMam": "8.8", "StyTR-2": "35.4", "AesPA-Net": "11.3", "Seedream 4.5": "—", "Latent-WCT": "0",
              "WEAVE": "1.04", "Identity": "—", "Target Style": "—"}
    tgt5 = t[("Target Style", D5)]
    valid5 = [m for m in order[2:] if mf.failures(t, m, D5) == ""]
    best = {k: (min if k == "lpips" else max)(t[(m, D5)][k] for m in valid5) for k in KEYS}
    data = [["方法", "DINO-S↑", "CLIP-S↑", "LPIPS↓", "DINO-C↑", "参数 (M)", "有效基准"]]
    fail_cells, best_cells = set(), set()
    for i, m in enumerate(order, start=1):
        r = t[(m, D5)]
        row = [name.get(m, m)] + [f"{r[k]:.3f}" for k in KEYS] + [params[m]]
        if m in ("Identity", "Target Style"):
            row.append("—")
        else:
            row.append(f"{sum(mf.failures(t, m, b) == '' for b in mf.BOARDS)}/3")
            for j, k in enumerate(KEYS, start=1):
                if (k in ("dino_s", "clip_s") and r[k] <= IDT5[k]) or (k == "lpips" and r[k] >= tgt5["lpips"]) or \
                        (k == "dino_c" and r[k] <= tgt5["dino_c"]):
                    fail_cells.add((i, j))
                if m in valid5 and abs(r[k] - best[k]) < 1e-9:
                    best_cells.add((i, j))
        data.append(row)
    weave_row = len(data) - 1

    def style_main(i, j, v):
        st = {}
        if i in (1, 2):
            st.update(color=LGRAY)
        if i == weave_row:
            st.update(fill="FDECEE", color=RED if j == 0 else TEXT, bold=j == 0)
        if (i, j) in best_cells:
            st.update(bold=True)
        if (i, j) in fail_cells:
            st.update(color=RED)
        if j == 6 and v == "3/3":
            st.update(color=GREEN, bold=True)
        return st
    table(s, 0.62, 1.72, [2.05, 1.08, 1.08, 1.08, 1.08, 1.0, 1.0], data, row_h=0.3, size=13, cell_style=style_main)
    text(s, 0.62, 6.3, 8.5, 0.32, "~~参数为可训练参数；免训练扩散方法推理时需加载约 1.1–2.4B 冻结参数，WEAVE 另有 84M 冻结 VAE。~~", size=11)
    callout(s, 9.35, 1.72, 3.36, 1.48, "有效方法中内容保持第一", f"DINO-C **{W5['dino_c']:.3f}**\nLPIPS **{W5['lpips']:.3f}**")
    sd = t[("Seedream 4.5", D5)]
    callout(s, 9.35, 3.34, 3.36, 1.48, "风格超过商用 Seedream 4.5",
            f"DINO-S **{W5['dino_s']:.3f}** vs {sd['dino_s']:.3f}\nLPIPS 约为其一半（{W5['lpips']:.3f} vs {sd['lpips']:.3f}）")
    callout(s, 9.35, 4.96, 3.36, 1.48, "帕累托最优（D5、R5）", "没有基线能同时在一项风格指标和一项内容指标上胜过 WEAVE")

    # ---------------------------------------------------------------- three boards
    s = d.content("三个基准：WEAVE 全部有效", "每个基准单独画出夹逼区间：绿色为有效区域，空心点未通过，红星为 WEAVE", f"""
            三个基准都画出来，WEAVE 在三个基准上都落在有效区域里。
            它是参数量小于 1000 万的学习型模型里，唯一三个基准全部有效的；在有效方法中，三个基准的 DINO-C 都是第一；
            在 512 像素的 D5 和 R5 上，{pareto_free}。
            P2A 是 256 像素，SaMam 在那里四项指标都略好于我们，附录的问答里专门解释。""")
    _, bh = picture_fit(s, "boards.png", 1.68, max_h=6.9 - 1.68 - 0.42 - 0.95)
    y = 1.68 + bh + 0.02
    text(s, 0.62, y, 12.09, 0.32, "~~SA StyleAligned · ZS Z-STAR · SS StyleShot · SI StyleID · CU CUT · ST SaMST · SM SaMam · T2 StyTr² · "
         "AP AesPA-Net · SD Seedream 4.5 · LW Latent-WCT · 黑色菱形 IDT~~", size=11, align="c")
    y += 0.4
    hcard = 6.88 - y
    dc = " · ".join(f"{t[('WEAVE', b)]['dino_c']:.3f}" for b in mf.BOARDS)
    for i, (big, lab) in enumerate([("3 / 3", "基准全部有效（<10M 参数的学习型模型中唯一）"),
                                    (dc, "DINO-C：有效方法中三个基准均第一"),
                                    ("0 / 11", "512px 下能同时在风格与内容上胜过 WEAVE 的基线")]):
        stat_card(s, 0.62 + i * 4.1, y, 3.85, hcard, big, lab, color=RED if i == 0 else BLUE, big_size=22, label_size=12)

    # ---------------------------------------------------------------- efficiency
    s = d.content("效率：1.04M 参数、1.4 分钟训练", "同为学习型模型，WEAVE 的训练时间和参数量低一到三个数量级", """
            WEAVE 只有 104 万可训练参数，比其他有效的学习型模型少 11 到 34 倍；
            RTX 3060 上训练 1.4 分钟，比学习型基线快 28 到 1000 倍；推理每张 512 像素图 168 毫秒，包括 VAE 解码。
            左图训练时间，右图 750 张图的推理时间，都是对数刻度。Latent-WCT 推理更快，但它是解析方法，风格不达标。""")
    for i, (big, lab) in enumerate([("1.04M", "可训练参数：比其他有效学习型模型少 11–34×"),
                                    ("1.4 min", "单卡训练：比学习型基线快 28–1000×"),
                                    ("168 ms", "每张 512px 图推理，含 VAE 解码")]):
        stat_card(s, 0.62 + i * 4.1, 1.72, 3.85, 1.12, big, lab, color=RED, big_size=30, label_size=12)
    _, ch = picture_fit(s, "cost.png", 2.98, max_h=6.62 - 2.98 - 0.32)
    text(s, 0.62, 2.98 + ch + 0.03, 12.09, 0.3, "~~* StyTr² 训练时间为作者报告（4 块 GPU）；† Z-STAR 在 12 GB 显存下无法完整运行 512px，为按步数外推的估计。~~",
         size=11)

    # ---------------------------------------------------------------- qualitative
    s = d.content("定性对比：保留结构，重写笔触", "极端请求：极简主义（Minimalism）→ 洛可可（Rococo）", """
            一个比较极端的例子：把极简主义的格子图迁移到洛可可风格。
            StyleID 几乎原样输出；CUT 和 SaMam 主要是把颜色调暗或洗淡；Seedream 4.5 风格很强，但用洛可可装饰把原来的格子结构整个替换掉了——内容崩塌。
            WEAVE 保留了格子的布局；放大看右边两格，原来平涂的色块被改写成了有笔触的绘画质感，色调更柔和。""")
    _, qh = picture(s, "qualitative.png", 0.52, 1.82, w=12.29)
    bullets(s, 0.65, 1.82 + qh + 0.35, 12.0, 1.8, [
        "**StyleID** 几乎原样输出；**CUT、SaMam** 主要是调暗或褪色",
        "**Seedream 4.5** 用洛可可装饰替换了原图的格子结构——内容崩塌",
        "**WEAVE** 保留格子布局；放大可见平涂被改写为绘画笔触，色调更柔和"], size=17, after=10)
    takeaway(s, "WEAVE 改的是“纹理与色调”，保留的是“几何与布局”", y=6.22)

    # ---------------------------------------------------------------- ablation
    d4, d3, d5 = epoch_rows("expA_D4_seed42"), epoch_rows("expA_D3_seed42"), epoch_rows("expA_D5_seed42")
    best_ep = lambda rows: (lambda r: f"{r['dino_s']:.4f}（{r['epoch']}）")(max(rows, key=lambda r: r["dino_s"]))  # noqa: E731
    variants = [("WEAVE", W0, f"{W0['dino_s']:.4f}（4）"), ("去掉逐步 AdaIN（β = 0）", ADAIN0, "—"),
                ("去掉纹理码", EXPD["D2_no_hf_route"], "—"), ("直接端点（以参考图为目标）", at_epoch(d4, 4), best_ep(d4)),
                ("λLL = 1.0", at_epoch(d3, 4), best_ep(d3)), ("学习 HH 输出头", at_epoch(d5, 4), best_ep(d5))]
    idt_clip = IDT5["clip_s"]
    s = d.content("消融：每个组件都在起作用", "同一 750 个请求，每次只改一个组件；重训变体与 WEAVE 同训 4 个 epoch，末列为 15 epoch 内最佳", f"""
            每次只改一个组件，重训的变体和 WEAVE 一样训练 4 个 epoch 比较。
            去掉逐步 AdaIN，CLIP-S 只比原样复制高 {ADAIN0['clip_s'] - idt_clip:.3f}（WEAVE 是 {W0['clip_s'] - idt_clip:.3f}），说明大部分风格余量来自这一步；内容指标变好，是风格换内容的取舍。
            去掉纹理码，DINO-S 掉 0.009，LPIPS 升 0.024。
            直接端点代价最大：同预算下 DINO-S 低 0.012、LPIPS 高 0.028，训练三倍时长也追不上，正好印证命题 1。
            λLL 改成 1，DINO-C 掉 0.025；学一个 HH 头，风格增益在噪声以内而内容变差，所以不要它。
            红色表示比 WEAVE 差、绿色表示比 WEAVE 好，且幅度超过种子间的标准差。""")
    data = [["变体", "DINO-S↑", "CLIP-S↑", "LPIPS↓", "DINO-C↑", "最佳（epoch）"]]
    for nm, r, bst in variants:
        data.append([nm] + [f"{r[k]:.4f}" for k in KEYS] + [bst])

    def style_abl(i, j, v):
        if i == 1:
            return {"fill": "FDECEE", "bold": True, "color": RED if j == 0 else TEXT}
        if 1 <= j <= 4:
            k = KEYS[j - 1]
            delta = float(v) - W0[k]
            if abs(delta) > SEED_SD[k]:
                better = delta < 0 if k == "lpips" else delta > 0
                return {"color": GREEN if better else RED, "bold": True}
        return {}
    table(s, 0.62, 1.72, [3.29, 1.6, 1.6, 1.6, 1.6, 2.4], data, row_h=0.42, size=14, cell_style=style_abl)
    text(s, 0.62, 4.72, 12.09, 0.3, f"~~红 / 绿：比 WEAVE 差 / 好，且幅度超过 3 个种子的标准差（DINO-S {SEED_SD['dino_s']:.4f}，CLIP-S {SEED_SD['clip_s']:.4f}，"
         f"LPIPS {SEED_SD['lpips']:.4f}，DINO-C {SEED_SD['dino_c']:.4f}）~~", size=11)
    findings = [("逐步 AdaIN", f"去掉后 CLIP-S 只比复制高 {ADAIN0['clip_s'] - idt_clip:.3f}（WEAVE {W0['clip_s'] - idt_clip:.3f}）"),
                ("纹理码", "去掉后 DINO-S −0.009，LPIPS +0.024"),
                ("源锚定端点", "同预算 DINO-S −0.012、LPIPS +0.028；训练 3 倍仍不及"),
                ("LL 权重 / HH 头", "λLL = 1：DINO-C −0.025；HH 头：增益在噪声内")]
    for i, (h, body) in enumerate(findings):
        x = 0.62 + i * 3.06
        box(s, x, 5.1, 2.91, 0.98, fill=CARD, radius=0.08)
        box(s, x, 5.1, 2.91, 0.07, fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
        text(s, x + 0.15, 5.2, 2.65, 0.34, h, size=14, color=BLUE, bold=True)
        text(s, x + 0.15, 5.54, 2.7, 0.52, body, size=12, spacing=1.1)
    takeaway(s, "三个核心设计各自贡献风格或内容，去掉任何一个都会退化", y=6.22)

    # ---------------------------------------------------------------- robustness
    mean_ds, sd_ds = statistics.mean(SEED["dino_s"]), SEED_SD["dino_s"]
    s = d.content("稳健性与迁移", "换种子、换参考池、换风格族、换主干，结论都成立", f"""
            第一，三个种子分别在第 4、4、3 个 epoch 停止，DINO-S {mean_ds:.3f} 正负 {sd_ds:.3f}；有完整曲线的两个种子都正好停在峰值。
            第二，DINO-S 依赖参考池，我们随机重抽 1000 次参考池，WEAVE 相对原样复制的优势每次都是正的。
            第三，零样本迁移到 5 个没见过的 WikiArt 风格族，CLIP-S、LPIPS、DINO-C 都是最好的。
            第四，把 Haar 对齐当作插件加到冻结的 SD1.5 编辑器上，不改任何权重，风格分显著提升，每张图只多 4.9 毫秒。""")
    quads = [("种子与早停", f"{mean_ds:.3f} ± {sd_ds:.3f}",
              "3 个种子分别在第 4 / 4 / 3 epoch 停止（DINO-S）；有完整曲线的两个种子都恰好停在峰值"),
             ("参考池重采样", "1000 / 1000",
              "随机抽取 8/30 张参考：对 IDT 的优势全部为正，0.030（95% 区间 [0.020, 0.039]），全池为 0.033"),
             ("未见风格族（零样本）", "3 项最佳",
              "5 个未见 WikiArt 风格族：CLIP-S 0.774、LPIPS 0.280、DINO-C 0.759 均为最佳；SaMam 的 DINO-S 高 0.015，但 LPIPS 高 0.12"),
             ("冻结 SD1.5 插件", "+4.9 ms",
              "只加 Haar 对齐、不改权重：DINO-S 0.397 → 0.408，CLIP-S 0.722 → 0.737（p < 10⁻¹⁵）")]
    for i, (h, big, body) in enumerate(quads):
        x = 0.62 + (i % 2) * 6.17
        y = 1.8 + (i // 2) * 2.5
        box(s, x, y, 5.92, 2.3, fill=CARD, radius=0.06)
        text(s, x + 0.28, y + 0.15, 3.2, 0.45, h, size=18, color=NAVY, bold=True, font=FONT_MED)
        text(s, x + 3.0, y + 0.08, 2.7, 0.6, big, size=24, color=RED, font=FONT_BLACK, align="r", anchor="m")
        text(s, x + 0.28, y + 0.78, 5.4, 1.45, body, size=15, spacing=1.25)

    # ---------------------------------------------------------------- summary
    s = d.content("总结：四项贡献", "标准指标奖励复制；两个零成本参照就能识破；病因在频谱；WEAVE 从源头消除它", f"""
            总结四项贡献。
            一，评测：IDT–TGT 夹逼判据，零成本识别恒等捷径和内容崩塌，并审计了 {N_BASE} 个基线。
            二，机理：通过频带测量和命题 1，把恒等捷径归因于低频梯度主导，并据此设计训练目标。
            三，方法：小波解耦的速度匹配，加上方向性纹理码、逐步高频对齐和无指标早停。
            四，结果：三个基准全部有效、512 像素下帕累托最优，只有 104 万参数、训练 1.4 分钟。
            论文全文已经按 ICME 格式完成，代码和数据处理流程都可以复现。""")
    for i, (h, body) in enumerate([("评测", f"IDT–TGT 夹逼判据：零成本识别**恒等捷径**与**内容崩塌**；审计 {N_BASE} 个基线 × 3 个基准"),
                                   ("机理", "频带测量 + **命题 1**：把恒等捷径归因于低频梯度主导，并据此设计训练目标"),
                                   ("方法", "小波解耦速度匹配 + 方向性纹理码 + 逐步高频对齐 + 无指标早停"),
                                   ("结果", "三个基准全部有效、512px 帕累托最优；**1.04M** 参数、**1.4 分钟**训练")]):
        y = 1.8 + i * 0.98
        box(s, 0.62, y, 12.09, 0.84, fill=CARD, radius=0.12)
        numbered(s, 0.82, y + 0.17, i + 1, color=BLUE, d=0.5, size=18)
        text(s, 1.55, y, 1.3, 0.84, h, size=19, color=NAVY, bold=True, font=FONT_MED, anchor="m")
        text(s, 2.85, y, 9.7, 0.84, body, size=16, anchor="m")
    box(s, 0.62, 5.85, 12.09, 0.95, fill=LIGHT, radius=0.1)
    pill(s, 0.85, 6.13, 1.1, 0.4, "项目产出", BLUE)
    text(s, 2.15, 5.85, 10.4, 0.95, "ICME 格式论文全文（6 页）+ 补充材料　·　可复现的图表生成脚本　·　实验数据索引与出处记录",
         size=15, color=NAVY, anchor="m")

    # ---------------------------------------------------------------- next steps
    s = d.content("下一步计划", "补齐主观评测与更多场景，推进论文投稿与端侧落地", """
            第一，补充用户主观评测，以及更多风格、更多参考图的定性对比；
            第二，换用 SDXL、FLUX 等不同 VAE，并尝试多级小波，覆盖跨尺度的风格；
            第三，扩展到更高分辨率和视频风格迁移，考虑时序一致性；
            第四，延续中期已经完成的 MNN 安卓端部署，104 万参数的速度场很适合移动端；
            最后，论文计划投稿 IEEE ICME 2027。""")
    for i, (h, body) in enumerate([("主观评测", "用户研究（两两比较）+ 更多风格与参考图的定性对比"),
                                   ("更多 VAE 与多级小波", "SDXL、FLUX 等潜空间；多级 Haar 覆盖跨尺度风格"),
                                   ("更高分辨率与视频", "1024px 与视频风格迁移，引入时序一致性"),
                                   ("端侧部署", "延续中期的 MNN 安卓部署；1.04M 参数的速度场适合移动端"),
                                   ("论文投稿", "IEEE ICME 2027（全文与补充材料已完成）")]):
        y = 1.8 + i * 0.95
        box(s, 0.62, y, 12.09, 0.82, fill=CARD if i % 2 == 0 else "FFFFFF", radius=0.12)
        numbered(s, 0.82, y + 0.16, i + 1, color=BLUE, d=0.5, size=18)
        text(s, 1.55, y, 3.5, 0.82, h, size=18, color=NAVY, bold=True, font=FONT_MED, anchor="m")
        text(s, 5.1, y, 7.5, 0.82, body, size=16, anchor="m")

    d.cover([("谢谢各位老师！", 54), ("敬请批评指正", 28)], "Q & A", """
            我的汇报就到这里，谢谢各位老师，请批评指正。""")

    # ================================================================ appendix
    d.section("A", "附录：答辩问答", "Backup slides", """
            以下是附录：研究历程和几个可能被问到的问题。""")

    s = d.content("研究历程：从 LANCET 到 WEAVE", "关键转折：从“加模块”转向“先诊断、再设计”", """
            一月从潜空间流匹配开始探索；二到四月是 Cycle-NCE 和 LANCET，在潜空间用 SWD 做风格监督，这是中期答辩的内容，还完成了 MNN 安卓部署；
            四到五月尝试薛定谔桥和最优传输耦合；六月做了“雾化、白化”诊断，发现端点收缩和风格门控失效；
            七月提出 WEAVE 和恒等捷径判据；十月完成 ICME 版论文。
            最大的转折是方法论上的：从不断加模块，转向先诊断问题、再对症设计。""")
    steps = [("2026.01", "探索", "潜空间流匹配\nSA-Flow · DiT"), ("2026.02–04", "LANCET", "潜空间 SWD 监督\n中期答辩\nMNN 端侧部署"),
             ("2026.04–05", "薛定谔桥", "SWD 引导的\n最优传输耦合"), ("2026.06", "诊断", "发现端点收缩\n与风格门控失效"),
             ("2026.07", "WEAVE", "小波解耦\n恒等捷径判据"), ("2026.10", "论文", "命题 1\n审计 11 个基线\nICME 全文定稿")]
    line(s, 0.95, 3.15, 12.4, 3.15, color=BLUE, width=3)
    for i, (date, h, body) in enumerate(steps):
        x = 0.62 + i * 2.03
        c = RED if h == "WEAVE" else BLUE
        box(s, x + 0.78, 2.98, 0.34, 0.34, fill=c, shape=MSO_SHAPE.OVAL)
        text(s, x, 2.35, 1.9, 0.45, date, size=14, color=GRAY, bold=True, align="c")
        box(s, x + 0.05, 3.6, 1.8, 2.2, fill=CARD, radius=0.08)
        text(s, x + 0.1, 3.72, 1.7, 0.45, h, size=17, color=c, bold=True, font=FONT_MED, align="c")
        text(s, x + 0.1, 4.25, 1.7, 1.5, body.split("\n"), size=13, align="c", spacing=1.25)
    takeaway(s, "中期的“潜空间风格信噪比低”判断，最终落实为频带级的测量与对症设计", y=6.12)

    p2a = "P2A-256"
    s = d.content("Q：为什么在 P2A 上不如 SaMam？", "256px 时高频子带被压缩；即便如此 WEAVE 在 P2A 上仍然有效", """
            P2A 是 256 像素。VAE 下采样 8 倍，Haar 再减半，每个子带只有 16 乘 16 个系数，高频风格信息被压缩了。
            即便如此，WEAVE 在 P2A 上仍然有效，并且在有效方法中 DINO-C 最高。
            SaMam 的可训练参数是我们的 8.5 倍，训练时间 300 倍以上；在 512 像素的两个基准上，没有基线能同时在风格和内容上胜过 WEAVE。""")
    bullets(s, 0.65, 1.85, 6.55, 4.2, [
        "P2A 为 256px：VAE 下采样 8 倍、Haar 再减半 → 每个子带仅 **16×16** 个系数，高频风格信息被压缩",
        f"WEAVE 在 P2A 上**仍然有效**，且有效方法中 DINO-C 最高（**{t[('WEAVE', p2a)]['dino_c']:.3f}**）",
        "SaMam：可训练参数为 WEAVE 的 **8.5×**，训练时间 **>300×**",
        "512px 的 D5、R5 上，没有基线能同时在风格和内容上胜过 WEAVE"], size=16, after=11)
    data = [["P2A-256", "DINO-S", "CLIP-S", "LPIPS", "DINO-C"]] + [
        [lab] + [f"{t[(m, p2a)][k]:.3f}" for k in KEYS] for lab, m in (("WEAVE", "WEAVE"), ("SaMam", "SaMam"),
                                                                       ("IDT（复制）", "Identity"))]
    table(s, 7.45, 2.0, [1.45, 0.95, 0.95, 0.95, 0.95], data, row_h=0.5, size=14,
          cell_style=lambda i, j, v: {"fill": "FDECEE", "bold": True} if i == 1 else ({"color": LGRAY} if i == 3 else {}))
    takeaway(s, "256px 时高频子带只有 16×16；在 512px 的两个基准上，没有基线能同时在风格和内容上胜过 WEAVE", y=6.12)

    margin = W5["dino_s"] - IDT5["dino_s"]
    s = d.content("Q：风格提升幅度是否偏小？", "原样复制本身就是强基线；WEAVE 的余量稳定，且来自设计", f"""
            原样复制本身就是很强的基线：D5 上 DINO-S {IDT5['dino_s']:.3f}、CLIP-S {IDT5['clip_s']:.3f}；{N_BASE} 个基线中 {N_STYLE_FAIL} 个在至少一个基准上风格不及它。
            WEAVE 的 DINO-S 比原样复制高 {margin:.3f}，与商用 Seedream 4.5 相当。这个优势在 1000 次参考池重采样中全部为正，种子间标准差只有 {SEED_SD['dino_s']:.3f}。
            消融显示 CLIP-S 余量主要来自逐步 AdaIN：去掉它，余量从 {W0['clip_s'] - idt_clip:.3f} 降到 {ADAIN0['clip_s'] - idt_clip:.3f}，余量来自设计而不是噪声。""")
    bullets(s, 0.65, 1.85, 7.2, 4.3, [
        f"复制就是强基线：D5 上 IDT 的 DINO-S **{IDT5['dino_s']:.3f}**、CLIP-S **{IDT5['clip_s']:.3f}**；{N_BASE} 个基线中 {N_STYLE_FAIL} 个在至少一个基准上风格不及它",
        f"WEAVE 的 DINO-S 比 IDT 高 **{margin:.3f}**（{W5['dino_s']:.3f} vs {IDT5['dino_s']:.3f}），与 Seedream 4.5（{sd['dino_s']:.3f}）相当",
        f"该优势在 1000 次参考池重采样中**全部为正**；种子间标准差仅 {SEED_SD['dino_s']:.3f}",
        f"CLIP-S 余量主要来自逐步 AdaIN：去掉后从 **{W0['clip_s'] - idt_clip:.3f}** 降到 **{ADAIN0['clip_s'] - idt_clip:.3f}**——余量来自设计而非噪声"],
        size=16, after=11)
    for i, (big, lab) in enumerate([(f"+{margin:.3f}", "DINO-S 相对 IDT（D5）"), ("1000 / 1000", "重采样中优势为正"),
                                    (f"{SEED_SD['dino_s']:.3f}", "种子间 DINO-S 标准差")]):
        stat_card(s, 8.35, 1.85 + i * 1.45, 4.36, 1.3, big, lab, color=RED if i == 0 else BLUE, big_size=28)

    s = d.content("Q：测试参考图就是 TGT 图，DINO-S 会被高估吗？", "参考池重采样表明结论不依赖这张参考图", """
            推理时每个目标风格用测试集第一张图作参考，它也在 DINO-S 的参考池里。
            所以我们做了参考池重采样：每个风格随机取 30 张中的 8 张作参考池，只有约 27% 的概率包含这张图。
            重采样 1000 次，WEAVE 相对原样复制的优势全部为正，平均 0.030，和全池的 0.033 几乎一样，结论不依赖这张参考图。""")
    bullets(s, 0.65, 1.85, 6.85, 4.3, [
        "推理时每个目标风格以测试集**第一张图**为参考，它也在 DINO-S 的参考池中",
        "重采样：每个风格随机取 **8 / 30** 张作参考池，只有约 **27%** 的概率包含这张图",
        "1000 次重采样：WEAVE 相对 IDT 的优势**全部为正**，0.030（95% 区间 [0.020, 0.039]）",
        "与全池的 0.033 几乎一致 → 结论**不依赖**这张参考图"], size=16, after=11)
    table(s, 7.75, 2.0, [1.5, 0.82, 0.72, 1.86], [["参考池大小", "WEAVE", "IDT", "优势（95% 区间）"],
                                                ["8 / 30", "0.345", "0.315", "0.030 [0.020, 0.039]"],
                                                ["16 / 30", "0.378", "0.346", "0.032 [0.023, 0.038]"],
                                                ["30 / 30（全池）", "0.403", "0.370", "0.033"]], row_h=0.52, size=13)
    takeaway(s, "参考池里有没有这张图，WEAVE 相对原样复制的优势都稳定为正", y=6.12)

    s = d.content("Q：WEAVE 需要在风格图集上训练，对比公平吗？", "同类学习型基线处于相同设定；WEAVE 还能零样本迁移", """
            CUT、SaMST、SaMam 同样在我们的数据集上训练，属于同一设定；WEAVE 训练只需要 82.8 秒，训练代价远低于其他学习型方法；
            它还能零样本迁移到 5 个没见过的风格族，三项指标仍是最好的；扩散编辑器虽然免训练，但每张图都要运行十亿级模型。""")
    bullets(s, 0.65, 1.85, 12.0, 3.0, [
        "CUT、SaMST、SaMam 同样在我们的数据集上训练，属于**同一设定**",
        "WEAVE 训练仅 **82.8 秒**，训练代价远低于其他学习型方法（39.5–1440 分钟）",
        "零样本迁移到 **5 个未见风格族**：CLIP-S、LPIPS、DINO-C 仍为最佳",
        "扩散编辑器虽免训练，但每张图需运行十亿级模型（StyleID 处理 750 张需 63 分钟）"], size=17, after=12)
    other5 = [["未见风格族（750 对）", "DINO-S", "CLIP-S", "LPIPS", "DINO-C"],
              ["WEAVE", "0.531", "0.774", "0.280", "0.759"],  # paper Table III
              ["SaMam", "0.547", "0.758", "0.399", "0.737"],
              ["SaMST", "0.391", "0.749", "0.621", "0.268"]]

    def style_other5(i, j, v):
        st = {"fill": "FDECEE"} if i == 1 else {}
        if j:
            col = [float(other5[k][j]) for k in (1, 2, 3)]
            if float(v) == (min(col) if j == 3 else max(col)):
                st["bold"] = True
        return st
    table(s, 3.0, 3.95, [2.4, 1.1, 1.1, 1.1, 1.1], other5, row_h=0.42, size=14, cell_style=style_other5)
    takeaway(s, "同一设定下 WEAVE 训练代价最低，且能零样本迁移到未见风格族", y=6.12)

    s = d.content("Q：基线是否都正确运行？", "所有数字都可追溯到原始评测文件；发现问题的基线已移除", """
            所有数字都能追溯到原始评测文件，主表每一格都记录了出处。
            核查时发现 SD-Turbo 原配置是 strength 0.8 乘 1 步，在 diffusers 里实际是 0 步去噪，输出等于输入，所以已从结果中移除，等待重跑；
            同时更正了两处抄录错误，†和‡标记由脚本按定义自动计算；
            Z-STAR 在 12 GB 显存下跑不了 512 像素，推理时间是按步数外推的估计；StyTr² 的训练时间用的是作者报告。""")
    bullets(s, 0.65, 1.85, 12.0, 4.2, [
        "所有数字可追溯到原始评测文件：主表**逐格记录出处**（main_table.csv）",
        "SD-Turbo 原配置 strength 0.8 × 1 步，在 diffusers 中实际为 **0 步去噪**、输出等于输入 → **已移除**，待重跑",
        "更正两处抄录错误（Z-STAR P2A、StyleID R5）；†/‡ 标记由脚本按定义**自动计算**",
        "Z-STAR 在 12 GB 显存下无法完整运行 512px，推理时间为按步数外推的估计；StyTr² 训练时间采用作者报告"],
        size=17, after=14)
    takeaway(s, "宁可少一个基线，也不报告一个配置错误的基线", y=6.12)

    s = d.content("附：超参数敏感性", "λLL 是平滑的风格–内容旋钮；β = 2 带来明显的风格提升", """
            λLL 在 0.1 到 0.5 之间，DINO-S 几乎不变，内容和风格指标单调地此消彼长，是一个平滑的旋钮；
            逐步 AdaIN 强度 β 在 1.0 到 1.5 之间很平稳，到 2.0 风格分明显提升；α 的影响在 0.002 以内。""")
    _, wh = picture(s, "sweeps.png", 0.62, 1.8, w=12.09)
    bullets(s, 0.65, 1.8 + wh + 0.25, 12.0, 1.5, ["λLL ∈ [0.1, 0.5]：DINO-S 稳定在 0.4847–0.4857，DINO-C 与 CLIP-S 单调此消彼长",
                                                 "β 扫描在同一 checkpoint 上只改推理，完全匹配；λLL、α 扫描为固定预算重训"],
            size=15, after=6)

    s = d.content("附：ArtFID 审计", "ArtFID 把原样复制排第一，不能单独用作风格排名", """
            在同一份 750 个请求的源图列表上计算，原样复制的 ArtFID 和原始 FID 都是最好的，所以 ArtFID 不能单独用来排风格。
            WEAVE 和 SaMam 的 ArtFID 接近，但 WEAVE 的 CLIP-S 高于原样复制，SaMam 在两个基准上低于原样复制。
            Z-STAR 和 StyleAligned 的输出来自另一份源图列表，没法公平比较，所以排除了。""")
    note_zh = {"IDT": "10,000 次 bootstrap", "WEAVE": "epoch 4 检查点", "SaMam": "",
               "Seedream 4.5": "API，720 / 750 个请求", "TGT": "30 次随机参考，±56.1"}
    data = [["方法", "FID", "LPIPS", "ArtFID ↓", "说明"]] + [
        [r["method"], f"{float(r['fid']):.1f}", f"{float(r['lpips']):.3f}", f"{float(r['artfid']):.1f}", note_zh.get(r["method"], "")]
        for r in read_csv(ICME / "data" / "artfid_d5.csv")]
    table(s, 0.62, 1.85, [2.2, 1.3, 1.3, 1.5, 3.0], data, row_h=0.52, size=15,
          cell_style=lambda i, j, v: {"fill": "FDECEE", "bold": True} if i == 2 else ({"bold": True} if i == 1 and j == 3 else {}))
    bullets(s, 0.65, 5.15, 12.0, 0.8, ["ArtFID = (1 + FID) × (1 + LPIPS)，按目标风格平均；Z-STAR、StyleAligned 源图列表不同，已排除"], size=15)
    takeaway(s, "IDT 的 ArtFID 与原始 FID 都最好 → 需要 IDT–TGT 判据给指标定方向", y=6.12)

    out = HERE / "WEAVE_defense.pptx"
    d.finish(out)
    print(f"wrote {out.name}: {len(d.prs.slides)} slides; baselines {N_BASE}, failing {N_FAIL}, style-failing {N_STYLE_FAIL}")


if __name__ == "__main__":
    build()
