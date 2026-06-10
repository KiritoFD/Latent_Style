from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "framework_lbm_main_unified.png"
OUT_PDF = ROOT / "framework_lbm_main_unified.pdf"

W, H = 1500, 820
BG = "#FCFBF8"
TEXT = "#1F1F1F"
MUTED = "#666666"

ORANGE = "#B45309"
ORANGE_FILL = "#F7E9DA"
BLUE = "#1D4ED8"
BLUE_FILL = "#EEF4FE"
GREEN = "#4F8F57"
GREEN_FILL = "#EDF7EF"
PURPLE = "#8E63C0"
PURPLE_FILL = "#F3ECFB"
RED = "#C0392B"
RED_FILL = "#FBE7E2"
GRAY = "#909090"
GRID_LINE = "#D8D8D8"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\timesbd.ttf",
            r"C:\Windows\Fonts\georgiab.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
    candidates += [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SECTION = load_font(21, bold=True)
FONT_BOX = load_font(17, bold=True)
FONT_BODY = load_font(15)
FONT_SMALL = load_font(13)
FONT_TINY = load_font(12)


def rounded_box(draw, box, outline, fill, width=3, radius=18):
    draw.rounded_rectangle(box, radius=radius, outline=outline, fill=fill, width=width)


def center_text(draw, box, text, font, fill=TEXT, spacing=4):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    metrics = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        metrics.append((bb[2] - bb[0], bb[3] - bb[1]))
    total_h = sum(h for _, h in metrics) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, (w, h) in zip(lines, metrics):
        x = x0 + (x1 - x0 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + spacing


def label(draw, xy, text, font, fill=MUTED, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def arrow(draw, start, end, color=TEXT, width=4, head=14, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = 22
        for i in range(steps):
            if i % 2 == 0:
                xa = x1 + (x2 - x1) * i / steps
                ya = y1 + (y2 - y1) * i / steps
                xb = x1 + (x2 - x1) * (i + 1) / steps
                yb = y1 + (y2 - y1) * (i + 1) / steps
                draw.line((xa, ya, xb, yb), fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)

    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - head * math.cos(ang) + 0.55 * head * math.sin(ang),
        y2 - head * math.sin(ang) - 0.55 * head * math.cos(ang),
    )
    right = (
        x2 - head * math.cos(ang) - 0.55 * head * math.sin(ang),
        y2 - head * math.sin(ang) + 0.55 * head * math.cos(ang),
    )
    draw.polygon([end, left, right], fill=color)


def tiny_grid(draw, box, rows, cols, tint, outline):
    x0, y0, x1, y1 = box
    w = (x1 - x0) / cols
    h = (y1 - y0) / rows
    for r in range(rows):
        for c in range(cols):
            fill = tint if (r + c) % 3 == 0 else "#FFFFFF"
            draw.rectangle(
                (x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h),
                outline=outline,
                fill=fill,
                width=1,
            )


def draw_content_thumbnail(draw, box, outline):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, outline, "#FFFFFF", width=2, radius=12)
    pad = 8
    draw.rectangle((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill="#EAF3FF", outline=outline, width=1)
    draw.rectangle((x0 + pad, y0 + pad, x1 - pad, y0 + 30), fill="#CFE6FF", outline=None)
    draw.ellipse((x0 + 18, y0 + 16, x0 + 34, y0 + 32), fill="#FFFFFF")
    draw.polygon([(x0 + 18, y1 - 18), (x0 + 48, y0 + 52), (x0 + 76, y1 - 18)], fill="#7FA4D9")
    draw.polygon([(x0 + 54, y1 - 18), (x0 + 82, y0 + 42), (x0 + 108, y1 - 18)], fill="#5F88C4")
    draw.rectangle((x0 + 80, y0 + 42, x0 + 94, y1 - 18), fill="#E6E0D1", outline=outline, width=1)
    draw.polygon([(x0 + 77, y0 + 42), (x0 + 87, y0 + 27), (x0 + 97, y0 + 42)], fill="#B46A39")
    draw.line((x0 + 18, y1 - 18, x1 - 16, y1 - 18), fill="#4E8B57", width=3)


def draw_vae_icon(draw, box, outline, frozen=False):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, outline, "#FFFFFF", width=2, radius=12)
    mid = (x0 + x1) / 2
    top = y0 + 18
    bot = y1 - 18
    draw.polygon([(x0 + 18, top), (mid - 8, (y0 + y1) / 2), (x0 + 18, bot)], fill="#DCE7FA", outline=outline)
    draw.polygon([(x1 - 18, top), (mid + 8, (y0 + y1) / 2), (x1 - 18, bot)], fill="#DCE7FA", outline=outline)
    if frozen:
        label(draw, (x1 - 14, y1 - 14), "frozen", FONT_TINY, fill=BLUE, anchor="rs")


def draw_node_chain(draw, box, outline):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, outline, "#FFFFFF", width=2, radius=12)
    pts = [
        (x0 + 18, y0 + 30),
        (x0 + 42, y0 + 54),
        (x0 + 70, y0 + 24),
        (x0 + 98, y0 + 48),
    ]
    draw.line(pts, fill=outline, width=2)
    for px, py in pts:
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#C6D9B5", outline="#56784F", width=1)


def draw_transport_field(draw, box):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, BLUE, "#FFFFFF", width=2, radius=14)
    center_text(draw, (x0, y0 + 4, x1, y0 + 30), "style-conditioned latent field v_theta", FONT_SECTION, fill=BLUE)
    panels = [
        (x0 + 18, y0 + 38, x0 + 144, y1 - 20),
        (x0 + 160, y0 + 38, x0 + 324, y1 - 20),
        (x0 + 340, y0 + 38, x1 - 18, y1 - 20),
    ]
    for p in panels:
        rounded_box(draw, p, BLUE, "#F7FAFF", width=1, radius=8)

    center_text(draw, (panels[0][0], panels[0][1] + 4, panels[0][2], panels[0][1] + 24), "time embedding t", FONT_TINY)
    cx = (panels[0][0] + panels[0][2]) / 2
    cy = (panels[0][1] + panels[0][3]) / 2 + 2
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), outline=TEXT, width=2)
    draw.line((cx, cy, cx, cy - 12), fill=TEXT, width=2)
    draw.line((cx, cy, cx + 9, cy + 7), fill=TEXT, width=2)

    center_text(draw, (panels[1][0], panels[1][1] + 4, panels[1][2], panels[1][1] + 24), "semantic routing / cross-attention", FONT_TINY)
    pts = [
        (panels[1][0] + 26, cy - 6),
        (panels[1][0] + 52, cy + 12),
        (panels[1][0] + 80, cy - 2),
        (panels[1][0] + 108, cy + 12),
        (panels[1][0] + 134, cy - 6),
    ]
    draw.line(pts, fill="#555555", width=2)
    colors = ["#76A5E3", "#A58AE0", "#8BC06E", "#E89A9A", "#76A5E3"]
    for (px, py), color in zip(pts, colors):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline="#555555", width=1)

    center_text(draw, (panels[2][0], panels[2][1] + 4, panels[2][2], panels[2][1] + 24), "latent U-Net", FONT_TINY)
    px0 = panels[2][0] + 20
    py0 = panels[2][1] + 34
    draw.rectangle((px0, py0, px0 + 12, py0 + 48), fill="#9CB9E8", outline=BLUE, width=1)
    draw.rectangle((px0 + 20, py0 + 10, px0 + 32, py0 + 38), fill="#9CB9E8", outline=BLUE, width=1)
    draw.rectangle((px0 + 56, py0 + 10, px0 + 68, py0 + 38), fill="#9CB9E8", outline=BLUE, width=1)
    draw.rectangle((px0 + 76, py0, px0 + 88, py0 + 48), fill="#9CB9E8", outline=BLUE, width=1)
    draw.line((px0 + 12, py0 + 24, px0 + 76, py0 + 24), fill=BLUE, width=3)


def draw_endpoint_selector(draw, box):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, GREEN, "#FFFFFF", width=2, radius=14)
    center_text(draw, (x0 + 10, y0 + 10, x1 - 10, y0 + 96), "prototype-aware /\nstructure-aware\nendpoint selector", FONT_BOX, fill=GREEN)
    label(draw, ((x0 + x1) / 2, y0 + 110), "pairing cache", FONT_SMALL, fill=GREEN, anchor="mm")
    clusters = [(x0 + 52, y1 - 54, "#5DAE4A"), (x0 + 124, y1 - 54, "#F39C12"), (x0 + 88, y1 - 18, PURPLE)]
    for cx, cy, color in clusters:
        draw.ellipse((cx - 26, cy - 18, cx + 26, cy + 18), outline=color, width=2)
        for dx, dy in [(-10, -5), (8, -8), (12, 8), (-8, 7)]:
            draw.ellipse((cx + dx - 5, cy + dy - 5, cx + dx + 5, cy + dy + 5), fill=color)


def draw_swd_panel(draw, box):
    x0, y0, x1, y1 = box
    rounded_box(draw, box, RED, "#FFFFFF", width=2, radius=14)
    center_text(draw, (x0, y0 + 2, x1, y0 + 30), "SA-SWD terminal matching", FONT_SECTION, fill=RED)
    center_text(draw, (x0 + 8, y0 + 36, x0 + 106, y0 + 74), "generated\nendpoint patches", FONT_TINY)
    center_text(draw, (x0 + 118, y0 + 36, x0 + 212, y0 + 74), "semantic\nbins", FONT_TINY)
    center_text(draw, (x0 + 222, y0 + 36, x1 - 10, y0 + 74), "target-style\npatches", FONT_TINY)
    for r in range(3):
        for c in range(3):
            x = x0 + 18 + c * 28
            y = y0 + 82 + r * 28
            draw.rectangle((x, y, x + 20, y + 20), fill="#7BA6E8" if (r + c) % 2 == 0 else "#5E86D0", outline=BLUE)
    bins = [(x0 + 138, y0 + 94), (x0 + 138, y0 + 126), (x0 + 138, y0 + 158)]
    bin_colors = [BLUE, GREEN, RED]
    bin_names = ["bin 1", "bin 2", "bin K"]
    for (bx, by), c, name in zip(bins, bin_colors, bin_names):
        rounded_box(draw, (bx, by, bx + 70, by + 24), c, "#FFFFFF", width=2, radius=8)
        center_text(draw, (bx, by, bx + 70, by + 24), name, FONT_TINY)
    for r in range(3):
        for c in range(3):
            x = x0 + 256 + c * 28
            y = y0 + 82 + r * 28
            draw.rectangle((x, y, x + 20, y + 20), fill="#A78BFA" if (r + c) % 2 == 0 else "#7C5FD6", outline=PURPLE)
    for y in [106, 138, 170]:
        arrow(draw, (x0 + 102, y0 + y), (x0 + 138, y0 + y), color=bin_colors[[106, 138, 170].index(y)], width=3)
        arrow(draw, (x0 + 208, y0 + y), (x0 + 256, y0 + y), color=bin_colors[[106, 138, 170].index(y)], width=3)
    label(draw, (x0 + 220, y0 + 48), "training refs only", FONT_TINY, fill=GREEN, anchor="mm")
    center_text(draw, (x0, y1 - 38, x1, y1 - 8), "match within semantic bins (distributional)", FONT_SMALL)


def main():
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    center_text(draw, (0, 12, W, 64), "Latent Bridge Matching (LBM)", FONT_TITLE)

    top = (28, 74, W - 28, 234)
    mid = (28, 252, W - 28, 494)
    bot = (28, 512, W - 28, 724)

    rounded_box(draw, top, ORANGE, ORANGE_FILL, width=3, radius=18)
    rounded_box(draw, mid, BLUE, BLUE_FILL, width=3, radius=18)
    rounded_box(draw, bot, GREEN, GREEN_FILL, width=3, radius=18)

    label(draw, (52, 104), "1. STYLE CONTROL", FONT_SECTION, fill=ORANGE)
    label(draw, (52, 280), "2. INFERENCE PATH (ACTIVE AT TEST TIME)", FONT_SECTION, fill=BLUE)
    label(draw, (52, 540), "3. TRAINING-SIDE ENDPOINT SUPERVISION", FONT_SECTION, fill=GREEN)

    style_id = (120, 122, 332, 188)
    tokenizer = (416, 100, 914, 206)
    style_code = (984, 122, 1146, 198)
    rounded_box(draw, style_id, ORANGE, "#FFFFFF", width=2, radius=12)
    rounded_box(draw, tokenizer, ORANGE, "#FFFDF9", width=2, radius=16)
    rounded_box(draw, style_code, ORANGE, "#FFFFFF", width=2, radius=12)
    center_text(draw, (style_id[0], style_id[1] + 2, style_id[2], style_id[1] + 28), "Style ID s", FONT_SMALL)
    tiny_grid(draw, (style_id[0] + 18, style_id[1] + 34, style_id[2] - 18, style_id[3] - 14), 1, 6, "#EFDDB0", ORANGE)
    center_text(draw, (tokenizer[0], tokenizer[1] - 2, tokenizer[2], tokenizer[1] + 28), "Style Tokenizer T_phi", FONT_SECTION)
    proto = (tokenizer[0] + 18, tokenizer[1] + 34, tokenizer[0] + 128, tokenizer[1] + 88)
    atoms = (tokenizer[0] + 154, tokenizer[1] + 34, tokenizer[0] + 284, tokenizer[1] + 88)
    prior = (tokenizer[0] + 310, tokenizer[1] + 34, tokenizer[0] + 420, tokenizer[1] + 88)
    for box, name in [(proto, "prototype"), (atoms, "atoms"), (prior, "prior")]:
        rounded_box(draw, box, ORANGE, "#FFFFFF", width=1, radius=8)
        center_text(draw, (box[0], box[1] - 1, box[2], box[1] + 18), name, FONT_TINY)
    for i, color in enumerate([PURPLE, "#FFD23F", "#F58518", "#F58518"]):
        draw.rectangle((proto[0] + 14 + i * 24, proto[1] + 20, proto[0] + 34 + i * 24, proto[1] + 44), fill=color, outline=ORANGE)
    atom_colors = ["#F3E39A", "#A7A7E8", "#F8D1A8", "#F3E39A", "#E8A1B5", "#F59EA8"]
    idx = 0
    for r in range(2):
        for c in range(3):
            x = atoms[0] + 18 + c * 32
            y = atoms[1] + 12 + r * 28
            draw.rectangle((x, y, x + 20, y + 18), fill=atom_colors[idx], outline=ORANGE)
            idx += 1
    for r in range(3):
        for c in range(4):
            draw.ellipse((prior[0] + 16 + c * 24, prior[1] + 14 + r * 18, prior[0] + 22 + c * 24, prior[1] + 20 + r * 18), fill=PURPLE)
    center_text(draw, (style_code[0], style_code[1] + 2, style_code[2], style_code[1] + 28), "style code\nc_s", FONT_SMALL)
    tiny_grid(draw, (style_code[0] + 20, style_code[1] + 40, style_code[2] - 20, style_code[3] - 16), 1, 4, PURPLE_FILL, ORANGE)
    label(draw, (W - 132, 148), "style-ID\nconditioning only", FONT_SMALL, fill=ORANGE, anchor="mm")

    arrow(draw, (332, 156), (416, 156), color=TEXT, width=4)
    arrow(draw, (914, 156), (984, 156), color=TEXT, width=4)
    arrow(draw, ((style_code[0] + style_code[2]) / 2, style_code[3]), ((style_code[0] + style_code[2]) / 2, mid[1]), color=TEXT, width=4)

    content = (48, 336, 180, 450)
    vae_enc = (224, 336, 336, 450)
    latent = (378, 336, 494, 450)
    field = (526, 310, 954, 466)
    euler = (988, 344, 1080, 440)
    end_latent = (1106, 336, 1198, 450)
    vae_dec = (1224, 336, 1318, 450)
    stylized = (1344, 336, 1444, 450)

    draw_content_thumbnail(draw, content, BLUE)
    label(draw, (content[0] + 16, content[1] + 12), "content image x", FONT_SMALL, fill=TEXT)
    draw_vae_icon(draw, vae_enc, BLUE, frozen=True)
    center_text(draw, vae_enc, "Frozen\nVAE\nEncoder", FONT_SMALL)
    rounded_box(draw, latent, BLUE, "#FFFFFF", width=2, radius=12)
    center_text(draw, latent, "source\nlatent z0", FONT_SMALL)
    tiny_grid(draw, (latent[0] + 22, latent[1] + 40, latent[2] - 22, latent[3] - 16), 4, 4, "#D9E7FB", BLUE)
    draw_transport_field(draw, field)
    draw_node_chain(draw, euler, GREEN)
    center_text(draw, euler, "K-step\nEuler", FONT_SMALL)
    rounded_box(draw, end_latent, BLUE, "#FFFFFF", width=2, radius=12)
    center_text(draw, end_latent, "integrated\nlatent z1_hat", FONT_SMALL)
    tiny_grid(draw, (end_latent[0] + 18, end_latent[1] + 40, end_latent[2] - 18, end_latent[3] - 16), 4, 4, "#D9E7FB", BLUE)
    draw_vae_icon(draw, vae_dec, BLUE, frozen=True)
    center_text(draw, vae_dec, "Frozen\nVAE\nDecoder", FONT_SMALL)
    draw_content_thumbnail(draw, stylized, BLUE)
    center_text(draw, (stylized[0], stylized[1] + 2, stylized[2], stylized[1] + 30), "stylized\noutput x_hat", FONT_TINY)

    for start, end in [
        ((180, 394), (224, 394)),
        ((336, 394), (378, 394)),
        ((494, 394), (526, 394)),
        ((954, 394), (988, 394)),
        ((1080, 394), (1106, 394)),
        ((1198, 394), (1224, 394)),
        ((1318, 394), (1344, 394)),
    ]:
        arrow(draw, start, end, color=TEXT, width=4)

    content_batch = (50, 560, 214, 660)
    style_batch = (50, 678, 214, 778)
    selector = (302, 596, 564, 780)
    paired = (602, 628, 716, 744)
    kinetic = (822, 596, 1060, 780)
    term = (1084, 596, 1440, 780)

    rounded_box(draw, content_batch, GREEN, "#FFFFFF", width=2, radius=12)
    rounded_box(draw, style_batch, GREEN, "#FFFFFF", width=2, radius=12)
    rounded_box(draw, selector, GREEN, "#FFFFFF", width=2, radius=14)
    rounded_box(draw, paired, GREEN, "#FFFFFF", width=2, radius=12)
    rounded_box(draw, kinetic, GREEN, "#FFFFFF", width=2, radius=14)
    rounded_box(draw, term, RED, "#FFFFFF", width=2, radius=14)

    center_text(draw, (content_batch[0], content_batch[1] + 4, content_batch[2], content_batch[1] + 28), "content latent\nbatch {z0}", FONT_SMALL)
    center_text(draw, (style_batch[0], style_batch[1] + 4, style_batch[2], style_batch[1] + 28), "target-style latent\nbatch {z_style}", FONT_SMALL)
    tiny_grid(draw, (content_batch[0] + 18, content_batch[1] + 44, content_batch[2] - 26, content_batch[1] + 82), 2, 4, "#E7EDF8", GREEN)
    tiny_grid(draw, (style_batch[0] + 18, style_batch[1] + 44, style_batch[2] - 26, style_batch[1] + 82), 2, 4, "#EFE3F8", GREEN)
    draw_endpoint_selector(draw, selector)
    center_text(draw, paired, "paired endpoint\nz1_tilde", FONT_SMALL)
    tiny_grid(draw, (paired[0] + 22, paired[1] + 44, paired[2] - 22, paired[3] - 18), 3, 3, "#E7E2F7", GREEN)
    center_text(draw, (kinetic[0], kinetic[1] + 4, kinetic[2], kinetic[1] + 28), "Kinetic regularization\nL_kin", FONT_SECTION, fill=GREEN)
    draw_node_chain(draw, (kinetic[0] + 44, kinetic[1] + 66, kinetic[2] - 44, kinetic[1] + 132), GREEN)
    center_text(draw, (kinetic[0], kinetic[1] + 138, kinetic[2], kinetic[3] - 12), "penalize excessive\nkinetic energy", FONT_SMALL)
    draw_swd_panel(draw, term)

    arrow(draw, (214, 610), (302, 688), color=TEXT, width=4)
    arrow(draw, (214, 728), (302, 688), color=TEXT, width=4)
    arrow(draw, (564, 688), (602, 688), color=TEXT, width=4)
    arrow(draw, (716, 688), (822, 688), color=TEXT, width=4)
    arrow(draw, (716, 688), (1084, 688), color=TEXT, width=4, dashed=True)

    arrow(draw, (1080, 596), (1080, 440), color=TEXT, width=3, dashed=True)
    arrow(draw, (1200, 596), (1200, 450), color=TEXT, width=3, dashed=True)
    arrow(draw, (756, 596), (756, 466), color=TEXT, width=3, dashed=True)

    legend = (510, 734, 980, 774)
    rounded_box(draw, legend, GRAY, "#FFFFFF", width=1, radius=10)
    arrow(draw, (548, 754), (620, 754), color=TEXT, width=3)
    label(draw, (636, 754), "inference (active)", FONT_SMALL, fill=TEXT, anchor="lm")
    arrow(draw, (744, 754), (840, 754), color=TEXT, width=3, dashed=True)
    label(draw, (856, 754), "training supervision only", FONT_SMALL, fill=TEXT, anchor="lm")

    im.save(OUT_PNG)


if __name__ == "__main__":
    main()
