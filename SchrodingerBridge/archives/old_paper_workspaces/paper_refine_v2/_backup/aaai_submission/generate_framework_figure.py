from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("fig_framework_overview_v3.png")
W, H = 1100, 1180
BG = "white"
EDGE = "#2f2f2f"
TEXT = "#181818"
MUTED = "#666666"
SOLID = "#2f3e52"
DASH = "#7b7b7b"
STYLE = "#946c1f"


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
        ]
    candidates += [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_HEAD = load_font(22, bold=True)
FONT_TEXT = load_font(18)
FONT_TEXT_SM = load_font(16)
FONT_NOTE = load_font(15)


def multiline_center(draw, box, text, font, fill=TEXT, spacing=5):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    line_heights = []
    widths = []
    for line in lines:
        w, h = draw.textsize(line, font=font)
        widths.append(w)
        line_heights.append(h)
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in zip(lines, widths, line_heights):
        x = x0 + (x1 - x0 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + spacing


def rounded_box(draw, box, fill, outline=EDGE, radius=28, width=4):
    del radius
    draw.rectangle(box, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=SOLID, width=5, head=18, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = 18
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
        x2 - head * math.cos(ang) + 0.6 * head * math.sin(ang),
        y2 - head * math.sin(ang) - 0.6 * head * math.cos(ang),
    )
    right = (
        x2 - head * math.cos(ang) - 0.6 * head * math.sin(ang),
        y2 - head * math.sin(ang) + 0.6 * head * math.cos(ang),
    )
    draw.polygon([end, left, right], fill=color)


def label(draw, xy, text, font, fill=MUTED, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def main():
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    # Column backgrounds
    draw.rectangle((35, 35, 355, 1125), outline="#efcfa3", fill="#fff8ee", width=2)
    draw.rectangle((390, 35, 1065, 1125), outline="#bfd2f8", fill="#f4f8ff", width=2)

    label(draw, (60, 58), "Conditioning branch", FONT_HEAD, fill=STYLE)
    label(draw, (415, 58), "U-Net-like latent transport trunk", FONT_HEAD, fill="#37557f")

    # Conditioning column
    c1 = (75, 95, 315, 165)
    c2 = (75, 205, 315, 290)
    c3 = (75, 330, 315, 430)
    c4 = (75, 470, 315, 570)
    for box in (c1, c2, c3, c4):
        rounded_box(draw, box, "#fff0d7", outline="#c85a0c", width=3)
    multiline_center(draw, c1, "Style ID s", FONT_TEXT)
    multiline_center(draw, c2, "style_emb + time code\n(global style vector)", FONT_TEXT_SM)
    multiline_center(draw, c3, "Reference latent z_s\nor learned prior P_s", FONT_TEXT_SM)
    multiline_center(draw, c4, "Spatial style map M_s\nprojected to 16 x 16 body", FONT_TEXT_SM)
    arrow(draw, (195, 165), (195, 205), color="#c85a0c", width=4)
    arrow(draw, (195, 290), (195, 330), color="#c85a0c", width=4)
    arrow(draw, (195, 430), (195, 470), color="#c85a0c", width=4)

    # Main trunk
    b_in = (510, 90, 840, 160)
    b_lift = (510, 195, 840, 275)
    b_hires = (470, 315, 880, 405)
    b_down = (545, 445, 805, 515)
    b_body = (470, 555, 880, 655)
    b_up = (545, 695, 805, 765)
    b_skip = (545, 805, 805, 885)
    b_dec = (835, 805, 1015, 885)
    b_out = (835, 900, 1015, 950)
    xattn = (560, 975, 1015, 1055)
    loss1 = (560, 1080, 770, 1145)
    loss2 = (800, 1080, 1015, 1145)

    rounded_box(draw, b_in, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_lift, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_hires, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_down, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_body, "#fff0d7", outline="#2962e3", width=3)
    rounded_box(draw, b_up, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_skip, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_dec, "#dbe8f7", outline="#2962e3", width=3)
    rounded_box(draw, b_out, "#fbe3e2", outline="#2962e3", width=3)
    rounded_box(draw, xattn, "#fff0d7", outline="#c85a0c", width=3)
    rounded_box(draw, loss1, "#e0f6ef", outline="#0e7c72", width=3)
    rounded_box(draw, loss2, "#e0f6ef", outline="#0e7c72", width=3)

    multiline_center(draw, b_in, "Input latent z_t\n4 x 32 x 32", FONT_TEXT_SM)
    multiline_center(draw, b_lift, "enc_in + SiLU\nlift to 128 channels", FONT_TEXT_SM)
    multiline_center(draw, b_hires, "Hi-res stem x 2 @ 32 x 32\nshallow feature blocks before downsampling", FONT_TEXT_SM)
    multiline_center(draw, b_down, "4x4 Conv, stride 2\n32 -> 16", FONT_TEXT_SM)
    multiline_center(draw, b_body, "SemanticCrossAttn body x 4 @ 16 x 16\nmain style-writing site\nQ = content, K/V = style map", FONT_TEXT_SM)
    multiline_center(draw, b_up, "Upsample + blur\n16 -> 32", FONT_TEXT_SM)
    multiline_center(draw, b_skip, "Skip routing + fusion\nupsampled body + 32x32 content skip", FONT_TEXT_SM)
    multiline_center(draw, b_dec, "Decoder blocks x 2\n+ dec_out", FONT_TEXT_SM)
    multiline_center(draw, b_out, "latent velocity v_theta", FONT_TEXT_SM)
    multiline_center(draw, xattn, "Cross-Attn at bottleneck\nQ = content features, K/V = style map\nA = softmax(QK^T / tau),  body <- A V", FONT_TEXT_SM)
    multiline_center(draw, loss1, "Terminal SWD / OT proxy\nmatch endpoint patch distributions", FONT_TEXT_SM)
    multiline_center(draw, loss2, "Kinetic loss\npenalize large latent motion", FONT_TEXT_SM)

    arrow(draw, (675, 160), (675, 195))
    arrow(draw, (675, 275), (675, 315))
    arrow(draw, (675, 405), (675, 445))
    arrow(draw, (675, 515), (675, 555))
    arrow(draw, (675, 655), (675, 695))
    arrow(draw, (675, 765), (675, 805))
    arrow(draw, (805, 845), (835, 845))
    arrow(draw, (925, 885), (925, 910))

    # Skip connection from hires to fusion
    arrow(draw, (470, 360), (430, 360), color="#2962e3", width=3)
    arrow(draw, (430, 360), (430, 845), color="#2962e3", width=3)
    arrow(draw, (430, 845), (545, 845), color="#2962e3", width=3)
    label(draw, (438, 635), "32x32 content skip", FONT_NOTE, fill="#2962e3")

    # Style map and global code into trunk
    arrow(draw, (315, 248), (470, 248), color="#c85a0c", width=3, dashed=True)
    arrow(draw, (470, 248), (470, 845), color="#c85a0c", width=3, dashed=True)
    arrow(draw, (470, 845), (835, 845), color="#c85a0c", width=3, dashed=True)
    label(draw, (770, 790), "global style code -> decoder modulation", FONT_NOTE, fill="#c85a0c", anchor="ma")

    arrow(draw, (315, 520), (470, 520), color="#c85a0c", width=3)
    arrow(draw, (470, 520), (470, 605), color="#c85a0c", width=3)
    label(draw, (520, 575), "style map M_s -> bottleneck CA blocks", FONT_NOTE, fill="#c85a0c")

    # Cross-attention and objective links
    arrow(draw, (675, 655), (675, 975), color="#c85a0c", width=3, dashed=True)
    arrow(draw, (925, 950), (925, 1080), color="#0e7c72", width=3)
    arrow(draw, (560, 1112), (505, 1112), color="#0e7c72", width=3)
    arrow(draw, (505, 1112), (505, 925), color="#0e7c72", width=3)
    arrow(draw, (505, 925), (835, 925), color="#0e7c72", width=3)
    arrow(draw, (800, 1112), (760, 1112), color="#0e7c72", width=3)
    arrow(draw, (760, 1112), (760, 925), color="#0e7c72", width=3)
    label(draw, (575, 957), "mechanism detail", FONT_NOTE, fill="#c85a0c")

    im.save(OUT)


if __name__ == "__main__":
    main()
