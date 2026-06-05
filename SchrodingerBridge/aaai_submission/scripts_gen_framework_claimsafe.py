"""Build a claim-safe framework figure without implying active online OT training.

The user wants to keep the stronger visual style of ``framework_lbm_main_v5.png``
while removing the misleading active-looking ``OT + Sinkhorn`` block. This script
uses that figure as a visual base, rewrites the bottom-left training band, and
keeps the successful middle-band / SA-SWD styling intact.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "framework_lbm_main_v5.png"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(25, bold=True)
FONT_LABEL = load_font(24, bold=True)
FONT_TEXT = load_font(20, bold=False)
FONT_SMALL = load_font(18, bold=False)
FONT_CHIP = load_font(17, bold=True)
FONT_TINY = load_font(15, bold=False)

COLORS = {
    "warm_bg": (255, 249, 240),
    "ink": (28, 28, 28),
    "batch_fill": (248, 244, 255),
    "batch_edge": (85, 68, 145),
    "cache_fill": (255, 245, 220),
    "cache_edge": (193, 132, 25),
    "endpoint_fill": (237, 238, 253),
    "endpoint_edge": (97, 106, 188),
    "loss_fill": (240, 244, 255),
    "loss_edge": (41, 66, 138),
    "kin_fill": (236, 246, 236),
    "kin_edge": (78, 120, 78),
    "note_fill": (248, 248, 248),
    "note_edge": (160, 160, 160),
    "dash": (55, 55, 55),
}


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
    radius: int,
    title: str | None = None,
    body: str | None = None,
    title_font: ImageFont.ImageFont = FONT_LABEL,
    body_font: ImageFont.ImageFont = FONT_TEXT,
    title_fill: tuple[int, int, int] = COLORS["ink"],
    body_fill: tuple[int, int, int] = COLORS["ink"],
) -> None:
    draw.rounded_rectangle(xyxy, radius=radius, fill=fill, outline=outline, width=width)
    x0, y0, x1, y1 = xyxy
    cx = (x0 + x1) / 2
    if title:
        ty = y0 + 22
        draw.text((cx, ty), title, font=title_font, fill=title_fill, anchor="ma")
    if body:
        by = (y0 + y1) / 2 + (12 if title else 0)
        draw.multiline_text(
            (cx, by),
            body,
            font=body_font,
            fill=body_fill,
            anchor="mm",
            align="center",
            spacing=4,
        )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int] = (30, 30, 30),
    width: int = 5,
    head: int = 18,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        tri = [(ex, ey), (ex - sign * head, ey - head // 2), (ex - sign * head, ey + head // 2)]
    else:
        sign = 1 if ey >= sy else -1
        tri = [(ex, ey), (ex - head // 2, ey - sign * head), (ex + head // 2, ey - sign * head)]
    draw.polygon(tri, fill=fill)


def dashed_vertical_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int] = COLORS["dash"],
    width: int = 4,
    dash: int = 12,
    gap: int = 10,
) -> None:
    sx, sy = start
    ex, ey = end
    direction = 1 if ey >= sy else -1
    y = sy
    while (y - ey) * direction < 0:
        y_next = y + direction * dash
        if (y_next - ey) * direction > 0:
            y_next = ey
        draw.line([(sx, y), (sx, y_next)], fill=fill, width=width)
        y = y_next + direction * gap
    head = 16
    tri = [(ex, ey), (ex - head // 2, ey - direction * head), (ex + head // 2, ey - direction * head)]
    draw.polygon(tri, fill=fill)


def main() -> None:
    canvas = Image.open(BASE).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # Redraw only the left/bottom active-training area. The right SA-SWD inset stays.
    redraw = (36, 566, 1000, 860)
    draw.rounded_rectangle(redraw, radius=18, fill=COLORS["warm_bg"])
    draw.text((52, 586), "3. TRAINING (ACTIVE SUPERVISION)", font=FONT_TITLE, fill=(140, 90, 14))

    rounded_box(
        draw,
        (54, 640, 276, 744),
        fill=COLORS["batch_fill"],
        outline=COLORS["batch_edge"],
        width=3,
        radius=16,
        title="content latent batch",
        body="{z0}",
        title_font=FONT_TEXT,
        body_font=FONT_LABEL,
        title_fill=(30, 30, 100),
        body_fill=(42, 42, 110),
    )
    rounded_box(
        draw,
        (54, 760, 276, 854),
        fill=(241, 248, 255),
        outline=(84, 130, 196),
        width=3,
        radius=16,
        title="target-domain latent pool",
        body="{zs}",
        title_font=FONT_TEXT,
        body_font=FONT_LABEL,
        title_fill=(30, 70, 120),
        body_fill=(30, 70, 120),
    )

    cache_box = (326, 646, 588, 850)
    rounded_box(
        draw,
        cache_box,
        fill=(255, 249, 233),
        outline=COLORS["cache_edge"],
        width=3,
        radius=20,
        title="offline pairing cache  C(z0, s)",
        body=None,
        title_font=FONT_TEXT,
    )
    rounded_box(
        draw,
        (352, 700, 562, 734),
        fill=(255, 252, 244),
        outline=COLORS["cache_edge"],
        width=2,
        radius=12,
        title="prototype-aligned queue",
        body=None,
        title_font=FONT_TINY,
        title_fill=(108, 78, 18),
    )
    draw.text((456, 748), "content-coherent target candidates", font=FONT_TINY, fill=(120, 95, 42), anchor="ma")
    # Draw a compact row of candidate tiles to make the cache read visually.
    tile_y0, tile_y1 = 760, 798
    tile_xs = [368, 410, 452, 494]
    tile_fills = [(245, 223, 168), (219, 205, 250), (247, 213, 138), (244, 199, 210)]
    for x0, fill in zip(tile_xs, tile_fills):
        draw.rounded_rectangle((x0, tile_y0, x0 + 30, tile_y1), radius=8, fill=fill, outline=COLORS["cache_edge"], width=2)
    draw.text((535, 779), "...", font=FONT_LABEL, fill=(120, 90, 25), anchor="mm")
    rounded_box(
        draw,
        (356, 812, 558, 844),
        fill=(255, 252, 242),
        outline=COLORS["cache_edge"],
        width=2,
        radius=12,
        title="cache draw",
        body=None,
        title_font=FONT_TINY,
        title_fill=(110, 75, 10),
    )

    rounded_box(
        draw,
        (628, 694, 858, 790),
        fill=COLORS["endpoint_fill"],
        outline=COLORS["endpoint_edge"],
        width=3,
        radius=18,
        title="selected endpoint",
        body="z~1",
        title_font=FONT_TEXT,
        body_font=FONT_TITLE,
        title_fill=(46, 58, 128),
        body_fill=(46, 58, 128),
    )
    rounded_box(
        draw,
        (598, 806, 726, 854),
        fill=COLORS["loss_fill"],
        outline=COLORS["loss_edge"],
        width=2,
        radius=14,
        title="endpoint OMF",
        body=None,
        title_font=FONT_CHIP,
        title_fill=(33, 55, 120),
    )
    rounded_box(
        draw,
        (752, 806, 880, 854),
        fill=COLORS["kin_fill"],
        outline=COLORS["kin_edge"],
        width=2,
        radius=14,
        title="Kinetic  L_kin",
        body=None,
        title_font=FONT_CHIP,
        title_fill=(55, 95, 55),
    )
    # OT is kept only as a muted family-background note, not as an active block.
    draw.text(
        (456, 861),
        "historical OT family: supplementary background only",
        font=FONT_TINY,
        fill=(118, 118, 118),
        anchor="ma",
    )

    arrow(draw, (276, 694), (326, 724), width=4)
    arrow(draw, (276, 808), (336, 808), width=4)
    arrow(draw, (588, 748), (628, 748), width=4)
    dashed_vertical_arrow(draw, (678, 694), (678, 568))
    dashed_vertical_arrow(draw, (816, 806), (816, 568))
    dashed_vertical_arrow(draw, (662, 806), (662, 568))

    out = OUT_DIR / "fig_framework_claimsafe.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
