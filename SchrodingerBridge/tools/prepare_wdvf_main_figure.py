"""Prepare the paper-facing WD-VF main figure from a draw.io SVG source.

The source SVG contains draw.io light/dark adaptive styles. For the paper we
freeze all colors to the light branch, enforce a static white background, and
render a high-resolution PNG with Microsoft Edge headless.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


SRC = Path(r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio.svg")
OUT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027_v2")
OUT_SVG = OUT_DIR / "framework_sfm_main.svg"
OUT_PNG = OUT_DIR / "framework_sfm_main.png"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def freeze_light_dark(svg_text: str) -> str:
    svg_text = re.sub(r"light-dark\(\s*([^,]+?)\s*,\s*([^)]+?)\)", r"\1", svg_text)
    svg_text = svg_text.replace("color-scheme: light dark;", "")
    svg_text = svg_text.replace("var(--ge-adaptive-bg, #ffffff)", "#ffffff")
    svg_text = svg_text.replace('background-color="#FFFFFF"', 'background-color="#FFFFFF"')
    svg_text = svg_text.replace("background-color: #FFFFFF;", "")
    svg_text = svg_text.replace("background: #FFFFFF; ", "background: #FFFFFF; ")

    # Keep the layout identical, but avoid the one heavy black highlight box in paper mode.
    svg_text = svg_text.replace(
        'fill="#111111" stroke="#92400e" stroke-width="2"',
        'fill="#FEF3C7" stroke="#92400e" stroke-width="2"',
    )
    svg_text = svg_text.replace(
        'font-size: 14px; color: rgb(228, 158, 115);',
        'font-size: 14px; color: rgb(146, 64, 14);',
    )
    svg_text = svg_text.replace(
        'font-size: 24px; color: rgb(17, 17, 17);',
        'font-size: 24px; color: rgb(146, 64, 14);',
    )
    svg_text = svg_text.replace(
        '<svg xmlns="http://www.w3.org/2000/svg" style="background: #FFFFFF; background-color: #FFFFFF; "',
        '<svg xmlns="http://www.w3.org/2000/svg" style="background: #FFFFFF;"',
    )
    return svg_text


def render_png(svg_path: Path, png_path: Path) -> None:
    cmd = [
        str(EDGE),
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-color-profile=srgb",
        "--force-device-scale-factor=2",
        "--window-size=2400,1000",
        f"--screenshot={png_path}",
        svg_path.as_uri(),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    text = freeze_light_dark(text)
    OUT_SVG.write_text(text, encoding="utf-8")
    render_png(OUT_SVG, OUT_PNG)
    print(f"saved {OUT_SVG}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
