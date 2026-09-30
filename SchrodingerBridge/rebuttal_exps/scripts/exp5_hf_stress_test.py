"""Exp5: Extreme High-Frequency Style Stress Test (Qualitative Visualization).

Goal: Show WEAVE's behavior on styles with strong high-frequency phase structure
(cross-hatching-like, wave patterns, detailed textures). This is a HONEST limitation
demonstration - no extra training needed, just picks specific generated images from
the existing 750-image production run and assembles side-by-side comparison panels.

Stress-test styles (picked for maximal high-frequency phase content):
  1. Hokusai "cargo-ship-and-wave" - 强波浪纹理 (phase-coherent curves)
  2. Kuniyoshi "tamatori-being-pursued-bya-dragon" - 强细节纹理
  3. Monet "rouen-cathedral-the-portal-at-midday" - 强光影笔触

For each stress style, we pick 2 content images from different families and assemble:
  [content原图 | style参考图 | WEAVE生成图]
"""
import json, os, sys, shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required. pip install Pillow")
    sys.exit(1)

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

PROD_IMAGES = WEAVE_ROOT / "exp" / "repro_weave_d5" / "images"
TEST_DIR = WEAVE_ROOT / "data" / "test"
OUTPUT_DIR = WEAVE_ROOT / "exp" / "rebuttal" / "exp5_hf_stress"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stress-test styles: (style_family, style_filename_substring, short_label)
# Picked for maximal high-frequency phase structure
STRESS_STYLES = [
    ("Ukiyo_e", "katsushika-hokusai_cargo-ship-and-wave", "hokusai_wave"),
    ("Ukiyo_e", "utagawa-kuniyoshi_tamatori-being-pursued-bya-dragon", "kuniyoshi_dragon"),
    ("Impressionism", "claude-monet_rouen-cathedral-the-portal-at-midday", "monet_rouen"),
]

# Content images: pick 2 from different families for each stress style
# (content_family, content_filename_substring, short_label)
CONTENT_IMAGES = [
    ("Early_Renaissance", "leonardo-da-vinci_study-of-the-effect-of-light-on-a-profile-head-facsimile", "leo_study"),
    ("Rococo", "thomas-gainsborough_a-coastal-landscape-1782", "gainsborough_coast"),
]


def find_style_image(family, sub):
    """Find the style reference image in data/test/<family>/."""
    family_dir = TEST_DIR / family
    if not family_dir.exists():
        return None
    for f in family_dir.iterdir():
        if sub.lower() in f.name.lower():
            return f
    return None


def find_generated_image(content_family, content_sub, target_family):
    """Find generated image: <cf>_<cf>__<content_sub>_to_<target_family>.png"""
    if not PROD_IMAGES.exists():
        return None
    prefix = f"{content_family}_{content_family}__"
    for f in PROD_IMAGES.iterdir():
        name = f.name
        if not name.startswith(prefix):
            continue
        if not name.endswith(f"_to_{target_family}.png"):
            continue
        if content_sub.lower() in name.lower():
            return f
    return None


def assemble_panel(content_img, style_img, generated_img, out_path, labels=None):
    """Assemble side-by-side: [content | style | generated]."""
    c = Image.open(content_img).convert("RGB")
    s = Image.open(style_img).convert("RGB")
    g = Image.open(generated_img).convert("RGB")
    # Resize to common height (256)
    H = 256
    c = c.resize((int(c.width * H / c.height), H), Image.LANCZOS)
    s = s.resize((int(s.width * H / s.height), H), Image.LANCZOS)
    g = g.resize((int(g.width * H / g.height), H), Image.LANCZOS)
    total_w = c.width + s.width + g.width + 20
    panel = Image.new("RGB", (total_w, H + 30), (255, 255, 255))
    panel.paste(c, (0, 30))
    panel.paste(s, (c.width + 10, 30))
    panel.paste(g, (c.width + s.width + 20, 30))
    out_path.write_bytes(b"")  # touch
    panel.save(out_path)
    return panel.size


def main():
    print("=" * 60)
    print("Exp5: Extreme High-Frequency Style Stress Test")
    print("=" * 60)

    results = []
    for style_family, style_sub, style_label in STRESS_STYLES:
        print(f"\n--- Stress style: {style_label} ({style_family}) ---")
        style_img = find_style_image(style_family, style_sub)
        if style_img is None:
            print(f"  ERROR: style image not found for {style_sub}")
            results.append({"style": style_label, "status": "style_not_found"})
            continue
        print(f"  Style image: {style_img.name}")

        for content_family, content_sub, content_label in CONTENT_IMAGES:
            print(f"  Content: {content_label} ({content_family})")
            # Find content image
            content_img = find_style_image(content_family, content_sub)
            if content_img is None:
                print(f"    ERROR: content image not found for {content_sub}")
                continue
            # Find generated image
            gen_img = find_generated_image(content_family, content_sub, style_family)
            if gen_img is None:
                print(f"    ERROR: generated image not found")
                continue
            print(f"    Content: {content_img.name}")
            print(f"    Generated: {gen_img.name}")

            # Assemble panel
            panel_name = f"panel_{style_label}__{content_label}.png"
            panel_path = OUTPUT_DIR / panel_name
            try:
                w, h = assemble_panel(content_img, style_img, gen_img, panel_path)
                print(f"    Panel saved: {panel_name} ({w}x{h})")
                results.append({
                    "style": style_label,
                    "style_family": style_family,
                    "content": content_label,
                    "content_family": content_family,
                    "panel": panel_name,
                    "status": "ok",
                })
            except Exception as e:
                print(f"    ERROR assembling panel: {e}")
                results.append({
                    "style": style_label,
                    "content": content_label,
                    "status": "error",
                    "error": str(e),
                })

    # Save manifest
    manifest = OUTPUT_DIR / "_results.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nManifest saved: {manifest}")
    print(f"Total panels: {sum(1 for r in results if r.get('status') == 'ok')}")
    print("\nNOTE: These panels demonstrate WEAVE's behavior on extreme high-frequency")
    print("phase-structured styles. Honest limitation: HH subband uses endpoint AdaIN")
    print("only, which may cause slight phase misalignment on cross-hatching-like patterns.")


if __name__ == "__main__":
    main()
