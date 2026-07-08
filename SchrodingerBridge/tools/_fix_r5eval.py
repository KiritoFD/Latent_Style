"""Fix Random5 eval: copy images and re-run evaluation."""
from pathlib import Path
import sys, shutil, json, subprocess

img_dir = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\images")
eval_dir = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\eval")
eval_img = eval_dir / "images"
eval_img.mkdir(parents=True, exist_ok=True)

# Copy images
n = 0
for f in img_dir.iterdir():
    if f.suffix.lower() in (".png", ".jpg", ".jpeg") and not f.name.startswith("_"):
        dst = eval_img / f.name
        if not dst.exists():
            shutil.copy2(str(f), str(dst))
        n += 1
print(f"Copied {n} images to eval/images/")

# Run eval
EVAL_SCRIPT = r"I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"
TEST_DIR = r"I:\datasets\wikiarts20_512_test"
STYLES = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

cmd = [
    sys.executable, EVAL_SCRIPT,
    str(eval_dir),
    "--reuse_generated",
    "--save_generated_images",
    "--style_subdirs", STYLES,
    "--test_dir", TEST_DIR,
    "--eval_only_lpips_clip_style",
    "--clip_style_idt_baseline", "0.7312",
]
print(f"Running: {cmd[:4]} ...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
print(result.stdout[-800:])
if result.returncode != 0:
    print(f"ERROR: {result.stderr[-500:]}")
else:
    summary = eval_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        cs = s.get("clip_style", "N/A")
        lp = s.get("content_lpips", "N/A")
        print(f"Random5: CLIP-S={cs:.4f} LPIPS={lp:.4f}" if isinstance(cs, float) else f"Random5: {s}")

# Also show P2A results
p2a_summary = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\eval\summary.json")
if p2a_summary.exists():
    s = json.loads(p2a_summary.read_text())
    cs = s.get("clip_style", "N/A")
    lp = s.get("content_lpips", "N/A")
    print(f"Photo2Art: CLIP-S={cs:.4f} LPIPS={lp:.4f}" if isinstance(cs, float) else f"Photo2Art: CLIP-S={cs}, LPIPS={lp}")
