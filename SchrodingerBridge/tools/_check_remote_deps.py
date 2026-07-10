import os, json
from pathlib import Path

OUT = {}
def sp(p): return str(p).replace("\\", "/")

# style_aligned handler (for StyleAligned)
cands = [
    r"I:/GitHub/Latent_Style/SchrodingerBridge/tools/style_aligned",
    r"I:/GitHub/Latent_Style/SchrodingerBridge/tools/style_aligned_sd15",
]
OUT["style_aligned"] = {sp(c): Path(c).exists() for c in cands}

# existing remote baseline scripts
OUT["remote_scripts"] = {}
for s in ["_run_stylealigned_remote.py", "_run_sdturbo_remote.py", "_run_zstar_remote.py",
          "_run_styleshot_remote.py", "_run_styleid_remote.py"]:
    p = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/tools") / s
    if not p.exists():
        p = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/_tmp/tools") / s
    OUT["remote_scripts"][s] = sp(p) if p.exists() else "MISSING"

# CLIP-ViT-H cache
OUT["clip_vit_h"] = []
for base in [r"C:/Users/Administrator/.cache/huggingface/hub",
             r"I:/modelscope_cache/laion",
             r"I:/GitHub/Latent_Style"]:
    bp = Path(base)
    if not bp.exists():
        continue
    for dp, dn, fn in os.walk(base):
        if "CLIP-ViT-H-14-laion2B-s32B-b79K" in dp:
            OUT["clip_vit_h"].append(sp(dp))
            dn[:] = []
        if len(OUT["clip_vit_h"]) >= 5:
            break

# style_aligned inversion module
OUT["inversion_sd15"] = any(Path(c).exists() for c in cands)

# StyleShot repo subdirs
ss = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/tools/styleshot")
OUT["styleshot_subdirs"] = sorted([d.name for d in ss.iterdir() if d.is_dir()]) if ss.exists() else "MISSING"

# xformers / diffusers availability (quick import)
OUT["imports"] = {}
for mod in ["diffusers", "xformers", "torch", "transformers", "cv2"]:
    try:
        m = __import__(mod)
        OUT["imports"][mod] = getattr(m, "__version__", "ok")
    except Exception as e:
        OUT["imports"][mod] = f"ERR {e}"

print(json.dumps(OUT, indent=2, ensure_ascii=False))
