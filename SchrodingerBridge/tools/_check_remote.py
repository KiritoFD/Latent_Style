import os, importlib
def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"MISSING({e})"
for mod in ["cv2", "transformers", "huggingface_hub", "accelerate", "diffusers", "torch", "einops"]:
    print("PKG", mod, ver(mod))
for name, d in [
    ("P2A", "I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/photo2art256/images"),
    ("R5",  "I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/random5/images"),
]:
    print("COUNT", name, len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else "NA")
