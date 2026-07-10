"""Verify D4 FiLM config loads correctly and style_film_heads is read."""
import sys
import os
ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config_schema import load_experiment_config

c = load_experiment_config("configs/d4_film_hf1_15ep.json")
print("=== Model config ===")
print("style_film_heads (getattr):", getattr(c.model, "style_film_heads", "NOT_FOUND"))
print("style_film_enabled:", getattr(c.model, "style_film_enabled", "NOT_FOUND"))
print("extra keys:", list(c.model.extra.keys()) if hasattr(c.model, "extra") else "no extra attr")
print("base_dim:", c.model.base_dim)
print("num_styles:", c.model.num_styles)

# Check if style_film_heads is in extra
if hasattr(c.model, "extra"):
    print("extra['style_film_heads']:", c.model.extra.get("style_film_heads", "NOT_IN_EXTRA"))

# Now test model construction
from spectral_bridge620 import SpectralODEBridge620
model = SpectralODEBridge620(c.model, c.bridge)
print("\n=== Model ===")
print("style_film_heads:", model.style_film_heads)
print("film_dim:", getattr(model, "film_dim", "N/A"))
# Check head_ll
head = model.head_ll
print("head_ll.film:", head.film)
print("head_ll.style_dim:", head.style_dim)
total_params = sum(p.numel() for p in model.parameters())
print("total_params:", total_params)
