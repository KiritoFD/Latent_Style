from pathlib import Path

p = Path(r"I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed")
print("exists", p.exists(), p)
if p.exists():
    for item in sorted(p.iterdir())[:50]:
        print(item.name)
