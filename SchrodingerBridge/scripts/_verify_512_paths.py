"""Verify the canonical 512 training/test paths on I drive."""
import os
from pathlib import Path

PATHS = [
    "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
    "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache",
    "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
    "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt",
    "/mnt/i/wikiart_distinct5_samam_512_classview/test",
    "/mnt/i/wikiart_distinct5_latents_512_ema",
    "/mnt/i/wikiart_distinct5_latents_512_ema/.latent_cache",
]


def main():
    for p in PATHS:
        path = Path(p)
        if path.exists():
            if path.is_dir():
                try:
                    entries = sorted(os.listdir(path))[:15]
                    print(f"[OK DIR ] {p}")
                    for e in entries:
                        print(f"           - {e}")
                except OSError as ex:
                    print(f"[ERR    ] {p}: {ex}")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"[OK FILE] {p}  ({size_mb:.2f} MB)")
        else:
            print(f"[MISSING] {p}")
    # Count files per style in samam_512_latents_ema/train
    train = Path("/mnt/i/wikiart_distinct5_samam_512_latents_ema/train")
    if train.exists():
        print(f"\n=== Style file counts in {train} ===")
        for style in sorted(os.listdir(train)):
            sp = train / style
            if sp.is_dir() and not style.startswith("."):
                n = sum(1 for _ in os.scandir(sp))
                print(f"  {style:30s}  {n:>6d} entries")
    # Count test files
    test = Path("/mnt/i/wikiart_distinct5_samam_512_classview/test")
    if test.exists():
        print(f"\n=== Test set structure {test} ===")
        for style in sorted(os.listdir(test)):
            sp = test / style
            if sp.is_dir():
                n = sum(1 for _ in os.scandir(sp))
                print(f"  {style:30s}  {n:>6d} entries")


if __name__ == "__main__":
    main()
