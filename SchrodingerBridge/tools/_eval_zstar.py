"""Z-STAR baseline evaluation: CLIP-S, LPIPS, MUSIQ on D5-512, P2A-256, R5-512.
Reuses the same evaluation protocol as StyleAligned.
"""
import sys
sys.path.insert(0, ".")

from _eval_stylealigned import (
    collect_pairs_d5, collect_pairs_p2a, eval_dataset,
)

D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
R5_STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]

if __name__ == "__main__":
    print("=" * 60)
    print("Z-STAR Baseline Evaluation")
    print("=" * 60)

    # ---- P2A-256 ----
    p2a_pairs = collect_pairs_p2a(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\P256\zstar",
        r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test",
        P2A_STYLES,
    )
    print(f"\n[P2A-256] {len(p2a_pairs)} pairs collected")
    cs1, lp1, mq1 = eval_dataset(
        p2a_pairs,
        r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test",
        P2A_STYLES,
        "P2A-256",
    )

    # ---- D5-512 ----
    d5_pairs = collect_pairs_d5(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\zstar",
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train",
        D5_STYLES,
    )
    print(f"\n[D5-512] {len(d5_pairs)} pairs collected")
    cs2, lp2, mq2 = eval_dataset(
        d5_pairs,
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
        D5_STYLES,
        "D5-512",
    )

    # ---- R5-512 ----
    r5_base = r"G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images"
    r5_pairs = collect_pairs_d5(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\zstar",
        rf"{r5_base}\test",
        rf"{r5_base}\train",
        R5_STYLES,
    )
    print(f"\n[R5-512] {len(r5_pairs)} pairs collected")
    cs3, lp3, mq3 = eval_dataset(
        r5_pairs,
        rf"{r5_base}\test",
        R5_STYLES,
        "R5-512",
    )

    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    if mq1 is not None:
        print(f"P2A-256 Z-STAR:  CLIP-S={cs1:.4f}  LPIPS={lp1:.4f}  MUSIQ={mq1:.4f}")
    else:
        print(f"P2A-256 Z-STAR:  CLIP-S={cs1:.4f}  LPIPS={lp1:.4f}")
    if mq2 is not None:
        print(f"D5-512  Z-STAR:  CLIP-S={cs2:.4f}  LPIPS={lp2:.4f}  MUSIQ={mq2:.4f}")
    else:
        print(f"D5-512  Z-STAR:  CLIP-S={cs2:.4f}  LPIPS={lp2:.4f}")
    if mq3 is not None:
        print(f"R5-512  Z-STAR:  CLIP-S={cs3:.4f}  LPIPS={lp3:.4f}  MUSIQ={mq3:.4f}")
    else:
        print(f"R5-512  Z-STAR:  CLIP-S={cs3:.4f}  LPIPS={lp3:.4f}")
    print("=" * 60)
