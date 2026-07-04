"""Generate T2 Soft LL Route experiment configs.
Base: 4J.1 DWT route (all_pairs_lpips=0.3068, best content preservation)
New: cross_attn_dwt_ll_route_alpha > 0 (LL 残差注入 style)

3 alpha values to sweep:
  T2a: alpha=0.05 (极少量 LL style, 最保守)
  T2b: alpha=0.10 (少量 LL style, 平衡)
  T2c: alpha=0.15 (中等 LL style, 更激进)
"""
import json
import copy
import os

CONFIG_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\configs"

# Remote paths (I: drive)
REMOTE_DATA_ROOT = "I:/wikiart_distinct5_samam_512_latents_ema/train"
REMOTE_LATENT_CACHE = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
REMOTE_TEST_DIR = "I:/wikiart_distinct5_samam_512_classview/test"
REMOTE_EXP_ROOT = "I:/Github/Latent_Style/SchrodingerBridge/exp"
REMOTE_EVAL_CACHE = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
REMOTE_EVAL_HF = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

FIVE_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IDT_CLIP_BASELINE = 0.639920825263


def main():
    # Load remote base 5style config (complete, no _base inheritance)
    base_path = os.path.join(CONFIG_DIR, "630_remote_base_5style.json")
    with open(base_path, 'r', encoding='utf-8') as f:
        base_cfg = json.load(f)

    print(f"Base config keys: {list(base_cfg.keys())}")
    model = base_cfg.get('model', {})
    print(f"model.cross_attn_dwt_route = {model.get('cross_attn_dwt_route', 'NOT SET')}")

    # Generate 3 T2 configs
    for alpha in [0.05, 0.10, 0.15]:
        name = f"t2a" if alpha == 0.05 else (f"t2b" if alpha == 0.10 else f"t2c")
        full_name = f"630_remote_t2_soft_ll_{name}"

        cfg = copy.deepcopy(base_cfg)

        # 4J.1 params: DWT route + cosine + heun + per_subband_wct
        cfg['model']['cross_attn_dwt_route'] = True
        cfg['model']['cross_attn_dwt_ll_route_alpha'] = alpha
        cfg['model']['solver_type'] = 'heun'
        cfg['model']['time_schedule'] = 'cosine'
        cfg['model']['endpoint_adain_mode'] = 'per_subband_wct'
        cfg['model']['endpoint_adain_scale_ll'] = 0.0  # keep LL bypass on endpoint side
        cfg['model']['endpoint_adain_scale_lh'] = 0.3
        cfg['model']['endpoint_adain_scale_hl'] = 0.3
        cfg['model']['endpoint_adain_scale_hh'] = 0.5
        cfg['model']['style_extrap_alpha'] = 0.4

        # Disable endpoint style loss (proven ineffective in 4J.6 v3)
        cfg['bridge']['spectral_w_endpoint_style_lh'] = 0.0
        cfg['bridge']['spectral_w_endpoint_style_hl'] = 0.0

        # Update save_dir
        cfg['checkpoint']['save_dir'] = f"{REMOTE_EXP_ROOT}/{full_name}"
        cfg['checkpoint']['resume_checkpoint'] = ""

        # Ablation info
        cfg['ablation'] = {
            'name': full_name,
            'axis': 'theory_soft_ll_route',
            'stage': full_name,
            'notes': f'T2(Theory): Soft LL Route alpha={alpha}. Base=4J.1 DWT route (lpips=0.3068). '
                     f'LL 以 alpha={alpha} 残差注入 style. 理论: 少量 LL style 提供色彩风格, 保结构. '
                     f'Target: all_pairs_clip>0.7319 AND all_pairs_lpips<0.3068'
        }

        out_path = os.path.join(CONFIG_DIR, f"{full_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"Written: {out_path}  alpha={alpha}")

    print(f"\n3 T2 configs generated")
    print(f"Target: all_pairs_clip > 0.7319 AND all_pairs_lpips < 0.3068")


if __name__ == "__main__":
    main()
