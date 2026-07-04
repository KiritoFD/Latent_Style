"""Generate LOCAL T2 Soft LL Route configs for 8GB GPU.
batch_size=2, all paths on G: drive.
"""
import json
import copy
import os

CONFIG_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\configs"

LOCAL_DATA_ROOT = "G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train"
LOCAL_LATENT_CACHE = "G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train/.latent_cache/packed"
LOCAL_TEST_DIR = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
LOCAL_EXP_ROOT = "G:/GitHub/Latent_Style/SchrodingerBridge/exp"
LOCAL_EVAL_CACHE = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
LOCAL_EVAL_HF = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

FIVE_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IDT_CLIP_BASELINE = 0.639920825263


def main():
    base_path = os.path.join(CONFIG_DIR, "630_remote_base_5style.json")
    with open(base_path, 'r', encoding='utf-8') as f:
        base_cfg = json.load(f)

    for alpha in [0.05, 0.10, 0.15]:
        name = f"t2a" if alpha == 0.05 else (f"t2b" if alpha == 0.10 else f"t2c")
        full_name = f"630_local_t2_soft_ll_{name}"

        cfg = copy.deepcopy(base_cfg)

        # T2 params: DWT route + cosine + heun + soft LL route
        cfg['model']['cross_attn_dwt_route'] = True
        cfg['model']['cross_attn_dwt_ll_route_alpha'] = alpha
        cfg['model']['solver_type'] = 'heun'
        cfg['model']['time_schedule'] = 'cosine'
        cfg['model']['endpoint_adain_mode'] = 'per_subband_wct'
        cfg['model']['endpoint_adain_scale_ll'] = 0.0
        cfg['model']['endpoint_adain_scale_lh'] = 0.3
        cfg['model']['endpoint_adain_scale_hl'] = 0.3
        cfg['model']['endpoint_adain_scale_hh'] = 0.5
        cfg['model']['style_extrap_alpha'] = 0.4

        cfg['bridge']['spectral_w_endpoint_style_lh'] = 0.0
        cfg['bridge']['spectral_w_endpoint_style_hl'] = 0.0

        # Fix learning rate: few-shot template uses 0.005, full training needs 0.0002
        cfg['training']['learning_rate'] = 0.0002

        # LOCAL paths (G: drive)
        cfg['data']['data_root'] = LOCAL_DATA_ROOT
        cfg['data']['latent_cache_dir'] = LOCAL_LATENT_CACHE
        cfg['data']['style_subdirs'] = FIVE_STYLES
        cfg['data']['dino_cache_path'] = ""
        cfg['data']['dino_cache_required'] = False

        # Training: from scratch, small batch for 8GB GPU
        cfg['training']['freeze_mode'] = "none"
        cfg['training']['few_shot_new_style_idx'] = None
        cfg['training']['few_shot_base_checkpoint'] = ""
        cfg['training']['resume_checkpoint'] = ""
        cfg['training']['resume_optimizer'] = False
        cfg['training']['resume_training_state'] = False
        cfg['training']['num_epochs'] = 5
        cfg['training']['patience'] = 2
        cfg['training']['batch_size'] = 2  # 8GB GPU constraint
        cfg['training']['full_eval_each_epoch'] = True
        cfg['training']['full_eval_batch_size'] = 2
        cfg['training']['full_eval_clip_style_idt_baseline'] = IDT_CLIP_BASELINE
        cfg['training']['test_image_dir'] = LOCAL_TEST_DIR
        cfg['training']['full_eval_cache_dir'] = LOCAL_EVAL_CACHE
        cfg['training']['full_eval_clip_hf_cache_dir'] = LOCAL_EVAL_HF

        cfg['checkpoint']['save_dir'] = f"{LOCAL_EXP_ROOT}/{full_name}"
        cfg['checkpoint']['resume_checkpoint'] = ""

        cfg['full_eval']['batch_size'] = 2
        cfg['full_eval']['ref_feature_batch_size'] = 2

        cfg['ablation'] = {
            'name': full_name,
            'axis': 'theory_soft_ll_route_local',
            'stage': full_name,
            'notes': f'T2 LOCAL: Soft LL Route alpha={alpha}. Base=4J.1 DWT route (all_pairs_lpips=0.3068). '
                     f'batch_size=2 for 8GB GPU. '
                     f'Target: all_pairs_clip>0.7319 AND all_pairs_lpips<0.3068'
        }

        out_path = os.path.join(CONFIG_DIR, f"{full_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"Written: {out_path}  alpha={alpha}")

    print(f"\n3 LOCAL T2 configs generated (batch_size=2 for 8GB GPU)")


if __name__ == "__main__":
    main()
