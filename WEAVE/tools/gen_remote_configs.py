"""
Generate 630 remote tuning experiment configs.
Base: 4J.6 v3 config (DWT route + cosine+heun + per_subband_wct)
Modified for remote 5-style dataset (I: drive paths, no few-shot).

5 Experiment Directions:
  A1 (Arch):     DWT route + stronger style injection (alpha=0.6, hh=0.7)
  A2 (Arch):     Cosine+Heun+DWT triple combo (alpha=0.5, balanced)
  P1 (Param):    Spectral loss rebalancing (hh=3.0, w_endpoint_style=12.0)
  T1 (Theory):   Low-freq style injection (ll=0.15, hh=0.6) - color/contrast style
  P2 (Param):    SWD + flow weight balancing (terminal_swd=0.3, w_flow=0.8)
"""
import json
import copy
import os

CONFIG_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\configs"
TEMPLATE = os.path.join(CONFIG_DIR, "630_phase4j6_fewshot_popart_v3.json")

# Remote paths (I: drive)
REMOTE_DATA_ROOT = "I:/wikiart_distinct5_samam_512_latents_ema/train"
REMOTE_LATENT_CACHE = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
REMOTE_TEST_DIR = "I:/wikiart_distinct5_samam_512_classview/test"
REMOTE_EXP_ROOT = "I:/Github/Latent_Style/SchrodingerBridge/exp"
REMOTE_EVAL_CACHE = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
REMOTE_EVAL_HF = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

FIVE_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e"
]

# SaMam identity baseline for 5-style (from baseline_v2 eval)
# Using approximate value - will be updated from actual eval if needed
IDT_CLIP_BASELINE = 0.639920825263


def create_base_config():
    """Create base config for remote 5-style training from 4J.6 v3 template."""
    with open(TEMPLATE, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # --- Model section: 5 styles, disable endpoint style loss (proven ineffective) ---
    cfg['model']['num_styles'] = 5
    # Keep DWT route + cosine + heun + per_subband_wct from 4J.1/4I.7a
    # Disable endpoint style loss (v3 proved it ineffective)
    # (spectral_w_endpoint_style_lh/hl are in bridge section, handle below)

    # --- Bridge section: disable endpoint style loss ---
    cfg['bridge']['spectral_w_endpoint_style_lh'] = 0.0
    cfg['bridge']['spectral_w_endpoint_style_hl'] = 0.0

    # --- Training section: from-scratch full training ---
    cfg['training']['freeze_mode'] = "none"
    cfg['training']['few_shot_new_style_idx'] = None
    cfg['training']['few_shot_base_checkpoint'] = ""
    cfg['training']['resume_checkpoint'] = ""
    cfg['training']['resume_optimizer'] = False
    cfg['training']['resume_training_state'] = False
    cfg['training']['num_epochs'] = 5
    cfg['training']['patience'] = 2
    cfg['training']['batch_size'] = 16
    cfg['training']['full_eval_each_epoch'] = True
    cfg['training']['full_eval_batch_size'] = 2
    cfg['training']['full_eval_clip_style_idt_baseline'] = IDT_CLIP_BASELINE
    cfg['training']['test_image_dir'] = REMOTE_TEST_DIR
    cfg['training']['full_eval_cache_dir'] = REMOTE_EVAL_CACHE
    cfg['training']['full_eval_clip_hf_cache_dir'] = REMOTE_EVAL_HF

    # --- Data section: 5-style remote paths ---
    cfg['data']['data_root'] = REMOTE_DATA_ROOT
    cfg['data']['latent_cache_dir'] = REMOTE_LATENT_CACHE
    cfg['data']['style_subdirs'] = FIVE_STYLES
    cfg['data']['dino_cache_path'] = ""
    cfg['data']['dino_cache_required'] = False

    # --- Checkpoint section ---
    cfg['checkpoint']['resume_checkpoint'] = ""

    # --- Full eval section ---
    cfg['full_eval']['batch_size'] = 2
    cfg['full_eval']['ref_feature_batch_size'] = 2
    cfg['full_eval']['target_chunk_size'] = 1

    # --- Remove standalone-only fields ---
    cfg.pop('name', None)
    cfg.pop('save_dir', None)
    cfg.pop('notes', None)

    return cfg


def make_experiment(base_cfg, name, model_overrides=None, bridge_overrides=None,
                    training_overrides=None, ablation_info=None):
    """Create an experiment config inheriting from base with overrides."""
    exp = {
        "_base": "630_remote_base_5style.json",
    }

    if model_overrides:
        exp['model'] = copy.deepcopy(model_overrides)
    if bridge_overrides:
        exp['bridge'] = copy.deepcopy(bridge_overrides)
    if training_overrides:
        exp['training'] = copy.deepcopy(training_overrides)

    exp['checkpoint'] = {
        'save_dir': f"{REMOTE_EXP_ROOT}/630_remote_{name}",
        'resume_checkpoint': ""
    }

    exp['ablation'] = ablation_info or {
        'name': f'remote_{name}',
        'axis': 'tuning',
        'stage': f'remote_{name}',
        'notes': f'Remote tuning experiment {name}'
    }

    return exp


def main():
    # Create base config
    base = create_base_config()
    base_path = os.path.join(CONFIG_DIR, "630_remote_base_5style.json")
    with open(base_path, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=2, ensure_ascii=False)
    print(f"Written: {base_path}")

    # --- Experiment A1: DWT + Stronger Style Injection ---
    # Theory: DWT route frees style_mem capacity → afford stronger style injection
    # Architecture: style_extrap_alpha 0.4→0.6, endpoint_adain_scale_hh 0.5→0.7
    a1 = make_experiment(
        base, "a1_dwt_strong_style",
        model_overrides={
            'style_extrap_alpha': 0.6,
            'endpoint_adain_scale_hh': 0.7,
            'endpoint_adain_scale': 0.6,
        },
        ablation_info={
            'name': 'remote_a1_dwt_strong_style',
            'axis': 'architecture_style_injection_strength',
            'stage': 'remote_a1',
            'notes': 'A1(Arch): DWT route + stronger style injection. Theory: DWT frees style_mem → afford alpha=0.6 + hh=0.7. Expected: clip↑, lpips slight↑ but <baseline.'
        }
    )
    a1_path = os.path.join(CONFIG_DIR, "630_remote_a1_dwt_strong_style.json")
    with open(a1_path, 'w', encoding='utf-8') as f:
        json.dump(a1, f, indent=2, ensure_ascii=False)
    print(f"Written: {a1_path}")

    # --- Experiment A2: Cosine+Heun+DWT Triple Combo (balanced alpha) ---
    # Theory: combine orthogonal improvements from 4I.7a (cosine+heun) and 4J.1 (DWT route)
    # alpha=0.5 as compromise between content (0.4) and style (0.9)
    a2 = make_experiment(
        base, "a2_cosine_heun_dwt_balanced",
        model_overrides={
            'style_extrap_alpha': 0.5,
            'solver_type': 'heun',
            'time_schedule': 'cosine',
            'cross_attn_dwt_route': True,
            'endpoint_adain_mode': 'per_subband_wct',
        },
        ablation_info={
            'name': 'remote_a2_cosine_heun_dwt_balanced',
            'axis': 'architecture_orthogonal_combo',
            'stage': 'remote_a2',
            'notes': 'A2(Arch): Cosine+Heun+DWT triple combo. Combines 4I.7a (best clip) + 4J.1 (best lpips). alpha=0.5 balanced. Expected: both metrics improve.'
        }
    )
    a2_path = os.path.join(CONFIG_DIR, "630_remote_a2_cosine_heun_dwt_balanced.json")
    with open(a2_path, 'w', encoding='utf-8') as f:
        json.dump(a2, f, indent=2, ensure_ascii=False)
    print(f"Written: {a2_path}")

    # --- Experiment P1: Spectral Loss Rebalancing ---
    # Theory: stronger high-freq supervision → better texture/style transfer
    # spectral_w_hh 2.0→3.0, w_endpoint_style 8.0→12.0
    p1 = make_experiment(
        base, "p1_spectral_rebalance",
        bridge_overrides={
            'spectral_w_hh': 3.0,
            'w_endpoint_style': 12.0,
            'spectral_w_lh': 1.5,
            'spectral_w_hl': 1.5,
        },
        ablation_info={
            'name': 'remote_p1_spectral_rebalance',
            'axis': 'parameter_spectral_weights',
            'stage': 'remote_p1',
            'notes': 'P1(Param): Spectral loss rebalancing. hh=3.0 (stronger high-freq texture), w_endpoint_style=12.0. Theory: stronger HF supervision → better style texture. Expected: clip↑ for textured styles.'
        }
    )
    p1_path = os.path.join(CONFIG_DIR, "630_remote_p1_spectral_rebalance.json")
    with open(p1_path, 'w', encoding='utf-8') as f:
        json.dump(p1, f, indent=2, ensure_ascii=False)
    print(f"Written: {p1_path}")

    # --- Experiment T1: Low-Freq Style Injection (Theory) ---
    # Theory: slight low-freq style for color/contrast, DWT route preserves content structure
    # endpoint_adain_scale_ll 0.0→0.15, hh 0.5→0.6
    t1 = make_experiment(
        base, "t1_lowfreq_style",
        model_overrides={
            'endpoint_adain_scale_ll': 0.15,
            'endpoint_adain_scale_hh': 0.6,
            'endpoint_adain_scale_lh': 0.4,
            'endpoint_adain_scale_hl': 0.4,
        },
        ablation_info={
            'name': 'remote_t1_lowfreq_style',
            'axis': 'theory_frequency_band_style_injection',
            'stage': 'remote_t1',
            'notes': 'T1(Theory): Low-freq style injection. ll=0.15 (color/contrast style), hh=0.6. Theory: DWT route preserves content → can afford slight low-freq style. Expected: color style improves, content preserved.'
        }
    )
    t1_path = os.path.join(CONFIG_DIR, "630_remote_t1_lowfreq_style.json")
    with open(t1_path, 'w', encoding='utf-8') as f:
        json.dump(t1, f, indent=2, ensure_ascii=False)
    print(f"Written: {t1_path}")

    # --- Experiment P2: SWD + Flow Weight Balancing ---
    # Theory: stronger style consistency loss, slightly relaxed flow matching
    # terminal_swd_weight 0.1→0.3, w_flow 1.0→0.8, single_step_swd_weight 8.0→10.0
    p2 = make_experiment(
        base, "p2_swd_flow_balance",
        bridge_overrides={
            'terminal_swd_weight': 0.3,
            'w_flow': 0.8,
            'single_step_swd_weight': 10.0,
            'w_endpoint_style': 10.0,
        },
        ablation_info={
            'name': 'remote_p2_swd_flow_balance',
            'axis': 'parameter_swd_flow_balance',
            'stage': 'remote_p2',
            'notes': 'P2(Param): SWD+Flow weight balancing. terminal_swd=0.3 (stronger style consistency), w_flow=0.8 (slightly relaxed FM). Theory: stronger style distribution matching. Expected: style distribution improves.'
        }
    )
    p2_path = os.path.join(CONFIG_DIR, "630_remote_p2_swd_flow_balance.json")
    with open(p2_path, 'w', encoding='utf-8') as f:
        json.dump(p2, f, indent=2, ensure_ascii=False)
    print(f"Written: {p2_path}")

    print(f"\nAll 6 configs generated (1 base + 5 experiments)")
    print(f"Target: transfer_clip > 0.6914 AND transfer_lpips < 0.3387 (double surpass SaMam)")


if __name__ == "__main__":
    main()
