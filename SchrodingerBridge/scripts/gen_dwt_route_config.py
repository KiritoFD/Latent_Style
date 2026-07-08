"""Generate DWT Route training config from clean_base.json."""
import json

with open('G:/GitHub/Latent_Style/SchrodingerBridge/configs/clean_base.json', 'r') as f:
    cfg = json.load(f)

# --- Model changes: enable DWT route ---
cfg['model']['cross_attn_dwt_route'] = True
cfg['model']['dwt_route_train_prob'] = 0.8

# --- Training changes: local paths, no resume ---
cfg['training']['resume_checkpoint'] = ''
cfg['training']['resume_optimizer'] = False
cfg['training']['resume_model_strict'] = False
cfg['training']['resume_training_state'] = False
cfg['training']['num_epochs'] = 10
cfg['training']['batch_size'] = 4
cfg['training']['full_eval_defer_until_training_end'] = True
cfg['training']['full_eval_each_epoch'] = False
cfg['training']['full_eval_force_regen'] = True
cfg['training']['test_image_dir'] = 'F:/wikiart_distinct5_samam_512_classview/test_new3'
cfg['training']['full_eval_cache_dir'] = 'G:/GitHub/Latent_Style/SchrodingerBridge/eval_cache'
cfg['training']['full_eval_clip_hf_cache_dir'] = 'C:/Users/xy/.cache/huggingface/hub'
cfg['training']['full_eval_batch_size'] = 2
cfg['training']['num_workers'] = 0
cfg['training']['pin_memory'] = False
cfg['training']['persistent_workers'] = False

# --- Data changes: local F paths ---
cfg['data']['data_root'] = 'F:/wikiart_distinct5_samam_512_latents_ema/train'
cfg['data']['pairing_cache_path'] = 'F:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt'
cfg['data']['latent_cache_dir'] = 'F:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed'

# --- Checkpoint ---
cfg['checkpoint']['save_dir'] = './exp/dwt_route_distinct5'

# --- Full eval ---
cfg['full_eval']['batch_size'] = 2
cfg['full_eval']['ref_feature_batch_size'] = 2

out_path = 'G:/GitHub/Latent_Style/SchrodingerBridge/configs/dwt_route_distinct5.json'
with open(out_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('Written to', out_path)
print('cross_attn_dwt_route =', cfg['model']['cross_attn_dwt_route'])
print('dwt_route_train_prob =', cfg['model']['dwt_route_train_prob'])
