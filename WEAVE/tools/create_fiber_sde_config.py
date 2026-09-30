import json

# Create complete Fiber-SDE config by copying topogate config and modifying solver
# Baseline: topogate e2 (0.671/0.314)
# Goal: break ODE mean collapse with fiber-aligned noise

base_config = {
    "data": {
        "data_root": "/mnt/i/wikiarts_5_full_notest_latents_ema/train",
        "batch_size": 12,
        "num_workers": 2,
        "shuffle": True,
        "drop_last": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "style_subdirs": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    },
    "training": {
        "num_epochs": 1,  # Run 1 epoch to trigger full_eval
        "full_eval_each_epoch": True,
        "full_eval_only_lpips_clip_style": True,
        "resume_model_strict": False,
        "resume_training_state": False,
        "learning_rate": 0.0  # Zero LR = eval-only, no training
    },
    "model": {
        # Copy all model settings from topogate (essential params only)
        "latent_channels": 4,
        "num_styles": 5,
        "style_dim": 160,
        "tokenizer_family": "pure_latent_spatial",
        "backbone_attention_family": "legacy_semantic_crossattn",
        "tokenizer_num_clusters": 32,
        "tokenizer_query_dim": 96,
        "tokenizer_query_num_blocks": 5,
        "tokenizer_pe_temperature": 0.75,
        "tokenizer_global_gate_scale": 1.15,
        "semantic_self_topology_gate": True,
        "semantic_self_topology_blend": 1.0,
        "transport_prediction_mode": "velocity",
        
        # Fiber-SDE overrides
        "solver_family": "solver_unsb_cycle",
        "solver_corrector_steps": 2,
        "solver_corrector_step_size": 0.06,
        "solver_corrector_mode": "latent_lowpass",
        "solver_corrector_lowpass_kernel": 5,
        "solver_stochastic_noise_scale": 0.02,
        "solver_fiber_aligned": True
    }
}

with open("G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_fiber_sde_complete_0p02.json", "w") as f:
    json.dump(base_config, f, indent=2)

print("Created complete Fiber-SDE config")
print("Expected: style pushes from 0.671 to 0.69-0.72, LPIPS stays <0.35")
