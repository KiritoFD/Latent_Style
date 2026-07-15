import json
import copy

# Load the full expanded config from the 20-style baseline experiment
with open(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep\config.json", "r", encoding="utf-8") as f:
    base_cfg = json.load(f)

# D5 dataset paths (local)
D5_DATA_ROOT = "G:/GitHub/Latent_Style/Dataset/wikiart_distinct5_samam_512_latents_ema/train"
D5_CACHE_DIR = "G:/GitHub/Latent_Style/Dataset/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
D5_TEST_DIR = "G:/GitHub/Latent_Style/Dataset/wikiart_distinct5_samam_512_classview/test"

# D5 dataset paths (remote I:)
D5_DATA_ROOT_R = "I:/datasets/wikiart_distinct5_samam_512_latents_ema/train"
D5_CACHE_DIR_R = "I:/datasets/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
D5_TEST_DIR_R = "I:/datasets/wikiart_distinct5_samam_512_classview/test"


def make_d5_config(name, overrides_model=None, overrides_training=None, remote=False):
    cfg = copy.deepcopy(base_cfg)
    # Remove _base since we already have full config
    cfg.pop("_base", None)

    # num_styles: 20 -> 5
    cfg["model"]["num_styles"] = 5

    # Dataset paths
    if remote:
        cfg["data"] = {
            "data_root": D5_DATA_ROOT_R,
            "latent_cache_dir": D5_CACHE_DIR_R,
            "pairing_cache_path": D5_DATA_ROOT_R + "/.latent_cache/prototype_pairing_top8.pt",
            "pairing_cache_active_topk": 0,
            "dataset_index_path": "",
        }
        cfg.setdefault("training", {})["test_image_dir"] = D5_TEST_DIR_R
    else:
        cfg["data"] = {
            "data_root": D5_DATA_ROOT,
            "latent_cache_dir": D5_CACHE_DIR,
            "pairing_cache_path": D5_DATA_ROOT + "/.latent_cache/prototype_pairing_top8.pt",
            "pairing_cache_active_topk": 0,
            "dataset_index_path": "",
        }
        cfg.setdefault("training", {})["test_image_dir"] = D5_TEST_DIR

    # Apply overrides
    if overrides_model:
        cfg["model"].update(overrides_model)
    if overrides_training:
        cfg["training"].update(overrides_training)

    # Checkpoint dir
    cfg["checkpoint"] = {
        "save_dir": f"G:/GitHub/Latent_Style/SchrodingerBridge/exp/evo_d5_{name}",
        "resume_checkpoint": ""
    }

    # Ablation metadata
    cfg["ablation"] = {
        "name": f"evo_d5_{name}",
        "axis": "d5_baseline" if name == "baseline" else name,
        "stage": "evo_d5",
        "notes": f"D5 baseline from random20_heun config (spatial_fiber+heun). num_styles=5, samam dataset. Baseline ref: 20-style eval_r5 clip=0.7213 lpips=0.2728."
    }

    return cfg


# Configs to generate
configs = [
    ("baseline", None, None),
    ("adain10", {"endpoint_adain_scale": 1.0}, None),
    ("long10", None, {"num_epochs": 10, "patience": 3}),
    ("extrap02", {"style_extrap_alpha": 0.2}, None),
    ("combo", {"endpoint_adain_scale": 1.0, "style_extrap_alpha": 0.2}, {"num_epochs": 10, "patience": 3}),
]

for name, om, ot in configs:
    # Local version
    cfg_local = make_d5_config(name, om, ot, remote=False)
    path_local = f"G:/GitHub/Latent_Style/SchrodingerBridge/configs/evo_d5_{name}.json"
    with open(path_local, "w", encoding="utf-8") as f:
        json.dump(cfg_local, f, indent=2, ensure_ascii=False)
    print(f"Written: {path_local}")

    # Remote version
    cfg_remote = make_d5_config(name, om, ot, remote=True)
    cfg_remote["checkpoint"]["save_dir"] = f"I:/Github/Latent_Style/SchrodingerBridge/exp/evo_d5_{name}"
    path_remote = f"G:/GitHub/Latent_Style/SchrodingerBridge/configs/remote_evo_d5_{name}.json"
    with open(path_remote, "w", encoding="utf-8") as f:
        json.dump(cfg_remote, f, indent=2, ensure_ascii=False)
    print(f"Written: {path_remote}")

print("\nDone. All configs generated.")
