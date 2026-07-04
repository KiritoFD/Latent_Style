"""生成 p4_fusion_breakout 系列训练配置 T1-T4。

基于 exp/p3_remote_10h/e4_long_10ep/config.json (E4-long) 派生四个配置:
  - T1: 架构增强 (endpoint_style_hidden_dim 128 -> 512)
  - T2: T1 + 训练侧 DWT (lowpass_mode=dwt_haar)            依赖 D2
  - T3: T1 + 频域 loss (spectral_w_*)                        依赖 D3
  - T4: T1 + DWT + 频域 loss + style_extrap_alpha            依赖 D2/D3/D4
"""

import json
import os
from copy import deepcopy

BASE_CONFIG = "exp/p3_remote_10h/e4_long_10ep/config.json"
CONFIGS_DIR = "configs"

# 四个实验的输出路径
OUTPUT_PATHS = {
    "T1": os.path.join(CONFIGS_DIR, "p4_t1_arch.json"),
    "T2": os.path.join(CONFIGS_DIR, "p4_t2_dwt.json"),
    "T3": os.path.join(CONFIGS_DIR, "p4_t3_spectral_loss.json"),
    "T4": os.path.join(CONFIGS_DIR, "p4_t4_full_fusion.json"),
}

# 各实验对应的 checkpoint.save_dir
SAVE_DIRS = {
    "T1": "./exp/p4_fusion_breakout/t1_arch",
    "T2": "./exp/p4_fusion_breakout/t2_dwt",
    "T3": "./exp/p4_fusion_breakout/t3_spectral_loss",
    "T4": "./exp/p4_fusion_breakout/t4_full_fusion",
}

# ablation.name 与 ablation.notes 描述
ABLATION_META = {
    "T1": {
        "name": "p4_t1_arch",
        "notes": "P4 T1 架构增强: E4-long 基线 + endpoint_style_hidden_dim 512 (原 128)",
    },
    "T2": {
        "name": "p4_t2_dwt",
        "notes": "P4 T2 训练侧 DWT: T1 + model.lowpass_mode=dwt_haar (依赖 D2 改动)",
    },
    "T3": {
        "name": "p4_t3_spectral_loss",
        "notes": (
            "P4 T3 频域 loss: T1 + bridge.spectral_w_ll=0.3, spectral_w_lh=1.0, "
            "spectral_w_hl=1.0, spectral_w_hh=1.5 (依赖 D3 改动)"
        ),
    },
    "T4": {
        "name": "p4_t4_full_fusion",
        "notes": (
            "P4 T4 全融合: T1 + lowpass_mode=dwt_haar + spectral_w_ll=0.3, "
            "spectral_w_hh=1.5 + style_extrap_alpha=0.1 (依赖 D2/D3/D4 改动)"
        ),
    },
}


def apply_common(base: dict) -> dict:
    """应用 T1-T4 共同的字段修改 (训练/数据相关)。"""
    cfg = deepcopy(base)

    t = cfg["training"]
    # resume 相关: 全部从头训练
    t["resume_checkpoint"] = ""
    t["resume_optimizer"] = False
    t["resume_model_strict"] = False
    t["resume_training_state"] = False
    # 训练超参
    t["num_epochs"] = 10
    t["batch_size"] = 16
    t["num_workers"] = 0
    t["persistent_workers"] = False
    t["pin_memory"] = False
    t["prefetch_factor"] = 1
    # 全量评估
    t["full_eval_each_epoch"] = True
    t["full_eval_in_process"] = False
    t["test_image_dir"] = "I:/wikiart_distinct5_samam_512_classview/test"
    t["full_eval_cache_dir"] = "I:/Github/Latent_Style/eval_cache"
    t["full_eval_clip_hf_cache_dir"] = "I:/Github/Latent_Style/eval_cache/hf"

    # 数据路径 (Windows native I:/ 风格)
    d = cfg["data"]
    d["data_root"] = "I:/wikiart_distinct5_samam_512_latents_ema/train"
    d["latent_cache_dir"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
    d["pairing_cache_path"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"

    return cfg


def apply_t1(cfg: dict) -> dict:
    """T1 架构增强: endpoint_style_hidden_dim 128 -> 512。"""
    cfg = deepcopy(cfg)
    cfg["model"]["endpoint_style_hidden_dim"] = 512
    return cfg


def apply_t2(cfg: dict) -> dict:
    """T2: T1 + 训练侧 DWT (lowpass_mode=dwt_haar)。"""
    cfg = deepcopy(cfg)
    # 先应用 T1 字段
    cfg["model"]["endpoint_style_hidden_dim"] = 512
    # 再叠加 DWT
    cfg["model"]["lowpass_mode"] = "dwt_haar"
    return cfg


def apply_t3(cfg: dict) -> dict:
    """T3: T1 + 频域 loss 权重。"""
    cfg = deepcopy(cfg)
    # 先应用 T1 字段
    cfg["model"]["endpoint_style_hidden_dim"] = 512
    # 再叠加频域 loss 权重
    b = cfg["bridge"]
    b["spectral_w_ll"] = 0.3
    b["spectral_w_lh"] = 1.0
    b["spectral_w_hl"] = 1.0
    b["spectral_w_hh"] = 1.5
    return cfg


def apply_t4(cfg: dict) -> dict:
    """T4: T1 + DWT + 部分 spectral 权重 + style_extrap_alpha。"""
    cfg = deepcopy(cfg)
    # 先应用 T1 字段
    cfg["model"]["endpoint_style_hidden_dim"] = 512
    # 叠加 DWT
    cfg["model"]["lowpass_mode"] = "dwt_haar"
    # 叠加部分频域 loss 权重
    b = cfg["bridge"]
    b["spectral_w_ll"] = 0.3
    b["spectral_w_hh"] = 1.5
    # 叠加风格外插
    cfg["model"]["style_extrap_alpha"] = 0.1
    return cfg


SPECIFIC_APPLIERS = {
    "T1": apply_t1,
    "T2": apply_t2,
    "T3": apply_t3,
    "T4": apply_t4,
}


def main() -> None:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base = json.load(f)

    # 先应用公共修改, 得到所有 T 共享的中间态
    common = apply_common(base)

    os.makedirs(CONFIGS_DIR, exist_ok=True)

    for tag in ("T1", "T2", "T3", "T4"):
        cfg = SPECIFIC_APPLIERS[tag](common)
        # checkpoint.save_dir
        cfg["checkpoint"]["save_dir"] = SAVE_DIRS[tag]
        # ablation.name / notes
        cfg["ablation"]["name"] = ABLATION_META[tag]["name"]
        cfg["ablation"]["notes"] = ABLATION_META[tag]["notes"]

        out_path = OUTPUT_PATHS[tag]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
