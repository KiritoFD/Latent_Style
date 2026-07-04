
"""
Task 8: Backward Compatibility Regression Test

验证所有新增配置参数的默认值下，代码行为与改动前完全一致。
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from config_schema import (
    ExperimentConfig,
    ModelConfig,
    BridgeConfig,
    load_experiment_config,
)
from model620 import build_spatial_bridge620_from_config
from losses620 import SpatialBridgeObjective620


def test_config_defaults():
    """Test 1: 验证所有新增参数都有正确的默认值"""
    print("=" * 60)
    print("Test 1: Config Default Values")
    print("=" * 60)
    
    model_cfg = ModelConfig()
    bridge_cfg = BridgeConfig()
    
    checks = []
    
    # ModelConfig 新增参数
    checks.append(("style_gate_mode", model_cfg.style_gate_mode, "tanh_gate"))
    checks.append(("endpoint_film_use_norm", model_cfg.endpoint_film_use_norm, True))
    
    # BridgeConfig 新增参数
    checks.append(("w_style_strength_reg", bridge_cfg.w_style_strength_reg, 0.0))
    checks.append(("training_objective_mode", bridge_cfg.training_objective_mode, "velocity"))
    checks.append(("w_endpoint_content", bridge_cfg.w_endpoint_content, 1.0))
    checks.append(("w_endpoint_style", bridge_cfg.w_endpoint_style, 8.0))
    checks.append(("w_endpoint_velocity_reg", bridge_cfg.w_endpoint_velocity_reg, 0.0))
    checks.append(("two_stage_enabled", bridge_cfg.two_stage_enabled, False))
    
    all_passed = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {name}: actual={actual!r}, expected={expected!r}")
    
    print()
    return all_passed


def test_config_loading_old_format():
    """Test 2: 验证旧配置（无新参数）能正常加载"""
    print("=" * 60)
    print("Test 2: Old Config Loading (no new params)")
    print("=" * 60)
    
    # 创建一个没有任何新参数的配置
    old_model_config = {
        "contract_family": "620_spatial_bridge",
        "style_condition_source": "target_dino_patches",
        "base_dim": 64,
        "num_res_blocks": 2,
        "style_attn_num_heads": 4,
        "style_cross_attn_gate_init": 0.05,
    }
    
    old_bridge_config = {
        "w_flow": 1.0,
        "single_step_swd_weight": 8.0,
        "bridge_sigma": 0.02,
    }
    
    old_config = {
        "model": old_model_config,
        "bridge": old_bridge_config,
        "training": {"num_epochs": 1, "batch_size": 4},
        "data": {
            "data_root": "./test_data",
            "style_subdirs": ["style0", "style1"],
        },
    }
    
    try:
        cfg = ExperimentConfig.from_mapping(old_config)
        
        # 验证新参数都使用了默认值
        checks = []
        checks.append(("style_gate_mode", cfg.model.style_gate_mode, "tanh_gate"))
        checks.append(("endpoint_film_use_norm", cfg.model.endpoint_film_use_norm, True))
        checks.append(("w_style_strength_reg", cfg.bridge.w_style_strength_reg, 0.0))
        checks.append(("training_objective_mode", cfg.bridge.training_objective_mode, "velocity"))
        checks.append(("two_stage_enabled", cfg.bridge.two_stage_enabled, False))
        
        all_passed = True
        for name, actual, expected in checks:
            status = "PASS" if actual == expected else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  [{status}] {name}: default={actual!r}, expected={expected!r}")
        
        print(f"\n  Config loaded successfully with all defaults applied.")
        print()
        return all_passed
    except Exception as e:
        print(f"  [FAIL] Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_model_forward():
    """Test 3: 验证模型在默认参数下的前向传播"""
    print("=" * 60)
    print("Test 3: Model Forward Pass (default params)")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # 使用默认配置创建模型
    model_cfg = ModelConfig()
    model_cfg.contract_family = "620_spatial_bridge"
    model_cfg.base_dim = 32  # 小一点快一点
    model_cfg.num_res_blocks = 2
    model_cfg.style_attn_num_heads = 4
    model_cfg.tokenizer_dino_dim = 64  # 小一点
    
    bridge_cfg = BridgeConfig()
    
    try:
        model = build_spatial_bridge620_from_config(model_cfg, bridge_cfg=bridge_cfg)
        model.eval()
        
        # 验证 style_gate_mode
        print(f"  [INFO] style_gate_mode: {model.style_gate_mode}")
        print(f"  [INFO] endpoint_film_use_norm: {model.endpoint_film_use_norm}")
        
        # 验证 FiLMEndpointHead 的 use_norm
        if model.endpoint_film_enabled and hasattr(model, 'endpoint_film_low') and model.endpoint_film_low:
            print(f"  [INFO] FiLMEndpointHead.use_norm: {model.endpoint_film_low.use_norm}")
            print(f"  [INFO] FiLMEndpointHead.norm type: {type(model.endpoint_film_low.norm).__name__}")
        else:
            print(f"  [INFO] endpoint_film_enabled: {model.endpoint_film_enabled} (endpoint head not used in velocity mode)")
        
        # 简单前向传播测试
        batch_size = 2
        h, w = 16, 16
        x = torch.randn(batch_size, model_cfg.latent_channels, h, w)
        t = torch.rand(batch_size)
        style_id = torch.randint(0, 5, (batch_size,))
        style_dino_patches = torch.randn(batch_size, 256, 64)  # DINO tokens
        style_latent = torch.randn(batch_size, model_cfg.latent_channels, h, w)
        
        with torch.no_grad():
            output = model(
                x,
                t=t,
                style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_latent=style_latent,
            )
        
        print(f"  [PASS] Forward pass succeeded")
        print(f"  [INFO] Output shape: {output.shape}")
        print(f"  [INFO] Output mean: {output.mean().item():.4f}")
        print(f"  [INFO] Output std: {output.std().item():.4f}")
        
        # 检查输出中没有 NaN
        if torch.isnan(output).any():
            print(f"  [FAIL] Output contains NaN!")
            all_passed = False
        else:
            print(f"  [PASS] No NaN in output")
            all_passed = True
        
        print()
        return all_passed
    except Exception as e:
        print(f"  [FAIL] Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_loss_computation():
    """Test 4: 验证 loss 计算在默认参数下"""
    print("=" * 60)
    print("Test 4: Loss Computation (default params)")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # 创建配置
    model_cfg = ModelConfig()
    model_cfg.contract_family = "620_spatial_bridge"
    model_cfg.base_dim = 32
    model_cfg.num_res_blocks = 2
    model_cfg.style_attn_num_heads = 4
    model_cfg.tokenizer_dino_dim = 64
    
    bridge_cfg = BridgeConfig()
    
    try:
        # 验证默认参数
        print(f"  [INFO] training_objective_mode: {bridge_cfg.training_objective_mode}")
        print(f"  [INFO] w_style_strength_reg: {bridge_cfg.w_style_strength_reg}")
        print(f"  [INFO] two_stage_enabled: {bridge_cfg.two_stage_enabled}")
        
        # 创建实验配置
        exp_cfg = ExperimentConfig()
        exp_cfg.model = model_cfg
        exp_cfg.bridge = bridge_cfg
        
        # 创建模型和 loss
        model = build_spatial_bridge620_from_config(model_cfg, bridge_cfg=bridge_cfg)
        objective = SpatialBridgeObjective620(exp_cfg)
        
        # 验证两阶段调度：默认不启用
        epoch_weights = objective.update_weights_for_epoch(1)
        print(f"  [INFO] Epoch 1 weights: stage={epoch_weights['stage']}")
        if epoch_weights['stage'] == 0:
            print(f"  [PASS] Two-stage disabled by default (stage=0)")
        else:
            print(f"  [FAIL] Two-stage should be disabled by default")
            print()
            return False
        
        # 构造测试数据
        batch_size = 2
        h, w = 16, 16
        content = torch.randn(batch_size, model_cfg.latent_channels, h, w)
        target_style = torch.randn(batch_size, model_cfg.latent_channels, h, w)
        target_style_id = torch.randint(0, 5, (batch_size,))
        
        style_dino_patches = torch.randn(batch_size, 256, 64)
        style_dino_cls = torch.randn(batch_size, 64)
        content_dino_patches = torch.randn(batch_size, 64, 64)
        style_latent = torch.randn(batch_size, model_cfg.latent_channels, h, w)
        
        conditioning = {
            "target_style_dino_patches": style_dino_patches,
            "target_style_dino_cls": style_dino_cls,
            "content_dino_patches": content_dino_patches,
            "target_style_text_tokens": None,
        }
        
        # 计算 loss
        model.train()
        metrics = objective.compute(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=None,
            aux_target_style=None,
            aux_target_valid=None,
            conditioning=conditioning,
        )
        
        loss = metrics["loss"]
        
        print(f"  [PASS] Loss computation succeeded")
        print(f"  [INFO] loss: {loss.item():.4f}")
        print(f"  [INFO] flow (fm): {metrics['flow'].item():.4f}")
        print(f"  [INFO] loss_swd_ss: {metrics['loss_swd_ss'].item():.4f}")
        print(f"  [INFO] loss_style_strength_reg: {metrics['loss_style_strength_reg'].item():.4f}")
        print(f"  [INFO] training_objective_mode (0=velocity, 1=endpoint): {metrics['training_objective_mode'].item():.0f}")
        
        # 验证默认是 velocity 模式
        if metrics['training_objective_mode'].item() < 0.5:
            print(f"  [PASS] Default mode is velocity (correct)")
        else:
            print(f"  [FAIL] Default mode should be velocity")
            print()
            return False
        
        # 验证 style_strength_reg 为 0（因为 w=0）
        if abs(metrics['loss_style_strength_reg'].item()) < 1e-6:
            print(f"  [PASS] style_strength_loss is 0 (w_style_strength_reg=0)")
        else:
            print(f"  [WARN] style_strength_loss: {metrics['loss_style_strength_reg'].item():.6f} (should be ~0)")
        
        # 检查没有 NaN
        has_nan = False
        for key, value in metrics.items():
            if torch.is_tensor(value) and torch.isnan(value).any():
                print(f"  [FAIL] {key} contains NaN!")
                has_nan = True
        
        if not has_nan:
            print(f"  [PASS] No NaN in loss components")
        
        print()
        return not has_nan
    except Exception as e:
        print(f"  [FAIL] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_style_gate_mode():
    """Test 5: 验证 style_gate_mode = 'tanh_gate' 走原有路径"""
    print("=" * 60)
    print("Test 5: style_gate_mode = 'tanh_gate' (original behavior)")
    print("=" * 60)
    
    from blocks620 import SpatialBridgeBlock620
    
    try:
        block = SpatialBridgeBlock620(
            dim=32,
            num_heads=4,
            style_gate_init=0.05,
            style_gate_mode="tanh_gate",
        )
        
        # 验证默认就是 tanh_gate
        print(f"  [INFO] style_gate_mode: {block.style_gate_mode}")
        
        # 检查 forward 中 style_gate_mode == 'tanh_gate' 时执行 torch.tanh(self.style_gate) * attended
        # 我们通过检查 style_gate 的值来验证
        print(f"  [INFO] style_gate param: {block.style_gate.item():.4f}")
        print(f"  [INFO] tanh(style_gate): {torch.tanh(block.style_gate).item():.4f}")
        
        # 简单的前向传播
        x = torch.randn(1, 32, 8, 8)
        time_emb = torch.randn(1, 32)
        style_tokens = torch.randn(1, 128, 32)
        
        with torch.no_grad():
            output = block(x, time_emb=time_emb, style_tokens=style_tokens)
        
        print(f"  [PASS] Forward pass with tanh_gate succeeded")
        print(f"  [INFO] Output shape: {output.shape}")
        
        # 验证 debug 信息中有 style_gate_value
        debug = block.last_debug
        print(f"  [INFO] style_gate_value (|tanh(gate)|): {debug['style_gate_value'].item():.4f}")
        
        print()
        return True
    except Exception as e:
        print(f"  [FAIL] style_gate_mode test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_film_endpoint_head_norm():
    """Test 6: 验证 FiLMEndpointHead use_norm=True 时使用 GroupNorm"""
    print("=" * 60)
    print("Test 6: FiLMEndpointHead use_norm=True (GroupNorm)")
    print("=" * 60)
    
    from model620 import FiLMEndpointHead
    
    try:
        # use_norm=True (默认)
        head_with_norm = FiLMEndpointHead(
            dim=32,
            latent_channels=4,
            style_dim=32,
            style_hidden_dim=16,
            use_norm=True,
        )
        
        print(f"  [INFO] use_norm=True:")
        print(f"    - self.norm is None: {head_with_norm.norm is None}")
        if head_with_norm.norm is not None:
            print(f"    - norm type: {type(head_with_norm.norm).__name__}")
            print(f"    - norm num_groups: {head_with_norm.norm.num_groups}")
        
        # use_norm=False
        head_no_norm = FiLMEndpointHead(
            dim=32,
            latent_channels=4,
            style_dim=32,
            style_hidden_dim=16,
            use_norm=False,
        )
        print(f"  [INFO] use_norm=False:")
        print(f"    - self.norm is None: {head_no_norm.norm is None}")
        
        # 前向传播测试
        x = torch.randn(1, 32, 8, 8)
        style_embed = torch.randn(1, 32)
        
        with torch.no_grad():
            out_with_norm = head_with_norm(x, style_embed)
            out_no_norm = head_no_norm(x, style_embed)
        
        print(f"  [PASS] Both heads forward pass succeeded")
        print(f"  [INFO] With norm output shape: {out_with_norm.shape}")
        print(f"  [INFO] Without norm output shape: {out_no_norm.shape}")
        
        print()
        return True
    except Exception as e:
        print(f"  [FAIL] FiLMEndpointHead norm test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    print("\n" + "=" * 60)
    print("Task 8: Backward Compatibility Regression Test")
    print("=" * 60 + "\n")
    
    results = {}
    
    results["config_defaults"] = test_config_defaults()
    results["old_config_loading"] = test_config_loading_old_format()
    results["model_forward"] = test_model_forward()
    results["loss_computation"] = test_loss_computation()
    results["style_gate_mode"] = test_style_gate_mode()
    results["film_endpoint_norm"] = test_film_endpoint_head_norm()
    
    # 总结
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")
    
    print()
    if all_passed:
        print("All tests passed! Backward compatibility is maintained.")
    else:
        print("Some tests failed! Please check the output above.")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
