"""Phase 4I.9 Probe: 诊断 velocity field 和 AdaIN 瓶颈.

用户指导: "结合理论文档，从模型全局出发，用 probe 去 debug 找出瓶颈去突破"

4 个诊断假设:
  A: velocity field 是否随 t 变化 (time conditioning 是否生效)
  B: AdaIN 高阶统计量匹配 (mean+std 够不够, 协方差是否丢失)
  C: HH 子带信息丢失 (无 v_hh, AdaIN 是否补偿)
  D: ODE 轨迹 vs 真实 target_delta 偏差 (velocity field 准确性)

用法: python probes/probe_velocity_adain.py
"""
from __future__ import annotations
import sys
import os
import math
import json
from pathlib import Path

# 确保 src 在 path 中
SRC_DIR = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\src").resolve()
sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from config_schema import load_experiment_config
from model import build_model_from_config
from utils.training import strip_compile_prefix
from style_families import prune_state_dict_for_tokenizer_family
from utils.dataset import AdaCUTLatentDataset
from wavelet import dwt2_haar, dwt2_lowpass, idwt2_haar


# ============================================================================
# 配置
# ============================================================================
CONFIG_PATH = r"g:\GitHub\Latent_Style\SchrodingerBridge\configs\630_phase4i7b_cosine_heun_a085_5ep.json"
CKPT_PATH = r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\630_phase4i7b_cosine_heun_a085_5ep\epoch_0005.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NUM_PROBE_SAMPLES = 4  # 诊断用的样本数


def log(msg: str) -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {title}")
    log("=" * 78)


# ============================================================================
# 模型加载
# ============================================================================
def load_model_and_config():
    """加载配置、模型、checkpoint."""
    log(f"[加载] 配置: {CONFIG_PATH}")
    config = load_experiment_config(CONFIG_PATH)

    log(f"[加载] 构建模型 (contract_family={config.model.contract_family})")
    model = build_model_from_config(
        config.model, bridge_cfg=config.bridge, use_checkpointing=False
    ).to(DEVICE).eval()

    log(f"[加载] checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model_state = strip_compile_prefix(ckpt["model_state_dict"])

    log("[加载] prune state dict for tokenizer family")
    model_state, _ = prune_state_dict_for_tokenizer_family(
        model_state,
        tokenizer_family=str(config.model.tokenizer_family),
        contract_family=str(config.model.contract_family),
        style_injection_mode=str(config.model.style_injection_mode),
        proximal_mode=str(config.model.proximal_mode),
        style_delta_mode=str(config.model.style_delta_mode),
        output_appearance_alignment_mode=str(config.model.output_appearance_alignment_mode),
    )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        log(f"[警告] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        log(f"[警告] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    log("[加载] 模型加载完成")
    return config, model


# ============================================================================
# 数据加载
# ============================================================================
def load_dataset(config):
    """加载数据集, 返回一个 batch."""
    log(f"[数据] data_root: {config.data.data_root}")
    dataset = AdaCUTLatentDataset(
        data_root=config.data.data_root,
        style_subdirs=config.data.style_subdirs,
        allow_hflip=False,
        pairing_cache_path="",
        latent_cache_mode="off",
        dino_cache_path="",
        device="cpu",
    )
    dataset.set_epoch(0)

    # 取前 NUM_PROBE_SAMPLES 个样本
    items = [dataset[i] for i in range(NUM_PROBE_SAMPLES)]
    # stack 成 batch
    batch = {}
    for key in items[0]:
        val = items[0][key]
        if isinstance(val, torch.Tensor):
            batch[key] = torch.stack([it[key] for it in items], dim=0)
        else:
            batch[key] = torch.tensor([it[key] for it in items])

    content = batch["content"].to(DEVICE, dtype=DTYPE)
    target_style = batch["target_style"].to(DEVICE, dtype=DTYPE)
    target_style_id = batch["target_style_id"].to(DEVICE)
    log(f"[数据] batch: content={content.shape}, target_style={target_style.shape}, "
        f"style_ids={target_style_id.tolist()}")
    return content, target_style, target_style_id


# ============================================================================
# 假设 A: velocity field 是否随 t 变化
# ============================================================================
def probe_a_velocity_vs_t(model, content, target_style, target_style_id):
    """诊断 velocity field 的 time conditioning 是否生效.

    如果 velocity 不随 t 变化 → time embedding 失效, velocity field 退化.
    """
    section("假设 A: velocity field 是否随 t 变化 (time conditioning)")

    t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    results = {}

    log(f"\n扫描 t 值: {t_values}")
    log(f"{'t':>6} | {'|v_ll|':>10} | {'|v_lh|':>10} | {'|v_hl|':>10} | {'|v_all|':>10}")
    log("-" * 60)

    with torch.no_grad():
        for t_val in t_values:
            t = torch.full((content.shape[0],), t_val, device=DEVICE, dtype=DTYPE)
            # x_t = (1-t)*content + t*target (训练时的插值方式)
            t_view = t.view(-1, 1, 1, 1)
            x_t = (1.0 - t_view) * content + t_view * target_style

            v_dict = model(
                x_t, t=t, style_id=target_style_id,
                style_dino_patches=None, style_dino_cls=None,
                content_dino_patches=None, style_latent=None,
            )
            v_ll_abs = v_dict["ll"].float().abs().mean().item()
            v_lh_abs = v_dict["lh"].float().abs().mean().item()
            v_hl_abs = v_dict["hl"].float().abs().mean().item()
            v_all = (v_ll_abs + v_lh_abs + v_hl_abs) / 3.0
            results[t_val] = {
                "v_ll": v_ll_abs, "v_lh": v_lh_abs, "v_hl": v_hl_abs, "v_all": v_all,
                "v_ll_tensor": v_dict["ll"].float(), "v_lh_tensor": v_dict["lh"].float(),
                "v_hl_tensor": v_dict["hl"].float(),
            }
            log(f"{t_val:>6.2f} | {v_ll_abs:>10.6f} | {v_lh_abs:>10.6f} | {v_hl_abs:>10.6f} | {v_all:>10.6f}")

    # 计算 t 之间的 velocity 差异
    log("\n--- velocity 差异 (L2 distance between t pairs) ---")
    log(f"{'t1':>6} {'t2':>6} | {'Δv_ll':>10} | {'Δv_lh':>10} | {'Δv_hl':>10}")
    log("-" * 55)
    for i, t1 in enumerate(t_values):
        for t2 in t_values[i+1:]:
            d_ll = (results[t1]["v_ll_tensor"] - results[t2]["v_ll_tensor"]).norm().item()
            d_lh = (results[t1]["v_lh_tensor"] - results[t2]["v_lh_tensor"]).norm().item()
            d_hl = (results[t1]["v_hl_tensor"] - results[t2]["v_hl_tensor"]).norm().item()
            log(f"{t1:>6.2f} {t2:>6.2f} | {d_ll:>10.6f} | {d_lh:>10.6f} | {d_hl:>10.6f}")

    # 判定
    log("\n--- 判定 ---")
    # 比较 t=0 和 t=1 的 velocity 差异 vs velocity 本身的大小
    v0_all = results[0.0]["v_all"]
    v1_all = results[1.0]["v_all"]
    d_01_ll = (results[0.0]["v_ll_tensor"] - results[1.0]["v_ll_tensor"]).norm().item()
    d_01_lh = (results[0.0]["v_lh_tensor"] - results[1.0]["v_lh_tensor"]).norm().item()
    d_01_hl = (results[0.0]["v_hl_tensor"] - results[1.0]["v_hl_tensor"]).norm().item()

    # velocity 变化幅度 vs velocity 本身
    ratio_ll = d_01_ll / (results[0.0]["v_ll_tensor"].norm().item() + 1e-8)
    ratio_lh = d_01_lh / (results[0.0]["v_lh_tensor"].norm().item() + 1e-8)
    ratio_hl = d_01_hl / (results[0.0]["v_hl_tensor"].norm().item() + 1e-8)

    log(f"velocity 变化/自身 比率 (t=0 vs t=1):")
    log(f"  LL: {ratio_ll:.4f}  ({'退化' if ratio_ll < 0.1 else '活跃'})")
    log(f"  LH: {ratio_lh:.4f}  ({'退化' if ratio_lh < 0.1 else '活跃'})")
    log(f"  HL: {ratio_hl:.4f}  ({'退化' if ratio_hl < 0.1 else '活跃'})")

    # velocity 幅度 vs target_delta 幅度
    target_delta = target_style - content
    target_ll, target_lh, target_hl, _ = dwt2_haar(target_delta)
    log(f"\n  target_delta 子带幅度:")
    log(f"    |target_ll| = {target_ll.float().abs().mean().item():.6f}")
    log(f"    |target_lh| = {target_lh.float().abs().mean().item():.6f}")
    log(f"    |target_hl| = {target_hl.float().abs().mean().item():.6f}")
    # 理论上 velocity ≈ target_delta (因为 ODE 是线性的: dx/dt = target - content)
    log(f"\n  理论: 若 ODE 为线性 x_t=(1-t)c+t*target, 则 velocity = target-content (常数)")
    log(f"  实测 v_ll(t=0.5) = {results[0.5]['v_ll']:.6f} vs target_ll = {target_ll.float().abs().mean().item():.6f}")
    log(f"  实测 v_lh(t=0.5) = {results[0.5]['v_lh']:.6f} vs target_lh = {target_lh.float().abs().mean().item():.6f}")
    log(f"  实测 v_hl(t=0.5) = {results[0.5]['v_hl']:.6f} vs target_hl = {target_hl.float().abs().mean().item():.6f}")

    return results


# ============================================================================
# 假设 B: AdaIN 高阶统计量匹配
# ============================================================================
def probe_b_adain_higher_order_stats(model, content, target_style, target_style_id, config):
    """诊断 AdaIN 的高阶统计量匹配.

    spatial_fiber 模式只做 mean+std 匹配, 缺失协方差等高阶统计.
    """
    section("假设 B: AdaIN 高阶统计量匹配")

    endpoint_adain_scale = float(getattr(config.model, "endpoint_adain_scale", 0.0))
    lowpass_levels = int(getattr(config.model, "endpoint_lowpass_levels", 1))
    style_extrap_alpha = float(getattr(config.model, "style_extrap_alpha", 0.0))

    log(f"配置: endpoint_adain_scale={endpoint_adain_scale}, "
        f"lowpass_levels={lowpass_levels}, style_extrap_alpha={style_extrap_alpha}")

    # 1. 运行完整推理 (with AdaIN)
    with torch.no_grad():
        output_with_adain = model.integrate_transport(
            content, style_id=target_style_id, num_steps=8, step_size=1.0,
            style_latent=target_style,  # 用 target_style 作为 style_latent
        )

    # 2. 运行推理 (without AdaIN — 临时设 scale=0)
    # 我们不能修改 config, 但可以手动模拟 "纯 ODE 积分" 的结果
    with torch.no_grad():
        # 手动运行 Heun 积分 (不调用 integrate_transport, 避免 AdaIN)
        h = content.clone()
        horizon = 1.0
        steps = 8
        dt = horizon / steps
        import math
        for i in range(steps):
            s = float(i) / steps
            t_curr = (1.0 - math.cos(math.pi * s)) / 2.0 * horizon
            s_next = float(i + 1) / steps
            t_next = (1.0 - math.cos(math.pi * s_next)) / 2.0 * horizon
            t_batch = torch.full((h.shape[0],), t_curr, device=DEVICE, dtype=DTYPE)
            v1 = model(h, t=t_batch, style_id=target_style_id,
                       style_dino_patches=None, style_dino_cls=None,
                       content_dino_patches=None, style_latent=None)
            ll1, lh1, hl1, hh1 = dwt2_haar(h)
            ll_pred = ll1 + v1["ll"] * dt
            lh_pred = lh1 + v1["lh"] * dt
            hl_pred = hl1 + v1["hl"] * dt
            h_pred = idwt2_haar(ll_pred, lh_pred, hl_pred, hh1)
            t_batch2 = torch.full((h.shape[0],), t_next, device=DEVICE, dtype=DTYPE)
            v2 = model(h_pred, t=t_batch2, style_id=target_style_id,
                       style_dino_patches=None, style_dino_cls=None,
                       content_dino_patches=None, style_latent=None)
            ll_new = ll1 + (v1["ll"] + v2["ll"]) / 2.0 * dt
            lh_new = lh1 + (v1["lh"] + v2["lh"]) / 2.0 * dt
            hl_new = hl1 + (v1["hl"] + v2["hl"]) / 2.0 * dt
            h = idwt2_haar(ll_new, lh_new, hl_new, hh1)
        output_ode_only = h

    # 3. 统计量对比函数
    def compute_stats(tensor, name):
        """计算 1-4 阶统计量 + 通道协方差."""
        t = tensor.float()
        B, C, H, W = t.shape
        # 一阶: mean
        mean = t.mean(dim=[2, 3])  # [B, C]
        # 二阶: std
        std = t.std(dim=[2, 3])  # [B, C]
        # 三阶: skewness
        centered = t - mean.unsqueeze(-1).unsqueeze(-1)
        std_exp = std.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-6)
        normalized = centered / std_exp
        skew = (normalized ** 3).mean(dim=[2, 3])  # [B, C]
        # 四阶: kurtosis (excess)
        kurt = (normalized ** 4).mean(dim=[2, 3]) - 3.0  # [B, C]
        # 通道间协方差 (flatten spatial, compute Cov[C,C])
        t_flat = t.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
        cov = torch.zeros(B, C, C)
        for b in range(B):
            cov[b] = torch.cov(t_flat[b].T)
        return {
            "mean": mean.mean(0),  # 平均 over batch → [C]
            "std": std.mean(0),
            "skew": skew.mean(0),
            "kurt": kurt.mean(0),
            "cov": cov.mean(0),  # [C, C]
        }

    # 4. 计算各阶段统计量
    log("\n--- 统计量对比 (batch-averaged) ---")
    stats_content = compute_stats(content, "content")
    stats_target = compute_stats(target_style, "target_style")
    stats_ode = compute_stats(output_ode_only, "ODE_only")
    stats_adain = compute_stats(output_with_adain, "ODE+AdaIN")

    # 5. 打印统计量差异
    def stats_diff(s1, s2, name1, name2):
        d_mean = (s1["mean"] - s2["mean"]).abs().mean().item()
        d_std = (s1["std"] - s2["std"]).abs().mean().item()
        d_skew = (s1["skew"] - s2["skew"]).abs().mean().item()
        d_kurt = (s1["kurt"] - s2["kurt"]).abs().mean().item()
        d_cov = (s1["cov"] - s2["cov"]).abs().mean().item()
        log(f"  {name1} vs {name2}:")
        log(f"    Δmean = {d_mean:.6f}")
        log(f"    Δstd  = {d_std:.6f}")
        log(f"    Δskew = {d_skew:.6f}")
        log(f"    Δkurt = {d_kurt:.6f}")
        log(f"    Δcov  = {d_cov:.6f}")
        return {"mean": d_mean, "std": d_std, "skew": d_skew, "kurt": d_kurt, "cov": d_cov}

    log("\n--- 基准差异 (content vs target_style = 风格差距) ---")
    base_diff = stats_diff(stats_content, stats_target, "content", "target_style")

    log("\n--- ODE_only vs target_style (ODE 积分后还差多少) ---")
    ode_diff = stats_diff(stats_ode, stats_target, "ODE_only", "target_style")

    log("\n--- ODE+AdaIN vs target_style (AdaIN 后还差多少) ---")
    adain_diff = stats_diff(stats_adain, stats_target, "ODE+AdaIN", "target_style")

    log("\n--- AdaIN 改善量 (ODE_only → ODE+AdaIN 的统计量修正) ---")
    log(f"  mean 修正: {ode_diff['mean'] - adain_diff['mean']:.6f} "
        f"({(1 - adain_diff['mean']/(ode_diff['mean']+1e-8))*100:.1f}%)")
    log(f"  std 修正:  {ode_diff['std'] - adain_diff['std']:.6f} "
        f"({(1 - adain_diff['std']/(ode_diff['std']+1e-8))*100:.1f}%)")
    log(f"  skew 修正: {ode_diff['skew'] - adain_diff['skew']:.6f} "
        f"({(1 - adain_diff['skew']/(ode_diff['skew']+1e-8))*100:.1f}%)")
    log(f"  kurt 修正: {ode_diff['kurt'] - adain_diff['kurt']:.6f} "
        f"({(1 - adain_diff['kurt']/(ode_diff['kurt']+1e-8))*100:.1f}%)")
    log(f"  cov 修正:  {ode_diff['cov'] - adain_diff['cov']:.6f} "
        f"({(1 - adain_diff['cov']/(ode_diff['cov']+1e-8))*100:.1f}%)")

    # 6. 判定
    log("\n--- 判定 ---")
    mean_fix_pct = (1 - adain_diff['mean']/(ode_diff['mean']+1e-8))*100
    std_fix_pct = (1 - adain_diff['std']/(ode_diff['std']+1e-8))*100
    cov_fix_pct = (1 - adain_diff['cov']/(ode_diff['cov']+1e-8))*100
    log(f"  AdaIN 对 mean 修正率: {mean_fix_pct:.1f}%  ({'充分' if mean_fix_pct > 80 else '不足'})")
    log(f"  AdaIN 对 std 修正率:  {std_fix_pct:.1f}%  ({'充分' if std_fix_pct > 80 else '不足'})")
    log(f"  AdaIN 对 cov 修正率:  {cov_fix_pct:.1f}%  ({'充分' if cov_fix_pct > 60 else '不足 — 协方差是瓶颈'})")

    return {
        "base_diff": base_diff, "ode_diff": ode_diff, "adain_diff": adain_diff,
        "output_ode_only": output_ode_only, "output_with_adain": output_with_adain,
    }


# ============================================================================
# 假设 C: HH 子带信息丢失
# ============================================================================
def probe_c_hh_subband(model, content, target_style, target_style_id, probe_b_results):
    """诊断 HH 子带信息丢失.

    模型没有 v_hh, HH 在 ODE 积分中保持不变.
    AdaIN (spatial_fiber 模式) 对 fiber = h - lp(h) 做全局匹配, 不专门处理 HH.
    """
    section("假设 C: HH 子带信息丢失")

    output_ode = probe_b_results["output_ode_only"]
    output_adain = probe_b_results["output_with_adain"]

    # 分解各阶段的 HH 子带
    ll_c, lh_c, hl_c, hh_c = dwt2_haar(content.float())
    ll_t, lh_t, hl_t, hh_t = dwt2_haar(target_style.float())
    ll_o, lh_o, hl_o, hh_o = dwt2_haar(output_ode.float())
    ll_a, lh_a, hl_a, hh_a = dwt2_haar(output_adain.float())

    log("\n--- HH 子带幅度对比 ---")
    log(f"{'阶段':>15} | {'|HH|':>10} | {'mean':>10} | {'std':>10}")
    log("-" * 55)
    for name, hh in [("content", hh_c), ("target_style", hh_t),
                     ("ODE_only", hh_o), ("ODE+AdaIN", hh_a)]:
        log(f"{name:>15} | {hh.abs().mean().item():>10.6f} | "
            f"{hh.mean().item():>10.6f} | {hh.std().item():>10.6f}")

    log("\n--- HH 子带差异 (L2) ---")
    def l2(a, b):
        return (a - b).float().norm().item()
    log(f"  content vs target_style HH: {l2(hh_c, hh_t):.6f} (风格差距)")
    log(f"  content vs ODE_only HH:     {l2(hh_c, hh_o):.6f} (ODE 是否改变 HH)")
    log(f"  content vs ODE+AdaIN HH:     {l2(hh_c, hh_a):.6f} (AdaIN 是否改变 HH)")
    log(f"  ODE_only vs ODE+AdaIN HH:    {l2(hh_o, hh_a):.6f} (AdaIN 对 HH 的净影响)")
    log(f"  ODE+AdaIN vs target HH:      {l2(hh_a, hh_t):.6f} (AdaIN 后 HH 还差多少)")

    # 判定
    log("\n--- 判定 ---")
    hh_ode_change = l2(hh_c, hh_o)
    hh_adain_change = l2(hh_o, hh_a)
    hh_residual = l2(hh_a, hh_t)
    hh_base = l2(hh_c, hh_t)

    log(f"  ODE 对 HH 的改变:     {hh_ode_change:.6f} ({'无改变 (HH 保持不变)' if hh_ode_change < 1e-4 else '有改变'})")
    log(f"  AdaIN 对 HH 的改变:   {hh_adain_change:.6f} ({'无改变' if hh_adain_change < 1e-4 else '有改变'})")
    log(f"  HH 残差/基准:         {hh_residual/(hh_base+1e-8):.4f} ({'HH 信息丢失' if hh_residual/hh_base > 0.5 else 'HH 匹配良好'})")

    # 也检查 LH/HL 子带
    log("\n--- LH/HL 子带对比 (参照) ---")
    log(f"{'阶段':>15} | {'|LH|':>10} | {'|HL|':>10}")
    log("-" * 45)
    for name, lh, hl in [("content", lh_c, hl_c), ("target_style", lh_t, hl_t),
                         ("ODE_only", lh_o, hl_o), ("ODE+AdaIN", lh_a, hl_a)]:
        log(f"{name:>15} | {lh.abs().mean().item():>10.6f} | {hl.abs().mean().item():>10.6f}")

    log(f"\n  LH: ODE 修正 {(1-l2(lh_o,lh_t)/(l2(lh_c,lh_t)+1e-8))*100:.1f}%, "
        f"AdaIN 额外修正 {(1-l2(lh_a,lh_t)/(l2(lh_o,lh_t)+1e-8))*100:.1f}%")
    log(f"  HL: ODE 修正 {(1-l2(hl_o,hl_t)/(l2(hl_c,hl_t)+1e-8))*100:.1f}%, "
        f"AdaIN 额外修正 {(1-l2(hl_a,hl_t)/(l2(hl_o,hl_t)+1e-8))*100:.1f}%")

    return {
        "hh_base": hh_base, "hh_ode_change": hh_ode_change,
        "hh_adain_change": hh_adain_change, "hh_residual": hh_residual,
    }


# ============================================================================
# 假设 D: ODE 轨迹 vs 真实 target_delta 偏差
# ============================================================================
def probe_d_velocity_accuracy(model, content, target_style, target_style_id):
    """诊断 velocity field 的准确性.

    理论: 若 ODE 为线性 x_t=(1-t)c+t*target, 则理想 velocity = target-content (常数).
    实际 velocity field 学到的可能偏离这个值.
    关键: ODE 积分 (无 AdaIN) 的最终结果 vs target 的偏差 = velocity field 误差.
    """
    section("假设 D: ODE 轨迹 vs 真实 target_delta 偏差")

    target_delta = target_style - content
    target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta.float())

    # 理想 velocity (线性 ODE: v = target - content, 恒定)
    ideal_v_ll = target_ll
    ideal_v_lh = target_lh
    ideal_v_hl = target_hl

    log("\n--- velocity 准确性 (t=0.5, 理想 v = target - content) ---")
    with torch.no_grad():
        t = torch.full((content.shape[0],), 0.5, device=DEVICE, dtype=DTYPE)
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * content + t_view * target_style
        v_dict = model(
            x_t, t=t, style_id=target_style_id,
            style_dino_patches=None, style_dino_cls=None,
            content_dino_patches=None, style_latent=None,
        )

    # 误差分析
    err_ll = (v_dict["ll"].float() - ideal_v_ll).abs().mean().item()
    err_lh = (v_dict["lh"].float() - ideal_v_lh).abs().mean().item()
    err_hl = (v_dict["hl"].float() - ideal_v_hl).abs().mean().item()

    ideal_ll_abs = ideal_v_ll.abs().mean().item()
    ideal_lh_abs = ideal_v_lh.abs().mean().item()
    ideal_hl_abs = ideal_v_hl.abs().mean().item()

    log(f"{'子带':>6} | {'|ideal_v|':>12} | {'|pred_v|':>12} | {'|error|':>12} | {'err/ideal':>10}")
    log("-" * 65)
    pred_ll_abs = v_dict["ll"].float().abs().mean().item()
    pred_lh_abs = v_dict["lh"].float().abs().mean().item()
    pred_hl_abs = v_dict["hl"].float().abs().mean().item()
    log(f"{'LL':>6} | {ideal_ll_abs:>12.6f} | {pred_ll_abs:>12.6f} | {err_ll:>12.6f} | {err_ll/(ideal_ll_abs+1e-8):>10.4f}")
    log(f"{'LH':>6} | {ideal_lh_abs:>12.6f} | {pred_lh_abs:>12.6f} | {err_lh:>12.6f} | {err_lh/(ideal_lh_abs+1e-8):>10.4f}")
    log(f"{'HL':>6} | {ideal_hl_abs:>12.6f} | {pred_hl_abs:>12.6f} | {err_hl:>12.6f} | {err_hl/(ideal_hl_abs+1e-8):>10.4f}")

    # 方向分析 (cosine similarity)
    def cos_sim(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    cos_ll = cos_sim(v_dict["ll"].float(), ideal_v_ll)
    cos_lh = cos_sim(v_dict["lh"].float(), ideal_v_lh)
    cos_hl = cos_sim(v_dict["hl"].float(), ideal_v_hl)

    log(f"\n--- 方向一致性 (cosine similarity, 1=完美) ---")
    log(f"  LL: {cos_ll:.4f}")
    log(f"  LH: {cos_lh:.4f}")
    log(f"  HL: {cos_hl:.4f}")

    # 积分后的累积误差
    log("\n--- 积分后累积误差 (8 步 Heun, 无 AdaIN) ---")
    with torch.no_grad():
        h = content.clone()
        horizon = 1.0
        steps = 8
        dt = horizon / steps
        import math
        for i in range(steps):
            s = float(i) / steps
            t_curr = (1.0 - math.cos(math.pi * s)) / 2.0 * horizon
            s_next = float(i + 1) / steps
            t_next = (1.0 - math.cos(math.pi * s_next)) / 2.0 * horizon
            t_batch = torch.full((h.shape[0],), t_curr, device=DEVICE, dtype=DTYPE)
            v1 = model(h, t=t_batch, style_id=target_style_id,
                       style_dino_patches=None, style_dino_cls=None,
                       content_dino_patches=None, style_latent=None)
            ll1, lh1, hl1, hh1 = dwt2_haar(h)
            ll_pred = ll1 + v1["ll"] * dt
            lh_pred = lh1 + v1["lh"] * dt
            hl_pred = hl1 + v1["hl"] * dt
            h_pred = idwt2_haar(ll_pred, lh_pred, hl_pred, hh1)
            t_batch2 = torch.full((h.shape[0],), t_next, device=DEVICE, dtype=DTYPE)
            v2 = model(h_pred, t=t_batch2, style_id=target_style_id,
                       style_dino_patches=None, style_dino_cls=None,
                       content_dino_patches=None, style_latent=None)
            ll_new = ll1 + (v1["ll"] + v2["ll"]) / 2.0 * dt
            lh_new = lh1 + (v1["lh"] + v2["lh"]) / 2.0 * dt
            hl_new = hl1 + (v1["hl"] + v2["hl"]) / 2.0 * dt
            h = idwt2_haar(ll_new, lh_new, hl_new, hh1)
        ode_output = h

    # ODE 积分结果 vs target 的偏差
    ll_o, lh_o, hl_o, hh_o = dwt2_haar(ode_output.float())
    ll_t, lh_t, hl_t, hh_t = dwt2_haar(target_style.float())

    log(f"{'子带':>6} | {'ODE结果':>12} | {'target':>12} | {'|误差|':>12} | {'修正率':>10}")
    log("-" * 65)
    for name, o, t_val in [("LL", ll_o, ll_t), ("LH", lh_o, lh_t), ("HL", hl_o, hl_t)]:
        o_abs = o.abs().mean().item()
        t_abs = t_val.abs().mean().item()
        err = (o - t_val).abs().mean().item()
        # 修正率 = 1 - |ODE - target| / |content - target|
        ll_c = dwt2_haar(content.float())
        c_val = ll_c[{"LL":0,"LH":1,"HL":2}[name]]
        base_err = (c_val - t_val).abs().mean().item()
        fix = (1 - err/(base_err+1e-8))*100
        log(f"{name:>6} | {o_abs:>12.6f} | {t_abs:>12.6f} | {err:>12.6f} | {fix:>9.1f}%")

    # HH 无法被 ODE 修正 (无 v_hh)
    hh_c = dwt2_haar(content.float())[3]
    hh_base = (hh_c - hh_t).abs().mean().item()
    hh_err = (hh_o - hh_t).abs().mean().item()
    log(f"{'HH':>6} | {hh_o.abs().mean().item():>12.6f} | {hh_t.abs().mean().item():>12.6f} | "
        f"{hh_err:>12.6f} | {'(无 v_hh)':>10}")

    # 判定
    log("\n--- 判定 ---")
    ll_fix = (1 - (ll_o - ll_t).abs().mean().item() / ((dwt2_haar(content.float())[0] - ll_t).abs().mean().item() + 1e-8)) * 100
    lh_fix = (1 - (lh_o - lh_t).abs().mean().item() / ((dwt2_haar(content.float())[1] - lh_t).abs().mean().item() + 1e-8)) * 100
    hl_fix = (1 - (hl_o - hl_t).abs().mean().item() / ((dwt2_haar(content.float())[2] - hl_t).abs().mean().item() + 1e-8)) * 100
    log(f"  LL ODE 修正率: {ll_fix:.1f}%  ({'良好' if ll_fix > 70 else '不足 — LL velocity 是瓶颈'})")
    log(f"  LH ODE 修正率: {lh_fix:.1f}%  ({'良好' if lh_fix > 70 else '不足 — LH velocity 是瓶颈'})")
    log(f"  HL ODE 修正率: {hl_fix:.1f}%  ({'良好' if hl_fix > 70 else '不足 — HL velocity 是瓶颈'})")
    log(f"  HH ODE 修正率: 0.0%  (无 v_hh — HH 完全依赖 AdaIN)")

    return {
        "cos_ll": cos_ll, "cos_lh": cos_lh, "cos_hl": cos_hl,
        "ll_fix": ll_fix, "lh_fix": lh_fix, "hl_fix": hl_fix,
    }


# ============================================================================
# 主函数
# ============================================================================
def main():
    log("=" * 78)
    log("  Phase 4I.9 Probe: Velocity Field & AdaIN 瓶颈诊断")
    log("  用户指导: 结合理论文档, 从模型全局出发, 用 probe debug 找瓶颈")
    log("=" * 78)

    # 1. 加载模型和数据
    config, model = load_model_and_config()
    content, target_style, target_style_id = load_dataset(config)

    # 2. 运行 4 个诊断
    results_a = probe_a_velocity_vs_t(model, content, target_style, target_style_id)
    results_b = probe_b_adain_higher_order_stats(model, content, target_style, target_style_id, config)
    results_c = probe_c_hh_subband(model, content, target_style, target_style_id, results_b)
    results_d = probe_d_velocity_accuracy(model, content, target_style, target_style_id)

    # 3. 汇总结论
    section("汇总: 瓶颈诊断结论")
    log("")
    log("假设 A (velocity time conditioning):")
    log("  → 查看上方 'velocity 变化/自身 比率' 判定")
    log("")
    log("假设 B (AdaIN 高阶统计量):")
    log("  → 查看上方 'AdaIN 对 cov 修正率' 判定")
    log("")
    log("假设 C (HH 子带信息丢失):")
    log("  → 查看上方 'HH 残差/基准' 判定")
    log("")
    log("假设 D (velocity field 准确性):")
    log("  → 查看上方 'ODE 修正率' 判定")
    log("")
    log("下一步: 根据以上诊断结果, 设计针对性结构性突破方案")


if __name__ == "__main__":
    main()
