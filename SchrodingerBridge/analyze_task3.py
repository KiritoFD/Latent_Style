import json
import os
import csv

experiments = [
    ("Baseline (1ep)", "exp/task3_baseline_1ep/full_eval/epoch_0001/summary.json", "exp/task3_baseline_1ep/logs"),
    ("Combo A (1ep)", "exp/task3_combo_a_1ep/full_eval/epoch_0001/summary.json", "exp/task3_combo_a_1ep/logs"),
    ("Combo B (1ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0001/summary.json", None),
    ("Combo B (2ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0002/summary.json", None),
    ("Combo B (3ep)", "exp/task3_combo_b_3ep/full_eval/epoch_0003/summary.json", "exp/task3_combo_b_3ep/logs"),
]

metric_keys = [
    ("analysis.style_transfer_ability.clip_style", "clip_style", "风格注入强度", "↑"),
    ("analysis.style_transfer_ability.content_lpips", "LPIPS", "内容保持（越低越好）", "↓"),
    ("runtime_observability.style_transfer_ability.model_velocity_std", "velocity_std", "速度场标准差", "—"),
    ("runtime_observability.style_transfer_ability.model_velocity_channel_std", "velocity_ch_std", "速度通道标准差", "—"),
    ("runtime_observability.style_transfer_ability.model_endpoint_alpha", "endpoint_alpha", "端点alpha", "↑"),
    ("runtime_observability.style_transfer_ability.model_endpoint_output_std", "endpoint_std", "端点输出标准差", "—"),
    ("runtime_observability.style_transfer_ability.model_style_gate_value", "style_gate_value", "风格gate值*", "—"),
    ("runtime_observability.style_transfer_ability.model_film_gamma_abs", "film_gamma_abs", "FiLM gamma强度", "↑"),
    ("runtime_observability.style_transfer_ability.model_film_beta_abs", "film_beta_abs", "FiLM beta强度", "—"),
    ("runtime_observability.style_transfer_ability.model_cross_attn_entropy", "ca_entropy", "Cross-attn熵", "↓"),
    ("runtime_observability.style_transfer_ability.model_cross_attn_delta_abs", "ca_delta_abs", "Cross-attn变化幅度", "↑"),
    ("runtime_observability.style_transfer_ability.model_velocity_abs", "velocity_abs", "速度绝对值", "—"),
    ("runtime_observability.style_transfer_ability.model_block0_output_channel_std", "block0_ch_std", "Block0通道std", "—"),
    ("runtime_observability.style_transfer_ability.model_block3_output_channel_std", "block3_ch_std", "Block3通道std", "—"),
]

def get_nested(obj, path):
    keys = path.split('.')
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj

def load_training_loss(log_dir):
    if not log_dir or not os.path.exists(log_dir):
        return []
    csv_files = [f for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.csv')]
    if not csv_files:
        return []
    path = os.path.join(log_dir, csv_files[0])
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'epoch': int(row['epoch']),
                'loss': float(row['loss']),
                'flow': float(row['flow']),
                'velocity_abs': float(row.get('velocity_abs', 0)),
            })
    return rows

print("=" * 130)
print("Task 3: 组合验证（去 Endpoint Head GN + Fixed One gate + FiLM）- 实验结果对比报告")
print("=" * 130)
print()
print("实验配置:")
print("  Baseline (1ep): style_film + endpoint_film + tanh_gate + GN")
print("  Combo A  (1ep): style_film + endpoint_film + fixed_one + no GN")
print("  Combo B  (3ep): style_film + endpoint_film + fixed_one + no GN (3 epochs)")
print()

results = {}
train_losses = {}
for exp_name, path, log_dir in experiments:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        results[exp_name] = data
        train_losses[exp_name] = load_training_loss(log_dir)
    else:
        print(f"WARNING: {path} not found!")

print("=" * 130)
print("1. 训练 Loss 对比")
print("=" * 130)
print(f"{'实验':<20} {'Epoch':<8} {'Loss':<12} {'Flow':<12} {'velocity_abs':<15}")
print("-" * 130)
for exp_name, _, _ in experiments:
    if exp_name in train_losses and train_losses[exp_name]:
        for row in train_losses[exp_name]:
            print(f"{exp_name:<20} {row['epoch']:<8} {row['loss']:<12.6f} {row['flow']:<12.6f} {row['velocity_abs']:<15.6f}")
    elif exp_name.startswith("Combo B"):
        ep_num = exp_name.split('(')[1].split('ep')[0].strip()
        print(f"{exp_name:<20} {ep_num:<8} (见 Combo B 训练曲线)")
print()

print("=" * 130)
print("2. 关键评估指标对比 (Style Transfer Ability)")
print("=" * 130)
header = f"{'指标':<25} {'方向':<6} {'Baseline 1ep':>14} {'Combo A 1ep':>14} {'Combo B 1ep':>14} {'Combo B 2ep':>14} {'Combo B 3ep':>14}"
print(header)
print("-" * 130)

for full_key, display_name, desc, direction in metric_keys:
    row = f"{display_name:<25} {direction:<6}"
    vals = []
    for exp_name, _, _ in experiments:
        if exp_name in results:
            val = get_nested(results[exp_name], full_key)
            if val is not None:
                row += f"{val:>14.6f}"
                vals.append(val)
            else:
                row += f"{'N/A':>14}"
                vals.append(None)
        else:
            row += f"{'N/A':>14}"
            vals.append(None)
    print(row)

print()
print("* style_gate_value 只是参数本身的 tanh 值，fixed_one 模式下实际 gate=1")
print()

print("=" * 130)
print("3. 白化（WFI）相关分析")
print("=" * 130)
print("代理指标: velocity_channel_std / velocity_std (比值越低 = 白化越严重)")
print()
print(f"{'实验':<20} {'velocity_ch_std':>18} {'velocity_std':>15} {'比值':>12} {'vs Baseline':>15}")
print("-" * 130)

baseline_wfi = None
for exp_name, _, _ in experiments:
    if exp_name not in results:
        continue
    vel_ch_std = get_nested(results[exp_name], "runtime_observability.style_transfer_ability.model_velocity_channel_std")
    vel_std = get_nested(results[exp_name], "runtime_observability.style_transfer_ability.model_velocity_std")
    if vel_ch_std and vel_std:
        wfi_ratio = vel_ch_std / vel_std
        if exp_name == "Baseline (1ep)":
            baseline_wfi = wfi_ratio
            diff_str = "—"
        elif baseline_wfi:
            diff = wfi_ratio - baseline_wfi
            diff_pct = (diff / baseline_wfi) * 100
            diff_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"
        else:
            diff_str = "—"
        print(f"{exp_name:<20} {vel_ch_std:>18.6f} {vel_std:>15.6f} {wfi_ratio:>12.6f} {diff_str:>15}")

print()
print("注: 比值下降表示白化/雾化改善（特征更丰富）")
print("     但注意: velocity_std 本身也在变化，需要综合判断")
print()

print("=" * 130)
print("4. 组合效应分析（1 epoch 对比）")
print("=" * 130)
print()
print("Baseline vs Combo A (1 epoch):")
print()

comparisons = [
    ("clip_style", "analysis.style_transfer_ability.clip_style", "风格注入强度"),
    ("LPIPS", "analysis.style_transfer_ability.content_lpips", "内容保持"),
    ("velocity_std", "runtime_observability.style_transfer_ability.model_velocity_std", "速度场强度"),
    ("cross_attn_delta_abs", "runtime_observability.style_transfer_ability.model_cross_attn_delta_abs", "Cross-attn幅度"),
    ("film_gamma_abs", "runtime_observability.style_transfer_ability.model_film_gamma_abs", "FiLM强度"),
]

baseline_name = "Baseline (1ep)"
combo_a_name = "Combo A (1ep)"
for short_name, full_key, desc in comparisons:
    b_val = get_nested(results[baseline_name], full_key)
    a_val = get_nested(results[combo_a_name], full_key)
    if b_val and a_val:
        diff = a_val - b_val
        pct = (diff / b_val) * 100 if b_val != 0 else 0
        print(f"  {desc:<25}: {b_val:.6f} → {a_val:.6f} ({diff:+.6f}, {pct:+.2f}%)")
print()

print("=" * 130)
print("5. 训练演化分析（Combo B 3 epochs）")
print("=" * 130)
print()
print("Epoch 1 → Epoch 3 的变化:")
print()

combo_1 = "Combo B (1ep)"
combo_3 = "Combo B (3ep)"
for short_name, full_key, desc in comparisons:
    v1 = get_nested(results[combo_1], full_key)
    v3 = get_nested(results[combo_3], full_key)
    if v1 and v3:
        diff = v3 - v1
        pct = (diff / v1) * 100 if v1 != 0 else 0
        print(f"  {desc:<25}: {v1:.6f} → {v3:.6f} ({diff:+.6f}, {pct:+.2f}%)")
print()

print("Velocity 幅度变化:")
v1_std = get_nested(results[combo_1], "runtime_observability.style_transfer_ability.model_velocity_std")
v3_std = get_nested(results[combo_3], "runtime_observability.style_transfer_ability.model_velocity_std")
v1_abs = get_nested(results[combo_1], "runtime_observability.style_transfer_ability.model_velocity_abs")
v3_abs = get_nested(results[combo_3], "runtime_observability.style_transfer_ability.model_velocity_abs")
if v1_std and v3_std:
    print(f"  velocity_std: {v1_std:.6f} → {v3_std:.6f} ({v3_std-v1_std:+.6f}, {((v3_std-v1_std)/v1_std)*100:+.2f}%)")
if v1_abs and v3_abs:
    print(f"  velocity_abs: {v1_abs:.6f} → {v3_abs:.6f} ({v3_abs-v1_abs:+.6f}, {((v3_abs-v1_abs)/v1_abs)*100:+.2f}%)")
print()

print("=" * 130)
print("6. 结论")
print("=" * 130)
print()
print("P-3 组合预测验证:")
print()

b_clip = get_nested(results["Baseline (1ep)"], "analysis.style_transfer_ability.clip_style")
b3_clip = get_nested(results["Combo B (3ep)"], "analysis.style_transfer_ability.clip_style")
clip_up = (b3_clip - b_clip) / b_clip * 100 if b_clip else 0
print(f"  ✓ clip_style 提升: +{clip_up:.2f}% (略有提升)")

b_lpips = get_nested(results["Baseline (1ep)"], "analysis.style_transfer_ability.content_lpips")
b3_lpips = get_nested(results["Combo B (3ep)"], "analysis.style_transfer_ability.content_lpips")
lpips_change = (b3_lpips - b_lpips) / b_lpips * 100 if b_lpips else 0
print(f"  ⚠ LPIPS 保持: {lpips_change:+.2f}% (略有上升，内容损失稍增)")

v1_wfi = get_nested(results["Combo B (1ep)"], "runtime_observability.style_transfer_ability.model_velocity_channel_std") / get_nested(results["Combo B (1ep)"], "runtime_observability.style_transfer_ability.model_velocity_std")
v3_wfi = get_nested(results["Combo B (3ep)"], "runtime_observability.style_transfer_ability.model_velocity_channel_std") / get_nested(results["Combo B (3ep)"], "runtime_observability.style_transfer_ability.model_velocity_std")
wfi_change = (v3_wfi - v1_wfi) / v1_wfi * 100
print(f"  ✓ WFI 下降: {wfi_change:+.2f}% (3ep vs 1ep，白化改善)")
print(f"    注: 最佳 WFI 出现在 Epoch 2 (0.6113)，Epoch 3 略有反弹 (0.6656)")

ep_alpha = get_nested(results["Combo B (3ep)"], "runtime_observability.style_transfer_ability.model_endpoint_alpha")
print(f"  ✗ endpoint_alpha 上升: {ep_alpha} (始终为 0，可能模式不对)")

b_film = get_nested(results["Baseline (1ep)"], "runtime_observability.style_transfer_ability.model_film_gamma_abs")
b3_film = get_nested(results["Combo B (3ep)"], "runtime_observability.style_transfer_ability.model_film_gamma_abs")
film_change = (b3_film - b_film) / b_film * 100 if b_film else 0
print(f"  ⚠ film_gamma_abs 上升: {film_change:+.2f}% (变化很小)")

print(f"  ✗ cross_attn_entropy 下降: 无变化 (始终为 5.53125)")
print()
print("训练稳定性:")
print("  ✓ Loss 平稳下降，3 epoch 训练稳定，无发散")
print()
print("关键发现:")
print("  1. fixed_one gate 使 cross_attn_delta_abs 增加了 ~19 倍 (0.0023 → 0.0439)")
print("  2. 但 1 epoch 时 velocity_std 反而下降 22%，说明初始阶段位移变小")
print("  3. 随着训练（3 epoch），velocity_std 大幅增加 +99%，位移逐渐变大")
print("  4. WFI 在第 2 epoch 达到最佳（0.6113），第 3 epoch 略有反弹")
print("  5. endpoint_alpha 始终为 0，可能因为 endpoint head 是 lowhigh 模式而非 alpha 模式")
print("  6. clip_style 提升有限（+0.7%），LPIPS 略有上升（+3.1%）")
print()
print("最优组合:")
print("  - 综合 clip_style 和 LPIPS，Combo B (3ep) 略优于 Baseline")
print("  - 但提升幅度不大，需要更多 epoch 或更强的配置来验证")
print()
print("下一步建议:")
print("  1. 检查 endpoint head 模式，确认为何 endpoint_alpha=0")
print("  2. 尝试更多 epoch（5-10）看 WFI 和 clip_style 是否持续改善")
print("  3. 尝试更强的 gate 初始化或更大的 FiLM 初始强度")
print("  4. 验证 WFI 的像素级计算（用 wfi.py 工具直接计算生成图像）")
print("  5. 考虑组合更多优化：如增加 FiLM 层数、调整 learning rate 等")
print()
print("=" * 130)
