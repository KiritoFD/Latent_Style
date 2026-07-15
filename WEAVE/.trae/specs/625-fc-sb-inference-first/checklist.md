# 625 FC-SB推理优先实验探索 - Verification Checklist

## 代码准备阶段
- [x] exp/625_fc_sb/ 目录结构创建完成（configs/子目录）
- [x] run_fc_inference_sweep.py 推理参数扫描脚本编写完成，支持sigma/kernel/fc_modes/sigma_schedule多值扫描
- [x] gen_configs.py 批量配置生成脚本编写完成（累积增量设计）
- [x] G0_baseline.json 干净基线配置创建完成，路径适配WSL，FC开关全关，Base Locking ON
- [x] G1-G7 七个FC增量配置创建完成，链式增量参数差异符合预期
- [x] run_all.sh 批量运行脚本编写完成（推理优先流程，错误处理）
- [x] summarize_results.py Pareto结果分析脚本编写完成（pandas+matplotlib）

## 代码审查阶段
- [x] 所有Python脚本通过 py_compile 语法检查
- [x] G0-G7配置参数矩阵验证通过（ep/noise/foep/kernel/path_mode/sigma/sched）
- [x] model620.py中Base Locking确认：bridge_path_mode=="vertical"时 h = x_base_lock + (h - lp(h))
- [x] model620.py中Fiber SDE噪声确认：noise_fiber = noise - lp(noise)，sigma_base>0且fiber_proj_noise=true时注入
- [x] model620.py中Fiber Velocity Projection确认：v_fiber = v_pred - lp(v_pred)，fiber_proj_ep=true时启用
- [x] model620.py中Fiber-Only Endpoint确认：ep_fiber = ep - lp(ep), endpoint = x_base_now + ep_fiber
- [x] model620.py中Curriculum sigma schedule确认（t<0.33: 0.25σ, t<0.66: 0.6σ, else: 1.0σ）
- [x] model620.py中self.model_cfg和self.bridge_cfg属性存在，apply_inference_overrides可正确设置
- [x] bridge_path_mode在apply_inference_overrides中强制为"vertical"
- [x] load_vae参数顺序修复（device=device, model_id=args.vae_model）
- [x] VAE scaling逻辑正确（scale_in/scale_out，无双重缩放）
- [x] bridge_sigma同时设置model.bridge_sigma和bcfg.bridge_sigma

## 0训练成本推理扫描阶段（远程执行）
- [ ] 远程3060 WSL环境可访问，I盘挂载正常
- [ ] 代码同步到远程，CHECKPOINT_BASE指向有效checkpoint目录
- [ ] HF_HOME和TRANSFORMERS_CACHE环境变量正确设置
- [ ] bash exp/625_fc_sb/run_all.sh 运行成功
- [ ] σ=[0.02,0.04,0.06,0.08,0.10] × kernel=[5,7] × fc_modes=[none,ep,ep_noise,ep_noise_foep] 扫描完成
- [ ] 生成 inference_sweep.json 结果文件
- [ ] summarize_results.py 成功生成 results.csv, pareto_optimal.csv, pareto.png
- [ ] Pareto前沿上至少有一个点clip_style>0.71

## 训练阶段（根据推理结果决定）
- [ ] 如需训练，G0基线3 epochs训练完成，无OOM
- [ ] G0评估完成：clip≈0.70±0.01, LPIPS≈0.34±0.02
- [ ] 根据Pareto结果选择2-3个最优配置训练
- [ ] 选定配置训练完成并评估

## 结果分析阶段
- [ ] 帕累托前沿图生成并分析
- [ ] 每个FC机制的独立贡献量化（none→ep→ep_noise→ep_noise_foep的边际增益）
- [ ] 最优σ和kernel确定
- [ ] 铜牌标准验证：是否达到clip_style>0.72且LPIPS<0.40
- [ ] 金牌标准验证：是否达到clip_style>0.73且LPIPS<0.35
- [ ] 下一轮实验建议（如需要：wavelet lowpass, CFG外推, σ=0.12等）
