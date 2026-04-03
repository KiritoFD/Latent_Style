# Pre-Feb-8 Detailed Code History

**Data Source**: `C:\Users\xy\repo.git` (Python-only history, 186 commits)
**Scope**: Jan 13 - Feb 7, 2026 (54 commits, BEFORE `Cycle-NCE/src/` existed)
**Project Names**: `DiT` → `Thermal` → `Cycle-NCE` transition

---

## Table of Contents
1. Phase 1: Data Preparation (Jan 13-15)
2. Phase 2: DiT Experiments (Jan 18-19)
3. Phase 3: Thermal Project Emerges (Jan 22-24)
4. Phase 4: LGT Theory & Verification (Jan 26-27)
5. Phase 5: Thermal Model Iterations (Jan 27-30)
6. Phase 6: Cross-Attention Era & Rollback (Jan 28-31)
7. Phase 7: Stability & Infra Optimization (Jan 30 - Feb 4)
8. Phase 8: Pre-February 8 Final Push (Feb 7)

---

## 1. Phase 1: Data Preparation (Jan 13-15)

**Project State**: No model exists yet, only data encoding scripts.

### Jan 13 - encode wikiarts (991f001)
- `debug_metadata.py` (25 lines)
- `encode.py` (395 lines)
  - Download WikiArt from HuggingFace
  - Filter styles (min 50 images, max 1000 per style)
  - VAE encoding: `stabilityai/sd-vae-ft-mse`, 512x512, batch size 4
  - Save to `./wikiart_latents/` as `.pt` files
  - FP16 for GPU, auto batch size detection
  - Reconstruction verification

### Jan 15: Infrastructure Addition
- `preprocess_latents.py` (169 lines) - Latent preprocessing
- `upscale.py` (178 lines) - Upscaling utilities
- `verify_lgt.py` (373 lines) - LGT (Latent Geometry Theory?) verification
- `plot_lgt_theory.py` (83 lines) - LGT theory plotting

**Key Files at this stage**: `encode.py`, `verify_lgt.py`, `upscale.py`

---

## 2. Phase 2: DiT Experiments (Jan 18-19)

### Jan 18: export (ea64eb6)
- Added OT (Optimal Transport) and android infra

### Jan 19: DiT Framework Added
**New Directory**: `DiT/` with full Diffusion Transformer implementation
- `DiT/train.py` (411 lines) - DiT training loop
- `DiT/models/dit_model.py` (184 lines) - DiT architecture
- `DiT/inference.py` (241 lines) - DiT inference
- `src/` - Initial source code directory
- Jan 19 18:07: "epoch 19 looks grate; should look at CUT" - Early success, looking at CUT method

**Commit**: 9798d90 - DiT experiments complete but likely unsatisfactory

### Jan 20-21: Loss Design Exploration
- `3ff5134` (Jan 20 17:00): identity & WSL (Windows Subsystem for Linux?) infra
- `ad93447` (Jan 21 11:53): "条件失效" - Conditional signal failure
- `7f30811` (Jan 20 22:39): auto search
- `492144a` (Jan 21 16:35): "SAF requires loss design and does not work,trying sth else" - SAF failed

**SAF (Style-Attention-Fusion?) was the first major technical dead end.**

---

## 3. Phase 3: Thermal Project Emerges (Jan 22-24)

**"Thermal" replaces DiT as the main codebase.**

### Jan 22: Infrastructure Leap
**New Files**:
- `Thermal/src/__init__.py` (47 lines)
- `Thermal/src/model.py` (605 lines) - First model definition
- `Thermal/src/losses.py` (415 lines) - First loss functions
- `Thermal/src/inference.py` (655 lines) - Inference pipeline
- `Thermal/src/trainer.py` - Training loop (initial version)

**Commit**: a2f7dca - "拆分了过长的train.py"
- Split monolithic training code into modular structure
- `Thermal/src/checkpoint.py` (293 lines)
- `Thermal/src/dataset.py` (175 lines)
- `Thermal/src/utils/` - Utility functions

### Jan 22-24: Early Loss Experiments
- `0e10148` (Jan 22 00:32): thermal dynamics
- `92ccc90` (Jan 22 00:51): infra & eval fix
- `9e8a72c` (Jan 22 09:59): "大batch下过度平均化，调整" - Large batch over-averaging
- `a3d579b` (Jan 22 01:14): "infra leap : 240 batchsize, group convolution + gradient checkpoint"

### Jan 23-24: Loss Refinement
- `cdb2389` (Jan 23 23:00): "自动定量评估" - Automatic quantitative evaluation
- `3a2a0ce` (Jan 24 20:34): "loss of details"
- `5a0d31a` (Jan 24 23:11): "增加MSE辅助保留内容，速度正则化解决亮度变化"

**Key insight**: Already dealing with brightness drift and detail loss - problems that persist throughout the project.

---

## 4. Phase 4: LGT Theory & Verification (Jan 26-27)

### Jan 26: Patch Size Exploration
- `0c8b670` (Jan 26 18:58): "patch3,7曲线离得很近，需要给大权重拉开"
  - Testing different patch sizes for SWD/Gram
  - Need larger weights to separate performance curves

### Jan 27: Major Research Day (8 commits!)
This was a CRITICAL research day with massive effort:

- `ed4c2ea` (00:55): "训练代理网络用于滤波" - Proxy network for filtering
  - Proxy Benchmark Report: IoU: 0.247 ± 0.048 ❌ FAILURE
  - "Proxy failed to learn"

- `49d3ecb` (00:38): "验证CNN滤波，不好，需要改为可学习CNN"
  - CNN filtering validation failed

- `8434a0d` (00:27): "一直在动亮度是高频没了；尝试视频，全是飘动色块。验证脚本，潜空间上的FFT不行"
  - **Key finding**: Brightness drift is due to high-frequency loss
  - Video generation showed "floating color blocks"
  - Latent space FFT rejected

- `1493ea0` (10:47): "verified lora on latents" - 4 tests passed:
  - ✅ Zero Initialization Safety (Max Diff: 0.000000000)
  - ✅ Batch Independence (Output Discrepancy: 0.773442)
  - ✅ Gradient Flow (Style Gradient Norm: 0.002790)
  - ✅ Memory Overhead (Base: 262,656 params, HyperNet: 2,171,136 params, 8.27x)

- `c03eaed` (11:22): "Verified Edge IoU: 0.5412 ✅ Correct: Identity path with LoRA perturbation maintains structure"
  - Edge IoU = 0.5412 - Structure maintained with LoRA

### LoRA Experiment Failed
- `7396e04` (17:00): "LORA没加进去，风格学的意外的好，就是图片质量不太好"
  - **LORA could NOT be integrated**
  - But style learned unexpectedly well (without LoRA?)
  - Image quality not good

---

## 5. Phase 5: Cross-Attention First Attempt (Jan 28-31)

### Jan 28: The First Cross-Attn Attempt
- `780d7e6` (11:15): "250 epoch，有提升但是噪点严重，改变太弱了"
  - 250 epochs showed improvement but severe noise
  - Changes too weak

- `b31c5ca` (14:20): **"风格强多了，加了cross_attn，encoder里面也加了AdaGN，原来的配置权重需要调整"**
  - **Cross-Attention INTRODUCED here!** (First attempt)
  - Encoder also got AdaGN
  - Style much stronger
  - Need to adjust weights (original configs incompatible)

- `8097ef7` (16:47): **"MSE完全爆炸"** - MSE completely exploded
  - The first cross-attn version destabilized training

### Jan 28 23:20: train.py Split
- `a2f7dca`: "拆分了过长的train.py" - Split the monolithic train.py
  - Moved to modular structure with separate model/loss/trainer files

### Jan 29-30: Stabilization Attempts
- `0527849` (Jan 29 00:24): "100epoch，很稳定，准备加强风格"
  - 100 epochs stable, ready to strengthen style
- `ca1371a` (Jan 29 10:54): "400epoch已经收敛" - Converged at 400 epochs
- `dfab37e` (Jan 29 11:26): "评估分频方式" - Evaluation frequency method
- `301440c` (Jan 30 00:24): "优化eval部分的infra，loss部分计算转为Conv避免unfold节省显存带宽，推理batch化"
  - Eval infra optimization
  - Loss computation changed from unfold to Conv (save VRAM)
  - Inference batched

- `f838690` (Jan 30 17:04): "修复从ckpt重训的问题；余弦退火调度lr"
  - Fixed resume-from-checkpoint issue
  - Cosine annealing LR scheduler added

- `9863944` (Jan 30 17:51): "两种Loss的权重归一化"
  - Loss weight normalization

### Jan 31: Cross-Attn ROLLBACK
After 3 days of cross-attn experiments:

- `ec1ac25` (16:59): "减小LR，增大MSE权重" - Reduce LR, increase MSE weight
- `59cffe2` (17:39): **"去掉cross_attn，用回AdaGN"**
  - **Cross-Attention REMOVED, back to AdaGN**
  - This was the first rollback

- `9b2be2e` (19:20): "去掉无效的CA Grad;1259的patch size是正确的"
  - CA (Cross-Attention) gradient ineffective
  - Correct patch size is 1259

- `1a8b94b` (22:27): **"Cross-Attention 在纯风格迁移中可能导致内容语义的过度纠缠（过拟合语义而非纹理）"**
  - **CRITICAL INSIGHT**: Cross-attention in pure style transfer causes content semantic over-entanglement (overfitting semantics rather than texture)
  - This insight will drive the next phase back to AdaGN

---

## 6. Phase 6: Thermal Model Optimization (Feb 1-4)

### Feb 1: Capacity & Channel Tweaks
- `d2ec738` (00:43): "调整推理batch为24" - Inference batch adjusted to 24
- `4a1555a` (15:58): "增大通道宽度" - Increase channel width
- `aabaadc` (17:17): "提升网络容量 减小层数" - Increase network capacity, reduce layers

### Feb 4: CNN Classifier
- `400e896` (13:50): "CNN分类器评估，效果很差" - CNN classifier evaluation, very poor results

### Feb 6: Code Simplification
- `8360a4c` (22:23): **"简化代码，修正训练目标：结构和风格损失不再对抗"**
  - CODE SIMPLIFIED
  - Training objectives corrected: **Structure and style losses no longer opposing each other**
  - This is a fundamental breakthrough - before this, structure and style were competing

### Feb 7: Baseline Established
- `9254f30` (12:10): "debug VRAM" - VRAM debugging
- `26476d0` (15:09): **"效果不理想，可作为基线"** - Results not ideal, can be used as baseline

---

## 7. Complete Commit Timeline (54 commits)

| 2026-01-13 | 21:30 | 991f001 | encode wikiarts |
| 2026-01-15 | 17:13 | ba2ac1f | transformer fail |
| 2026-01-15 | 17:59 | dfe37bc | added cfg |
| 2026-01-15 | 22:22 | 366cf20 | reflow |
| 2026-01-16 | 12:01 | 056356f | 强制数据平均分配 |
| 2026-01-16 | 21:11 | 868e88c | identity loss |
| 2026-01-18 | 21:15 | ea64eb6 | export |
| 2026-01-19 | 16:13 | ebec0eb | added OT and android infra |
| 2026-01-19 | 18:07 | 9798d90 | epoch 19 looks grate; should look at CUT |
| 2026-01-19 | 21:49 | 41d5741 | eval |
| 2026-01-19 | 23:49 | b53db2c | vgg |
| 2026-01-20 | 17:00 | 3ff5134 | identity & wsl infra |
| 2026-01-20 | 22:06 | b1a6c82 | 频谱幅度Loss |
| 2026-01-20 | 22:20 | 80b5cf8 | 减小了学习率；增大速度会跑飞，不可取 |
| 2026-01-20 | 22:39 | 7f30811 | auto search |
| 2026-01-21 | 11:53 | ad93447 | 条件失效 |
| 2026-01-21 | 16:35 | 492144a | SAF reqiures loss design and does not work,trying sth else |
| 2026-01-22 | 00:32 | 0e10148 | thermal dynamics |
| 2026-01-22 | 00:51 | 92ccc90 | infra & eval fix |
| 2026-01-22 | 01:14 | a3d579b | infra leap : 240 barchsize,group convolution+gradient checkpoint |
| 2026-01-22 | 09:59 | 9e8a72c | 大batch下过度平均化，调整 |
| 2026-01-23 | 23:00 | cdb2389 | 自动定量评估 |
| 2026-01-24 | 20:34 | 3a2a0ce | loss of details |
| 2026-01-24 | 23:11 | 5a0d31a | 增加MSE辅助保留内容，速度正则化解决亮度变化 |
| 2026-01-26 | 18:58 | 0c8b670 | patch3,7曲线离得很近，需要给大权重拉开 |
| 2026-01-27 | 00:27 | 8434a0d | 一直在动亮度是高频没了；尝试视频，全是飘动色块。验证脚本，潜空间上的FFT不行 |
| 2026-01-27 | 00:38 | 49d3ecb | 验证CNN滤波，不好，需要改为可学习CNN |
| 2026-01-27 | 00:55 | ed4c2ea | 训练代理网络用于滤波📊 Proxy Benchmark Report (N=50) -------------------------------------- |
| 2026-01-27 | 10:39 | 6dc08e4 | 100epoch best |
| 2026-01-27 | 10:47 | 1493ea0 | verified lora on latents : [Test 1] Zero Initialization Safety    Max Difference |
| 2026-01-27 | 11:22 | c03eaed | Verified Edge IoU: 0.5412 ✅ Correct: Identity path with LoRA perturbation mainta |
| 2026-01-27 | 17:00 | 7396e04 | LORA没加进去，风格学的意外的好，就是图片质量不太好 |
| 2026-01-27 | 23:20 | a2f7dca | 拆分了过长的train.py |
| 2026-01-28 | 11:15 | 780d7e6 | 250 epoch，有提升但是噪点严重，改变太弱了 |
| 2026-01-28 | 14:20 | b31c5ca | 风格强多了，加了cross_attn，encoder里面也加了AdaGN，原来的配置权重需要调整 |
| 2026-01-28 | 16:47 | 8097ef7 | MSE完全爆炸 |
| 2026-01-29 | 00:24 | 0527849 | 100epoch,很稳定，准备加强风格 |
| 2026-01-29 | 10:54 | ca1371a | 400epoch已经收敛 |
| 2026-01-29 | 11:26 | dfab37e | 评估分频方式 |
| 2026-01-30 | 00:24 | 301440c | 优化eval部分的infra，loss部分计算转为Conv避免unfold节省显存带宽，推理batch化 |
| 2026-01-30 | 17:04 | f838690 | 修复从ckpt重训的问题；余弦退火调度lr |
| 2026-01-30 | 17:51 | 9863944 | 两种Loss的权重归一化 |
| 2026-01-31 | 16:59 | ec1ac25 | 减小LR，增大MSE权重 |
| 2026-01-31 | 17:39 | 59cffe2 | 去掉cross_attn，用回AdaGN |
| 2026-01-31 | 19:20 | 9b2be2e | 去掉无效的CA Grad;1259的patch size是正确的 |
| 2026-01-31 | 22:27 | 1a8b94b | Cross-Attention 在纯风格迁移中可能导致内容语义的过度纠缠（过拟合语义而非纹理） |
| 2026-02-01 | 00:43 | d2ec738 | 调整推理batch为24 |
| 2026-02-01 | 15:58 | 4a1555a | 增大通道宽度 |
| 2026-02-01 | 17:17 | aabaadc | 提升网络容量 减小层数 |
| 2026-02-04 | 13:50 | 400e896 | CNN分类器评估，效果很差 |
| 2026-02-06 | 22:23 | 8360a4c | 简化代码，修正训练目标：结构和风格损失不再对抗 |
| 2026-02-07 | 12:10 | 9254f30 | debug VRAM |
| 2026-02-07 | 15:09 | 26476d0 | 效果不理想，可作为基线 |
| 2026-02-08 | 00:32 | 47ed14b | 6M model tryout |

---

## 8. Key Learnings from Pre-Scratch Phase

### 8.1 Technical Dead Ends
1. **SAF (Style-Attention-Fusion)**: Failed (Jan 21) - needed different loss design
2. **Proxy Network for Filtering**: IoU only 0.247 (Jan 27)
3. **CNN Filter**: Rejected, needed learnable CNN (Jan 27)
4. **Latent Space FFT**: "不行" (Jan 27)
5. **LoRA**: "没加进去" - couldn't integrate LoRA (Jan 27)
6. **DiT**: Replaced by Thermal by Jan 22
7. **Cross-Attn (first attempt)**: Rolled back due to semantic over-entanglement (Jan 31)

### 8.2 Key Breakthroughs
1. **Infra leap** (Jan 22): 240 batch size, group conv + gradient checkpoint
2. **Train.py split** (Jan 27): Modular architecture
3. **Loss weight normalization** (Jan 30): Balanced training
4. **Structure vs style no longer opposing** (Feb 6): Critical realization
5. **Edge IoU verified**: 0.5412 with LoRA perturbation (Jan 27)

### 8.3 Persistent Problems (carried forward)
1. **Brightness drift**: High-frequency loss problem
2. **Detail loss**: "loss of details" (Jan 24) persists
3. **Batch over-averaging**: Large batches cause over-averaging
4. **MSE instability**: "MSE完全爆炸" when style is too strong (Jan 28)
5. **VRAM constraints**: Constant VRAM optimization needed

### 8.4 Architecture Evolution
1. **Jan 13**: Data pipeline only (encode.py)
2. **Jan 18**: DiT experiments
3. **Jan 22**: Thermal born (model.py, losses.py, trainer.py)
4. **Jan 28**: Cross-Attn attempt (failed)
5. **Jan 31**: Rollback to AdaGN + MSE
6. **Feb 6**: Simplified code, objectives aligned
7. **Feb 7**: Baseline established

### 8.5 File Count Evolution
- Jan 13: 2 files (encode.py, debug_metadata.py)
- Jan 19: ~13 files (DiT framework added)
- Jan 22: ~19 files (Thermal framework)
- Jan 27: ~34 files (modular split)
- Jan 30: ~412 files (many .compile_cache files from torch.compile)

---

*End of Pre-Scratch Phase (Jan 13 - Feb 7)*
*Next: src/ becomes Cycle-NCE/src on Feb 8*


## 2. The "Thermal" Model Era (Jan 22 - Feb 7)

The `Cycle-NCE` architecture is the successor to a project named **Thermal**.

### 2.1 LGTUNet Origins (Jan 22)
The primary model was `LGTUNet` (approx. 605 lines).
- **Key Classes:** `TimestepEmbedding`, `StyleEmbedding`, `AdaGN`, `ResidualBlock`, `StyleResidualBlock`, `SelfAttention`, `StyleGate`.
- **Architecture:** Likely a UNet-style backbone for latent space (4x32x32), heavily relying on `AdaGN` (Adaptive Group Normalization) conditioned on style embeddings.
- **Losses:** `Thermal/src/losses.py` (415 lines) included complex loss terms, likely Gram matrices and NCE given the timeline context.

### 2.2 The "Cross-Attn Explosion" (Jan 28 - Jan 31)
A massive experiment added Transformers into the mix:
- **New Classes:** `StyleCrossAttention`, `StyleController`, `CCMLite`, `LCTXBlock`.
- **Code Size:** Ballooned from ~600 lines to **1033 lines**.
- **Outcome:** Failed. Commit messages cite "MSE explosion" and "semantic over-entanglement" (model learned to overfit semantic content rather than style texture).

### 2.3 The "Rollback" & Distillation (Jan 31 - Feb 1)
- **Feb 01:** Reverted `model.py` to a lean **217 lines**, removing all complex attention blocks.
- **Focus Shift:** Moved from architecture experiments to **training infra**:
  - Fixing checkpoint reloading.
  - Cosine annealing.
  - Optimizing eval (replacing `unfold` with `Conv` for speed).
  - Loss weight normalization.

### 2.4 The Birth of Cycle-NCE (Feb 08)
- `Cycle-NCE/src/model.py` (251 lines) was born directly from the simplified Thermal code.
- Classes: `AdaGN`, `ResBlock`, `LatentAdaCUT` (The original name of the project).
- This marked the definitive end of the DiT and pure Thermal/LGTUNet experiments.

