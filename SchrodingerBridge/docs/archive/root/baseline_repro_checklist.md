# WEAVE — Strong Baseline Reproduction Checklist

> 规则：**未复现的内容一律不写进 `paper.tex`**，只记在此清单。
> 图例：`[done]` 已复现 · `[todo]` 待复现 · `[blocked]` 无公开代码/不可行 · `[check]` 待核实代码

---

## 0. 三个目标数据集（已确认）

| 简称 | 全称 | 风格 | 分辨率 | 对数 |
|------|------|------|--------|------|
| **D5** | Distinct5 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | 512 | 750 (5×5×30) |
| **P2A** | Photo2Art-256 | cezanne, Hayao, monet, photo, vangogh | 256 | 750 |
| **R5** | Random5 / WikiArt20 hold-out | Cubism, Expressionism, Pop_Art, Romanticism, Symbolism | 512 | 750 |

本地副本：`datasets/wikiart5_test`(D5)、`datasets/p2a_test`(P2A)、`datasets/wikiart20_test`(R5)。

---

## 1. 已复现（用户确认 / 代码已在 `tools/`）

- [done] CUT（P2A-256，DINO-C/DINO-S = 0.702 / 0.539）
- [done] AdaIN / WCT（VGG19）
- [done] **S2WAT**（AAAI 2024，Zhang et al.）— 全部/部分复现
- [done] **StyleID**（CVPR 2024）— 已复现
- [done] **StyleAligned**（2024）— 已复现（D5 本地跑完；P2A+R5 走远程）
- [done] SaMam / SaMST — `tools/samam_distinct5_scratch/` 已有代码与结果（确认后标 done）

---

## 2. 用户指定：StyDiff

- [blocked] **StyDiff**（Scientific Reports 2025, diffusion + AdaIN）— **无公开 GitHub 代码**（仅 Nature 论文 + 摘要站）。
  要从零复现需自己实现 diffusion+AdaIN 管线；本地 GPU = RTX 4070 Laptop 8GB（跑 SD 偏紧）。→ 跳过。

---

## 3. 2025 / 2026 更新的工作（替代 StyDiff 或补 Related Work）

> **网络约束（已实测）**：本地 Windows 无法直连 github.com（443 被拒）；仅 `ghproxy.net` 代理的
> `raw.githubusercontent.com` 与 git 协议可用，但 git clone 在 300s 工具上限内拉不完，且 codeload/API 被代理 403。
> 因此**本地无法直接获取新仓库代码**；新方法的复现应在远程机（RTX 3060 12GB，已有 GitHub 访问与 baseline 基础设施）进行。

- [blocked] **FAST** — Flexibly Controllable AST via Latent Diffusion, ACM ToMM 2025. 仓库 `github.com/wd1511/FAST`
  **实测：main 分支仅有 README.md + environment.yml，无任何可运行源码**（项目页也只指向同一仓库）。
  → 公开仓库无可复现代码，**跳过**（与用户"没有的跳过"一致）。
- [check] **TransferAnything** — Frequency-Aware AST via self-attention constraints in latent space, IEEE 2026.
  **与本文频域主题最贴合**，待核实是否有公开可运行代码。
- [check] **"Compressing Image Style Training into a Single Model Forward"** — arXiv:2606.13809, 2026.
  轻量"单次前向"风格训练，主题贴合，待核实代码。
- [check] **DGPST** — Domain Generalizable Portrait ST, ICCV 2025, 用 AdaIN-Wavelet 潜空间初始化。
  与本文 Haar+AdaIN 近邻，待核实代码（注意：人像专用，可比性受限）。
- [check] **SPAST** — Arbitrary ST with Style Priors, arXiv:2505.08695, 2025. 待核实代码。
- [info] StyleStudio — Text-Driven ST w/ Selective Control, CVPR 2025（文本驱动，可比性较弱）。
- [info] Style Transfer: A Decade Survey, arXiv:2506.19278, 2025（综述，作 Related Work 引用）。

---

## 4. 人类偏好验证（化解"换指标才变好"质疑）

- [todo] Pairwise A/B 人类偏好研究：Ours vs 各强 baseline，作为 IDT 自动指标之外佐证。

---

## 5. 下一步决策（待用户确认）

- 本地 GPU：RTX 4070 Laptop，**8GB**（约 3.6GB 空闲）。
- StyDiff 无代码 → 建议改复现 **FAST**（有代码，2025）或另一有代码的 2025/26 方法，在 D5/P2A/R5 上跑。
- 或：仅把 2025/26 新工作补进 Related Work + 本清单，暂不跑复现。
