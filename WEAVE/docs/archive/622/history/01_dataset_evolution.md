# 数据集演变史

## 总览

| 阶段 | 时间 | 数据集 | 分辨率 | 风格数 | 格式 | VAE |
|------|------|--------|--------|--------|------|-----|
| V0 | 2026-01-13 | WikiArt子集 | 64×64×4 | ~9 | .pt文件 | SD1.5 VAE (0.18215 scale) |
| V1 | 2026-01-15 | monet2photo | 64×64×4 | 2 (monet, photo) | .pt文件 | SD1.5 VAE |
| V2 | 2026-01-28 | CycleGAN-T | 32×32×4 | 4 (monet, photo, vangogh, cezanne) | latent | SD1.5 VAE |
| V3 | 2026-03-29 | 5-style自定义 | 16×16×4 (body) | 5 (photo, Hayao, monet, vangogh, cezanne) | latent, GPU预加载 | SD1.5 VAE |
| V4 | 2026-05-07 | horse2zebra | 64×64×4 | 2 | .pt文件 | SD1.5 VAE |
| V5 | 2026-05-19 | SB标准数据集 | 64×64×4 | 5 (photo, Hayao, monet, vangogh, cezanne) | AdaCUTLatentDataset | SD1.5 VAE |
| V6 | 2026-05-30 | Tokenizer数据集 | 64×64×4 | 5 | latent | SD1.5 VAE |
| V7 | 2026-06-05 | WikiArt512 (SaMAM/SaMST) | 512px | 5+ | per-style models | SD1.5 VAE |
| V8 | 2026-06-11 | WikiArt5 (SaMST新数据集) | 512px | 5 (Baroque, Impressionism, Cubism, Symbolism, Art_Nouveau) | per-style segmented | SD1.5 VAE |
| V9 | 2026-06-17 | WikiArt OT实验 | 64×64×4 | 5+ | latent | SD1.5 VAE |
| V10 | 2026-06-19 | 620 Spatial Bridge | 64×64×4 | open-set (DINO条件) | StyleConditioner620 | SD1.5 VAE |

## 详细演变

### V0: WikiArt子集 (2026-01-13)
- **Commit**: `c7547f456` "encode wikiarts"
- **内容**: 32,000+ .pt文件，WikiArt子集包括Baroque, Color_Field_Painting等
- **问题**: 多风格联合训练效果差，类别不平衡严重
- **后续**: 被monet2photo替代

### V1: monet2photo (2026-01-15)
- **Commit**: `810cdc32f2` "transformer fail"
- **内容**: 2类（Monet绘画 ↔ 照片），CycleGAN标准数据集
- **位置**: `F:\monet2photo\latents\monet2photo`
- **编码**: SD1.5 VAE latent scale 0.18215
- **优点**: 简单2类对比，训练稳定
- **缺点**: 只有2个风格，风格迁移范围太窄
- **遗留**: 这个数据集一直被保留，horse2zebra变体在5月也被测试

### V2: CycleGAN-T (2026-01-28)
- **Commit**: `c04376700f` "加了cross_attn风格强多了"
- **内容**: 4类（monet, photo, vangogh, cezanne）
- **位置**: `../../data/latents` 子目录
- **变化**: 风格数从2→4，分辨率从64→32
- **权重**: style_weights=[2.0, 0.5, 0.5, 2.0]（Monet和Cezanne upweighted）
- **问题**: 4类仍不够多样，32×32分辨率限制了细节

### V3: 5-style自定义 (2026-03-29)
- **Commit**: `60b3bfef86` "加入attention效果明显"
- **内容**: 5类（photo, Hayao, monet, vangogh, cezanne）
- **位置**: `../../latent-256`，256px latent
- **特点**: GPU预加载（`preload_to_gpu: true`），batch=256
- **分辨率**: body工作在16×16×4
- **优点**: 更多风格，更大batch，GPU预加载加速
- **问题**: "num_style刚才居然是写错了"（commit `7535a9c3b7`），实际写了错误数字但影响不大

### V4: horse2zebra (2026-05-08)
- **Commit**: `af9d0b2384` "改了backbone"
- **用途**: 跨数据集验证，测试SB在非艺术风格上的泛化性
- **结果**: 不如艺术风格数据集，跨域泛化困难
- **后续**: 放弃此方向

### V5-V6: SB标准 + Tokenizer (2026-05-19 ~ 05-30)
- **内容**: 5类（同V3），但通过`AdaCUTLatentDataset`加载
- **Tokenizer**: factorized style tokenizer (identity 24d + texture 32d + geometry 24d)
- **位置**: 共享远程数据路径
- **问题**: low-cell probe失败（`4a5f9f9316`），弱源/目标cell的重采样无收益

### V7: WikiArt512 SaMAM/SaMST (2026-06-05)
- **Commit**: `7114e14cf` "Land remote 3060 SaMST step packet"
- **内容**: 512px WikiArt，每风格单独训练
- **模型**: SaMAM (Style-adaptive Multi-scale Attention Model), SaMST (Style-adaptive Multi-scale Style Transfer)
- **结果**: SaMST grand average clip_style=0.7597, LPIPS=0.3374（per-style模型，非通用）
- **问题**: 3060 12G显存严重限制batch size和训练速度

### V8: WikiArt5 SaMST新数据集 (2026-06-11)
- **Commit**: `6f4b88c47` "samam 新数据集复现"
- **风格**: Baroque, Impressionism, Cubism, Symbolism, Art_Nouveau_Modern
- **训练策略**: segmented resume to convergence（分段训练到收敛）
- **结果**:
  - Baroque: clip_style=0.7234, LPIPS=0.2939
  - Impressionism: clip_style=0.7361, LPIPS=0.2815
  - Cubism: clip_style=0.7766, LPIPS=0.4270
  - Symbolism: clip_style=0.7929, LPIPS=0.3339
  - Art_Nouveau: clip_style=0.7694, LPIPS=0.3509

### V9: WikiArt OT实验 (2026-06-17)
- **Commit**: `828151b2d` "Fix: bridge_vertical_base_stride"
- **virtual_length**: 0.1 (大幅减少数据量)
- **问题**: OT匹配在latent space中不稳定，PureLatentSpatial tokenizer零ROI
- **结论**: OT需要结构指纹而非视觉相似度

### V10: 620 Spatial Bridge (2026-06-19)
- **Commit**: `d94b5d4f6` "Add 620 spatial bridge mainline"
- **突破**: **open-set风格**，不再需要固定风格数
- **方式**: DINOv2 patch tokens作为style条件，reference image直接编码
- **数据**: WikiArt DINO cache, 风格caption路径
- **配置**: `num_memory_tokens=256` (DINO patches), `style_caption_path`处理修正
- **WFI评估**: 在wikiart上测量白化程度

## 数据集选择的关键教训

1. **2风格太窄，5风格刚好，open-set才是目标** — monet2photo→CycleGAN-T→5-style→DINO open-set
2. **分辨率很重要** — 32×32→64×64→512px，低分辨率latent丢失风格细节
3. **GPU预加载是大batch的前提** — batch 12→96→256→4(620)
4. **数据量不是瓶颈，表达力才是** — virtual_length=0.1就够训练
5. **Per-style模型的天花板是通用模型的参考基线** — SaMST 0.76 vs 620 0.67
