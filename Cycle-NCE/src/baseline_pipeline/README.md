# Baseline Reproduction Pipeline
For AAAI 2027 Paper Experiments - Fair Comparison Baseline Suite

## ⚠️ Hardware Requirements
- GPU with ≥8GB VRAM (optimized for RTX 3060)
- 16GB+ System RAM
- 50GB+ Disk Space for models and datasets

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Run setup script (Windows)
.\setup.ps1

# Or manual setup
pip install -r requirements.txt
```

### 2. Data Configuration
**不需要额外准备数据，自动复用项目现有数据集：**
- 测试集：`/style_data/overfit50/` (30张content，每个style30张测试图)
- 训练集：`/style_data/` (全量风格训练数据)
```
Latent_Style/
└── style_data/
    ├── overfit50/         # 统一测试集
    │   ├── photo/         # 30 test content images
    │   ├── monet/
    │   ├── vangogh/
    │   └── ...
    ├── monet/             # 全量训练集
    ├── vangogh/
    └── ...
```

### 3. Run Zero-shot Baselines (Fast, ~1 hour total)
```powershell
# Run StyleID and CycleGAN-Turbo on Monet style
python main.py --baselines styleid cyclegan_turbo --styles monet

# Run on all styles
python main.py --baselines styleid cyclegan_turbo --styles monet vangogh ukiyoe cezanne
```

### 4. Run Training Baselines (Slow, multiple days)
```powershell
# Run CUT training and evaluation for Monet
python main.py --baselines cut --styles monet
```

### 5. Evaluate Only (if you already have generated images)
```powershell
python main.py --baselines styleid --styles monet --skip-generation
```

## 📊 Metrics
All baselines are evaluated on these standardized metrics:
| Metric | Description |
|--------|-------------|
| **CMMD** | Clustered Mean Maximum Distance (replaces FID, lower = better style alignment) |
| **LPIPS** | Learned Perceptual Image Patch Similarity (lower = better content preservation) |
| **DINO_Struct** | DINOv2 feature similarity (higher = better structure preservation) |
| **CLIP_Style** | CLIP similarity to style references (higher = better style transfer) |
| **CLIP_Content** | CLIP similarity to original content (higher = better content preservation) |

## 🎯 Baseline Overview

### Zero-shot (No Training Required)
| Baseline | Year | Speed | VRAM Usage | Description |
|----------|------|-------|------------|-------------|
| StyleID | 2024 | ~30s/img | ~6GB | Diffusion-based zero-shot style transfer with attention injection |
| CycleGAN-Turbo | 2024 | ~0.1s/img | ~4GB | 1-step distilled diffusion for domain-specific translation |

### Training Required
| Baseline | Year | Train Time | VRAM Usage | Description |
|----------|------|------------|------------|-------------|
| CUT | 2020 | ~1-2 days/style | ~6GB | Classic contrastive unpaired translation baseline |
| S2WAT | 2024 | ~2 days | ~8GB | Wavelet transformer based style transfer |
| StyleAligned | 2024 | ~1 day | ~7GB | Diffusion with attention sharing + ControlNet |
| B-LoRA | 2024 | ~15min/style | ~7GB | SDXL LoRA fine-tuning per style |

## 📁 Directory Structure
```
baseline_pipeline/
├── baselines/              # Cloned baseline repositories
├── checkpoints/            # Model checkpoints
├── datasets/               # Test datasets
├── evaluation/             # Unified evaluation scripts
├── results/                # Generated images and metrics
├── scripts/                # Individual baseline execution scripts
├── utils/                  # Shared utilities
├── main.py                 # Main orchestration script
├── setup.ps1               # Windows setup script
└── requirements.txt        # Dependencies
```

## ⚡ Optimization for 8GB VRAM
All scripts include these memory optimizations:
- Reduced batch sizes (1-4)
- Attention slicing
- xFormers memory efficient attention
- VAE slicing
- Gradient checkpointing (for training)
- Aggressive CUDA cache cleaning

## 📝 Paper Citation Format
For baselines without available code:
> "For closed-source state-of-the-art models (e.g., UltraStyle and SaMST), we adopt the same evaluation protocol and test set as described in [X], and directly report their published metrics for fair comparison."

For all other baselines:
> "All baseline models were either executed using official pre-trained checkpoints or retrained on our dataset following the optimal configurations provided in their original repositories."
