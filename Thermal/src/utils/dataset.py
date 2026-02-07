import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Optional ModelScope support (robust import)
try:
    from modelscope.hub import snapshot_download as ms_snapshot_download  # type: ignore
    MODELSCOPE_AVAILABLE = True
except Exception:
    try:
        import modelscope.hub as ms_hub  # type: ignore
        ms_snapshot_download = getattr(ms_hub, 'snapshot_download', ms_hub)
        MODELSCOPE_AVAILABLE = True
    except Exception:
        ms_snapshot_download = None
        MODELSCOPE_AVAILABLE = False


def _call_modelscope_snapshot(repo_id: str, dest: str):
    """
    Small helper to normalize ModelScope snapshot download calls.
    """
    if not MODELSCOPE_AVAILABLE or ms_snapshot_download is None:
        raise RuntimeError("ModelScope snapshot downloader not available")

    logger.debug(f"ModelScope object type={type(ms_snapshot_download)}")

    if callable(ms_snapshot_download):
        last_exc = None
        for attempt in (
            lambda: ms_snapshot_download(repo_id, cache_dir=dest),
            lambda: ms_snapshot_download(repo_id, dest),
            lambda: ms_snapshot_download(repo_id=repo_id, cache_dir=dest),
        ):
            try:
                return attempt()
            except TypeError as e:
                last_exc = e
                continue
        raise last_exc or RuntimeError("Callable ms_snapshot_download failed")
    else:
        func = getattr(ms_snapshot_download, 'snapshot_download', None) or getattr(ms_snapshot_download, 'download', None)
        if callable(func):
            return func(repo_id, cache_dir=dest)
        raise RuntimeError("No callable snapshot_download available in ModelScope")


def elastic_deform(x: torch.Tensor, alpha: float = 15.0, sigma: int = 3, seed: Optional[int] = None) -> torch.Tensor:
    """
    [7940HX Optimized] 弹性形变算子
    修复了归一化逻辑，增加了确定性控制。
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if x.ndim != 4:
        x = x.unsqueeze(0)

    b, _, h, w = x.shape
    device = x.device

    # 1. 噪声场生成 (中心化分布)
    dx = torch.randn(b, 1, h, w, device=device)
    dy = torch.randn(b, 1, h, w, device=device)

    # 2. 缓存友好的平滑 (替代高斯模糊)
    # 3次 AvgPool 近似高斯，在 CPU 上利用 AVX 指令集极快
    for _ in range(3):
        dx = F.avg_pool2d(dx, kernel_size=5, stride=1, padding=2)
        dy = F.avg_pool2d(dy, kernel_size=5, stride=1, padding=2)

    flow = torch.cat([dx, dy], dim=1) * alpha

    # 3. 网格构建与归一化
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
    )
    # [B, H, W, 2]
    grid_norm = torch.stack([x_grid, y_grid], dim=-1).float().unsqueeze(0).repeat(b, 1, 1, 1)
    
    # 坐标归一化: [0, W-1] -> [-1, 1]
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (w - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (h - 1) - 1.0

    # 位移归一化: 像素距离 -> 相对距离
    # ✅ 修复核心 Bug: 相对位移不需要减 1，因为它本身就是 delta
    flow_norm = flow.permute(0, 2, 3, 1)
    flow_norm[..., 0] = 2.0 * flow_norm[..., 0] / (w - 1)
    flow_norm[..., 1] = 2.0 * flow_norm[..., 1] / (h - 1)

    # 4. 采样
    return F.grid_sample(x, grid_norm + flow_norm, mode='bilinear', padding_mode='reflection', align_corners=True)


class LatentDataset(Dataset):
    """
    [SWD-laplas] 均衡化数据集
    特性：内存驻留、确定性采样、动态增强调度
    """
    def __init__(
        self, 
        data_root: str, 
        num_styles: int, 
        style_subdirs: Optional[List[str]] = None, 
        config: dict = None
    ):
        self.data_root = Path(data_root)
        self.num_styles = num_styles
        self.style_subdirs = style_subdirs or [f"style{i}" for i in range(num_styles)]
        
        # 配置读取
        train_cfg = config['training'] if config else {}
        # 🔥 Fix: Force disable elastic deformation to preserve geometric integrity of style targets
        self.apply_elastic = False # train_cfg.get('use_elastic_deform', True)
        self.base_alpha = train_cfg.get('elastic_alpha', 15.0)
        self.current_alpha = self.base_alpha
        self.current_epoch = 0

        # 数据加载
        self.style_indices: Dict[int, List[int]] = {}
        latents_list = []
        current_idx = 0

        logger.info(f"Loading {self.num_styles} styles from {self.data_root}...")
        for style_id, subdir in enumerate(self.style_subdirs):
            files = sorted((self.data_root / subdir).glob("*.pt"))
            self.style_indices[style_id] = list(range(current_idx, current_idx + len(files)))
            for f in files:
                # 移除 squeeze，保持 [C, H, W] 统一处理
                latents_list.append(torch.load(f, map_location='cpu').float().squeeze(0))
                current_idx += 1
        
        if not latents_list:
            raise RuntimeError(f"No data found in {self.data_root}")

        self.latents_tensor = torch.stack(latents_list)
        
        # 自动缩放检测
        if self.latents_tensor.std() < 0.5:
            logger.info("Auto-scaling VAE latents by 1/0.18215")
            self.latents_tensor = self.latents_tensor / 0.18215
            
        # 🚀 Infra Optimization: GPU Cache Strategy
        # 4070 Laptop (8GB) can easily hold 100k latents (~1.6GB)
        self.preload_gpu = train_cfg.get('preload_data_to_gpu', False)
        
        if self.preload_gpu and torch.cuda.is_available():
            logger.info("🚀 Pre-loading ALL latents to GPU VRAM (Zero-Copy DataLoader)")
            self.latents_tensor = self.latents_tensor.to('cuda')
        else:
            logger.info("💾 Keeping latents in CPU RAM (Pinned)")
            self.latents_tensor = self.latents_tensor.pin_memory()

        # 虚拟长度：保证每个风格都充分覆盖
        max_len = max(len(x) for x in self.style_indices.values())
        self.virtual_len = max_len * num_styles 

    def set_epoch(self, epoch: int):
        """外部调用：更新增强强度"""
        self.current_epoch = epoch
        # 动态调度：100 epoch 后开始增强力度
        scale = 1.0 + 0.3 * (max(0, epoch - 100) // 50)
        self.current_alpha = self.base_alpha * scale

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, index):
        # ✅ 确定性采样逻辑
        style_id = index % self.num_styles
        indices = self.style_indices[style_id]
        # 伪随机但确定性的内部索引
        intra_idx = (index * 31337 + self.current_epoch) % len(indices)
        real_idx = indices[intra_idx]

        latent = self.latents_tensor[real_idx]
        
        # 异步增强 (CPU Worker 执行)
        if self.apply_elastic:
            # 传入 index 作为种子，保证同一个样本在同一个 epoch 增强结果一致
            latent_deformed = elastic_deform(
                latent.unsqueeze(0), 
                alpha=self.current_alpha, 
                seed=index + self.current_epoch * 10000
            ).squeeze(0)
        else:
            latent_deformed = latent

        return {
            'latent': latent,
            'latent_deformed': latent_deformed,
            'style_id': torch.tensor(style_id, dtype=torch.long)
        }

    def sample_style_batch(self, target_style_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """SWD Loss 专用的快速采样"""
        b = target_style_ids.shape[0]
        out = torch.empty((b, *self.latents_tensor.shape[1:]), device=device)
        target_cpu = target_style_ids.cpu()

        for style_id, indices in self.style_indices.items():
            mask = (target_cpu == style_id)
            if not mask.any(): continue
            
            # 这里的随机性是为了 SWD 统计特性，保留 random
            rand_idxs = torch.tensor(indices)[torch.randint(len(indices), (int(mask.sum()),))]
            out[mask] = self.latents_tensor[rand_idxs].to(device, non_blocking=True)
            
        return out