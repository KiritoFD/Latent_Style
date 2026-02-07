"""
LGT Geometric Free Energy Loss Functions - Clean Version

Core principle: MSE maintains structure + brightness, SWD controls texture.
Removed: CosineSSMLoss, NeighborhoodMatchingLoss (deprecated Content Losses)

Loss functions:
1. PatchSlicedWassersteinLoss: Texture matching (with brightness normalization)
2. MultiScaleSWDLoss: Multi-scale texture matching
3. GeometricFreeEnergyLoss: Unified wrapper for SWD only

All losses operate in FP32 for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StyleClassificationLoss(nn.Module):
    """
    Style classifier guidance loss.
    """
    def __init__(self, classifier: nn.Module, weight: float = 0.1):
        super().__init__()
        self.classifier = classifier
        self.weight = weight

    def forward(self, x_style_feat: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x_style_feat)
        loss = F.cross_entropy(logits, style_ids)
        return self.weight * loss


class VelocityRegularizationLoss(nn.Module):
    """
    Velocity Magnitude Regularization for Flow Matching.
    
    Purpose: Prevent model from learning excessively large velocity vectors,
    which causes brightness explosion and instability during inference (especially with CFG).
    
    Physics: Flow Matching assumes the velocity field v(x,t,c) is bounded in magnitude.
    If training data has small variance (e.g., VAE latents with std=0.18) but noise has std=1.0,
    the model learns to output huge velocities to bridge this gap. This causes:
    - Large gradient magnitudes → training instability
    - CFG amplifies these huge vectors → color saturation/clipping
    - Batch-wise brightness variance due to different energy scales
    
    Solution: Regularize L2 norm of velocity vectors during training.
    Loss = weight * mean(v_pred²) across spatial and batch dimensions.
    
    Trade-off: Too large regularization → slower learning, washed-out images.
    Recommended: weight ∈ [0.05, 0.2] for typical latent-space diffusion.
    """
    
    def __init__(self, weight=0.1):
        """
        Args:
            weight: Regularization strength (default 0.1 = 5-10% of total loss)
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, v_pred):
        """
        Compute velocity regularization loss.
        
        Args:
            v_pred: [B, C, H, W] Predicted velocity field from model
        
        Returns:
            loss: scalar regularization penalty
        """
        # L2 norm squared: sum across channel and spatial dims, mean over batch
        # This penalizes all velocity pixels equally regardless of position/channel
        return self.weight * torch.mean(v_pred ** 2)




class PatchSlicedWassersteinLoss(nn.Module):
    """
    [Optimized] Patch-Sliced Wasserstein Distance
    
    Optimization:
    1. Replaces memory-heavy F.unfold with F.conv2d (Math equivalent).
       [Image of convolution operation vs sliding window extraction matrix multiplication]
    2. Uses persistent buffers for projections to avoid random generation overhead.
    3. Implements "Post-Projection Normalization" to handle brightness decoupling mathematically.
    
    Math Equivalence:
    Projection of (Patch - Mean) * Vector 
    = (Patch * Vector) - (Mean * Sum(Vector))
    = Conv2d(x, Vector) - Avg(Patch) * Sum(Vector)
    """
    
    def __init__(
        self,
        patch_size=3,
        num_projections=64,
        max_samples=4096,
        use_fp32=True,
        normalize_patch='mean',
        in_channels: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_projections = num_projections
        self.max_samples = max_samples
        self.use_fp32 = use_fp32
        self.normalize_patch = normalize_patch
        
        # Default latent channels is 4, but feature-level SWD may use other channel sizes.
        self.in_channels = int(in_channels)
        
        # 1. 预先注册随机投影向量 (作为卷积核)
        # Shape: [Out(Projections), In, K, K]
        # 这样避免了每次 forward 重新生成随机数，也方便模型保存/加载
        self.register_buffer('projections', None)
        self.register_buffer('proj_sums', None) # 用于快速计算归一化项
        
        # 2. 预先注册求和核 (用于计算 Patch Mean)
        # 这是一个固定权重的卷积核，所有权重为 1.0
        ones_kernel = torch.ones(1, self.in_channels, patch_size, patch_size)
        self.register_buffer('ones_kernel', ones_kernel)

    def _init_projections(self, device, dtype):
        """Lazy initialization ensuring correct device/dtype"""
        if self.projections is None:
            # Generate random projections
            # Shape: [num_projections, C, K, K]
            projections = torch.randn(
                self.num_projections, self.in_channels, self.patch_size, self.patch_size,
                device=device, dtype=dtype
            )
            
            # Normalize: standard SWD requires unit vectors
            # Flatten to [N, C*K*K] for norm calculation
            flat_proj = projections.view(self.num_projections, -1)
            norms = flat_proj.norm(dim=1, keepdim=True) + 1e-8
            projections = projections / norms.view(-1, 1, 1, 1)
            
            self.projections = projections
            
            # Pre-compute sum of each projection vector for mathematical normalization
            # This is \sum \theta term
            self.proj_sums = projections.sum(dim=(1, 2, 3)).view(1, -1, 1, 1)

    def forward(self, x_pred, x_style):
        # x_pred: [B, 4, H, W]
        
        # FP32 Stability
        with torch.amp.autocast('cuda', enabled=False):
            if self.use_fp32:
                x_pred = x_pred.float()
                x_style = x_style.float()

            if x_pred.shape[1] != self.in_channels or x_style.shape[1] != self.in_channels:
                raise ValueError(
                    f"PatchSWD channel mismatch: expected {self.in_channels}, "
                    f"got pred={x_pred.shape[1]}, style={x_style.shape[1]}"
                )
            
            # Lazy Init
            if self.projections is None:
                self._init_projections(x_pred.device, x_pred.dtype)
                
            # ==========================================
            # 🚀 Optimization 1: Conv2d Projection
            # ==========================================
            # Project ALL patches at once using Convolution. 
            # This is mathematically equivalent to Unfold + MatMul but highly optimized by Tensor Cores.
            # Output: [B, num_projections, H, W]
            padding = self.patch_size // 2
            
            pred_proj = F.conv2d(x_pred, self.projections, padding=padding)
            style_proj = F.conv2d(x_style, self.projections, padding=padding)
            
            # ==========================================
            # 🔥 Optimization 2: Math-Equivalent Normalization
            # ==========================================
            if self.normalize_patch == 'mean':
                # Calculate Patch Mean efficiently using the constant 'ones_kernel'
                # patch_sum: [B, 1, H, W]
                num_elements = self.in_channels * self.patch_size * self.patch_size
                
                pred_patch_sum = F.conv2d(x_pred, self.ones_kernel, padding=padding)
                pred_patch_mean = pred_patch_sum / num_elements
                
                style_patch_sum = F.conv2d(x_style, self.ones_kernel, padding=padding)
                style_patch_mean = style_patch_sum / num_elements
                
                # Apply normalization correction: 
                # (Patch - Mean) * Theta = (Patch * Theta) - (Mean * Sum(Theta))
                # Broadcasting: [B, 1, H, W] * [1, N_proj, 1, 1] -> [B, N_proj, H, W]
                pred_proj = pred_proj - (pred_patch_mean * self.proj_sums)
                style_proj = style_proj - (style_patch_mean * self.proj_sums)

            # ==========================================
            # 3. Sampling & Sorting
            # ==========================================
            # Flatten spatial dimensions: [B, N_proj, H, W] -> [B * H * W, N_proj]
            # We treat all patches from the batch as a single distribution
            B, N, H, W = pred_proj.shape
            pred_flat = pred_proj.permute(0, 2, 3, 1).reshape(-1, N)
            style_flat = style_proj.permute(0, 2, 3, 1).reshape(-1, N)
            
            # Random Sampling (only if total pixels > max_samples)
            total_samples = pred_flat.shape[0]
            if total_samples > self.max_samples:
                # Generate random indices once
                indices = torch.randperm(total_samples, device=x_pred.device)[:self.max_samples]
                pred_sampled = pred_flat[indices]
                style_sampled = style_flat[indices]
            else:
                pred_sampled = pred_flat
                style_sampled = style_flat
            
            # Sort columns (Quantile alignment)
            # [max_samples, num_projections]
            pred_sorted, _ = torch.sort(pred_sampled, dim=0)
            style_sorted, _ = torch.sort(style_sampled, dim=0)
            
            # L1 (W1-style) for stabler texture matching
            loss = F.l1_loss(pred_sorted, style_sorted)
            
            return loss

class MultiScaleSWDLoss(nn.Module):
    """
    Multi-Scale Sliced Wasserstein Distance for unified annealing/quenching.
    
    🔥 Update: Automatically enables brightness normalization for scales > 1.
    This ensures high-frequency texture matching without brightness explosion.
    
    Physics motivation:
    - Unifies optimization objective for both phase transitions
    - Matching photo's high-freq distribution → auto-sharpens (quenching)
    - Matching painting's low-freq distribution → auto-smooths (annealing)
    - No need for w_freq or asymmetric hard constraints
    
    Architecture:
    - Scale 1×1 (Pixel): Color palette distribution (no normalization - patch_size=1 would collapse)
    - Scale 3×3 (Texture): High-frequency details (mean-normalized for brightness invariance)
    - Scale 7×7 (Structure): Local structural patterns (mean-normalized for texture focus)
    
    Default weights: [1.0, 1.0, 1.0] - let model balance frequency bands naturally
    """
    
    def __init__(
        self,
        scales=[1, 3, 7],
        scale_weights=[1.0, 1.0, 1.0],
        num_projections=64,
        max_samples=4096,
        use_fp32=True,
        in_channels: int = 4,
    ):
        super().__init__()
        
        assert len(scales) == len(scale_weights), "scales and scale_weights must have same length"
        
        self.scales = scales
        self.scale_weights = scale_weights
        
        # Create SWD loss for each scale
        self.swd_losses = nn.ModuleList([
            PatchSlicedWassersteinLoss(
                patch_size=scale,
                num_projections=num_projections,
                max_samples=max_samples,
                use_fp32=use_fp32,
                in_channels=in_channels,
                # 🔥 Fix: Only normalize for very large scales to preserve brightness/color
                # Scale 1-5: 'none' preserves palette and local contrast
                normalize_patch='mean' if scale > 5 else 'none'
            )
            for scale in scales
        ])
    
    def forward(self, x_pred, x_style):
        """
        Args:
            x_pred: [B, 4, H, W] predicted latent
            x_style: [B, 4, H, W] style reference latent
        
        Returns:
            loss: scalar weighted multi-scale SWD
            loss_dict: dictionary with per-scale losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        for scale, weight, swd_loss in zip(self.scales, self.scale_weights, self.swd_losses):
            scale_loss = swd_loss(x_pred, x_style)
            total_loss = total_loss + weight * scale_loss
            loss_dict[f'swd_scale_{scale}'] = scale_loss
        
        loss_dict['swd_total'] = total_loss
        
        return total_loss, loss_dict


class GeometricFreeEnergyLoss(nn.Module):
    """
    [LGT-X Pro] Optimized Multi-Style SWD Loss with Style-Indexed Cache
    
    🔥 Key Innovation: Style Look-Up Table (LUT)
    - Pre-compute sorted projections for each style during initialization
    - During training: O(1) target retrieval via index_select, only Input needs sorting
    - 4070 Friendly: Fixed LUT is read-only, better L2 cache utilization
    
    Performance:
    - Traditional: Sort Input + Sort Target = 2 × O(N log N)
    - LUT-based:  Sort Input only = O(N log N) + O(1) LUT retrieval
    - Memory trade-off: ~10-50MB LUT for significant speedup
    
    Architecture:
    - Fixed orthogonal projections per scale
    - Pre-sorted style distributions stored in GPU buffers
    - Dynamic style routing during forward pass via style_ids
    """
    
    def __init__(
        self,
        num_styles=4,
        w_style=40.0,
        swd_scales=[1, 3, 5, 7, 15],
        swd_scale_weights=[1.0, 5.0, 5.0, 5.0, 3.0],
        num_projections=64,
        max_samples=4096,
        moment_lowpass_size: int = 8,
        moment_weight: float = 10.0,
        fixed_sample_indices: bool = True,
        swd_sample_seed: int = 1234,
        **kwargs  # Accept but ignore deprecated parameters for backward compatibility
    ):
        super().__init__()
        
        self.num_styles = num_styles
        self.w_style = w_style
        self.scales = swd_scales
        self.scale_weights = swd_scale_weights
        self.num_projections = num_projections
        self.max_samples = max_samples
        self.moment_lowpass_size = moment_lowpass_size
        self.moment_weight = float(moment_weight)
        self.fixed_sample_indices = fixed_sample_indices
        self.swd_sample_seed = swd_sample_seed
        self._fixed_sample_cache = {}
        
        # Register initialization flag
        self.register_buffer('_is_initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('_moment_cache_ready', torch.tensor(False, dtype=torch.bool))
        # Default channel size is 4 for SD latents; replaced on initialize_cache
        self.register_buffer('_style_mu', torch.zeros(self.num_styles, 4))
        self.register_buffer('_style_std', torch.ones(self.num_styles, 4))
        
        logger.info(
            f"🎯 GeometricFreeEnergyLoss initialized (Multi-Style Mode)\n"
            f"   Scales: {swd_scales}\n"
            f"   Styles: {num_styles} | Max Samples: {max_samples}\n"
            f"   Projections: {num_projections} | Memory: ~{self._estimate_memory():.1f}MB"
        )
    
    def _estimate_memory(self) -> float:
        """Estimate GPU memory usage for LUT in MB"""
        # 每个尺度的 LUT: [num_styles, max_samples, num_projections]
        bytes_per_scale = self.num_styles * self.max_samples * self.num_projections * 4  # float32
        total_bytes = bytes_per_scale * len(self.scales)
        
        # 投影矩阵: [num_projections, C]，其中 C 因尺度而异
        # 粗估：4 + 4 + 4 个通道（缩放后）
        proj_bytes = self.num_projections * (4 + 4 + 4) * 4
        
        return (total_bytes + proj_bytes) / (1024 * 1024)
    
    def _get_orthogonal_projections(self, n_dims: int, n_projections: int, device: torch.device) -> torch.Tensor:
        """使用 QR 分解生成正交投影矩阵（比随机高斯更稳定）"""
        # 生成高斯随机矩阵并进行 QR 分解
        mat = torch.randn(n_dims, n_projections, device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat)
        
        # 返回 [n_projections, n_dims] 形状用于矩阵乘法
        return q[:, :n_projections].t()  # [n_projections, n_dims]
    
    def initialize_cache(self, style_latents_dict: Dict[int, torch.Tensor], device: torch.device) -> None:
        """
        初始化 Style LUT 缓存。必须在训练开始前调用一次。
        
        Args:
            style_latents_dict: 字典 {style_id: latent_tensor [B, 4, H, W]}
                               包含每个风格的代表性样本
            device: 目标设备（通常是 'cuda'）
        
        Example:
            swd_loss = GeometricFreeEnergyLoss(num_styles=4)
            style_dict = {0: monet_latent, 1: photo_latent, 2: vangogh_latent, 3: cezanne_latent}
            swd_loss.initialize_cache(style_dict, device='cuda')
        """
        if self._is_initialized:
            logger.info("⚠️ Cache already initialized. Skipping re-initialization.")
            return
        
        logger.info("🔥 Pre-computing SWD Style Cache (this may take a minute)...")
        
        with torch.no_grad():
            # Pre-compute per-style global moments for stability
            mu_list = []
            std_list = []
            for style_id in range(self.num_styles):
                assert style_id in style_latents_dict, f"Style {style_id} not found in input dict"
                style_latent = style_latents_dict[style_id].to(device, non_blocking=True).float()
                if self.moment_lowpass_size > 0:
                    lp = F.interpolate(style_latent, size=(self.moment_lowpass_size, self.moment_lowpass_size), mode='area')
                    lp = F.interpolate(lp, size=style_latent.shape[-2:], mode='bilinear', align_corners=False)
                    moment_source = lp
                else:
                    moment_source = style_latent
                mu = moment_source.mean(dim=(0, 2, 3))  # [C]
                std = moment_source.std(dim=(0, 2, 3))  # [C]
                mu_list.append(mu)
                std_list.append(std)

            self._style_mu = torch.stack(mu_list, dim=0)
            self._style_std = torch.stack(std_list, dim=0)
            self._moment_cache_ready.fill_(True)

            for scale_idx, scale in enumerate(self.scales):
                logger.info(f"  Processing scale {scale}×{scale} ({scale_idx + 1}/{len(self.scales)})...")
                
                # 1. 确定当前尺度的通道数（探测一个样本）
                sample_latent = style_latents_dict[0].to(device, non_blocking=True)
                
                # 缩放采样（模拟多尺度）
                if scale > 1:
                    sample_latent = F.interpolate(sample_latent, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                
                c_dim = sample_latent.shape[1]  # 通常是 4
                
                # 2. 生成固定的正交投影矩阵（一次性计算）
                projections = self._get_orthogonal_projections(c_dim, self.num_projections, device)
                self.register_buffer(f'_proj_{scale}', projections, persistent=False)
                
                # 3. 为每个风格构建 LUT 项
                lut_list = []
                
                for style_id in range(self.num_styles):
                    assert style_id in style_latents_dict, f"Style {style_id} not found in input dict"
                    
                    style_latent = style_latents_dict[style_id].to(device, non_blocking=True)
                    
                    # 缩放到当前尺度
                    if scale > 1:
                        style_latent = F.interpolate(style_latent, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                    
                    # 展开为像素级特征：[B, C, H, W] → [N, C]
                    b, c, h, w = style_latent.shape
                    style_flat = style_latent.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)  # [B*H*W, C]
                    
                    # 采样到固定大小（保证 LUT 维度固定）
                    n_pixels = style_flat.shape[0]
                    if n_pixels > self.max_samples:
                        idx = torch.randperm(n_pixels, device=device)[:self.max_samples]
                        style_flat = style_flat[idx]
                    else:
                        # 如果样本不足，进行循环复制填充
                        repeat_times = (self.max_samples // max(n_pixels, 1)) + 1
                        style_flat = style_flat.repeat(repeat_times, 1)[:self.max_samples]
                    
                    # 投影：[max_samples, C] @ [C, num_projections] → [max_samples, num_projections]
                    proj_style = torch.matmul(style_flat, projections.t())
                    
                    # 排序（这是 LUT 初始化时的唯一开销，之后就不再做）
                    proj_style_sorted, _ = torch.sort(proj_style, dim=0)
                    
                    lut_list.append(proj_style_sorted)
                
                # 4. 堆叠所有风格的排序结果：[num_styles, max_samples, num_projections]
                lut_tensor = torch.stack(lut_list, dim=0)
                self.register_buffer(f'_lut_{scale}', lut_tensor, persistent=False)
                
                logger.info(f"    ✓ Scale {scale}: LUT shape {lut_tensor.shape}")
        
        self._is_initialized.fill_(True)
        logger.info("✅ SWD Cache Initialization Complete")
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_style: torch.Tensor,
        style_ids: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        moments_pred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        moments_style: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算多尺度 SWD 损失（使用预计算的 Style LUT）
        
        Args:
            x_pred: [B, 4, H, W] 模型预测的终端状态
            x_style: [B, 4, H, W] 风格参考（用于旧式直接计算，仅在未初始化缓存时使用）
            style_ids: [B] 每个样本的目标风格 ID（与 LUT 索引对应）
            sample_weights: [B] Optional per-sample weights for loss masking
        
        Returns:
            loss_dict: {
                'style_swd': scalar 总 SWD 损失,
                'swd_scale_1': 尺度 1 的损失,
                'swd_scale_3': 尺度 3 的损失,
                ...
            }
        """
        # Fallback：如果没有初始化缓存，降级到旧式多尺度 SWD（兼容性保证）
        if not self._is_initialized:
            logger.warning(
                "⚠️ Style LUT not initialized. Falling back to traditional SWD "
                "(slower, but backward compatible). Call loss.initialize_cache() before training."
            )
            return self._forward_traditional(x_pred, x_style)
        
        # 推理路径：使用 LUT 加速
        device = x_pred.device
        dtype = x_pred.dtype
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        loss_dict = {}
        
        # 如果没有提供 style_ids，假设所有样本都是第一个风格（用于评估）
        if style_ids is None:
            style_ids = torch.zeros(x_pred.shape[0], device=device, dtype=torch.long)
        
        with torch.autocast('cuda', enabled=False):  # 强制 FP32 以保证数值稳定性
            x_pred_fp32 = x_pred.float()
            
            for scale_idx, scale in enumerate(self.scales):
                # 1. 尺度缩放
                if scale > 1:
                    x_scaled = F.interpolate(x_pred_fp32, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                else:
                    x_scaled = x_pred_fp32
                
                b, c, h, w = x_scaled.shape
                
                # 2. 读取固定投影矩阵
                projections = getattr(self, f'_proj_{scale}')  # [num_projections, C]
                
                # 3. 投影 Input（每次 Forward 都要排序 Input）
                x_flat = x_scaled.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
                
                n_pixels = x_flat.shape[1]
                if n_pixels >= self.max_samples:
                    # 固定采样索引（减少分位数噪声）
                    if self.fixed_sample_indices:
                        cache_key = (device, scale, n_pixels)
                        idx = self._fixed_sample_cache.get(cache_key)
                        if idx is None or idx.device != device:
                            gen = torch.Generator(device=device)
                            gen.manual_seed(self.swd_sample_seed + int(scale))
                            idx = torch.randint(0, n_pixels, (self.max_samples,), device=device, generator=gen)
                            self._fixed_sample_cache[cache_key] = idx
                    else:
                        idx = torch.randint(0, n_pixels, (self.max_samples,), device=device)
                    x_sampled = x_flat[:, idx, :]  # [B, max_samples, C]
                else:
                    # 不足时循环填充，避免 0 padding 造成分布偏移
                    repeat_times = (self.max_samples // n_pixels) + 1
                    x_repeat = x_flat.repeat(1, repeat_times, 1)
                    x_sampled = x_repeat[:, :self.max_samples, :]
                
                # 投影：[B, max_samples, C] @ [C, num_projections] → [B, max_samples, num_projections]
                proj_input = torch.matmul(x_sampled, projections.t())  # [B, max_samples, num_projections]
                
                # 4. 排序 Input（唯一的排序操作）
                proj_input_sorted, _ = torch.sort(proj_input, dim=1)
                
                # 5. 从 LUT 中检索对应风格的目标分布（这是 O(1) 操作）
                lut = getattr(self, f'_lut_{scale}')  # [num_styles, max_samples, num_projections]
                
                # index_select：高效地从 LUT 中根据 style_id 提取目标
                # [num_styles, max_samples, num_projections] → [B, max_samples, num_projections]
                proj_target_sorted = lut.index_select(0, style_ids)
                
                # 6. 计算 SWD 损失（L2 距离）
                if sample_weights is not None:
                    # [B, max_samples, num_projections] -> [B]
                    raw_loss = (proj_input_sorted.float() - proj_target_sorted.float()).abs().mean(dim=(1, 2))
                    scale_loss = (raw_loss * sample_weights).mean()
                else:
                    scale_loss = (proj_input_sorted.float() - proj_target_sorted.float()).abs().mean()
                
                total_loss = total_loss + self.scale_weights[scale_idx] * scale_loss
                loss_dict[f'swd_scale_{scale}'] = scale_loss.detach()
        
        # 🔥 Fix: Global Moment Matching (Color/Brightness Lock)
        # SWD handles local texture, this handles global atmosphere
        if moments_pred is None:
            mu_pred = x_pred.float().mean(dim=(2, 3))
            std_pred = x_pred.float().std(dim=(2, 3))
        else:
            mu_pred, std_pred = moments_pred

        if moments_style is None:
            if self._moment_cache_ready and style_ids is not None:
                mu_style = self._style_mu.index_select(0, style_ids)
                std_style = self._style_std.index_select(0, style_ids)
            else:
                mu_style = x_style.float().mean(dim=(2, 3))
                std_style = x_style.float().std(dim=(2, 3))
        else:
            mu_style, std_style = moments_style
        
        loss_moments = F.mse_loss(mu_pred, mu_style) + F.mse_loss(std_pred, std_style)
        if self.moment_weight > 0.0:
            total_loss = total_loss + self.moment_weight * loss_moments
        loss_dict['moments'] = loss_moments.detach()
        
        loss_dict['style_swd'] = total_loss
        
        return loss_dict
    
    def _forward_traditional(self, x_pred: torch.Tensor, x_style: torch.Tensor) -> Dict[str, torch.Tensor]:
        """降级方案：不使用 LUT，直接计算（兼容性保证）"""
        # 使用原来的 MultiScaleSWDLoss 进行计算
        if not hasattr(self, '_fallback_loss'):
            self._fallback_loss = MultiScaleSWDLoss(
                scales=self.scales,
                scale_weights=self.scale_weights,
                num_projections=self.num_projections,
                max_samples=self.max_samples,
                use_fp32=True,
                in_channels=x_pred.shape[1],
            ).to(x_pred.device)
        
        style_potential, swd_dict = self._fallback_loss(x_pred, x_style)
        
        result = {
            'style_swd': style_potential,
        }
        result.update(swd_dict)
        
        return result


class PyramidStructuralLoss(nn.Module):
    """
    Pyramid Structural Lock for Flow Matching.
    
    🔥 Key Innovation: Frequency-Domain Separation
    - Low-freq (8x8): Locks macro layout/composition → fast convergence
    - Mid-freq (16x16): Preserves object contours
    - High-freq (32x32): Soft constraint allowing artistic deformation
    
    Physics: By downsampling with 'area' mode (anti-aliased box filter),
    we surgically remove high-frequency texture from structure supervision.
    This eliminates gradient interference between MSE (structure) and SWD (style).
    
    Trade-off:
    - w_low too high → rigid, no artistic freedom
    - w_high too high → texture bleeds into structure loss
    Recommended: {'low': 5.0, 'mid': 1.0, 'high': 0.2}
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        # Default weights optimized for latent space (32x32 base resolution)
        self.w = weights or {'low': 5.0, 'mid': 1.0, 'high': 0.2}
    
    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-scale structural MSE loss.
        
        Args:
            v_pred: [B, C, H, W] Predicted velocity field
            v_target: [B, C, H, W] Target velocity field
            sample_weights: [B] Optional per-sample weights
        
        Returns:
            loss: Weighted sum of pyramid MSE losses
        """
        # L2: 8x8 Macro Layout (Low-frequency anchor)
        # At this scale, only composition/color palette survives
        v_p_low = F.interpolate(v_pred, size=(8, 8), mode='area')
        v_t_low = F.interpolate(v_target, size=(8, 8), mode='area')
        
        # L1: 16x16 Contours (Mid-frequency)
        # Objects and major edges are preserved
        v_p_mid = F.interpolate(v_pred, size=(16, 16), mode='area')
        v_t_mid = F.interpolate(v_target, size=(16, 16), mode='area')
        
        # L0: 32x32 Details (High-frequency, soft protection)
        # Full resolution - texture can still be modified by SWD
        
        if sample_weights is not None:
            # Helper to compute weighted mean
            def weighted_mse(pred, tgt, w):
                raw = F.mse_loss(pred, tgt, reduction='none').mean(dim=(1, 2, 3)) # [B]
                return (raw * w).mean()
            
            loss_low = weighted_mse(v_p_low, v_t_low, sample_weights)
            loss_mid = weighted_mse(v_p_mid, v_t_mid, sample_weights)
            loss_high = weighted_mse(v_pred, v_target, sample_weights)
        else:
            loss_low = F.mse_loss(v_p_low, v_t_low)
            loss_mid = F.mse_loss(v_p_mid, v_t_mid)
            loss_high = F.mse_loss(v_pred, v_target)

        total = self.w['low'] * loss_low + self.w['mid'] * loss_mid + self.w['high'] * loss_high

        return total, {
            "l_8x8": loss_low.detach(),
            "l_16x16": loss_mid.detach(),
            "l_32x32": loss_high.detach()
        }


if __name__ == "__main__":
    # Test loss functions (clean version)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    B, C, H, W = 4, 4, 32, 32
    x_pred = torch.randn(B, C, H, W, device=device)
    x_style = torch.randn(B, C, H, W, device=device)
    v_pred = torch.randn(B, C, H, W, device=device)

    print("Testing Patch-SWD Loss (mean-normalized)...")
    swd_loss = PatchSlicedWassersteinLoss(normalize_patch='mean').to(device)
    swd_value = swd_loss(x_pred, x_style)
    print(f"  SWD Loss: {swd_value.item():.6f}")

    print("\nTesting Multi-Scale SWD Loss...")
    multi_swd = MultiScaleSWDLoss(scales=[2, 4, 8], scale_weights=[2.0, 5.0, 5.0]).to(device)
    multi_loss, scale_dict = multi_swd(x_pred, x_style)
    print(f"  Multi-Scale SWD Total: {multi_loss.item():.6f}")
    for scale, loss_val in scale_dict.items():
        if isinstance(loss_val, torch.Tensor):
            print(f"    {scale}: {loss_val.item():.6f}")

    print("\nTesting Geometric Free Energy Loss (SWD only)...")
    energy_loss = GeometricFreeEnergyLoss(w_style=60.0).to(device)
    loss_dict = energy_loss(x_pred, x_style)
    print(f"  Total Energy: {loss_dict['style_swd']:.6f}")
    print(f"  Style SWD: {loss_dict['style_swd']:.6f}")

    print("\nTesting Pyramid Structural Loss...")
    pyramid_loss = PyramidStructuralLoss().to(device)
    v_target = x_style - torch.randn_like(x_style)
    pyramid_value = pyramid_loss(v_pred, v_target)
    print(f"  Pyramid Structural Loss: {pyramid_value.item():.6f}")

    print("\n✓ All loss functions tested successfully!")
