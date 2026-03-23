"""
PatchNCE Loss for Semantic Structure Preservation

This module implements the Patch-wise Contrastive Learning loss used in 
CycleGAN and CUT for enforcing semantic consistency during style transfer.

Key idea: 
- Extract deep features from both content and generated images
- Create positive pairs from spatially aligned patches
- Use InfoNCE to push positive pairs closer while pushing negative pairs apart
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PatchNCELoss(nn.Module):
    """
    Patch-wise Noise Contrastive Estimation Loss.
    
    For each spatial location in the source feature map, we treat the 
    corresponding location in the target feature map as a positive sample,
    and all other locations as negative samples.
    """
    
    def __init__(
        self,
        nce_layers: List[int] = [2, 3, 4],
        nce_tau: float = 0.07,
        num_patches: int = 256,
        use_projection: bool = True,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.nce_layers = nce_layers
        self.nce_tau = nce_tau
        self.num_patches = num_patches
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        
        # Projection heads will be created dynamically based on input channels
        self.projection_heads: nn.ModuleDict = nn.ModuleDict()
        
    def _get_projection_head(self, channels: int, device: torch.device) -> nn.Module:
        """Get or create projection head for given channel dimension."""
        key = f"proj_{channels}"
        if key not in self.projection_heads:
            head = nn.Sequential(
                nn.Conv2d(channels, self.projection_dim, kernel_size=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.projection_dim, self.projection_dim, kernel_size=1, bias=False),
            )
            # Initialize
            nn.init.xavier_uniform_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            nn.init.xavier_uniform_(head[2].weight)
            self.projection_heads[key] = head
        return self.projection_heads[key].to(device)
    
    def _sample_patches(
        self, 
        feat: torch.Tensor, 
        num_patches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample spatial locations from feature map.
        
        Args:
            feat: [B, C, H, W] feature map
            num_patches: number of patches to sample per image
            
        Returns:
            patches: [B, num_patches, C] sampled features
            indices: [B, num_patches, 2] spatial indices (y, x)
        """
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, H * W)  # [B, C, HW]
        
        # Random sample without replacement
        num_samples = min(num_patches, H * W)
        
        # Generate random indices
        indices = torch.rand(B, H * W, device=feat.device).argsort(dim=1)[:, :num_samples]
        
        # Gather sampled features
        indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)  # [B, C, num_samples]
        patches = torch.gather(feat_flat, 2, indices_expanded)  # [B, C, num_samples]
        patches = patches.permute(0, 2, 1)  # [B, num_samples, C]
        
        # Convert linear indices to spatial coordinates
        y_idx = indices // W
        x_idx = indices % W
        spatial_indices = torch.stack([y_idx, x_idx], dim=-1)  # [B, num_samples, 2]
        
        return patches, spatial_indices
    
    def _get_target_patches(
        self,
        feat: torch.Tensor,
        spatial_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get patches from target feature map at same spatial locations.
        
        Args:
            feat: [B, C, H, W] target feature map
            spatial_indices: [B, num_patches, 2] (y, x) indices
            
        Returns:
            patches: [B, num_patches, C]
        """
        B, C, H, W = feat.shape
        num_patches = spatial_indices.shape[1]
        
        # Clamp indices to valid range
        y_idx = spatial_indices[..., 0].clamp(0, H - 1)
        x_idx = spatial_indices[..., 1].clamp(0, W - 1)
        
        # Convert to linear indices
        linear_idx = y_idx * W + x_idx  # [B, num_patches]
        
        # Gather features
        feat_flat = feat.view(B, C, H * W)  # [B, C, HW]
        indices_expanded = linear_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, num_patches]
        patches = torch.gather(feat_flat, 2, indices_expanded)  # [B, C, num_patches]
        patches = patches.permute(0, 2, 1)  # [B, num_samples, C]
        
        return patches
    
    def compute_nce_loss(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss between source and target features.
        
        Args:
            source_feat: [B, C, H, W] features from content image
            target_feat: [B, C, H, W] features from generated image
            
        Returns:
            loss: scalar NCE loss
        """
        B, C, H, W = source_feat.shape
        device = source_feat.device
        dtype = source_feat.dtype
        
        # Apply projection head if enabled
        if self.use_projection:
            proj_head = self._get_projection_head(C, device)
            source_feat = proj_head(source_feat)
            target_feat = proj_head(target_feat)
            C = self.projection_dim
        
        # Sample patches from source
        source_patches, spatial_indices = self._sample_patches(source_feat, self.num_patches)
        # source_patches: [B, num_patches, C]
        
        # Get corresponding patches from target
        target_patches = self._get_target_patches(target_feat, spatial_indices)
        # target_patches: [B, num_patches, C]
        
        # Normalize features
        source_patches = F.normalize(source_patches, dim=-1)  # [B, num_patches, C]
        target_patches = F.normalize(target_patches, dim=-1)  # [B, num_patches, C]
        
        # Compute similarity matrix
        # [B, num_patches, num_patches]
        sim_matrix = torch.bmm(source_patches, target_patches.transpose(1, 2))
        sim_matrix = sim_matrix / self.nce_tau
        
        # Create labels: diagonal elements are positive pairs
        # For each row, the positive sample is at the same index
        labels = torch.arange(self.num_patches, device=device, dtype=torch.long)
        labels = labels.unsqueeze(0).expand(B, -1)  # [B, num_patches]
        
        # Cross entropy loss
        # sim_matrix[i, j] = similarity between source_patch[i] and target_patch[j]
        # We want source_patch[i] to be most similar to target_patch[i]
        loss = F.cross_entropy(
            sim_matrix.view(B * self.num_patches, self.num_patches),
            labels.view(B * self.num_patches),
            reduction='mean'
        )
        
        return loss
    
    def forward(
        self,
        source_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute NCE loss across multiple feature layers.
        
        Args:
            source_features: list of [B, C, H, W] features from content
            target_features: list of [B, C, H, W] features from generated
            
        Returns:
            total_loss: sum of NCE losses across all layers
        """
        assert len(source_features) == len(target_features), \
            "Source and target features must have same number of layers"
        
        total_loss = torch.tensor(0.0, device=source_features[0].device)
        
        for src_feat, tgt_feat in zip(source_features, target_features):
            loss = self.compute_nce_loss(src_feat, tgt_feat)
            total_loss = total_loss + loss
        
        return total_loss / len(source_features)


def extract_features(
    model: nn.Module,
    x: torch.Tensor,
    layers: List[int],
) -> List[torch.Tensor]:
    """
    Extract intermediate features from model at specified layers.
    
    This is a helper function that hooks into the model to extract
    features from specific ResBlocks.
    
    Args:
        model: LatentAdaCUT model
        x: input latent [B, 4, 32, 32]
        layers: list of layer indices to extract from
        
    Returns:
        features: list of feature tensors
    """
    features = []
    hooks = []
    
    def make_hook(idx):
        def hook(module, input, output):
            features.append((idx, output))
        return hook
    
    # Register hooks on ResBlocks
    # This needs to be adapted based on the actual model structure
    # For LatentAdaCUT, we hook into body ResBlocks
    
    return features
