from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    style_ids: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 256,
    use_high_freq: bool = True,
    projection_chunk_size: int = 0,
    distance_mode: str = "cdf",
    cdf_num_bins: int = 64,
    cdf_tau: float = 0.01,
    cdf_sample_size: int = 256,
    cdf_bin_chunk_size: int = 16,
    cdf_sample_chunk_size: int = 128,
    sobel_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
    projection_bank: Dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    del style_ids
    x = x.contiguous()
    y = y.contiguous()
    device = x.device
    if use_high_freq:
        c = x.shape[1]
        if sobel_kernels is None:
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                device=device,
                dtype=x.dtype,
            )
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                device=device,
                dtype=x.dtype,
            )
            weight_x = sobel_x.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
            weight_y = sobel_y.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
        else:
            weight_x, weight_y = sobel_kernels

        x_gx = F.conv2d(x, weight_x, padding=1, groups=c)
        x_gy = F.conv2d(x, weight_y, padding=1, groups=c)
        y_gx = F.conv2d(y, weight_x, padding=1, groups=c)
        y_gy = F.conv2d(y, weight_y, padding=1, groups=c)

        x = torch.cat([x, x_gx, x_gy], dim=1)
        y = torch.cat([y, y_gx, y_gy], dim=1)
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    c_total = x.shape[1]
    chunk = int(projection_chunk_size)
    if chunk <= 0 or chunk >= num_projections:
        chunk = num_projections
    mode = str(distance_mode).lower()
    use_cdf = mode in {"cdf", "softcdf", "cdf_soft"}
    cdf_bins = max(8, int(cdf_num_bins))
    tau = max(1e-5, float(cdf_tau))
    sample_size = max(32, int(cdf_sample_size))
    bin_chunk = max(1, int(cdf_bin_chunk_size))
    sample_chunk = max(32, int(cdf_sample_chunk_size))

    def _soft_cdf_mean(proj: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # proj: [B, P, N], grid: [K]
        # Returns smoothed CDF sampled on grid: [B, P, K]
        bsz, n_proj, n_pts = proj.shape
        k_bins = int(grid.shape[0])
        acc = torch.zeros((bsz, n_proj, k_bins), device=proj.device, dtype=proj.dtype)
        for n0 in range(0, n_pts, sample_chunk):
            n1 = min(n_pts, n0 + sample_chunk)
            p = proj[:, :, n0:n1].unsqueeze(-1)  # [B,P,n,1]
            g = grid.view(1, 1, 1, k_bins)       # [1,1,1,K]
            # chunk reduction on N avoids materializing full [B,P,N,K]
            acc = acc + torch.sigmoid((g - p) / tau).sum(dim=2)
        return acc / float(n_pts)

    for p in patch_sizes:
        if projection_bank is not None and p in projection_bank:
            rand_weights = projection_bank[p]
        else:
            rand_weights = torch.randn(
                num_projections,
                c_total,
                p,
                p,
                device=device,
                dtype=x.dtype,
            )
            rand_weights = F.normalize(rand_weights.view(num_projections, -1), p=2, dim=1).view_as(rand_weights)

        patch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for start in range(0, num_projections, chunk):
            end = min(num_projections, start + chunk)
            w = rand_weights[start:end]
            x_proj = F.conv2d(x, w, padding=p // 2).view(x.shape[0], end - start, -1)
            y_proj = F.conv2d(y, w, padding=p // 2).view(y.shape[0], end - start, -1)

            if use_cdf:
                n_pts = int(x_proj.shape[-1])
                if n_pts > sample_size:
                    sample_idx = torch.randint(0, n_pts, (sample_size,), device=x_proj.device)
                    x_proj = x_proj.index_select(2, sample_idx)
                    y_proj = y_proj.index_select(2, sample_idx)
                min_val = torch.minimum(x_proj.amin().detach(), y_proj.amin().detach())
                max_val = torch.maximum(x_proj.amax().detach(), y_proj.amax().detach())
                span = (max_val - min_val).clamp_min(1e-6)
                dx = span / float(cdf_bins - 1)
                grid = torch.linspace(min_val, max_val, cdf_bins, device=x_proj.device, dtype=x_proj.dtype)
                swd_chunk = torch.tensor(0.0, device=device, dtype=torch.float32)
                for b0 in range(0, cdf_bins, bin_chunk):
                    b1 = min(cdf_bins, b0 + bin_chunk)
                    g = grid[b0:b1]
                    cdf_x = _soft_cdf_mean(x_proj, g)
                    cdf_y = _soft_cdf_mean(y_proj, g)
                    swd_chunk = swd_chunk + (cdf_x - cdf_y).abs().sum(dim=-1).mean() * dx
                patch_loss = patch_loss + swd_chunk * ((end - start) / float(num_projections))
            else:
                x_sorted, _ = torch.sort(x_proj, dim=2)
                y_sorted, _ = torch.sort(y_proj, dim=2)
                patch_loss = patch_loss + F.l1_loss(x_sorted, y_sorted) * ((end - start) / float(num_projections))
        total_loss += patch_loss
    return total_loss / max(len(patch_sizes), 1)


def calc_hf_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 256,
    projection_chunk_size: int = 0,
    distance_mode: str = "cdf",
    cdf_num_bins: int = 64,
    cdf_tau: float = 0.01,
    cdf_sample_size: int = 256,
    cdf_bin_chunk_size: int = 16,
    cdf_sample_chunk_size: int = 128,
    sobel_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
    projection_bank: Dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    c = int(x.shape[1])
    device = x.device
    if sobel_kernels is None:
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=x.dtype,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=x.dtype,
        )
        weight_x = sobel_x.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
        weight_y = sobel_y.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    else:
        weight_x, weight_y = sobel_kernels

    x_gx = F.conv2d(x, weight_x, padding=1, groups=c)
    x_gy = F.conv2d(x, weight_y, padding=1, groups=c)
    y_gx = F.conv2d(y, weight_x, padding=1, groups=c)
    y_gy = F.conv2d(y, weight_y, padding=1, groups=c)
    hf_x = torch.sqrt(x_gx.pow(2) + x_gy.pow(2) + 1e-8)
    hf_y = torch.sqrt(y_gx.pow(2) + y_gy.pow(2) + 1e-8)
    hf_x = hf_x / (hf_x.mean(dim=(2, 3), keepdim=True) + 1e-5)
    hf_y = hf_y / (hf_y.mean(dim=(2, 3), keepdim=True) + 1e-5)

    dummy_ids = torch.zeros((hf_x.shape[0],), device=hf_x.device, dtype=torch.long)
    return calc_swd_loss(
        hf_x,
        hf_y,
        dummy_ids,
        patch_sizes,
        num_projections=num_projections,
        use_high_freq=False,
        projection_chunk_size=projection_chunk_size,
        distance_mode=distance_mode,
        cdf_num_bins=cdf_num_bins,
        cdf_tau=cdf_tau,
        cdf_sample_size=cdf_sample_size,
        cdf_bin_chunk_size=cdf_bin_chunk_size,
        cdf_sample_chunk_size=cdf_sample_chunk_size,
        sobel_kernels=None,
        projection_bank=projection_bank,
    )


def calc_patch_nce_loss(
    feat_q: torch.Tensor,
    feat_k: torch.Tensor,
    *,
    tau: float = 0.07,
    num_patches: int = 128,
) -> torch.Tensor:
    """
    Infra-optimized latent-space PatchNCE.
    Negatives are restricted within each image to avoid O((B*N)^2) VRAM blow-up.
    """
    if feat_q.shape != feat_k.shape:
        raise ValueError(f"PatchNCE feature shape mismatch: q={tuple(feat_q.shape)} k={tuple(feat_k.shape)}")
    bsz, channels, h, w = feat_q.shape
    if bsz <= 0 or channels <= 0 or h <= 0 or w <= 0:
        return feat_q.new_tensor(0.0)
    feat_k = feat_k.detach()
    q = feat_q.view(bsz, channels, -1).transpose(1, 2)
    k = feat_k.view(bsz, channels, -1).transpose(1, 2)
    n_tokens = h * w
    n_pick = min(max(1, int(num_patches)), n_tokens)
    sel = torch.randperm(n_tokens, device=feat_q.device)[:n_pick]
    q_sel = F.normalize(q[:, sel, :], dim=-1)
    k_sel = F.normalize(k[:, sel, :], dim=-1)
    logits = torch.bmm(q_sel, k_sel.transpose(1, 2)) / max(float(tau), 1e-6)
    labels = torch.arange(n_pick, device=logits.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    return F.cross_entropy(logits.reshape(-1, n_pick), labels.reshape(-1))


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        self.w_swd = float(loss_cfg.get("w_swd", 30.0))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 256))
        self.swd_projection_chunk_size = int(loss_cfg.get("swd_projection_chunk_size", 64))
        self.swd_distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
        self.swd_cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 64))
        self.swd_cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
        self.swd_cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
        self.swd_cdf_bin_chunk_size = int(loss_cfg.get("swd_cdf_bin_chunk_size", 16))
        self.swd_cdf_sample_chunk_size = int(loss_cfg.get("swd_cdf_sample_chunk_size", 128))
        self.swd_batch_size = int(loss_cfg.get("swd_batch_size", 0))
        self.swd_use_high_freq = bool(loss_cfg.get("swd_use_high_freq", True))
        self.swd_hf_weight_ratio = float(loss_cfg.get("swd_hf_weight_ratio", 2.0))
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_color = float(loss_cfg.get("w_color", 15.0))
        self.w_nce = float(loss_cfg.get("w_nce", 0.0))
        self.w_nce_identity = float(loss_cfg.get("w_nce_identity", 1.0))
        self.nce_tau = float(loss_cfg.get("nce_tau", 0.07))
        self.nce_num_patches = int(loss_cfg.get("nce_num_patches", 128))
        layer_weights_cfg = loss_cfg.get("nce_layer_weights", [1.0, 1.0, 1.0])
        if isinstance(layer_weights_cfg, (list, tuple)):
            self.nce_layer_weights = [float(v) for v in layer_weights_cfg if float(v) > 0.0]
        else:
            self.nce_layer_weights = [float(layer_weights_cfg)]
        if not self.nce_layer_weights:
            self.nce_layer_weights = [1.0]
        self.w_gate_reg = float(loss_cfg.get("w_gate_reg", 0.0))
        self.gate_reg_target_var = float(loss_cfg.get("gate_reg_target_var", 0.05))
        # Kept for trainer hot-reload compatibility; probe monitoring is optional.
        self.style_probe_enabled = bool(loss_cfg.get("style_probe_enabled", False))
        self.style_probe_layer_index = int(loss_cfg.get("style_probe_layer_index", -1))
        self.style_probe_batch_size = int(loss_cfg.get("style_probe_batch_size", 64))
        self.style_probe_lr = float(loss_cfg.get("style_probe_lr", 1e-3))
        self.style_probe_weight_decay = float(loss_cfg.get("style_probe_weight_decay", 0.0))
        self.style_probe_use_spectral_norm = bool(loss_cfg.get("style_probe_use_spectral_norm", True))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._sobel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._projection_cache: Dict[tuple[int, int, int, str, str], torch.Tensor] = {}

    def _get_sobel_kernels(
        self, channels: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(channels), str(device), str(dtype))
        cached = self._sobel_cache.get(key)
        if cached is not None:
            return cached
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        w_x = sobel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        w_y = sobel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_cache[key] = (w_x, w_y)
        return w_x, w_y

    def _get_projection_bank(self, channels: int, *, device: torch.device, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        for p in self.swd_patch_sizes:
            key = (int(channels), int(p), int(self.swd_num_projections), str(device), str(dtype))
            w = self._projection_cache.get(key)
            if w is None:
                with torch.no_grad():
                    w = torch.randn(
                        self.swd_num_projections,
                        channels,
                        p,
                        p,
                        device=device,
                        dtype=dtype,
                    )
                    w = F.normalize(w.view(self.swd_num_projections, -1), p=2, dim=1).view_as(w)
                self._projection_cache[key] = w
            bank[p] = w
        return bank

    def _select_xid_indices(self, xid_mask: torch.Tensor) -> torch.Tensor:
        valid_idx = torch.nonzero(xid_mask, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return valid_idx
        if self.swd_batch_size <= 0 or valid_idx.numel() == self.swd_batch_size:
            return valid_idx
        if valid_idx.numel() > self.swd_batch_size:
            pick = torch.randint(0, valid_idx.numel(), (self.swd_batch_size,), device=valid_idx.device)
            return valid_idx.index_select(0, pick)
        pad_pick = torch.randint(0, valid_idx.numel(), (self.swd_batch_size - valid_idx.numel(),), device=valid_idx.device)
        return torch.cat([valid_idx, valid_idx.index_select(0, pad_pick)], dim=0)

    @contextmanager
    def _nvtx_range(self, name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_feat: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        debug_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred = model(content, style_feat=target_style_feat, step_size=1.0, style_strength=1.0)

        work_dtype = pred.dtype
        target_cast = target_style.to(dtype=work_dtype)
        content_cast = content.to(dtype=work_dtype)
        total_loss = torch.tensor(0.0, device=content.device)
        metrics = {"identity_ratio": id_ratio.detach()}
        xid_valid_idx: torch.Tensor | None = None
        if xid_mask.any():
            # Reuse one sampled xid subset for both SWD and color to reduce gradient jitter.
            xid_valid_idx = self._select_xid_indices(xid_mask)

        gate_t = model.get_skip_gate_tensor()
        if gate_t is not None:
            gate_det = gate_t.detach()
            metrics["skip_gate_mean"] = gate_det.mean()
            metrics["skip_gate_std"] = gate_det.std(unbiased=False)
            metrics["skip_gate_min"] = gate_det.min()
            metrics["skip_gate_max"] = gate_det.max()
            # Channel variance as a weak "gate should diversify" signal.
            gate_var = gate_det.var(dim=1, unbiased=False).mean()
            metrics["gate_var"] = gate_var
            if self.w_gate_reg > 0.0:
                gate_var_live = gate_t.var(dim=1, unbiased=False).mean()
                loss_gate_reg = F.relu(gate_t.new_tensor(self.gate_reg_target_var) - gate_var_live)
                total_loss = total_loss + self.w_gate_reg * loss_gate_reg
                metrics["gate_reg"] = loss_gate_reg.detach()
        # Release module-held gate tensor reference ASAP to avoid unnecessary graph retention.
        if hasattr(model, "clear_skip_gate_tensor"):
            model.clear_skip_gate_tensor()

        if xid_mask.any() and self.w_swd > 0.0:
            assert xid_valid_idx is not None
            swd_x = pred.index_select(0, xid_valid_idx)
            projection_bank_base = self._get_projection_bank(
                int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype
            )
            loss_swd = calc_swd_loss(
                swd_x,
                target_cast.index_select(0, xid_valid_idx),
                target_style_id.index_select(0, xid_valid_idx),
                self.swd_patch_sizes,
                self.swd_num_projections,
                use_high_freq=False,
                projection_chunk_size=self.swd_projection_chunk_size,
                distance_mode=self.swd_distance_mode,
                cdf_num_bins=self.swd_cdf_num_bins,
                cdf_tau=self.swd_cdf_tau,
                cdf_sample_size=self.swd_cdf_sample_size,
                cdf_bin_chunk_size=self.swd_cdf_bin_chunk_size,
                cdf_sample_chunk_size=self.swd_cdf_sample_chunk_size,
                sobel_kernels=None,
                projection_bank=projection_bank_base,
            )
            total_loss += self.w_swd * loss_swd
            metrics["swd"] = loss_swd.detach()
            if self.swd_use_high_freq:
                sobel_kernels = self._get_sobel_kernels(
                    int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype
                )
                projection_bank_hf = self._get_projection_bank(
                    int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype
                )
                loss_swd_hf = calc_hf_swd_loss(
                    swd_x,
                    target_cast.index_select(0, xid_valid_idx),
                    self.swd_patch_sizes,
                    num_projections=self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    cdf_bin_chunk_size=self.swd_cdf_bin_chunk_size,
                    cdf_sample_chunk_size=self.swd_cdf_sample_chunk_size,
                    sobel_kernels=sobel_kernels,
                    projection_bank=projection_bank_hf,
                )
                total_loss += (self.w_swd * self.swd_hf_weight_ratio) * loss_swd_hf
                metrics["swd_hf"] = loss_swd_hf.detach()

        if xid_mask.any() and self.w_color > 0.0:
            assert xid_valid_idx is not None
            pred_f32 = pred.float()
            target_f32 = target_cast.float()
            p_pool = F.adaptive_avg_pool2d(pred_f32.index_select(0, xid_valid_idx), (1, 1))
            t_pool = F.adaptive_avg_pool2d(target_f32.index_select(0, xid_valid_idx), (1, 1))
            loss_color = F.mse_loss(p_pool, t_pool)
            total_loss += self.w_color * loss_color
            metrics["color"] = loss_color.detach()

        if self.w_nce > 0.0:
            loss_nce_total = pred.new_tensor(0.0)
            loss_nce_terms = 0

            if xid_mask.any():
                assert xid_valid_idx is not None
                xid_content = content_cast.index_select(0, xid_valid_idx)
                xid_pred = pred.index_select(0, xid_valid_idx)
                q_feats = model.get_nce_features(xid_pred)
                with torch.no_grad():
                    k_feats = model.get_nce_features(xid_content)
                loss_nce_xid = pred.new_tensor(0.0)
                for i, (qf, kf) in enumerate(zip(q_feats, k_feats)):
                    w_layer = self.nce_layer_weights[min(i, len(self.nce_layer_weights) - 1)]
                    loss_nce_xid = loss_nce_xid + calc_patch_nce_loss(
                        qf,
                        kf,
                        tau=self.nce_tau,
                        num_patches=self.nce_num_patches,
                    ) * float(w_layer)
                loss_nce_xid = loss_nce_xid / max(len(q_feats), 1)
                loss_nce_total = loss_nce_total + loss_nce_xid
                loss_nce_terms += 1
                metrics["patch_nce_xid"] = loss_nce_xid.detach()

            if id_mask.any() and self.w_nce_identity > 0.0:
                id_valid_idx = torch.nonzero(id_mask, as_tuple=False).squeeze(1)
                id_content = content_cast.index_select(0, id_valid_idx)
                id_pred = pred.index_select(0, id_valid_idx)
                q_feats = model.get_nce_features(id_pred)
                with torch.no_grad():
                    k_feats = model.get_nce_features(id_content)
                loss_nce_id = pred.new_tensor(0.0)
                for i, (qf, kf) in enumerate(zip(q_feats, k_feats)):
                    w_layer = self.nce_layer_weights[min(i, len(self.nce_layer_weights) - 1)]
                    loss_nce_id = loss_nce_id + calc_patch_nce_loss(
                        qf,
                        kf,
                        tau=self.nce_tau,
                        num_patches=self.nce_num_patches,
                    ) * float(w_layer)
                loss_nce_id = loss_nce_id / max(len(q_feats), 1)
                loss_nce_total = loss_nce_total + self.w_nce_identity * loss_nce_id
                loss_nce_terms += 1
                metrics["patch_nce_id"] = loss_nce_id.detach()

            if loss_nce_terms > 0:
                loss_nce = loss_nce_total / float(loss_nce_terms)
                total_loss += self.w_nce * loss_nce
                metrics["patch_nce"] = loss_nce.detach()

        if self.w_identity > 0.0 and id_mask.any():
            with self._nvtx_range("loss/identity", nvtx_enabled):
                id_per_sample = (pred - content_cast).abs().mean(dim=(1, 2, 3))
            loss_identity = (id_per_sample * id_mask.float()).sum() / id_mask.float().sum().clamp_min(1.0)
            total_loss += self.w_identity * loss_identity
            metrics["identity"] = loss_identity.detach()

        if self.w_delta_tv > 0.0:
            delta = pred - content_cast
            loss_delta_tv = _tv_per_sample(delta).mean()
            total_loss += self.w_delta_tv * loss_delta_tv
            metrics["delta_tv"] = loss_delta_tv.detach()

        metrics["loss"] = total_loss
        return metrics
