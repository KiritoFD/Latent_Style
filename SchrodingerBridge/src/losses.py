from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from scipy.optimize import linear_sum_assignment

from config_schema import BridgeConfig, ExperimentConfig, ModelConfig, TrainingConfig
from model import TimeConditionedLANCETBridge
from ot_cost import SWDTransportCost


class KantorovichPotential(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        hidden = max(8, int(hidden_channels))
        groups = max(1, min(4, hidden))
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, hidden, kernel_size=1)),
            nn.SiLU(),
            spectral_norm(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=groups)),
            nn.SiLU(),
            spectral_norm(nn.Conv2d(hidden, 1, kernel_size=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).mean(dim=(1, 2, 3))


class OTFlowMatchingObjective:
    """Training objective kept to the components that survived experiments.

    The historical heuristic losses (PatchNCE, local color, low-frequency
    anchor, cycle, and repulsive terms) are intentionally absent here. They are
    documented as disabled/dead in docs/cleanup_report.md and were already zero
    in the active S-add K1/W20 configuration.
    """

    def __init__(self, config: Dict | ExperimentConfig) -> None:
        if isinstance(config, ExperimentConfig):
            bridge_cfg = config.bridge
            train_cfg = config.training
            model_cfg = config.model
        else:
            bridge_cfg = BridgeConfig.from_mapping(config.get("bridge", {}))
            train_cfg = TrainingConfig.from_mapping(config.get("training", {}))
            model_cfg = ModelConfig.from_mapping(config.get("model", {}))

        self.objective_mode = str(bridge_cfg.objective_mode).strip().lower()
        self.t_min = float(bridge_cfg.t_min)
        self.t_max = float(bridge_cfg.t_max)
        self.loss_type = str(bridge_cfg.loss_type).strip().lower()
        self.identity_endpoint = bool(bridge_cfg.identity_endpoint)
        self.eps = float(bridge_cfg.eps)

        self.coupling_solver = str(bridge_cfg.coupling_solver).strip().lower()
        self.coupling_feature_mode = str(bridge_cfg.coupling_feature_mode).strip().lower()
        self.coupling_lowfreq_kernel = max(1, int(bridge_cfg.coupling_lowfreq_kernel))
        if self.coupling_lowfreq_kernel % 2 == 0:
            self.coupling_lowfreq_kernel += 1
        self.coupling_edge_weight = max(0.0, float(bridge_cfg.coupling_edge_weight))
        self.sinkhorn_epsilon = max(float(bridge_cfg.sinkhorn_epsilon), 1e-5)
        self.sinkhorn_iters = max(int(bridge_cfg.sinkhorn_iters), 1)
        self.sinkhorn_stabilize = bool(bridge_cfg.sinkhorn_stabilize)

        self.bridge_sigma = max(0.0, float(bridge_cfg.bridge_sigma))
        self.bridge_noise_mode = str(bridge_cfg.bridge_noise_mode).strip().lower()
        self.bridge_style_noise_kernel = max(1, int(bridge_cfg.bridge_style_noise_kernel))
        if self.bridge_style_noise_kernel % 2 == 0:
            self.bridge_style_noise_kernel += 1
        self.bridge_style_noise_flat_gamma = max(0.0, float(bridge_cfg.bridge_style_noise_flat_gamma))
        self.terminal_swd_weight = max(0.0, float(bridge_cfg.terminal_swd_weight))
        self.w_variance_penalty = max(0.0, float(bridge_cfg.w_variance_penalty))
        self.w_content_anchor = max(0.0, float(bridge_cfg.w_content_anchor))
        self.w_edge_anchor = max(0.0, float(bridge_cfg.w_edge_anchor))
        self.w_style_energy_floor = max(0.0, float(bridge_cfg.w_style_energy_floor))
        self.w_lowfreq_velocity = max(0.0, float(bridge_cfg.w_lowfreq_velocity))
        self.w_style_contrastive = max(0.0, float(bridge_cfg.w_style_contrastive))
        self.style_contrastive_temperature = max(1e-4, float(bridge_cfg.style_contrastive_temperature))
        self.style_contrastive_pool_size = max(1, int(bridge_cfg.style_contrastive_pool_size))
        self.w_residual_style_direction = max(0.0, float(bridge_cfg.w_residual_style_direction))
        self.w_semantic_entropy = max(0.0, float(bridge_cfg.w_semantic_entropy))
        self.semantic_entropy_target = max(0.0, float(bridge_cfg.semantic_entropy_target))
        self.w_spectral_amplitude = max(0.0, float(bridge_cfg.w_spectral_amplitude))
        self.spectral_amplitude_channels = max(1, int(bridge_cfg.spectral_amplitude_channels))
        self.spectral_amplitude_highpass = bool(bridge_cfg.spectral_amplitude_highpass)
        self.w_divergence = max(0.0, float(bridge_cfg.w_divergence))
        self.divergence_samples = max(1, int(bridge_cfg.divergence_samples))
        self.w_feature_riemannian = max(0.0, float(bridge_cfg.w_feature_riemannian))
        self.w_kantorovich = max(0.0, float(bridge_cfg.w_kantorovich))
        self.kantorovich_steps = max(0, int(bridge_cfg.kantorovich_steps))
        self.kantorovich_lr = max(1e-8, float(bridge_cfg.kantorovich_lr))
        self.sb_noise_epsilon = max(0.0, float(bridge_cfg.sb_noise_epsilon))
        self.retinex_target_blend = max(0.0, min(1.0, float(bridge_cfg.retinex_target_blend)))
        self.retinex_kernel_size = max(3, int(bridge_cfg.retinex_kernel_size))
        if self.retinex_kernel_size % 2 == 0:
            self.retinex_kernel_size += 1
        self.w_anisotropic_kinetic = max(0.0, float(bridge_cfg.w_anisotropic_kinetic))
        self.anisotropic_normal_weight = max(0.0, float(bridge_cfg.anisotropic_normal_weight))
        self.anisotropic_tangent_weight = max(0.0, float(bridge_cfg.anisotropic_tangent_weight))
        self.w_stokes_viscous = max(0.0, float(bridge_cfg.w_stokes_viscous))
        self.w_phase_separation = max(0.0, float(bridge_cfg.w_phase_separation))
        self.phase_gradient_weight = max(0.0, float(bridge_cfg.phase_gradient_weight))
        self.w_fourier_phase_lock = max(0.0, float(bridge_cfg.w_fourier_phase_lock))
        self.fourier_phase_lock_highpass = bool(bridge_cfg.fourier_phase_lock_highpass)
        self.w_head_color_tv = max(0.0, float(bridge_cfg.w_head_color_tv))
        self.w_head_color_energy = max(0.0, float(bridge_cfg.w_head_color_energy))
        self.w_head_amp_energy = max(0.0, float(bridge_cfg.w_head_amp_energy))
        self.w_warp_curl_reward = max(0.0, float(bridge_cfg.w_warp_curl_reward))
        self.kantorovich_potential = (
            KantorovichPotential(int(model_cfg.latent_channels), int(bridge_cfg.kantorovich_channels))
            if self.w_kantorovich > 0.0
            else None
        )
        self.style_energy_floor_ratio = max(0.0, float(bridge_cfg.style_energy_floor_ratio))
        self.anchor_pool_size = max(1, int(bridge_cfg.anchor_pool_size))
        if self.anchor_pool_size % 2 == 0:
            self.anchor_pool_size += 1
        self.terminal_num_steps = max(1, int(bridge_cfg.terminal_num_steps))
        self.terminal_swd_on_identity = bool(bridge_cfg.terminal_swd_on_identity)
        self.semantic_swd_num_projections = max(1, int(bridge_cfg.semantic_swd_num_projections))
        self.terminal_swd_mode = str(bridge_cfg.terminal_swd_mode).strip().lower()
        if self.terminal_swd_mode not in {"standard", "spectral_orthogonal", "semantic_quotient"}:
            self.terminal_swd_mode = "standard"
        self.spectral_swd_low_weight = max(0.0, float(bridge_cfg.spectral_swd_low_weight))
        self.spectral_swd_high_weight = max(0.0, float(bridge_cfg.spectral_swd_high_weight))
        self.spectral_swd_low_kernel = max(1, int(bridge_cfg.spectral_swd_low_kernel))
        if self.spectral_swd_low_kernel % 2 == 0:
            self.spectral_swd_low_kernel += 1
        self.semantic_quotient_bins = max(2, int(bridge_cfg.semantic_quotient_bins))

        self.w_kinetic = max(0.0, float(bridge_cfg.w_kinetic))
        self.w_flow = max(0.0, float(bridge_cfg.w_flow))
        self.w_curvature = max(0.0, float(bridge_cfg.w_curvature))
        self.curvature_dt = max(0.01, float(bridge_cfg.curvature_dt))
        self.kinetic_mode = str(bridge_cfg.kinetic_mode).strip().lower()
        if self.kinetic_mode not in {"endpoint", "path", "time_gated"}:
            self.kinetic_mode = "endpoint"
        self.kinetic_gate_exponent = max(0.0, float(bridge_cfg.kinetic_gate_exponent))

        self.normalize_eps = max(1e-8, float(bridge_cfg.normalize_eps))
        self.velocity_clamp = max(1.0, float(bridge_cfg.velocity_clamp))
        self.endpoint_clamp = max(self.velocity_clamp, float(bridge_cfg.endpoint_clamp))
        self.transport_cost = SWDTransportCost(config)

        distill_cfg = train_cfg.distill
        self.distill_velocity_weight = max(0.0, float(distill_cfg.get("velocity_weight", 1.0)))
        self.distill_endpoint_weight = max(0.0, float(distill_cfg.get("endpoint_weight", 0.0)))

    def potential_parameters(self) -> list[nn.Parameter]:
        if self.kantorovich_potential is None:
            return []
        return list(self.kantorovich_potential.parameters())

    def _ensure_potential_device(self, x: torch.Tensor) -> None:
        if self.kantorovich_potential is not None:
            self.kantorovich_potential.to(device=x.device)

    def _sanitize_tensor(self, x: torch.Tensor, *, clamp_value: float) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        return x.clamp(min=-clamp_value, max=clamp_value)

    def _sinkhorn_plan(self, cost: torch.Tensor) -> torch.Tensor:
        n_src, n_tgt = int(cost.shape[0]), int(cost.shape[1])
        mu = torch.full((n_src,), 1.0 / max(n_src, 1), device=cost.device, dtype=cost.dtype)
        nu = torch.full((n_tgt,), 1.0 / max(n_tgt, 1), device=cost.device, dtype=cost.dtype)
        if self.sinkhorn_stabilize:
            log_k = (-cost / self.sinkhorn_epsilon).clamp(min=-80.0, max=80.0)
            log_u = torch.zeros_like(mu)
            log_v = torch.zeros_like(nu)
            log_mu = torch.log(mu.clamp_min(1e-12))
            log_nu = torch.log(nu.clamp_min(1e-12))
            for _ in range(self.sinkhorn_iters):
                log_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
                log_v = log_nu - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
            plan = torch.exp(log_k + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        else:
            kernel = torch.exp((-cost / self.sinkhorn_epsilon).clamp(min=-80.0, max=80.0))
            u = torch.ones_like(mu)
            v = torch.ones_like(nu)
            for _ in range(self.sinkhorn_iters):
                u = mu / (kernel @ v).clamp_min(1e-12)
                v = nu / (kernel.transpose(0, 1) @ u).clamp_min(1e-12)
            plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return plan / plan.sum().clamp_min(1e-12)

    def _sample_from_plan(self, plan: torch.Tensor, target_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_probs = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        sampled_cols = torch.multinomial(row_probs, num_samples=1, replacement=True).squeeze(1)
        matched = target_group.index_select(0, sampled_cols)
        entropy = -(row_probs * row_probs.clamp_min(1e-12).log()).sum(dim=1).mean()
        return matched, entropy

    def _coupling_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.sqrt(gx.square().mean(dim=1, keepdim=True) + gy.square().mean(dim=1, keepdim=True) + 1e-8)

    def _kernel_lowpass(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel = max(1, int(kernel_size))
        if kernel <= 1:
            return x.float()
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=pad)

    def _coupling_feature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.coupling_feature_mode
        if mode in {"latent", "raw", ""}:
            return x
        low = self._kernel_lowpass(x, self.coupling_lowfreq_kernel)
        if mode == "lowfreq":
            return low
        if mode in {"lowfreq_edge", "edge_lowfreq"}:
            edge = self._coupling_edge_map(low)
            if self.coupling_edge_weight > 0.0:
                edge = edge * self.coupling_edge_weight
            return torch.cat([low, edge], dim=1)
        return x

    def _style_bridge_noise(self, content: torch.Tensor, matched_target: torch.Tensor) -> torch.Tensor:
        mode = self.bridge_noise_mode
        if mode in {"gaussian", "", "randn"}:
            return torch.randn_like(content)

        low = self._kernel_lowpass(matched_target, self.bridge_style_noise_kernel)
        noise = matched_target.float() - low
        if mode in {"style_highfreq_flat", "style_hf_flat", "flat_style_highfreq"} and self.bridge_style_noise_flat_gamma > 0.0:
            edge = self._coupling_edge_map(self._kernel_lowpass(content, self.bridge_style_noise_kernel))
            flat_mask = torch.exp(-self.bridge_style_noise_flat_gamma * edge)
            noise = noise * flat_mask

        noise_std = noise.flatten(1).std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
        return noise / noise_std

    def _solve_group_coupling(
        self,
        content_group: torch.Tensor,
        target_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost = self.transport_cost.pairwise_cost(
            self._coupling_feature_tensor(content_group),
            self._coupling_feature_tensor(target_group),
        )
        if self.coupling_solver == "hungarian":
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            row_t = torch.from_numpy(row_ind).to(device=cost.device, dtype=torch.long)
            col_t = torch.from_numpy(col_ind).to(device=cost.device, dtype=torch.long)
            matched = target_group.index_select(0, col_t)
            return matched, cost[row_t, col_t].mean(), cost.new_tensor(0.0)

        plan = self._sinkhorn_plan(cost)
        matched, entropy = self._sample_from_plan(plan, target_group)
        expected_cost = (plan * cost).sum() * float(cost.shape[0])
        return matched, expected_cost, entropy

    def _ot_match_targets(
        self,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched = torch.empty_like(target_style)
        total_cost = content.new_tensor(0.0, dtype=torch.float32)
        total_entropy = content.new_tensor(0.0, dtype=torch.float32)

        for style_id in torch.unique(target_style_id.long(), sorted=True).tolist():
            mask = target_style_id.long() == int(style_id)
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue

            content_group = content.index_select(0, indices)
            target_group = target_style.index_select(0, indices)
            if source_style_id is not None and self.identity_endpoint:
                same_style_mask = source_style_id.index_select(0, indices).long() == int(style_id)
            else:
                same_style_mask = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.bool)

            cross_indices = torch.nonzero(~same_style_mask, as_tuple=False).squeeze(1)
            if cross_indices.numel() > 0:
                matched_group, group_cost, group_entropy = self._solve_group_coupling(
                    content_group.index_select(0, cross_indices),
                    target_group.index_select(0, cross_indices),
                )
                matched.index_copy_(0, indices.index_select(0, cross_indices), matched_group)
                total_cost = total_cost + group_cost * float(cross_indices.numel())
                total_entropy = total_entropy + group_entropy * float(cross_indices.numel())

            if same_style_mask.any():
                same_indices = indices.index_select(0, torch.nonzero(same_style_mask, as_tuple=False).squeeze(1))
                matched.index_copy_(0, same_indices, content.index_select(0, same_indices))

        denom = max(int(content.shape[0]), 1)
        return matched, total_cost / float(denom), total_entropy / float(denom)

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        t_min = min(max(float(self.t_min), 0.0), 1.0)
        t_max = min(max(float(self.t_max), 0.0), 1.0)
        if self.bridge_sigma > 0.0:
            t_min = max(t_min, self.eps)
            t_max = min(t_max, 1.0 - self.eps)
        if t_max < t_min:
            t_min, t_max = t_max, t_min
        if abs(t_max - t_min) < self.eps:
            return torch.full((content.shape[0],), t_min, device=content.device, dtype=content.dtype)
        return torch.empty((content.shape[0],), device=content.device, dtype=content.dtype).uniform_(t_min, t_max)

    def _bridge_state_and_velocity(
        self,
        *,
        content: torch.Tensor,
        matched_target: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t4 = t.view(-1, 1, 1, 1)
        base = (1.0 - t4) * content + t4 * matched_target
        velocity = matched_target - content
        if self.bridge_sigma <= 0.0:
            return base, velocity

        bridge_var = (t * (1.0 - t)).clamp_min(self.eps)
        bridge_std = torch.sqrt(bridge_var).view(-1, 1, 1, 1)
        noise = self._style_bridge_noise(content, matched_target)
        x_t = base + (self.bridge_sigma * bridge_std) * noise
        d_std_dt = ((1.0 - 2.0 * t) / (2.0 * torch.sqrt(bridge_var))).view(-1, 1, 1, 1)
        return x_t, velocity + (self.bridge_sigma * d_std_dt) * noise

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, beta=0.5)
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.anchor_pool_size // 2
        return F.avg_pool2d(x.float(), kernel_size=self.anchor_pool_size, stride=1, padding=pad)

    def _retinex_target(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.retinex_target_blend <= 0.0:
            return style
        pad = self.retinex_kernel_size // 2
        illum_c = F.avg_pool2d(content.float(), self.retinex_kernel_size, stride=1, padding=pad)
        illum_s = F.avg_pool2d(style.float(), self.retinex_kernel_size, stride=1, padding=pad)
        ref_c = content.float() - illum_c
        ref_s = style.float() - illum_s
        mu_c = ref_c.mean(dim=(2, 3), keepdim=True)
        std_c = ref_c.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        mu_s = ref_s.mean(dim=(2, 3), keepdim=True)
        std_s = ref_s.std(dim=(2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
        target = illum_c + ((ref_c - mu_c) / std_c) * std_s + mu_s
        return style.float().lerp(target, self.retinex_target_blend)

    def _anisotropic_kinetic_loss(self, pred_velocity: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.w_anisotropic_kinetic <= 0.0:
            return content.new_tensor(0.0, dtype=torch.float32)
        channels = int(content.shape[1])
        kx = content.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = content.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        kx = kx.expand(channels, 1, 3, 3).contiguous()
        ky = ky.expand(channels, 1, 3, 3).contiguous()
        dx = F.conv2d(content.float(), kx, padding=1, groups=channels).detach()
        dy = F.conv2d(content.float(), ky, padding=1, groups=channels).detach()
        norm = torch.sqrt(dx.square() + dy.square() + self.normalize_eps)
        nx, ny = dx / norm, dy / norm
        tx, ty = -ny, nx
        v = pred_velocity.float()
        normal = 0.5 * ((v * nx).square().mean() + (v * ny).square().mean())
        tangent = 0.5 * ((v * tx).square().mean() + (v * ty).square().mean())
        return (normal * self.anisotropic_normal_weight + tangent * self.anisotropic_tangent_weight) * self.w_anisotropic_kinetic

    def _stokes_viscous_loss(self, pred_velocity: torch.Tensor) -> torch.Tensor:
        if self.w_stokes_viscous <= 0.0:
            return pred_velocity.new_tensor(0.0, dtype=torch.float32)
        channels = int(pred_velocity.shape[1])
        kernel = pred_velocity.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        kernel = kernel.expand(channels, 1, 3, 3).contiguous()
        lap = F.conv2d(pred_velocity.float(), kernel, padding=1, groups=channels)
        return lap.square().mean() * self.w_stokes_viscous

    def _phase_separation_loss(self, pred_endpoint: torch.Tensor) -> torch.Tensor:
        if self.w_phase_separation <= 0.0:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        state = F.instance_norm(pred_endpoint.float(), eps=1e-5)
        double_well = (state.square() - 1.0).square().mean()
        if self.phase_gradient_weight <= 0.0:
            return double_well * self.w_phase_separation
        dx = state[..., :, 1:] - state[..., :, :-1]
        dy = state[..., 1:, :] - state[..., :-1, :]
        grad = 0.5 * (dx.square().mean() + dy.square().mean())
        return (double_well + self.phase_gradient_weight * grad) * self.w_phase_separation

    def _fourier_phase_lock_loss(self, pred_endpoint: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.w_fourier_phase_lock <= 0.0:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        pred = pred_endpoint.float()
        base = content.float().detach()
        if self.fourier_phase_lock_highpass:
            pred = pred - self._lowpass(pred)
            base = base - self._lowpass(base)
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        base_fft = torch.fft.rfft2(base, norm="ortho")
        phase_delta = torch.angle(pred_fft * torch.conj(base_fft))
        return (1.0 - torch.cos(phase_delta)).mean() * self.w_fourier_phase_lock

    def _raw_head_tax_loss(self, model: TimeConditionedLANCETBridge, content: torch.Tensor) -> torch.Tensor:
        raw = getattr(model, "last_raw_diffeomorphic", None)
        if raw is None:
            return content.new_tensor(0.0, dtype=torch.float32)
        channels = int(content.shape[1])
        raw = raw.float()
        color = raw[:, :channels]
        total = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_head_color_energy > 0.0:
            total = total + color.square().mean() * self.w_head_color_energy
        if self.w_head_color_tv > 0.0:
            dx = color[..., :, 1:] - color[..., :, :-1]
            dy = color[..., 1:, :] - color[..., :-1, :]
            total = total + (dx.abs().mean() + dy.abs().mean()) * self.w_head_color_tv
        if self.w_head_amp_energy > 0.0 and raw.shape[1] >= channels + 3:
            mode = str(getattr(model, "diffeomorphic_head_mode", "standard"))
            if mode == "factorized_amp":
                amp = raw[:, channels : channels + 1]
                total = total + amp.square().mean() * self.w_head_amp_energy
        if self.w_warp_curl_reward > 0.0 and raw.shape[1] >= channels + 2:
            mode = str(getattr(model, "diffeomorphic_head_mode", "standard"))
            if mode == "factorized_amp":
                warp = raw[:, channels + 1 : channels + 3]
            else:
                warp = raw[:, channels : channels + 2]
            if warp.shape[1] == 2:
                dv_dx = warp[:, 1:2, :, 1:] - warp[:, 1:2, :, :-1]
                du_dy = warp[:, 0:1, 1:, :] - warp[:, 0:1, :-1, :]
                dv_dx = F.pad(dv_dx, (0, 1, 0, 0))
                du_dy = F.pad(du_dy, (0, 0, 0, 1))
                curl = dv_dx - du_dy
                # Negative term is intentional: it subsidizes coherent rotational flow.
                total = total - curl.abs().mean() * self.w_warp_curl_reward
        return total

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        channels = int(x.shape[1])
        kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        kx = kx.expand(channels, 1, 3, 3).contiguous()
        ky = ky.expand(channels, 1, 3, 3).contiguous()
        gx = F.conv2d(x.float(), kx, padding=1, groups=channels)
        gy = F.conv2d(x.float(), ky, padding=1, groups=channels)
        return torch.sqrt(gx.square() + gy.square() + self.normalize_eps)

    def _diff_x(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))

    def _diff_y(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))

    def _content_anchor_loss(self, pred_endpoint: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.w_content_anchor <= 0.0:
            return content.new_tensor(0.0, dtype=torch.float32)
        return F.l1_loss(self._lowpass(pred_endpoint), self._lowpass(content)) * self.w_content_anchor

    def _edge_anchor_loss(self, pred_endpoint: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if self.w_edge_anchor <= 0.0:
            return content.new_tensor(0.0, dtype=torch.float32)
        pred_edges = self._gradient_magnitude(self._lowpass(pred_endpoint))
        content_edges = self._gradient_magnitude(self._lowpass(content))
        return F.l1_loss(pred_edges, content_edges) * self.w_edge_anchor

    def _style_energy_floor_loss(self, pred_endpoint: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        if self.w_style_energy_floor <= 0.0:
            return target_style.new_tensor(0.0, dtype=torch.float32)
        pred_high = pred_endpoint.float() - self._lowpass(pred_endpoint)
        target_high = target_style.float() - self._lowpass(target_style)
        pred_energy = pred_high.std(dim=(2, 3), unbiased=False)
        target_energy = target_high.std(dim=(2, 3), unbiased=False)
        floor = target_energy.detach() * self.style_energy_floor_ratio
        return F.relu(floor - pred_energy).mean() * self.w_style_energy_floor

    def _lowfreq_velocity_loss(self, pred_velocity: torch.Tensor) -> torch.Tensor:
        if self.w_lowfreq_velocity <= 0.0:
            return pred_velocity.new_tensor(0.0, dtype=torch.float32)
        return self._lowpass(pred_velocity).abs().mean() * self.w_lowfreq_velocity

    def _style_signature(self, x: torch.Tensor) -> torch.Tensor:
        high = x.float() - self._lowpass(x)
        pooled = F.adaptive_avg_pool2d(high, self.style_contrastive_pool_size).flatten(1)
        stats = torch.cat(
            [
                pooled,
                high.mean(dim=(2, 3)),
                high.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        return F.normalize(stats, p=2, dim=1, eps=self.normalize_eps)

    def _style_contrastive_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.w_style_contrastive <= 0.0 or pred_endpoint.shape[0] < 2:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        pred_feat = self._style_signature(pred_endpoint)
        target_feat = self._style_signature(target_style).detach()
        logits = pred_feat @ target_feat.transpose(0, 1) / self.style_contrastive_temperature
        labels = torch.arange(pred_endpoint.shape[0], device=pred_endpoint.device)
        loss = F.cross_entropy(logits, labels)

        same_style = target_style_id.view(-1, 1).long() == target_style_id.view(1, -1).long()
        same_style.fill_diagonal_(False)
        if same_style.any():
            log_prob = F.log_softmax(logits, dim=1)
            soft_pos = -(log_prob * same_style.float()).sum(dim=1) / same_style.float().sum(dim=1).clamp_min(1.0)
            loss = 0.5 * (loss + soft_pos.mean())
        return loss * self.w_style_contrastive

    def _residual_style_direction_loss(
        self,
        pred_endpoint: torch.Tensor,
        content: torch.Tensor,
        target_style: torch.Tensor,
    ) -> torch.Tensor:
        if self.w_residual_style_direction <= 0.0:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        pred_delta = (pred_endpoint.float() - content.float()) - self._lowpass(pred_endpoint.float() - content.float())
        style_delta = (target_style.float() - content.float()) - self._lowpass(target_style.float() - content.float())
        pred_vec = pred_delta.flatten(1)
        style_vec = style_delta.detach().flatten(1)
        return (1.0 - F.cosine_similarity(pred_vec, style_vec, dim=1, eps=self.normalize_eps)).mean() * self.w_residual_style_direction

    def _semantic_entropy_loss(self, attn_plan: torch.Tensor | None, content: torch.Tensor) -> torch.Tensor:
        if self.w_semantic_entropy <= 0.0 or attn_plan is None:
            return content.new_tensor(0.0, dtype=torch.float32)
        probs = attn_plan.float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        target = entropy.new_tensor(self.semantic_entropy_target)
        return (entropy - target).square() * self.w_semantic_entropy

    def _spectral_amplitude_loss(self, pred_endpoint: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        if self.w_spectral_amplitude <= 0.0:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        channels = min(self.spectral_amplitude_channels, int(pred_endpoint.shape[1]), int(target_style.shape[1]))
        pred = pred_endpoint[:, :channels].float()
        target = target_style[:, :channels].float()
        if self.spectral_amplitude_highpass:
            pred = pred - self._lowpass(pred)
            target = target - self._lowpass(target)
        amp_pred = torch.log(torch.abs(torch.fft.rfft2(pred, norm="ortho")) + 1e-8)
        amp_target = torch.log(torch.abs(torch.fft.rfft2(target, norm="ortho")) + 1e-8)
        return F.l1_loss(amp_pred, amp_target.detach()) * self.w_spectral_amplitude

    def _divergence_penalty(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.w_divergence <= 0.0:
            return x.new_tensor(0.0, dtype=torch.float32)
        endpoint = (x + v).float()
        source = x.float().detach()
        src_dx = source[..., :, 1:] - source[..., :, :-1]
        src_dy = source[..., 1:, :] - source[..., :-1, :]
        end_dx = endpoint[..., :, 1:] - endpoint[..., :, :-1]
        end_dy = endpoint[..., 1:, :] - endpoint[..., :-1, :]
        src_energy = 0.5 * (src_dx.abs().mean(dim=(1, 2, 3)) + src_dy.abs().mean(dim=(1, 2, 3)))
        end_energy = 0.5 * (end_dx.abs().mean(dim=(1, 2, 3)) + end_dy.abs().mean(dim=(1, 2, 3)))
        collapse = F.relu(src_energy * 0.85 - end_energy)
        return collapse.mean() * self.w_divergence

    def _feature_riemannian_loss(
        self,
        model: TimeConditionedLANCETBridge,
        content: torch.Tensor,
        pred_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if self.w_feature_riemannian <= 0.0:
            return content.new_tensor(0.0, dtype=torch.float32)
        enc_in = getattr(model, "enc_in", None)
        enc_act = getattr(model, "enc_in_act", None)
        if enc_in is None or enc_act is None:
            return (pred_velocity.float() ** 2).mean() * self.w_feature_riemannian
        feat_0 = enc_act(enc_in(content.float()))
        feat_1 = enc_act(enc_in((content + pred_velocity).float()))
        return F.mse_loss(feat_1, feat_0.detach()) * self.w_feature_riemannian

    def _kantorovich_generator_loss(self, pred_endpoint: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        if self.w_kantorovich <= 0.0 or self.kantorovich_potential is None:
            return pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        self._ensure_potential_device(pred_endpoint)
        for param in self.kantorovich_potential.parameters():
            param.requires_grad_(False)
        pred_score = self.kantorovich_potential(pred_endpoint).mean()
        target_score = self.kantorovich_potential(target_style.detach()).mean()
        return (target_score - pred_score) * self.w_kantorovich

    def compute_kantorovich_critic(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.w_kantorovich <= 0.0 or self.kantorovich_potential is None or self.kantorovich_steps <= 0:
            return None
        self._ensure_potential_device(content)
        for param in self.kantorovich_potential.parameters():
            param.requires_grad_(True)
        with torch.no_grad():
            t_fixed = content.new_ones(content.shape[0])
            pred_velocity = model(content, t=t_fixed, style_id=target_style_id)
            pred_endpoint = self._sanitize_tensor(content + pred_velocity, clamp_value=self.endpoint_clamp)
        pred_score = self.kantorovich_potential(pred_endpoint.detach()).mean()
        target_score = self.kantorovich_potential(target_style.detach()).mean()
        return -(target_score - pred_score)

    def _terminal_active_indices(
        self,
        pred_endpoint: torch.Tensor,
        source_style_id: torch.Tensor | None,
        target_style_id: torch.Tensor,
    ) -> torch.Tensor:
        if source_style_id is None or self.terminal_swd_on_identity:
            return torch.arange(pred_endpoint.shape[0], device=pred_endpoint.device)
        mask = source_style_id.long() != target_style_id.long()
        return torch.nonzero(mask, as_tuple=False).squeeze(1)

    def _semantic_guided_swd(
        self,
        pred_hf: torch.Tensor,
        target_hf: torch.Tensor,
        k_matrix: torch.Tensor,
    ) -> torch.Tensor:
        bsz, channels, _, _ = pred_hf.shape
        pred_flat = pred_hf.float().view(bsz, channels, -1)
        target_flat = target_hf.float().view(bsz, channels, -1)

        semantic_scores = k_matrix.float().abs().mean(dim=1)
        if semantic_scores.shape[-1] > self.semantic_swd_num_projections:
            idx = torch.topk(
                semantic_scores,
                k=self.semantic_swd_num_projections,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
        else:
            idx = torch.arange(semantic_scores.shape[-1], device=k_matrix.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        gather_idx = idx.unsqueeze(1).expand(-1, channels, -1)
        semantic_axes = torch.gather(target_flat, 2, gather_idx)
        theta = F.normalize(semantic_axes, p=2, dim=1, eps=self.normalize_eps)
        proj_pred = torch.bmm(theta.transpose(1, 2), pred_flat)
        proj_target = torch.bmm(theta.transpose(1, 2), target_flat)
        proj_pred = torch.nan_to_num(proj_pred, nan=0.0, posinf=self.endpoint_clamp, neginf=-self.endpoint_clamp)
        proj_target = torch.nan_to_num(proj_target, nan=0.0, posinf=self.endpoint_clamp, neginf=-self.endpoint_clamp)
        mean_loss = (torch.sort(proj_pred, dim=-1).values - torch.sort(proj_target, dim=-1).values).abs().mean()
        if self.w_variance_penalty <= 0.0:
            return mean_loss
        pred_var = proj_pred.var(dim=-1, unbiased=False)
        target_var = proj_target.var(dim=-1, unbiased=False)
        var_loss = (pred_var - target_var).abs().mean()
        return mean_loss + self.w_variance_penalty * var_loss

    def _calc_terminal_swd_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_style: torch.Tensor,
        source_style_id: torch.Tensor | None,
        target_style_id: torch.Tensor,
        semantic_k: torch.Tensor | None = None,
        content: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.terminal_swd_weight <= 0.0:
            return None
        if semantic_k is not None:
            return self._semantic_guided_swd(pred_endpoint, target_style, semantic_k)
        active = self._terminal_active_indices(pred_endpoint, source_style_id, target_style_id)
        if active.numel() == 0:
            return None
        pred_active = pred_endpoint.index_select(0, active)
        target_active = target_style.index_select(0, active)
        if self.terminal_swd_mode == "spectral_orthogonal":
            return self._spectral_orthogonal_swd(pred_active, target_active)
        if self.terminal_swd_mode == "semantic_quotient" and content is not None:
            content_active = content.index_select(0, active)
            return self._semantic_quotient_swd(pred_active, target_active, content_active)
        return self.transport_cost.aligned_cost(pred_active, target_active)

    def _spectral_orthogonal_swd(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self.spectral_swd_low_kernel
        pred_low = F.avg_pool2d(pred.float(), kernel_size=kernel, stride=1, padding=kernel // 2)
        target_low = F.avg_pool2d(target.float(), kernel_size=kernel, stride=1, padding=kernel // 2)
        pred_high = pred.float() - pred_low
        target_high = target.float() - target_low
        total = pred.new_tensor(0.0, dtype=torch.float32)
        weight = self.spectral_swd_low_weight + self.spectral_swd_high_weight
        if self.spectral_swd_low_weight > 0.0:
            total = total + self.transport_cost.aligned_cost(pred_low, target_low) * self.spectral_swd_low_weight
        if self.spectral_swd_high_weight > 0.0:
            total = total + self.transport_cost.aligned_cost(pred_high, target_high) * self.spectral_swd_high_weight
        return total / max(weight, 1e-8)

    def _semantic_quotient_swd(self, pred: torch.Tensor, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        bsz, channels, h_dim, w_dim = pred.shape
        guide_low = F.avg_pool2d(guide.float().mean(dim=1, keepdim=True), kernel_size=5, stride=1, padding=2)
        flat_guide = guide_low.view(bsz, -1)
        pred_flat = pred.float().view(bsz, channels, -1)
        target_flat = target.float().view(bsz, channels, -1)
        losses: list[torch.Tensor] = []
        bins = self.semantic_quotient_bins
        for idx in range(bins):
            lo = torch.quantile(flat_guide.detach(), idx / bins, dim=1, keepdim=True)
            hi = torch.quantile(flat_guide.detach(), (idx + 1) / bins, dim=1, keepdim=True)
            if idx == bins - 1:
                mask = (flat_guide >= lo) & (flat_guide <= hi)
            else:
                mask = (flat_guide >= lo) & (flat_guide < hi)
            if not bool(mask.any().item()):
                continue
            masked_pred = pred_flat * mask.unsqueeze(1).to(dtype=pred_flat.dtype)
            masked_target = target_flat * mask.unsqueeze(1).to(dtype=target_flat.dtype)
            losses.append(
                self.transport_cost.aligned_cost(
                    masked_pred.view(bsz, channels, h_dim, w_dim),
                    masked_target.view(bsz, channels, h_dim, w_dim),
                )
            )
        if not losses:
            return self.transport_cost.aligned_cost(pred, target)
        return torch.stack(losses).mean()

    def _terminal_swd(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        matched_target: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.terminal_swd_weight <= 0.0:
            return None, content.new_tensor(0.0, dtype=torch.float32)

        endpoint = model.integrate(
            content,
            style_id=target_style_id,
            num_steps=self.terminal_num_steps,
            step_size=1.0,
            style_strength=1.0,
        )
        active = self._terminal_active_indices(endpoint, source_style_id, target_style_id)
        if active.numel() == 0:
            return None, endpoint.new_tensor(0.0, dtype=torch.float32)
        term = self.transport_cost.aligned_cost(
            endpoint.index_select(0, active),
            matched_target.index_select(0, active),
        )
        return term, endpoint.abs().mean().detach()

    def _compute_omf_details(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor | None]]:
        t_fixed = content.new_ones(content.shape[0])
        content_for_model = content
        target_for_loss = self._retinex_target(content, target_style)
        pred_velocity = model(content_for_model, t=t_fixed, style_id=target_style_id)
        pred_velocity = self._sanitize_tensor(pred_velocity, clamp_value=self.velocity_clamp)
        pred_endpoint = self._sanitize_tensor(content_for_model + pred_velocity, clamp_value=self.endpoint_clamp)
        attn_plan = model.last_semantic_attn
        semantic_k = model.last_semantic_k

        total_loss = content.new_tensor(0.0, dtype=torch.float32)
        flow_loss = content.new_tensor(0.0, dtype=torch.float32)
        ot_cost = content.new_tensor(0.0, dtype=torch.float32)
        plan_entropy = content.new_tensor(0.0, dtype=torch.float32)

        if self.w_flow > 0.0:
            if content.device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda", enabled=False)
            else:
                autocast_ctx = torch.autocast("cpu", enabled=False)
            with torch.no_grad():
                with autocast_ctx:
                    matched_target, ot_cost, plan_entropy = self._ot_match_targets(
                        content,
                        target_style,
                        target_style_id,
                        source_style_id,
                    )
            flow_loss = self._loss(pred_endpoint, matched_target) * self.w_flow
            total_loss = total_loss + flow_loss

        kinetic_loss = (pred_velocity.float() ** 2).mean() * self.w_kinetic if self.w_kinetic > 0.0 else content.new_tensor(0.0)
        anisotropic_kinetic = self._anisotropic_kinetic_loss(pred_velocity, content)
        stokes_viscous = self._stokes_viscous_loss(pred_velocity)
        phase_separation = self._phase_separation_loss(pred_endpoint)
        fourier_phase_lock = self._fourier_phase_lock_loss(pred_endpoint, content)
        head_tax = self._raw_head_tax_loss(model, content)
        total_loss = total_loss + kinetic_loss

        content_anchor = self._content_anchor_loss(pred_endpoint, content)
        edge_anchor = self._edge_anchor_loss(pred_endpoint, content)
        style_energy_floor = self._style_energy_floor_loss(pred_endpoint, target_for_loss)
        lowfreq_velocity = self._lowfreq_velocity_loss(pred_velocity)
        style_contrastive = self._style_contrastive_loss(pred_endpoint, target_for_loss, target_style_id)
        residual_style_direction = self._residual_style_direction_loss(pred_endpoint, content, target_for_loss)
        semantic_entropy = self._semantic_entropy_loss(attn_plan, content)
        spectral_amplitude = self._spectral_amplitude_loss(pred_endpoint, target_for_loss)
        divergence = self._divergence_penalty(content_for_model, pred_velocity)
        feature_riemannian = self._feature_riemannian_loss(model, content, pred_velocity)
        kantorovich = self._kantorovich_generator_loss(pred_endpoint, target_for_loss)
        total_loss = (
            total_loss
            + content_anchor
            + edge_anchor
            + style_energy_floor
            + lowfreq_velocity
            + style_contrastive
            + residual_style_direction
            + semantic_entropy
            + spectral_amplitude
            + divergence
            + feature_riemannian
            + kantorovich
            + anisotropic_kinetic
            + stokes_viscous
            + phase_separation
            + fourier_phase_lock
            + head_tax
        )

        terminal_swd = self._calc_terminal_swd_loss(
            pred_endpoint,
            target_for_loss,
            source_style_id,
            target_style_id,
            semantic_k=semantic_k,
            content=content,
        )
        terminal_loss = content.new_tensor(0.0, dtype=torch.float32)
        if terminal_swd is not None:
            terminal_loss = terminal_swd * self.terminal_swd_weight
            total_loss = total_loss + terminal_loss

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "kinetic_energy": kinetic_loss.detach(),
            "anisotropic_kinetic": anisotropic_kinetic.detach(),
            "stokes_viscous": stokes_viscous.detach(),
            "phase_separation": phase_separation.detach(),
            "fourier_phase_lock": fourier_phase_lock.detach(),
            "head_tax": head_tax.detach(),
            "terminal_swd": terminal_loss.detach(),
            "content_anchor": content_anchor.detach(),
            "edge_anchor": edge_anchor.detach(),
            "style_energy_floor": style_energy_floor.detach(),
            "lowfreq_velocity": lowfreq_velocity.detach(),
            "style_contrastive": style_contrastive.detach(),
            "residual_style_direction": residual_style_direction.detach(),
            "semantic_entropy": semantic_entropy.detach(),
            "spectral_amplitude": spectral_amplitude.detach(),
            "divergence": divergence.detach(),
            "feature_riemannian": feature_riemannian.detach(),
            "kantorovich": kantorovich.detach(),
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(0.0, dtype=torch.float32),
            "t_mean": t_fixed.mean().detach(),
            "velocity_abs": pred_velocity.abs().mean().detach(),
            "velocity_max": pred_velocity.abs().amax().detach(),
            "endpoint_abs": pred_endpoint.abs().mean().detach(),
            "endpoint_max": pred_endpoint.abs().amax().detach(),
            "semantic_attn_mean": attn_plan.mean().detach() if attn_plan is not None else content.new_tensor(0.0),
            "semantic_k_abs": semantic_k.abs().mean().detach() if semantic_k is not None else content.new_tensor(0.0),
        }
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()

        components = {
            "flow": flow_loss,
            "kinetic_energy": kinetic_loss,
            "anisotropic_kinetic": anisotropic_kinetic,
            "stokes_viscous": stokes_viscous,
            "phase_separation": phase_separation,
            "fourier_phase_lock": fourier_phase_lock,
            "head_tax": head_tax,
            "terminal_swd": terminal_loss,
            "content_anchor": content_anchor,
            "edge_anchor": edge_anchor,
            "style_energy_floor": style_energy_floor,
            "lowfreq_velocity": lowfreq_velocity,
            "style_contrastive": style_contrastive,
            "residual_style_direction": residual_style_direction,
            "semantic_entropy": semantic_entropy,
            "spectral_amplitude": spectral_amplitude,
            "divergence": divergence,
            "feature_riemannian": feature_riemannian,
            "kantorovich": kantorovich,
        }
        debug_state: Dict[str, torch.Tensor | None] = {
            "pred_velocity": pred_velocity.detach(),
            "pred_endpoint": pred_endpoint.detach(),
            "semantic_attn": attn_plan.detach() if attn_plan is not None else None,
            "semantic_k": semantic_k.detach() if semantic_k is not None else None,
            "content": content.detach(),
            "target_style": target_for_loss.detach(),
        }
        return metrics, components, debug_state

    def _compute_omf(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        metrics, _, _ = self._compute_omf_details(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        return metrics

    def compute(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.objective_mode == "omf":
            return self._compute_omf(
                model,
                content=content,
                target_style=target_style,
                target_style_id=target_style_id,
                source_style_id=source_style_id,
            )

        if content.device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = torch.autocast("cpu", enabled=False)
        with torch.no_grad():
            with autocast_ctx:
                matched_target, ot_cost, plan_entropy = self._ot_match_targets(
                    content,
                    target_style,
                    target_style_id,
                    source_style_id,
                )
                if self.retinex_target_blend > 0.0:
                    matched_target = self._retinex_target(content, matched_target)

        t = self._sample_t(content)
        x_t, target_velocity = self._bridge_state_and_velocity(content=content, matched_target=matched_target, t=t)
        if self.sb_noise_epsilon > 0.0:
            bridge_gate = torch.sqrt((t.float() * (1.0 - t.float())).clamp_min(0.0)).view(-1, 1, 1, 1)
            x_t = x_t + torch.randn_like(x_t) * (self.sb_noise_epsilon ** 0.5) * bridge_gate
        pred_velocity = model(x_t, t=t, style_id=target_style_id)
        flow_loss = self._loss(pred_velocity, target_velocity)
        total_loss = flow_loss

        kinetic_loss = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_kinetic > 0.0 and self.kinetic_mode in {"path", "time_gated"}:
            v_sq = pred_velocity.float() ** 2
            if self.kinetic_mode == "time_gated":
                gate = t.view(-1, 1, 1, 1).float() ** self.kinetic_gate_exponent
                kinetic_loss = (gate * v_sq).mean()
            else:
                kinetic_loss = v_sq.mean()
            total_loss = total_loss + kinetic_loss * self.w_kinetic
        anisotropic_kinetic = self._anisotropic_kinetic_loss(pred_velocity, content)
        stokes_viscous = self._stokes_viscous_loss(pred_velocity)
        phase_separation = self._phase_separation_loss(x_t + pred_velocity)
        fourier_phase_lock = self._fourier_phase_lock_loss(x_t + pred_velocity, content)
        head_tax = self._raw_head_tax_loss(model, content)
        total_loss = total_loss + anisotropic_kinetic + stokes_viscous + phase_separation + fourier_phase_lock + head_tax

        curvature_loss = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_curvature > 0.0:
            dt = self.curvature_dt
            t2 = (t + dt).clamp(max=self.t_max)
            pred_v2 = model(x_t + pred_velocity * dt, t=t2, style_id=target_style_id)
            curvature_loss = self._loss(pred_v2, pred_velocity)
            total_loss = total_loss + curvature_loss * self.w_curvature

        terminal_swd, endpoint_abs = self._terminal_swd(
            model,
            content=content,
            matched_target=matched_target,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        if terminal_swd is not None:
            total_loss = total_loss + terminal_swd * self.terminal_swd_weight

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "kinetic_energy": (kinetic_loss * self.w_kinetic).detach(),
            "anisotropic_kinetic": anisotropic_kinetic.detach(),
            "stokes_viscous": stokes_viscous.detach(),
            "phase_separation": phase_separation.detach(),
            "fourier_phase_lock": fourier_phase_lock.detach(),
            "head_tax": head_tax.detach(),
            "curvature": (curvature_loss * self.w_curvature).detach(),
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(self.bridge_sigma, dtype=torch.float32),
            "t_mean": t.mean().detach(),
            "velocity_abs": target_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
            "terminal_swd": terminal_swd.detach() if terminal_swd is not None else content.new_tensor(0.0),
        }
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()
        return metrics

    def compute_debug(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor] | Dict[str, torch.Tensor | None]]:
        if self.objective_mode != "omf":
            raise NotImplementedError("compute_debug currently supports objective_mode='omf' only.")
        metrics, components, state = self._compute_omf_details(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        return {"metrics": metrics, "components": components, "state": state}

    def compute_distill(
        self,
        model: TimeConditionedLANCETBridge,
        teacher_model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if content.device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = torch.autocast("cpu", enabled=False)
        with torch.no_grad():
            with autocast_ctx:
                matched_target, ot_cost, plan_entropy = self._ot_match_targets(
                    content,
                    target_style,
                    target_style_id,
                    source_style_id,
                )
            t = self._sample_t(content)
            x_t, _ = self._bridge_state_and_velocity(content=content, matched_target=matched_target, t=t)
            teacher_velocity = teacher_model(x_t, t=t, style_id=target_style_id)
            teacher_endpoint = None
            if self.distill_endpoint_weight > 0.0:
                teacher_endpoint = teacher_model.integrate(
                    content,
                    style_id=target_style_id,
                    num_steps=self.terminal_num_steps,
                    step_size=1.0,
                    style_strength=1.0,
                )

        pred_velocity = model(x_t, t=t, style_id=target_style_id)
        velocity_loss = F.mse_loss(pred_velocity, teacher_velocity)
        total_loss = velocity_loss * self.distill_velocity_weight

        endpoint_loss = content.new_tensor(0.0, dtype=torch.float32)
        endpoint_abs = content.new_tensor(0.0, dtype=torch.float32)
        if self.distill_endpoint_weight > 0.0 and teacher_endpoint is not None:
            student_endpoint = model.integrate(
                content,
                style_id=target_style_id,
                num_steps=self.terminal_num_steps,
                step_size=1.0,
                style_strength=1.0,
            )
            endpoint_loss = F.mse_loss(student_endpoint, teacher_endpoint)
            endpoint_abs = student_endpoint.abs().mean().detach()
            total_loss = total_loss + endpoint_loss * self.distill_endpoint_weight

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "distill_velocity": velocity_loss.detach(),
            "distill_endpoint": endpoint_loss.detach(),
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(self.bridge_sigma, dtype=torch.float32),
            "t_mean": t.mean().detach(),
            "velocity_abs": teacher_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
        }
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()
        return metrics
