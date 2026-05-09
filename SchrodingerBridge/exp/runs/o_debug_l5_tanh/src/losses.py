from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model import TimeConditionedLANCETBridge
from ot_cost import SWDTransportCost


def calc_latent_patch_nce_loss(
    pred: torch.Tensor,
    content: torch.Tensor,
    *,
    num_patches: int = 256,
    temperature: float = 0.07,
    normalize_eps: float = 1e-8,
    logit_clamp: float = 50.0,
) -> torch.Tensor:
    """
    Pure latent-space PatchNCE that anchors local semantics at matched positions.
    """
    bsz, channels, height, width = pred.shape
    total_patches = height * width
    num_patches = max(1, min(int(num_patches), int(total_patches)))
    if total_patches <= 1 or num_patches <= 1:
        return pred.new_tensor(0.0, dtype=torch.float32)

    feat_q = pred.reshape(bsz, channels, total_patches).transpose(1, 2)
    feat_k = content.reshape(bsz, channels, total_patches).transpose(1, 2)
    perm = torch.randperm(total_patches, device=pred.device)[:num_patches]

    q_sampled = feat_q[:, perm, :]
    k_sampled = feat_k[:, perm, :]

    q_norm = F.normalize(q_sampled.float(), p=2, dim=-1, eps=normalize_eps)
    k_norm = F.normalize(k_sampled.float(), p=2, dim=-1, eps=normalize_eps)
    logits = torch.bmm(q_norm, k_norm.transpose(1, 2)) / max(float(temperature), 1e-6)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=logit_clamp, neginf=-logit_clamp)
    logits = logits.clamp(min=-logit_clamp, max=logit_clamp)
    labels = torch.arange(num_patches, device=pred.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    return F.cross_entropy(logits.reshape(-1, num_patches), labels.reshape(-1))


def calc_low_freq_structure_loss(
    pred: torch.Tensor,
    content: torch.Tensor,
    *,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    Color-decoupled low-frequency anchor that preserves only coarse geometry.
    """
    kernel_size = max(1, int(kernel_size))
    pred_in = F.instance_norm(pred.float(), eps=1e-3)
    content_in = F.instance_norm(content.float(), eps=1e-3)
    if kernel_size <= 1:
        return F.mse_loss(pred_in, content_in)

    pad = kernel_size // 2
    pred_low = F.avg_pool2d(pred_in, kernel_size=kernel_size, stride=1, padding=pad)
    content_low = F.avg_pool2d(content_in, kernel_size=kernel_size, stride=1, padding=pad)
    return F.mse_loss(pred_low, content_low)


class OTFlowMatchingObjective:
    def __init__(self, config: Dict) -> None:
        bridge_cfg = config.get("bridge", {})
        self.objective_mode = str(bridge_cfg.get("objective_mode", "flow_matching")).strip().lower()
        self.t_min = float(bridge_cfg.get("t_min", 0.0))
        self.t_max = float(bridge_cfg.get("t_max", 1.0))
        self.loss_type = str(bridge_cfg.get("loss_type", "mse")).strip().lower()
        self.identity_endpoint = bool(bridge_cfg.get("identity_endpoint", False))
        self.eps = float(bridge_cfg.get("eps", 1e-4))
        self.coupling_solver = str(bridge_cfg.get("coupling_solver", "sinkhorn")).strip().lower()
        self.sinkhorn_epsilon = max(float(bridge_cfg.get("sinkhorn_epsilon", 0.05)), 1e-5)
        self.sinkhorn_iters = max(int(bridge_cfg.get("sinkhorn_iters", 60)), 1)
        self.sinkhorn_stabilize = bool(bridge_cfg.get("sinkhorn_stabilize", True))
        self.bridge_sigma = max(0.0, float(bridge_cfg.get("bridge_sigma", 0.05)))
        self.terminal_swd_weight = max(0.0, float(bridge_cfg.get("terminal_swd_weight", 0.1)))
        self.terminal_num_steps = max(1, int(bridge_cfg.get("terminal_num_steps", 4)))
        self.terminal_swd_on_identity = bool(bridge_cfg.get("terminal_swd_on_identity", False))
        self.w_kinetic = max(0.0, float(bridge_cfg.get("w_kinetic", 1.0)))
        self.w_color = max(0.0, float(bridge_cfg.get("w_color", 15.0)))
        self.w_repulsive = max(0.0, float(bridge_cfg.get("w_repulsive", 10.0)))
        self.w_flow = max(0.0, float(bridge_cfg.get("w_flow", 0.0)))
        self.w_nce = max(0.0, float(bridge_cfg.get("w_nce", 0.0)))
        self.w_low_freq = max(0.0, float(bridge_cfg.get("w_low_freq", 0.0)))
        self.w_cycle = max(0.0, float(bridge_cfg.get("w_cycle", 0.0)))
        self.nce_num_patches = max(1, int(bridge_cfg.get("nce_num_patches", 256)))
        self.nce_temperature = max(1e-6, float(bridge_cfg.get("nce_temperature", 0.07)))
        self.low_freq_kernel_size = max(1, int(bridge_cfg.get("low_freq_kernel_size", 7)))
        self.semantic_swd_num_projections = max(1, int(bridge_cfg.get("semantic_swd_num_projections", 64)))
        self.swd_use_high_freq = bool(bridge_cfg.get("swd_use_high_freq", True))
        self.color_patch_size = max(1, int(bridge_cfg.get("omf_color_patch_size", 5)))
        self.repulsive_pool_size = max(1, int(bridge_cfg.get("repulsive_pool_size", 4)))
        self.repulsive_temperature = max(1e-4, float(bridge_cfg.get("repulsive_temperature", 0.25)))
        self.normalize_eps = max(1e-8, float(bridge_cfg.get("normalize_eps", 1e-8)))
        self.logit_clamp = max(1.0, float(bridge_cfg.get("logit_clamp", 50.0)))
        self.velocity_clamp = max(1.0, float(bridge_cfg.get("velocity_clamp", 20.0)))
        self.endpoint_clamp = max(self.velocity_clamp, float(bridge_cfg.get("endpoint_clamp", 24.0)))
        self.similarity_clamp = max(1.0, float(bridge_cfg.get("similarity_clamp", 50.0)))
        self.transport_cost = SWDTransportCost(config)
        train_cfg = config.get("training", {})
        distill_cfg = train_cfg.get("distill", {})
        self.distill_velocity_weight = max(0.0, float(distill_cfg.get("velocity_weight", 1.0)))
        self.distill_endpoint_weight = max(0.0, float(distill_cfg.get("endpoint_weight", 0.0)))

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

    def _solve_group_coupling(
        self,
        content_group: torch.Tensor,
        target_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost = self.transport_cost.pairwise_cost(content_group, target_group)
        if self.coupling_solver == "hungarian":
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            row_t = torch.from_numpy(row_ind).to(device=cost.device, dtype=torch.long)
            col_t = torch.from_numpy(col_ind).to(device=cost.device, dtype=torch.long)
            matched = target_group.index_select(0, col_t)
            selected_cost = cost[row_t, col_t].mean()
            entropy = cost.new_tensor(0.0)
            return matched, selected_cost, entropy

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
        unique_styles = torch.unique(target_style_id.long(), sorted=True)

        for style_id in unique_styles.tolist():
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
        noise = torch.randn_like(content)
        x_t = base + (self.bridge_sigma * bridge_std) * noise
        d_std_dt = ((1.0 - 2.0 * t) / (2.0 * torch.sqrt(bridge_var))).view(-1, 1, 1, 1)
        target_velocity = velocity + (self.bridge_sigma * d_std_dt) * noise
        return x_t, target_velocity

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, beta=0.5)
        return F.mse_loss(pred, target)

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
        if source_style_id is None or self.terminal_swd_on_identity:
            mask = torch.ones_like(target_style_id, dtype=torch.bool)
        else:
            mask = source_style_id.long() != target_style_id.long()
        active = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return None, endpoint.new_tensor(0.0, dtype=torch.float32)
        term = self.transport_cost.aligned_cost(
            endpoint.index_select(0, active),
            matched_target.index_select(0, active),
        )
        return term, endpoint.abs().mean().detach()

    def _calc_terminal_swd_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_style: torch.Tensor,
        source_style_id: torch.Tensor | None,
        target_style_id: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.terminal_swd_weight <= 0.0:
            return None
        if source_style_id is None or self.terminal_swd_on_identity:
            active = torch.arange(pred_endpoint.shape[0], device=pred_endpoint.device)
        else:
            mask = source_style_id.long() != target_style_id.long()
            active = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return None
        return self.transport_cost.aligned_cost(
            pred_endpoint.index_select(0, active),
            target_style.index_select(0, active),
        )

    def _avg_pool_exact(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if patch_size <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size, ceil_mode=False)

    def _calc_local_contextual_color_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_style: torch.Tensor,
        *,
        patch_size: int,
    ) -> torch.Tensor:
        pred_low = self._avg_pool_exact(pred_endpoint.float(), patch_size)
        target_low = self._avg_pool_exact(target_style.float(), patch_size)

        bsz, channels, h_dim, w_dim = pred_low.shape
        q = pred_low.mean(dim=1, keepdim=True).flatten(2).transpose(1, 2)
        k = target_low.mean(dim=1, keepdim=True).flatten(2)
        q_norm = F.normalize(q, dim=-1, eps=self.normalize_eps)
        k_norm = F.normalize(k, dim=1, eps=self.normalize_eps)
        attn = torch.bmm(q_norm, k_norm)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=self.logit_clamp, neginf=-self.logit_clamp)
        attn = F.softmax((attn / 0.07).clamp(min=-self.logit_clamp, max=self.logit_clamp), dim=-1)

        v = target_low.flatten(2).transpose(1, 2)
        warped = torch.bmm(attn, v).transpose(1, 2).reshape(bsz, channels, h_dim, w_dim)
        return F.l1_loss(pred_low, warped)

    def _freq_split(self, x: torch.Tensor, kernel_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_size = max(1, int(kernel_size))
        if kernel_size <= 1:
            low = x.float()
            return low, x.float() - low
        low = F.avg_pool2d(x.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        high = x.float() - low
        return low, high

    def _infer_attn_hw(self, attn: torch.Tensor) -> tuple[int, int] | None:
        hw = int(attn.shape[2])
        side = int(round(hw ** 0.5))
        if side * side != hw:
            return None
        return side, side

    def _barycentric_target(self, target_style: torch.Tensor, attn: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
        bsz, channels, _, _ = target_style.shape
        attn_hw = self._infer_attn_hw(attn)
        if attn_hw is None:
            if target_style.shape[-2:] != output_size:
                return F.interpolate(target_style.float(), size=output_size, mode="bilinear", align_corners=False)
            return target_style.float()
        target_resized = target_style.float()
        if attn_hw is not None and target_resized.shape[-2:] != attn_hw:
            target_resized = F.interpolate(target_resized, size=attn_hw, mode="bilinear", align_corners=False)
        value = target_resized.view(bsz, channels, -1).transpose(1, 2)
        projected = torch.bmm(attn.float(), value).transpose(1, 2)
        if attn_hw is None:
            projected = projected.view(bsz, channels, 1, -1)
        else:
            projected = projected.view(bsz, channels, attn_hw[0], attn_hw[1])
        if projected.shape[-2:] != output_size:
            projected = F.interpolate(projected, size=output_size, mode="bilinear", align_corners=False)
        return projected

    def _cosine_lock_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8).mean()

    def _semantic_guided_swd(
        self,
        pred_hf: torch.Tensor,
        target_hf: torch.Tensor,
        k_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Slice high-frequency distributions along semantic key directions instead of random axes.
        """
        bsz, channels, _, _ = pred_hf.shape
        pred_flat = pred_hf.float().view(bsz, channels, -1)
        target_flat = target_hf.float().view(bsz, channels, -1)

        # The deepest semantic keys live in backbone feature space rather than
        # latent channel space, so we use them to pick the most style-relevant
        # spatial positions and then form projection axes from target latents.
        semantic_scores = k_matrix.float().abs().mean(dim=1)
        if semantic_scores.shape[-1] > self.semantic_swd_num_projections:
            idx = torch.topk(semantic_scores, k=self.semantic_swd_num_projections, dim=-1, largest=True, sorted=False).indices
        else:
            idx = torch.arange(semantic_scores.shape[-1], device=k_matrix.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        gather_idx = idx.unsqueeze(1).expand(-1, channels, -1)
        semantic_axes = torch.gather(target_flat, 2, gather_idx)
        theta = F.normalize(semantic_axes, p=2, dim=1, eps=self.normalize_eps)
        proj_pred = torch.bmm(theta.transpose(1, 2), pred_flat)
        proj_target = torch.bmm(theta.transpose(1, 2), target_flat)
        proj_pred = torch.nan_to_num(proj_pred, nan=0.0, posinf=self.endpoint_clamp, neginf=-self.endpoint_clamp)
        proj_target = torch.nan_to_num(proj_target, nan=0.0, posinf=self.endpoint_clamp, neginf=-self.endpoint_clamp)
        proj_pred_sorted, _ = torch.sort(proj_pred, dim=-1)
        proj_target_sorted, _ = torch.sort(proj_target, dim=-1)
        return (proj_pred_sorted - proj_target_sorted).abs().mean()

    def _collect_repulsive_components(
        self,
        pred_endpoint: torch.Tensor,
        cross_domain_mask: torch.Tensor,
        *,
        target_style_id: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        active = torch.nonzero(cross_domain_mask, as_tuple=False).squeeze(1)
        if active.numel() <= 1:
            zero = pred_endpoint.new_tensor(0.0, dtype=torch.float32)
            return None, zero

        pooled = F.avg_pool2d(
            pred_endpoint.index_select(0, active).float(),
            kernel_size=self.repulsive_pool_size,
            stride=self.repulsive_pool_size,
            ceil_mode=False,
        )
        feats = F.normalize(pooled.flatten(1), dim=1, eps=self.normalize_eps)
        tgt_ids = target_style_id.index_select(0, active).long()
        sim = feats @ feats.transpose(0, 1)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=self.similarity_clamp, neginf=-self.similarity_clamp)
        sim = sim.clamp(min=-self.similarity_clamp, max=self.similarity_clamp)
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        same_target = tgt_ids.unsqueeze(1) == tgt_ids.unsqueeze(0)
        pair_mask = same_target & (~eye)
        if not pair_mask.any():
            zero = pred_endpoint.new_tensor(0.0, dtype=torch.float32)
            return None, zero
        repel_raw = torch.logsumexp(sim[pair_mask] / self.repulsive_temperature, dim=0)
        mean_pair_sim = sim[pair_mask].mean().detach()
        return repel_raw, mean_pair_sim

    def _compute_omf(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        t_fixed = content.new_ones(content.shape[0])
        pred_velocity = model(
            content,
            t=t_fixed,
            style_id=target_style_id,
        )
        pred_velocity = self._sanitize_tensor(pred_velocity, clamp_value=self.velocity_clamp)
        pred_endpoint = content + pred_velocity
        pred_endpoint = self._sanitize_tensor(pred_endpoint, clamp_value=self.endpoint_clamp)
        attn_plan = model.last_semantic_attn
        semantic_k = model.last_semantic_k

        total_loss = content.new_tensor(0.0, dtype=torch.float32)
        ot_cost = content.new_tensor(0.0, dtype=torch.float32)
        plan_entropy = content.new_tensor(0.0, dtype=torch.float32)
        matched_target: torch.Tensor | None = None
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

        metrics: Dict[str, torch.Tensor] = {
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(0.0, dtype=torch.float32),
            "t_mean": t_fixed.mean().detach(),
            "velocity_abs": pred_velocity.abs().mean().detach(),
            "velocity_max": pred_velocity.abs().amax().detach(),
            "endpoint_abs": pred_endpoint.abs().mean().detach(),
            "endpoint_max": pred_endpoint.abs().amax().detach(),
            "semantic_attn_mean": attn_plan.mean().detach() if attn_plan is not None else content.new_tensor(0.0, dtype=torch.float32),
            "semantic_attn_max": attn_plan.abs().amax().detach() if attn_plan is not None else content.new_tensor(0.0, dtype=torch.float32),
            "semantic_k_abs": semantic_k.abs().mean().detach() if semantic_k is not None else content.new_tensor(0.0, dtype=torch.float32),
            "semantic_k_max": semantic_k.abs().amax().detach() if semantic_k is not None else content.new_tensor(0.0, dtype=torch.float32),
        }

        if self.w_flow > 0.0 and matched_target is not None:
            flow_loss = self._loss(pred_endpoint, matched_target)
            flow_weighted = flow_loss * self.w_flow
            total_loss = total_loss + flow_weighted
            metrics["flow"] = flow_weighted.detach()
        else:
            metrics["flow"] = content.new_tensor(0.0, dtype=torch.float32)

        kinetic_loss = (pred_velocity.float() ** 2).mean()
        if self.w_kinetic > 0.0:
            kinetic_weighted = kinetic_loss * self.w_kinetic
            total_loss = total_loss + kinetic_weighted
            metrics["kinetic_energy"] = kinetic_weighted.detach()
        else:
            metrics["kinetic_energy"] = content.new_tensor(0.0, dtype=torch.float32)

        if not self.swd_use_high_freq:
            if semantic_k is not None:
                swd_loss = self._semantic_guided_swd(pred_endpoint, target_style, semantic_k)
            else:
                swd_loss = self._calc_terminal_swd_loss(
                    pred_endpoint,
                    target_style,
                    source_style_id,
                    target_style_id,
                )
            if swd_loss is not None:
                total_loss = total_loss + swd_loss * self.terminal_swd_weight
                metrics["terminal_swd"] = (swd_loss * self.terminal_swd_weight).detach()
            else:
                metrics["terminal_swd"] = content.new_tensor(0.0, dtype=torch.float32)
            metrics["low_freq_anchor"] = content.new_tensor(0.0, dtype=torch.float32)
        else:
            pred_low, pred_high = self._freq_split(pred_endpoint, kernel_size=self.low_freq_kernel_size)
            content_low, _ = self._freq_split(content, kernel_size=self.low_freq_kernel_size)
            _, target_high = self._freq_split(target_style, kernel_size=self.low_freq_kernel_size)

            # AdaIN color anchoring: preserve the coarse geometry from content_low,
            # but force its global palette statistics to match the target style.
            b_low, c_low, h_low, w_low = content_low.shape
            content_flat = content_low.reshape(b_low, c_low, -1)
            target_flat = target_style.reshape(target_style.shape[0], target_style.shape[1], -1)

            mu_c = content_flat.mean(dim=2, keepdim=True)
            std_c = content_flat.std(dim=2, keepdim=True) + 1e-5
            mu_t = target_flat.mean(dim=2, keepdim=True)
            std_t = target_flat.std(dim=2, keepdim=True) + 1e-5

            color_anchored_low = ((content_flat - mu_c) / std_c) * std_t + mu_t
            color_anchored_low = color_anchored_low.reshape(b_low, c_low, h_low, w_low)

            if self.w_low_freq > 0.0:
                # Low-frequency channels carry exposure and coarse tonal layout, so
                # we lock them to a content-preserving, target-colored anchor.
                low_freq_loss = F.l1_loss(pred_low, color_anchored_low.detach())
                total_loss = total_loss + low_freq_loss * self.w_low_freq
                metrics["low_freq_anchor"] = (low_freq_loss * self.w_low_freq).detach()
            else:
                metrics["low_freq_anchor"] = content.new_tensor(0.0, dtype=torch.float32)

            if semantic_k is not None:
                swd_loss = self._semantic_guided_swd(pred_high, target_high, semantic_k)
            else:
                swd_loss = self._calc_terminal_swd_loss(
                    pred_high,
                    target_high,
                    source_style_id,
                    target_style_id,
                )
            if swd_loss is not None:
                total_loss = total_loss + swd_loss * self.terminal_swd_weight
                metrics["terminal_swd"] = (swd_loss * self.terminal_swd_weight).detach()
            else:
                metrics["terminal_swd"] = content.new_tensor(0.0, dtype=torch.float32)

        if self.w_color > 0.0:
            color_loss = self._calc_local_contextual_color_loss(
                pred_endpoint,
                target_style,
                patch_size=self.color_patch_size,
            )
            total_loss = total_loss + color_loss * self.w_color
            metrics["color"] = (color_loss * self.w_color).detach()
        else:
            metrics["color"] = content.new_tensor(0.0, dtype=torch.float32)

        if self.w_nce > 0.0:
            nce_loss = calc_latent_patch_nce_loss(
                pred_endpoint,
                content,
                num_patches=self.nce_num_patches,
                temperature=self.nce_temperature,
                normalize_eps=self.normalize_eps,
                logit_clamp=self.logit_clamp,
            )
            total_loss = total_loss + nce_loss * self.w_nce
            metrics["patch_nce"] = (nce_loss * self.w_nce).detach()
        else:
            metrics["patch_nce"] = content.new_tensor(0.0, dtype=torch.float32)

        if self.w_cycle > 0.0 and source_style_id is not None:
            cycle_velocity = model(
                pred_endpoint,
                t=t_fixed,
                style_id=source_style_id,
            )
            cycle_velocity = self._sanitize_tensor(cycle_velocity, clamp_value=self.velocity_clamp)
            z_aba = pred_endpoint + cycle_velocity
            z_aba = self._sanitize_tensor(z_aba, clamp_value=self.endpoint_clamp)
            metrics["cycle_velocity_max"] = cycle_velocity.abs().amax().detach()
            metrics["cycle_endpoint_max"] = z_aba.abs().amax().detach()
            cycle_loss = self._cosine_lock_loss(z_aba, content)
            total_loss = total_loss + cycle_loss * self.w_cycle
            metrics["cycle"] = (cycle_loss * self.w_cycle).detach()
        else:
            metrics["cycle"] = content.new_tensor(0.0, dtype=torch.float32)
            metrics["cycle_velocity_max"] = content.new_tensor(0.0, dtype=torch.float32)
            metrics["cycle_endpoint_max"] = content.new_tensor(0.0, dtype=torch.float32)

        xid_mask = (
            source_style_id.long() != target_style_id.long()
            if source_style_id is not None
            else torch.ones_like(target_style_id, dtype=torch.bool)
        )
        repel_raw, mean_pair_sim = self._collect_repulsive_components(
            pred_endpoint,
            xid_mask,
            target_style_id=target_style_id,
        )
        if repel_raw is not None and self.w_repulsive > 0.0:
            repel_clamped = torch.clamp(repel_raw, max=1.0)
            total_loss = total_loss + repel_clamped * self.w_repulsive
            metrics["repulsive"] = (repel_clamped * self.w_repulsive).detach()
        else:
            metrics["repulsive"] = content.new_tensor(0.0, dtype=torch.float32)
        metrics["repulsive_pair_sim"] = mean_pair_sim

        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()

        metrics["loss"] = total_loss
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

        t = self._sample_t(content)
        x_t, target_velocity = self._bridge_state_and_velocity(
            content=content,
            matched_target=matched_target,
            t=t,
        )
        pred_velocity = model(
            x_t,
            t=t,
            style_id=target_style_id,
        )
        flow_loss = self._loss(pred_velocity, target_velocity)
        total_loss = flow_loss

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
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(self.bridge_sigma, dtype=torch.float32),
            "t_mean": t.mean().detach(),
            "velocity_abs": target_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
        }
        if terminal_swd is not None:
            metrics["terminal_swd"] = terminal_swd.detach()
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()
        return metrics

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
            x_t, _ = self._bridge_state_and_velocity(
                content=content,
                matched_target=matched_target,
                t=t,
            )
            teacher_velocity = teacher_model(
                x_t,
                t=t,
                style_id=target_style_id,
            )
            teacher_endpoint = None
            if self.distill_endpoint_weight > 0.0:
                teacher_endpoint = teacher_model.integrate(
                    content,
                    style_id=target_style_id,
                    num_steps=self.terminal_num_steps,
                    step_size=1.0,
                    style_strength=1.0,
                )

        pred_velocity = model(
            x_t,
            t=t,
            style_id=target_style_id,
        )
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
