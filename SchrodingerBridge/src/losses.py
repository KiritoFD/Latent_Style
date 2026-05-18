from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model import TimeConditionedLANCETBridge
from ot_cost import SWDTransportCost


class OTFlowMatchingObjective:
    def __init__(self, config: Dict) -> None:
        bridge_cfg = config.get("bridge", {})
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
        self.w_flow = max(0.0, float(bridge_cfg.get("w_flow", 0.0)))
        self.w_curvature = max(0.0, float(bridge_cfg.get("w_curvature", 0.0)))
        self.curvature_dt = max(0.01, float(bridge_cfg.get("curvature_dt", 0.15)))
        self.kinetic_mode = str(bridge_cfg.get("kinetic_mode", "endpoint")).strip().lower()
        if self.kinetic_mode not in {"endpoint", "path", "time_gated"}:
            self.kinetic_mode = "endpoint"
        self.kinetic_gate_exponent = max(0.0, float(bridge_cfg.get("kinetic_gate_exponent", 1.0)))
        self.semantic_swd_num_projections = max(1, int(bridge_cfg.get("semantic_swd_num_projections", 64)))
        self.kinetic_entropy_gate_weight = max(0.0, float(bridge_cfg.get("kinetic_entropy_gate_weight", 0.0)))
        self.normalize_eps = max(1e-8, float(bridge_cfg.get("normalize_eps", 1e-8)))
        self.velocity_clamp = max(1.0, float(bridge_cfg.get("velocity_clamp", 20.0)))
        self.endpoint_clamp = max(self.velocity_clamp, float(bridge_cfg.get("endpoint_clamp", 24.0)))
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
            return None, endpoint.abs().mean().detach()
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

    def _semantic_entropy_weight(self, attn_plan: torch.Tensor | None, output_size: tuple[int, int]) -> torch.Tensor | None:
        if attn_plan is None or self.kinetic_entropy_gate_weight <= 0.0:
            return None
        if attn_plan.ndim != 3 or attn_plan.shape[-1] <= 1:
            return None
        bsz, query_hw, key_hw = attn_plan.shape
        side = int(round(query_hw ** 0.5))
        if side * side != query_hw:
            return None
        attn = attn_plan.float().clamp_min(1e-8)
        entropy = -(attn * attn.log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(key_hw)), 1e-8)
        entropy = entropy.clamp(0.0, 1.0).view(bsz, 1, side, side)
        if entropy.shape[-2:] != output_size:
            entropy = F.interpolate(entropy, size=output_size, mode="bilinear", align_corners=False)
        return entropy

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

    def compute(
        self,
        model: TimeConditionedLANCETBridge,
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

        kinetic_loss = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_kinetic > 0.0:
            v_sq = pred_velocity.float() ** 2
            if self.kinetic_mode == "endpoint":
                kinetic_loss = v_sq.mean()
            elif self.kinetic_mode == "time_gated":
                gate = t.view(-1, 1, 1, 1).float() ** self.kinetic_gate_exponent
                kinetic_loss = (gate * v_sq).mean()
            else:
                kinetic_loss = v_sq.mean()
            total_loss = total_loss + kinetic_loss * self.w_kinetic

        curvature_loss = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_curvature > 0.0:
            dt = self.curvature_dt
            t2 = (t + dt).clamp(max=self.t_max)
            x_t2 = x_t + pred_velocity * dt
            pred_v2 = model(
                x_t2,
                t=t2,
                style_id=target_style_id,
            )
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

        attn_plan = model.last_semantic_attn
        semantic_k = model.last_semantic_k
        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "kinetic_energy": (kinetic_loss * self.w_kinetic).detach(),
            "curvature": (curvature_loss * self.w_curvature).detach(),
            "ot_cost": ot_cost.detach(),
            "terminal_swd": terminal_swd.detach() if terminal_swd is not None else content.new_tensor(0.0, dtype=torch.float32),
            "t_mean": t.mean().detach(),
            "velocity_abs": pred_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
        }
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
