from __future__ import annotations

import math
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import BridgeConfig, ExperimentConfig, TrainingConfig
from model import TimeConditionedLANCETBridge
from ot_cost import SWDTransportCost
from style_families import (
    resolves_exact_brownian_schedule,
    validate_i2sb_contract,
    validate_pure_latent_contract,
)


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
            model_cfg = config.get("model", {})

        self.objective_mode = str(bridge_cfg.objective_mode).strip().lower()
        if self.objective_mode in {"i2sb", "i2sb_endpoint", "bridge_endpoint"}:
            self.objective_mode = "i2sb_endpoint"
        self.t_min = float(bridge_cfg.t_min)
        self.t_max = float(bridge_cfg.t_max)
        self.loss_type = str(bridge_cfg.loss_type).strip().lower()
        self.transport_prediction_mode = str(getattr(model_cfg, "transport_prediction_mode", "velocity")).strip().lower()
        validate_i2sb_contract(
            solver_family=str(getattr(model_cfg, "solver_family", "euler_legacy")),
            transport_prediction_mode=self.transport_prediction_mode,
            objective_mode=self.objective_mode,
            loss_type=str(getattr(bridge_cfg, "loss_type", "")),
            bridge_noise_schedule=str(getattr(bridge_cfg, "bridge_noise_schedule", "auto")),
        )
        validate_pure_latent_contract(
            tokenizer_family=str(getattr(model_cfg, "tokenizer_family", "legacy_factorized")),
            style_tokenizer=str(getattr(model_cfg, "style_tokenizer", "")),
            semantic_supervision_family=str(getattr(bridge_cfg, "semantic_supervision_family", "legacy_terminal_swd")),
            dino_masked_swd_weight=float(getattr(bridge_cfg, "dino_masked_swd_weight", 0.0)),
            tokenizer_content_adaptive=bool(getattr(model_cfg, "tokenizer_content_adaptive", False)),
        )
        self.identity_endpoint = bool(bridge_cfg.identity_endpoint)
        self.eps = float(bridge_cfg.eps)

        self.coupling_solver = str(bridge_cfg.coupling_solver).strip().lower()
        if self.coupling_solver == "hungarian" and not bool(getattr(bridge_cfg, "allow_cpu_hungarian", False)):
            raise ValueError(
                "bridge.coupling_solver='hungarian' offloads OT matching to CPU via scipy.linear_sum_assignment. "
                "Use bridge.coupling_solver='sinkhorn' for GPU training, or set bridge.allow_cpu_hungarian=true "
                "only for explicit small diagnostic runs."
            )
        if self.coupling_solver not in {"sinkhorn", "sinkhorn_unbalanced", "hungarian"}:
            raise ValueError(
                f"Unsupported bridge.coupling_solver={self.coupling_solver!r}; expected 'sinkhorn', 'sinkhorn_unbalanced', or 'hungarian'."
            )
        self.coupling_feature_mode = str(bridge_cfg.coupling_feature_mode).strip().lower()
        self.coupling_lowfreq_kernel = max(1, int(bridge_cfg.coupling_lowfreq_kernel))
        if self.coupling_lowfreq_kernel % 2 == 0:
            self.coupling_lowfreq_kernel += 1
        self.coupling_edge_weight = max(0.0, float(bridge_cfg.coupling_edge_weight))
        self.coupling_cost_composition = str(
            getattr(bridge_cfg, "coupling_cost_composition", "structure_only")
        ).strip().lower()
        if self.coupling_cost_composition not in {
            "appearance_only",
            "appearance_plus_structure",
            "structure_only",
        }:
            self.coupling_cost_composition = "structure_only"
        self.coupling_structure_cost_mode = str(
            getattr(bridge_cfg, "coupling_structure_cost_mode", "none")
        ).strip().lower()
        if self.coupling_structure_cost_mode not in {
            "none",
            "lowedge",
            "self_affinity_gw",
            "lowedge_self_affinity_gw",
            "encoder_self_affinity_gw",
            "encoder_hybrid_affinity_gw",
            "tokenizer_aux_self_affinity_gw",
            "tokenizer_aux_hybrid_affinity_gw",
            "tokenizer_entropy_affinity_gw",
        }:
            self.coupling_structure_cost_mode = "none"
        self.coupling_structure_cost_weight = min(
            1.0,
            max(0.0, float(getattr(bridge_cfg, "coupling_structure_cost_weight", 0.0))),
        )
        self.coupling_structure_lowpass_kernel = max(
            1,
            int(getattr(bridge_cfg, "coupling_structure_lowpass_kernel", self.coupling_lowfreq_kernel)),
        )
        if self.coupling_structure_lowpass_kernel % 2 == 0:
            self.coupling_structure_lowpass_kernel += 1
        self.coupling_structure_edge_weight = max(
            0.0,
            float(getattr(bridge_cfg, "coupling_structure_edge_weight", 1.0)),
        )
        self.coupling_structure_affinity_grid = max(
            2,
            int(getattr(bridge_cfg, "coupling_structure_affinity_grid", 8)),
        )
        self.coupling_structure_hybrid_stats_weight = min(
            1.0,
            max(0.0, float(getattr(bridge_cfg, "coupling_structure_hybrid_stats_weight", 0.5))),
        )
        self.coupling_target_mode = str(getattr(bridge_cfg, "coupling_target_mode", "barycentric_full")).strip().lower()
        if self.coupling_target_mode not in {"sample", "barycentric_full", "barycentric_topk"}:
            self.coupling_target_mode = "barycentric_full"
        if (
            self.coupling_cost_composition == "structure_only"
            and (
                self.coupling_structure_cost_mode == "none"
                or self.coupling_structure_cost_weight <= 0.0
            )
        ):
            raise ValueError(
                "bridge.coupling_cost_composition='structure_only' requires a nonzero structural OT term. "
                "Set bridge.coupling_structure_cost_mode to a structure descriptor such as 'self_affinity_gw' "
                "and bridge.coupling_structure_cost_weight > 0."
            )
        self.coupling_barycentric_topk = max(0, int(getattr(bridge_cfg, "coupling_barycentric_topk", 0)))
        self.sinkhorn_epsilon = max(float(bridge_cfg.sinkhorn_epsilon), 1e-5)
        self.sinkhorn_iters = max(int(bridge_cfg.sinkhorn_iters), 1)
        self.sinkhorn_stabilize = bool(bridge_cfg.sinkhorn_stabilize)
        self.sinkhorn_unbalanced_tau_src = max(
            1e-6,
            float(getattr(bridge_cfg, "sinkhorn_unbalanced_tau_src", 1.0)),
        )
        self.sinkhorn_unbalanced_tau_tgt = max(
            1e-6,
            float(getattr(bridge_cfg, "sinkhorn_unbalanced_tau_tgt", 1.0)),
        )
        self.sinkhorn_unbalanced_dummy_cost = max(
            0.0,
            float(getattr(bridge_cfg, "sinkhorn_unbalanced_dummy_cost", 0.0)),
        )
        self.sinkhorn_unbalanced_dummy_offdiag_cost = max(
            self.sinkhorn_unbalanced_dummy_cost,
            float(getattr(bridge_cfg, "sinkhorn_unbalanced_dummy_offdiag_cost", 8.0)),
        )

        self.bridge_sigma = max(0.0, float(bridge_cfg.bridge_sigma))
        self.bridge_noise_mode = str(bridge_cfg.bridge_noise_mode).strip().lower()
        self.bridge_noise_schedule = str(getattr(bridge_cfg, "bridge_noise_schedule", "auto")).strip().lower()
        if self.bridge_noise_schedule not in {"auto", "exact_brownian", "delayed_window"}:
            self.bridge_noise_schedule = "auto"
        self.bridge_path_mode = str(getattr(bridge_cfg, "bridge_path_mode", "linear")).strip().lower()
        if self.bridge_path_mode not in {"linear", "latent_slerp", "vertical"}:
            raise ValueError(
                f"Unsupported bridge.bridge_path_mode={self.bridge_path_mode!r}; expected 'linear', 'latent_slerp', or 'vertical'."
            )
        if self.bridge_path_mode not in {"linear", "vertical"} and self.transport_prediction_mode != "endpoint":
            raise ValueError(
                f"bridge_path_mode={self.bridge_path_mode!r} is currently only supported for endpoint transport "
                "(or for 'vertical' which works with velocity)."
            )
        self.bridge_path_slerp_eps = max(1e-8, float(getattr(bridge_cfg, "bridge_path_slerp_eps", 1e-4)))
        # Vertical flow base-manifold lowpass stride (avg-pool kernel == stride).
        # bridge_vertical_base_stride is a known BridgeConfig field (default 2);
        # read it once here so call sites can use the attribute directly instead
        # of redundant getattr-with-default lookups.
        self.bridge_vertical_base_stride = max(1, int(getattr(bridge_cfg, "bridge_vertical_base_stride", 2)))

        # Auto-default noise projection to fiber-only when using vertical FM:
        # when bridge_path_mode="vertical" and sigma>0, training noise must also be
        # highpass-filtered to avoid injecting low-frequency noise into the base manifold.
        raw_noise_proj = str(getattr(bridge_cfg, "training_bridge_noise_projection_mode", "none")).strip().lower()
        if self.bridge_path_mode == "vertical" and raw_noise_proj == "none":
            raw_noise_proj = "pure_vertical_flow_wavelet"
        self.bridge_noise_window_start = float(getattr(bridge_cfg, "bridge_noise_window_start", 0.18))
        self.bridge_noise_window_end = float(getattr(bridge_cfg, "bridge_noise_window_end", 0.82))
        self.bridge_style_noise_kernel = max(1, int(bridge_cfg.bridge_style_noise_kernel))
        if self.bridge_style_noise_kernel % 2 == 0:
            self.bridge_style_noise_kernel += 1
        self.bridge_style_noise_flat_gamma = max(0.0, float(bridge_cfg.bridge_style_noise_flat_gamma))
        self.training_target_projection_mode = str(
            getattr(bridge_cfg, "training_target_projection_mode", "legacy")
        ).strip().lower()
        if self.training_target_projection_mode not in {
            "legacy",
            "source_low_target_high",
            "wavelet_source_low_target_high",
            "pure_vertical_flow",
            "pure_vertical_flow_wavelet",
        }:
            self.training_target_projection_mode = "legacy"
        self.training_target_projection_kernel = max(
            1,
            int(getattr(bridge_cfg, "training_target_projection_kernel", 5)),
        )
        if self.training_target_projection_kernel % 2 == 0:
            self.training_target_projection_kernel += 1
        self.training_target_projection_low_anchor = min(
            1.0,
            max(0.0, float(getattr(bridge_cfg, "training_target_projection_low_anchor", 1.0))),
        )
        self.training_target_projection_low_mode = str(
            getattr(bridge_cfg, "training_target_projection_low_mode", "all")
        ).strip().lower()
        if self.training_target_projection_low_mode not in {"all", "channel_mean"}:
            self.training_target_projection_low_mode = "all"
        self.training_bridge_noise_projection_mode = str(
            getattr(bridge_cfg, "training_bridge_noise_projection_mode", "none")
        ).strip().lower()
        # Auto-default: when bridge_path_mode="vertical", training noise must also
        # be fiber-only to avoid low-frequency noise leaking into the base manifold.
        if self.bridge_path_mode == "vertical" and self.training_bridge_noise_projection_mode == "none":
            self.training_bridge_noise_projection_mode = "pure_vertical_flow_wavelet"
        if self.training_bridge_noise_projection_mode not in {
            "none",
            "source_low_target_high",
            "wavelet_source_low_target_high",
            "pure_vertical_flow",
            "pure_vertical_flow_wavelet",
        }:
            self.training_bridge_noise_projection_mode = "none"
        self.training_bridge_noise_projection_kernel = max(
            1,
            int(getattr(bridge_cfg, "training_bridge_noise_projection_kernel", self.training_target_projection_kernel)),
        )
        if self.training_bridge_noise_projection_kernel % 2 == 0:
            self.training_bridge_noise_projection_kernel += 1
        self.training_bridge_noise_projection_preserve_rms = bool(
            getattr(bridge_cfg, "training_bridge_noise_projection_preserve_rms", True)
        )
        self.terminal_swd_weight = max(0.0, float(bridge_cfg.terminal_swd_weight))
        self.terminal_swd_aux_weight = max(0.0, float(bridge_cfg.terminal_swd_aux_weight))
        self.semantic_supervision_family = str(getattr(bridge_cfg, "semantic_supervision_family", "legacy_terminal_swd")).strip().lower()
        self.dino_masked_swd_weight = max(0.0, float(getattr(bridge_cfg, "dino_masked_swd_weight", 0.0)))
        self.w_variance_penalty = max(0.0, float(bridge_cfg.w_variance_penalty))
        self.w_style_energy_floor = max(0.0, float(bridge_cfg.w_style_energy_floor))
        self.w_lowfreq_velocity = max(0.0, float(bridge_cfg.w_lowfreq_velocity))
        self.w_content_lowpass_anchor = max(0.0, float(getattr(bridge_cfg, "w_content_lowpass_anchor", 0.0)))
        self.w_content_edge_anchor = max(0.0, float(getattr(bridge_cfg, "w_content_edge_anchor", 0.0)))
        self.content_anchor_lowpass_kernel = max(1, int(getattr(bridge_cfg, "content_anchor_lowpass_kernel", 9)))
        if self.content_anchor_lowpass_kernel % 2 == 0:
            self.content_anchor_lowpass_kernel += 1
        self.w_style_contrastive = max(0.0, float(bridge_cfg.w_style_contrastive))
        self.style_contrastive_temperature = max(1e-4, float(bridge_cfg.style_contrastive_temperature))
        self.style_contrastive_pool_size = max(1, int(bridge_cfg.style_contrastive_pool_size))
        self.w_residual_style_direction = max(0.0, float(bridge_cfg.w_residual_style_direction))
        self.w_generated_delta_diversity = max(0.0, float(bridge_cfg.w_generated_delta_diversity))
        self.generated_delta_diversity_margin = float(bridge_cfg.generated_delta_diversity_margin)
        self.w_spectral_amplitude = max(0.0, float(bridge_cfg.w_spectral_amplitude))
        self.spectral_amplitude_channels = max(1, int(bridge_cfg.spectral_amplitude_channels))
        self.spectral_amplitude_highpass = bool(bridge_cfg.spectral_amplitude_highpass)
        self.sb_noise_epsilon = max(0.0, float(bridge_cfg.sb_noise_epsilon))
        self.retinex_target_blend = max(0.0, min(1.0, float(bridge_cfg.retinex_target_blend)))
        self.retinex_kernel_size = max(3, int(bridge_cfg.retinex_kernel_size))
        if self.retinex_kernel_size % 2 == 0:
            self.retinex_kernel_size += 1
        self.w_anisotropic_kinetic = max(0.0, float(bridge_cfg.w_anisotropic_kinetic))
        self.anisotropic_normal_weight = max(0.0, float(bridge_cfg.anisotropic_normal_weight))
        self.anisotropic_tangent_weight = max(0.0, float(bridge_cfg.anisotropic_tangent_weight))
        self.anisotropic_edge_gate_gamma = max(0.0, float(getattr(bridge_cfg, "anisotropic_edge_gate_gamma", 0.0)))
        self.anisotropic_edge_gate_quantile = min(
            0.999,
            max(0.0, float(getattr(bridge_cfg, "anisotropic_edge_gate_quantile", 0.0))),
        )
        self.anisotropic_edge_gate_power = max(1e-3, float(getattr(bridge_cfg, "anisotropic_edge_gate_power", 1.0)))
        self.w_stokes_viscous = max(0.0, float(bridge_cfg.w_stokes_viscous))
        self.kinetic_penalty_mode = str(getattr(bridge_cfg, "kinetic_penalty_mode", "global_l2")).strip().lower()
        if self.kinetic_penalty_mode not in {
            "global_l2",
            "spatial_laplacian_split",
            "spectral_orthogonal_split",
            "manifold_adaptive_split",
        }:
            self.kinetic_penalty_mode = "global_l2"
        self.kinetic_lambda_low = max(0.0, float(getattr(bridge_cfg, "kinetic_lambda_low", 1.0)))
        self.kinetic_lambda_high = max(0.0, float(getattr(bridge_cfg, "kinetic_lambda_high", 0.02)))
        self.kinetic_lowpass_kernel = max(1, int(getattr(bridge_cfg, "kinetic_lowpass_kernel", 5)))
        if self.kinetic_lowpass_kernel % 2 == 0:
            self.kinetic_lowpass_kernel += 1
        self.kinetic_spectral_cutoff = max(1e-3, float(getattr(bridge_cfg, "kinetic_spectral_cutoff", 12.0)))
        self.kinetic_manifold_gamma = max(0.0, float(getattr(bridge_cfg, "kinetic_manifold_gamma", 10.0)))
        self.structure_penalty_mode = str(getattr(bridge_cfg, "structure_penalty_mode", "off")).strip().lower()
        if self.structure_penalty_mode not in {
            "off",
            "anisotropic",
            "stokes",
            "anisotropic_plus_stokes",
            "edge_gated_anisotropic",
            "edge_gated_anisotropic_plus_stokes",
            "quantile_edge_gated_anisotropic",
            "quantile_edge_gated_anisotropic_plus_stokes",
        }:
            self.structure_penalty_mode = "off"
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
        self.terminal_swd_axis_source = str(bridge_cfg.terminal_swd_axis_source).strip().lower()
        if self.terminal_swd_axis_source not in {"semantic", "random"}:
            self.terminal_swd_axis_source = "semantic"
        self.spectral_swd_low_weight = max(0.0, float(bridge_cfg.spectral_swd_low_weight))
        self.spectral_swd_high_weight = max(0.0, float(bridge_cfg.spectral_swd_high_weight))
        self.spectral_swd_low_kernel = max(1, int(bridge_cfg.spectral_swd_low_kernel))
        if self.spectral_swd_low_kernel % 2 == 0:
            self.spectral_swd_low_kernel += 1
        self.semantic_quotient_bins = max(2, int(bridge_cfg.semantic_quotient_bins))
        self.target_teacher_mode = str(getattr(bridge_cfg, "target_teacher_mode", "off")).strip().lower()
        if self.target_teacher_mode not in {"off", "style_lowfreq_ema", "style_endpoint_ema"}:
            self.target_teacher_mode = "off"
        self.target_teacher_decay = min(0.999999, max(0.0, float(getattr(bridge_cfg, "target_teacher_decay", 0.99))))
        self.target_teacher_weight = max(0.0, float(getattr(bridge_cfg, "target_teacher_weight", 0.0)))
        self.cycle_consistency_weight = max(0.0, float(getattr(bridge_cfg, "cycle_consistency_weight", 0.0)))
        self.cycle_consistency_num_steps = max(1, int(getattr(bridge_cfg, "cycle_consistency_num_steps", 4)))

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
        self.proximal_target_weight = max(0.0, float(getattr(bridge_cfg, "proximal_target_weight", 0.0)))
        self.transport_cost = SWDTransportCost(config)

        distill_cfg = train_cfg.distill
        self.distill_velocity_weight = max(0.0, float(distill_cfg.get("velocity_weight", 1.0)))
        self.distill_endpoint_weight = max(0.0, float(distill_cfg.get("endpoint_weight", 0.0)))
        self.profile_modules = bool(train_cfg.profile_modules)
        self.profile_sync_cuda = bool(train_cfg.profile_sync_cuda)
        self.last_profile: dict[str, float] = {}
        self.target_teacher_state: dict[int, torch.Tensor] = {}
        self.last_fiberwise_swd_debug: dict[str, float] = {}

    def state_dict(self) -> dict[str, object]:
        return {
            "target_teacher_state": {
                int(key): value.detach().cpu().clone()
                for key, value in self.target_teacher_state.items()
            }
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        self.target_teacher_state.clear()
        if not isinstance(state, dict):
            return
        raw = state.get("target_teacher_state", {})
        if not isinstance(raw, dict):
            return
        for key, value in raw.items():
            if torch.is_tensor(value):
                self.target_teacher_state[int(key)] = value.detach().cpu().clone()

    def _profile_start(self, ref: torch.Tensor) -> float:
        if not self.profile_modules:
            return 0.0
        if self.profile_sync_cuda and ref.device.type == "cuda":
            torch.cuda.synchronize(ref.device)
        return time.perf_counter()

    def _profile_end(self, name: str, start: float, ref: torch.Tensor) -> None:
        if not self.profile_modules:
            return
        if self.profile_sync_cuda and ref.device.type == "cuda":
            torch.cuda.synchronize(ref.device)
        self.last_profile[name] = self.last_profile.get(name, 0.0) + max(0.0, time.perf_counter() - start)

    def _profile_metrics(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.profile_modules:
            return {}
        return {
            f"profile_{name}_sec": ref.new_tensor(float(value), dtype=torch.float32)
            for name, value in self.last_profile.items()
        }

    def _model_profile_metrics(self, model: nn.Module, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.profile_modules:
            return {}
        raw = getattr(model, "last_profile", {})
        if not isinstance(raw, dict):
            return {}
        return {
            f"profile_{name}_sec": ref.new_tensor(float(value), dtype=torch.float32)
            for name, value in raw.items()
            if isinstance(value, (int, float))
        }

    @staticmethod
    def _attention_entropy(attn: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if attn is None:
            return ref.new_tensor(0.0, dtype=torch.float32)
        probs = attn.float().clamp_min(1e-8)
        return (-(probs * probs.log()).sum(dim=-1).mean()).to(device=ref.device, dtype=torch.float32)

    def _sanitize_tensor(self, x: torch.Tensor, *, clamp_value: float) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        return x.clamp(min=-clamp_value, max=clamp_value)

    @staticmethod
    def _conditioning_payload(conditioning: dict | None) -> dict | None:
        if not isinstance(conditioning, dict):
            return None
        known = {
            "content",
            "target_style",
            "target_style_id",
            "source_style_id",
            "aux_target_style",
            "aux_target_valid",
        }
        payload = {key: value for key, value in conditioning.items() if key not in known}
        return payload or None

    def _set_model_conditioning(self, model: nn.Module, conditioning: dict | None) -> None:
        setter = getattr(model, "set_runtime_conditioning", None)
        if callable(setter):
            setter(self._conditioning_payload(conditioning))

    def _sinkhorn_plan(self, cost: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n_src, n_tgt = int(cost.shape[0]), int(cost.shape[1])
        mu = torch.full((n_src,), 1.0 / max(n_src, 1), device=cost.device, dtype=cost.dtype)
        nu = torch.full((n_tgt,), 1.0 / max(n_tgt, 1), device=cost.device, dtype=cost.dtype)
        if self.coupling_solver == "sinkhorn_unbalanced":
            kernel = torch.exp((-cost / self.sinkhorn_epsilon).clamp(min=-80.0, max=80.0))
            u = torch.ones_like(mu)
            v = torch.ones_like(nu)
            tau_src = self.sinkhorn_unbalanced_tau_src
            tau_tgt = self.sinkhorn_unbalanced_tau_tgt
            rho_src = tau_src / (tau_src + self.sinkhorn_epsilon)
            rho_tgt = tau_tgt / (tau_tgt + self.sinkhorn_epsilon)
            for _ in range(self.sinkhorn_iters):
                kv = (kernel @ v).clamp_min(1e-12)
                ktu = (kernel.transpose(0, 1) @ u).clamp_min(1e-12)
                u = torch.pow(mu / kv, rho_src)
                v = torch.pow(nu / ktu, rho_tgt)
            raw_plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
            raw_total_mass = raw_plan.sum().clamp_min(1e-12)
            plan = raw_plan / raw_total_mass
            raw_source_mass = raw_plan.sum(dim=1)
            raw_target_mass = raw_plan.sum(dim=0)
            debug = {
                "ot_raw_total_mass": raw_total_mass.detach(),
                "ot_source_mass_mean": plan.sum(dim=1).mean().detach(),
                "ot_source_mass_min": plan.sum(dim=1).min().detach(),
                "ot_source_mass_max": plan.sum(dim=1).max().detach(),
                "ot_source_mass_entropy": self._mass_entropy(plan.sum(dim=1)).detach(),
                "ot_source_marginal_l1": (raw_source_mass - mu).abs().sum().detach(),
                "ot_source_truncation": torch.relu(mu - raw_source_mass).sum().detach(),
                "ot_target_marginal_l1": (raw_target_mass - nu).abs().sum().detach(),
                "ot_target_truncation": torch.relu(nu - raw_target_mass).sum().detach(),
            }
            return plan, debug
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
        plan = plan / plan.sum().clamp_min(1e-12)
        debug = {
            "ot_raw_total_mass": plan.new_tensor(1.0, dtype=torch.float32),
            "ot_source_mass_mean": plan.sum(dim=1).mean().detach(),
            "ot_source_mass_min": plan.sum(dim=1).min().detach(),
            "ot_source_mass_max": plan.sum(dim=1).max().detach(),
            "ot_source_mass_entropy": self._mass_entropy(plan.sum(dim=1)).detach(),
            "ot_source_marginal_l1": (plan.sum(dim=1) - mu).abs().sum().detach(),
            "ot_source_truncation": plan.new_tensor(0.0, dtype=torch.float32),
            "ot_target_marginal_l1": (plan.sum(dim=0) - nu).abs().sum().detach(),
            "ot_target_truncation": plan.new_tensor(0.0, dtype=torch.float32),
        }
        return plan, debug

    def _sample_from_plan(self, plan: torch.Tensor, target_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matched, entropy, _ = self._sample_or_project_from_plan(plan, target_group)
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

    def _wavelet_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        down = F.avg_pool2d(x.float(), kernel_size=2, stride=2, ceil_mode=False)
        return F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def _split_base_fiber(
        self,
        x: torch.Tensor,
        *,
        mode: str,
        kernel: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode in {"wavelet_source_low_target_high", "pure_vertical_flow_wavelet"}:
            base = self._wavelet_lowpass(x)
        else:
            base = self._kernel_lowpass(x, kernel)
        fiber = x.float() - base
        return base, fiber

    def _target_mass_gini(self, mass: torch.Tensor) -> torch.Tensor:
        mass = mass.float().flatten()
        if mass.numel() <= 1:
            return mass.new_tensor(0.0)
        total = mass.sum().clamp_min(1e-12)
        sorted_mass, _ = torch.sort(mass / total)
        n = int(sorted_mass.numel())
        index = torch.arange(1, n + 1, device=sorted_mass.device, dtype=sorted_mass.dtype)
        gini = (2.0 * (index * sorted_mass).sum() / float(n) - (float(n) + 1.0) / float(n))
        return gini.clamp(min=0.0, max=1.0)

    def _mass_entropy(self, mass: torch.Tensor) -> torch.Tensor:
        mass = mass.float().flatten()
        if mass.numel() <= 1:
            return mass.new_tensor(0.0)
        probs = (mass / mass.sum().clamp_min(1e-12)).clamp_min(1e-12)
        return -(probs * probs.log()).sum()

    def _coupling_structure_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        low = self._kernel_lowpass(x, self.coupling_structure_lowpass_kernel)
        edge = self._coupling_edge_map(low) * self.coupling_structure_edge_weight
        high = x.float() - low
        desc = torch.cat(
            [
                low.mean(dim=(2, 3)),
                low.std(dim=(2, 3), unbiased=False),
                edge.mean(dim=(2, 3)),
                edge.std(dim=(2, 3), unbiased=False),
                high.abs().mean(dim=(2, 3)),
                high.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        return desc.float()

    def _coupling_affinity_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        low = self._kernel_lowpass(x, self.coupling_structure_lowpass_kernel)
        edge = self._coupling_edge_map(low) * self.coupling_structure_edge_weight
        high = x.float() - low
        high_mag = high.abs().mean(dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(
            torch.cat([low, edge, high_mag], dim=1),
            output_size=(self.coupling_structure_affinity_grid, self.coupling_structure_affinity_grid),
        )
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens - tokens.mean(dim=-1, keepdim=True)
        tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
        affinity = torch.matmul(tokens, tokens.transpose(1, 2))
        tri = torch.triu_indices(
            affinity.shape[1],
            affinity.shape[2],
            offset=1,
            device=affinity.device,
        )
        desc = affinity[:, tri[0], tri[1]]
        return desc.float()

    def _affinity_descriptor_from_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(
            feat.float(),
            output_size=(self.coupling_structure_affinity_grid, self.coupling_structure_affinity_grid),
        )
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens - tokens.mean(dim=-1, keepdim=True)
        tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
        affinity = torch.matmul(tokens, tokens.transpose(1, 2))
        tri = torch.triu_indices(
            affinity.shape[1],
            affinity.shape[2],
            offset=1,
            device=affinity.device,
        )
        return affinity[:, tri[0], tri[1]].float()

    def _stats_descriptor_from_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        low = self._kernel_lowpass(feat, self.coupling_structure_lowpass_kernel)
        edge = self._coupling_edge_map(low) * self.coupling_structure_edge_weight
        high = feat.float() - low
        desc = torch.cat(
            [
                low.mean(dim=(2, 3)),
                low.std(dim=(2, 3), unbiased=False),
                edge.mean(dim=(2, 3)),
                edge.std(dim=(2, 3), unbiased=False),
                high.abs().mean(dim=(2, 3)),
                high.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        return desc.float()

    def _hybrid_descriptor_from_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        stats = self._stats_descriptor_from_feature_map(feat)
        affinity = self._affinity_descriptor_from_feature_map(feat)
        stats_weight = float(self.coupling_structure_hybrid_stats_weight)
        affinity_weight = 1.0 - stats_weight
        if stats_weight <= 0.0:
            return F.normalize(affinity, p=2, dim=1, eps=1e-6)
        if affinity_weight <= 0.0:
            return F.normalize(stats, p=2, dim=1, eps=1e-6)
        stats = F.normalize(stats, p=2, dim=1, eps=1e-6) * stats_weight
        affinity = F.normalize(affinity, p=2, dim=1, eps=1e-6) * affinity_weight
        return torch.cat([stats, affinity], dim=1)

    def _routing_entropy_affinity_descriptor(
        self,
        aux_map: torch.Tensor,
    ) -> torch.Tensor:
        probs = aux_map.float().clamp_min(1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=1, keepdim=True)
        max_entropy = max(math.log(float(max(int(probs.shape[1]), 2))), 1e-8)
        entropy = entropy / max_entropy
        sharpness = probs.amax(dim=1, keepdim=True)
        entropy_low = self._kernel_lowpass(entropy, self.coupling_structure_lowpass_kernel)
        entropy_edge = self._coupling_edge_map(entropy_low) * self.coupling_structure_edge_weight
        entropy_high = entropy - entropy_low
        stats = torch.cat(
            [
                entropy.mean(dim=(2, 3)),
                entropy.std(dim=(2, 3), unbiased=False),
                sharpness.mean(dim=(2, 3)),
                sharpness.std(dim=(2, 3), unbiased=False),
                entropy_edge.mean(dim=(2, 3)),
                entropy_edge.std(dim=(2, 3), unbiased=False),
                entropy_high.abs().mean(dim=(2, 3)),
                entropy_high.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        affinity_feat = torch.cat([probs, entropy, sharpness, entropy_edge], dim=1)
        affinity = self._affinity_descriptor_from_feature_map(affinity_feat)
        stats_weight = float(self.coupling_structure_hybrid_stats_weight)
        affinity_weight = 1.0 - stats_weight
        if stats_weight <= 0.0:
            return F.normalize(affinity, p=2, dim=1, eps=1e-6)
        if affinity_weight <= 0.0:
            return F.normalize(stats, p=2, dim=1, eps=1e-6)
        stats = F.normalize(stats, p=2, dim=1, eps=1e-6) * stats_weight
        affinity = F.normalize(affinity, p=2, dim=1, eps=1e-6) * affinity_weight
        return torch.cat([stats, affinity], dim=1)

    @staticmethod
    def _expand_style_id_tensor(
        style_id: torch.Tensor | int,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(style_id):
            style_id_t = style_id.to(device=device).long().view(-1)
            if style_id_t.numel() == batch:
                return style_id_t
            if style_id_t.numel() == 1:
                return style_id_t.expand(batch)
            raise ValueError(f"style_id batch mismatch: expected 1 or {batch}, got {style_id_t.numel()}")
        return torch.full((batch,), int(style_id), device=device, dtype=torch.long)

    @staticmethod
    def _ot_neutral_style_code(
        model: TimeConditionedLANCETBridge,
        x: torch.Tensor,
    ) -> torch.Tensor:
        width = int(getattr(model, "bridge_style_dim", getattr(model, "style_code_dim", 0)))
        if width <= 0:
            raise ValueError("model bridge/style code width must be positive for OT feature probes")
        return x.new_zeros((x.shape[0], width))

    def _ot_encoder_feature_map(
        self,
        model: TimeConditionedLANCETBridge,
        x: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> torch.Tensor:
        del style_id
        style_code = self._ot_neutral_style_code(model, x)
        feat_c = x / max(float(getattr(model, "latent_scale_factor", 1.0)), 1e-8)
        h_c = model.enc_in_act(model.enc_in(feat_c))
        # Keep this probe explicitly content-side: gate=0.0 makes the style
        # modulator return the normalized input path, and the zero style code
        # keeps that intent obvious in the OT audit.
        for block in model.hires_body:
            h_c = block(h_c, style_code, gate=0.0)
        return model.down(h_c).float()

    def _ot_tokenizer_aux_feature_map(
        self,
        model: TimeConditionedLANCETBridge,
        x: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> torch.Tensor:
        style_id_t = self._expand_style_id_tensor(style_id, batch=int(x.shape[0]), device=x.device)
        content_feat_16 = self._ot_encoder_feature_map(model, x, style_id=style_id_t).to(dtype=x.dtype)
        tokenizer = getattr(model, "structured_style_tokenizer", None)
        family = str(getattr(model, "tokenizer_family", "legacy_factorized"))
        if tokenizer is None or family not in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
            return content_feat_16.float()
        structured = tokenizer(
            style_id=style_id_t,
            base_style_code=self._ot_neutral_style_code(model, x),
            content_latent=x,
            target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
        )
        # For the latent-native tokenizers used in phase616 OT probes, aux_16 is
        # the content-driven routing attention map. Prefer it, then degrade
        # gracefully to mask/spatial outputs if a family does not expose aux_16.
        aux_map = getattr(structured, "aux_map", None)
        if torch.is_tensor(aux_map):
            return aux_map.float()
        mask_map = getattr(structured, "mask_map", None)
        if torch.is_tensor(mask_map):
            return mask_map.float()
        spatial_map = getattr(structured, "spatial_map", None)
        if torch.is_tensor(spatial_map):
            return spatial_map.float()
        return content_feat_16.float()

    def _hybrid_structure_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        stats = self._coupling_structure_descriptor(x)
        affinity = self._coupling_affinity_descriptor(x)
        stats_weight = float(self.coupling_structure_hybrid_stats_weight)
        affinity_weight = 1.0 - stats_weight
        if stats_weight <= 0.0:
            return F.normalize(affinity, p=2, dim=1, eps=1e-6)
        if affinity_weight <= 0.0:
            return F.normalize(stats, p=2, dim=1, eps=1e-6)
        stats = F.normalize(stats, p=2, dim=1, eps=1e-6) * stats_weight
        affinity = F.normalize(affinity, p=2, dim=1, eps=1e-6) * affinity_weight
        return torch.cat([stats, affinity], dim=1)

    def _structure_pairwise_cost(
        self,
        model: TimeConditionedLANCETBridge,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> torch.Tensor:
        if self.coupling_structure_cost_mode == "self_affinity_gw":
            x_desc = self._coupling_affinity_descriptor(x)
            y_desc = self._coupling_affinity_descriptor(y)
        elif self.coupling_structure_cost_mode == "lowedge_self_affinity_gw":
            x_desc = self._hybrid_structure_descriptor(x)
            y_desc = self._hybrid_structure_descriptor(y)
        elif self.coupling_structure_cost_mode == "encoder_self_affinity_gw":
            x_desc = self._affinity_descriptor_from_feature_map(self._ot_encoder_feature_map(model, x, style_id=style_id))
            y_desc = self._affinity_descriptor_from_feature_map(self._ot_encoder_feature_map(model, y, style_id=style_id))
        elif self.coupling_structure_cost_mode == "encoder_hybrid_affinity_gw":
            x_desc = self._hybrid_descriptor_from_feature_map(self._ot_encoder_feature_map(model, x, style_id=style_id))
            y_desc = self._hybrid_descriptor_from_feature_map(self._ot_encoder_feature_map(model, y, style_id=style_id))
        elif self.coupling_structure_cost_mode == "tokenizer_aux_self_affinity_gw":
            x_desc = self._affinity_descriptor_from_feature_map(self._ot_tokenizer_aux_feature_map(model, x, style_id=style_id))
            y_desc = self._affinity_descriptor_from_feature_map(self._ot_tokenizer_aux_feature_map(model, y, style_id=style_id))
        elif self.coupling_structure_cost_mode == "tokenizer_aux_hybrid_affinity_gw":
            x_desc = self._hybrid_descriptor_from_feature_map(self._ot_tokenizer_aux_feature_map(model, x, style_id=style_id))
            y_desc = self._hybrid_descriptor_from_feature_map(self._ot_tokenizer_aux_feature_map(model, y, style_id=style_id))
        elif self.coupling_structure_cost_mode == "tokenizer_entropy_affinity_gw":
            x_desc = self._routing_entropy_affinity_descriptor(
                self._ot_tokenizer_aux_feature_map(model, x, style_id=style_id)
            )
            y_desc = self._routing_entropy_affinity_descriptor(
                self._ot_tokenizer_aux_feature_map(model, y, style_id=style_id)
            )
        else:
            x_desc = self._coupling_structure_descriptor(x)
            y_desc = self._coupling_structure_descriptor(y)
        return torch.cdist(x_desc, y_desc, p=2).pow(2)

    def _fiber_probe_metrics(
        self,
        *,
        pred_endpoint: torch.Tensor,
        target_endpoint: torch.Tensor,
        content: torch.Tensor,
        mode: str,
        kernel: int,
    ) -> dict[str, torch.Tensor]:
        split_mode = str(mode or "source_low_target_high").strip().lower()
        if split_mode == "legacy":
            split_mode = "source_low_target_high"
        content_low, content_high = self._split_base_fiber(content, mode=split_mode, kernel=kernel)
        pred_low, pred_high = self._split_base_fiber(pred_endpoint, mode=split_mode, kernel=kernel)
        target_low, _ = self._split_base_fiber(target_endpoint, mode=split_mode, kernel=kernel)
        pred_low_delta = pred_low - content_low
        target_low_delta = target_low - content_low
        return {
            "base_structural_drift": F.mse_loss(pred_low, content_low).detach(),
            "fiber_energy_ratio": (
                pred_high.std(dim=(1, 2, 3), unbiased=False).mean()
                / content_high.std(dim=(1, 2, 3), unbiased=False).mean().clamp_min(self.eps)
            ).detach(),
            "low_freq_leak": (
                pred_low_delta.square().mean().sqrt()
                / target_low_delta.square().mean().sqrt().clamp_min(self.eps)
            ).detach(),
            "target_base_shift": F.mse_loss(target_low, content_low).detach(),
        }

    def _project_training_target(
        self,
        *,
        content: torch.Tensor,
        matched_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = content.new_tensor(0.0, dtype=torch.float32)
        metrics = {
            "training_target_projection_active": content.new_tensor(0.0, dtype=torch.float32),
            "training_target_projection_mode_source_low_target_high": content.new_tensor(0.0, dtype=torch.float32),
            "training_target_projection_mode_wavelet_source_low_target_high": content.new_tensor(0.0, dtype=torch.float32),
            "training_target_projection_mode_pure_vertical_flow": content.new_tensor(0.0, dtype=torch.float32),
            "training_target_projection_mode_pure_vertical_flow_wavelet": content.new_tensor(0.0, dtype=torch.float32),
            "training_target_projection_low_anchor": content.new_tensor(
                float(self.training_target_projection_low_anchor),
                dtype=torch.float32,
            ),
            "training_target_projection_low_drift": zero,
            "training_target_projection_target_delta": zero,
            "training_target_projection_high_energy_ratio": zero,
        }
        if self.training_target_projection_mode == "legacy":
            return matched_target, metrics

        kernel = self.training_target_projection_kernel
        split_mode = self.training_target_projection_mode
        content_low, content_high = self._split_base_fiber(content, mode=split_mode, kernel=kernel)
        target_low, target_high = self._split_base_fiber(matched_target, mode=split_mode, kernel=kernel)
        if self.training_target_projection_low_mode == "channel_mean":
            anchor_low = content_low.mean(dim=1, keepdim=True).expand_as(target_low)
        else:
            anchor_low = content_low
        low_anchor = float(self.training_target_projection_low_anchor)
        if self.training_target_projection_mode in {"pure_vertical_flow", "pure_vertical_flow_wavelet"}:
            projected = (anchor_low + target_high).to(dtype=matched_target.dtype)
            if self.training_target_projection_mode == "pure_vertical_flow_wavelet":
                metrics["training_target_projection_mode_pure_vertical_flow_wavelet"] = content.new_tensor(1.0, dtype=torch.float32)
            else:
                metrics["training_target_projection_mode_pure_vertical_flow"] = content.new_tensor(1.0, dtype=torch.float32)
        else:
            projected_low = target_low.lerp(anchor_low, low_anchor)
            projected = (projected_low + target_high).to(dtype=matched_target.dtype)
            if self.training_target_projection_mode == "wavelet_source_low_target_high":
                metrics["training_target_projection_mode_wavelet_source_low_target_high"] = content.new_tensor(1.0, dtype=torch.float32)
            else:
                metrics["training_target_projection_mode_source_low_target_high"] = content.new_tensor(1.0, dtype=torch.float32)

        projected_low_eval, projected_high = self._split_base_fiber(projected, mode=split_mode, kernel=kernel)
        metrics.update(
            {
                "training_target_projection_active": content.new_tensor(1.0, dtype=torch.float32),
                "training_target_projection_low_drift": F.mse_loss(projected_low_eval, content_low).detach(),
                "training_target_projection_target_delta": (projected.float() - matched_target.float()).abs().mean().detach(),
                "training_target_projection_high_energy_ratio": (
                    projected_high.std(dim=(1, 2, 3), unbiased=False).mean()
                    / content_high.std(dim=(1, 2, 3), unbiased=False).mean().clamp_min(self.eps)
                ).detach(),
            }
        )
        return projected, metrics

    def _training_bridge_noise_projection_metrics_template(
        self,
        ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = ref.new_tensor(0.0, dtype=torch.float32)
        return {
            "training_bridge_noise_projection_active": zero,
            "training_bridge_noise_projection_mode_source_low_target_high": zero,
            "training_bridge_noise_projection_mode_wavelet_source_low_target_high": zero,
            "training_bridge_noise_projection_mode_pure_vertical_flow": zero,
            "training_bridge_noise_projection_mode_pure_vertical_flow_wavelet": zero,
            "training_bridge_noise_projection_kernel": ref.new_tensor(
                float(self.training_bridge_noise_projection_kernel),
                dtype=torch.float32,
            ),
            "training_bridge_noise_projection_preserve_rms": ref.new_tensor(
                float(self.training_bridge_noise_projection_preserve_rms),
                dtype=torch.float32,
            ),
            "training_bridge_noise_projection_pre_rms": zero,
            "training_bridge_noise_projection_post_rms": zero,
            "training_bridge_noise_projection_low_rms": zero,
            "training_bridge_noise_projection_high_rms": zero,
        }

    def _project_training_bridge_noise(
        self,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = self._training_bridge_noise_projection_metrics_template(noise)
        mode = self.training_bridge_noise_projection_mode
        if mode == "none":
            return noise, metrics

        low, high = self._split_base_fiber(
            noise,
            mode=mode,
            kernel=self.training_bridge_noise_projection_kernel,
        )
        projected = high.to(dtype=noise.dtype)
        pre_rms = noise.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(self.eps)
        projected_rms = projected.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(self.eps)
        if self.training_bridge_noise_projection_preserve_rms:
            projected = projected * (pre_rms / projected_rms).to(dtype=projected.dtype)
        post_rms = projected.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        metrics.update(
            {
                "training_bridge_noise_projection_active": noise.new_tensor(1.0, dtype=torch.float32),
                "training_bridge_noise_projection_pre_rms": pre_rms.detach().mean().to(dtype=torch.float32),
                "training_bridge_noise_projection_post_rms": post_rms.detach().mean().to(dtype=torch.float32),
                "training_bridge_noise_projection_low_rms": low.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().mean().to(dtype=torch.float32),
                "training_bridge_noise_projection_high_rms": high.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().mean().to(dtype=torch.float32),
            }
        )
        if mode == "pure_vertical_flow_wavelet":
            metrics["training_bridge_noise_projection_mode_pure_vertical_flow_wavelet"] = noise.new_tensor(1.0, dtype=torch.float32)
        elif mode == "pure_vertical_flow":
            metrics["training_bridge_noise_projection_mode_pure_vertical_flow"] = noise.new_tensor(1.0, dtype=torch.float32)
        elif mode == "wavelet_source_low_target_high":
            metrics["training_bridge_noise_projection_mode_wavelet_source_low_target_high"] = noise.new_tensor(1.0, dtype=torch.float32)
        else:
            metrics["training_bridge_noise_projection_mode_source_low_target_high"] = noise.new_tensor(1.0, dtype=torch.float32)
        return projected, metrics

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

    def _coupling_cost_matrix(
        self,
        model: TimeConditionedLANCETBridge,
        content_group: torch.Tensor,
        target_group: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metric_ref = content_group.new_tensor(0.0, dtype=torch.float32)
        metrics: dict[str, torch.Tensor] = {
            "ot_appearance_cost_mean": metric_ref,
            "ot_appearance_cost_var": metric_ref,
            "ot_structure_cost_mean": metric_ref,
            "ot_structure_cost_var": metric_ref,
            "ot_structure_cost_active": metric_ref,
            "ot_cost_composition_appearance_only": metric_ref,
            "ot_cost_composition_appearance_plus_structure": metric_ref,
            "ot_cost_composition_structure_only": metric_ref,
        }

        structure_active = not (
            self.coupling_structure_cost_mode == "none" or self.coupling_structure_cost_weight <= 0.0
        )
        if self.coupling_cost_composition == "structure_only" and structure_active:
            structure_cost = self._structure_pairwise_cost(
                model,
                content_group,
                target_group,
                style_id=style_id,
            )
            metrics.update(
                {
                    "ot_structure_cost_mean": structure_cost.mean().detach(),
                    "ot_structure_cost_var": structure_cost.var(unbiased=False).detach(),
                    "ot_structure_cost_active": metric_ref.new_tensor(1.0, dtype=torch.float32),
                    "ot_cost_composition_structure_only": metric_ref.new_tensor(1.0, dtype=torch.float32),
                }
            )
            struct_scale = structure_cost.detach().mean().clamp_min(self.eps)
            return structure_cost / struct_scale, metrics

        appearance_cost = self.transport_cost.pairwise_cost(
            self._coupling_feature_tensor(content_group),
            self._coupling_feature_tensor(target_group),
        )
        metrics["ot_appearance_cost_mean"] = appearance_cost.mean().detach()
        metrics["ot_appearance_cost_var"] = appearance_cost.var(unbiased=False).detach()
        if not structure_active:
            metrics["ot_cost_composition_appearance_only"] = metric_ref.new_tensor(1.0, dtype=torch.float32)
            return appearance_cost, metrics

        structure_cost = self._structure_pairwise_cost(
            model,
            content_group,
            target_group,
            style_id=style_id,
        )
        app_scale = appearance_cost.detach().mean().clamp_min(self.eps)
        struct_scale = structure_cost.detach().mean().clamp_min(self.eps)
        weight = self.coupling_structure_cost_weight
        appearance_term = appearance_cost / app_scale
        structure_term = structure_cost / struct_scale
        if self.coupling_cost_composition == "appearance_only":
            total_cost = appearance_term
            metrics["ot_cost_composition_appearance_only"] = metric_ref.new_tensor(1.0, dtype=torch.float32)
        else:
            total_cost = (1.0 - weight) * appearance_term + weight * structure_term
            metrics["ot_cost_composition_appearance_plus_structure"] = metric_ref.new_tensor(1.0, dtype=torch.float32)
        metrics.update(
            {
                "ot_structure_cost_mean": structure_cost.mean().detach(),
                "ot_structure_cost_var": structure_cost.var(unbiased=False).detach(),
                "ot_structure_cost_active": metric_ref.new_tensor(1.0, dtype=torch.float32),
            }
        )
        return total_cost, metrics

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

    def _augment_cost_with_source_dummies(
        self,
        cost: torch.Tensor,
        content_group: torch.Tensor,
        target_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.coupling_solver != "sinkhorn_unbalanced" or self.sinkhorn_unbalanced_dummy_cost <= 0.0:
            return cost, target_group, int(target_group.shape[0])
        n_src = int(cost.shape[0])
        if n_src <= 0:
            return cost, target_group, int(target_group.shape[0])
        dummy_cost = torch.full(
            (n_src, n_src),
            float(self.sinkhorn_unbalanced_dummy_offdiag_cost),
            device=cost.device,
            dtype=cost.dtype,
        )
        diag = torch.arange(n_src, device=cost.device)
        dummy_cost[diag, diag] = float(self.sinkhorn_unbalanced_dummy_cost)
        augmented_cost = torch.cat([cost, dummy_cost], dim=1)
        augmented_targets = torch.cat([target_group, content_group.to(dtype=target_group.dtype)], dim=0)
        return augmented_cost, augmented_targets, int(target_group.shape[0])

    def _solve_group_coupling(
        self,
        model: TimeConditionedLANCETBridge,
        content_group: torch.Tensor,
        target_group: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cost, cost_metrics = self._coupling_cost_matrix(
            model,
            content_group,
            target_group,
            style_id=style_id,
        )
        cost, transport_targets, real_target_count = self._augment_cost_with_source_dummies(
            cost,
            content_group,
            target_group,
        )
        if self.coupling_solver == "hungarian":
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            row_t = torch.from_numpy(row_ind).to(device=cost.device, dtype=torch.long)
            col_t = torch.from_numpy(col_ind).to(device=cost.device, dtype=torch.long)
            matched = target_group.index_select(0, col_t)
            target_mass = torch.bincount(col_t, minlength=int(cost.shape[1])).to(device=cost.device, dtype=cost.dtype)
            cost_metrics.update(
                {
                    "ot_plan_entropy": cost.new_tensor(0.0, dtype=torch.float32),
                    "ot_barycentric_entropy": cost.new_tensor(0.0, dtype=torch.float32),
                    "ot_target_gini": self._target_mass_gini(target_mass).detach(),
                    "ot_target_mass_entropy": self._mass_entropy(target_mass).detach(),
                    "ot_target_max_mass": (target_mass.max() / target_mass.sum().clamp_min(1e-12)).detach(),
                    "ot_cost_mean": cost.mean().detach(),
                    "ot_cost_var": cost.var(unbiased=False).detach(),
                }
            )
            return matched, cost[row_t, col_t].mean(), cost.new_tensor(0.0), cost.new_tensor(0.0), cost_metrics

        plan, plan_debug = self._sinkhorn_plan(cost)
        matched, entropy, barycentric_entropy = self._sample_or_project_from_plan(plan, transport_targets)
        expected_cost = (plan * cost).sum() * float(cost.shape[0])
        target_mass = plan.sum(dim=0)
        real_target_mass = target_mass[:real_target_count]
        dummy_mass = target_mass[real_target_count:].sum() if real_target_count < target_mass.numel() else target_mass.new_tensor(0.0)
        real_mass = real_target_mass.sum()
        cost_metrics.update(
            {
                "ot_plan_entropy": entropy.detach(),
                "ot_barycentric_entropy": barycentric_entropy.detach(),
                "ot_target_gini": self._target_mass_gini(real_target_mass).detach(),
                "ot_target_mass_entropy": self._mass_entropy(real_target_mass).detach(),
                "ot_target_max_mass": real_target_mass.max().detach() if real_target_mass.numel() > 0 else cost.new_tensor(0.0),
                "ot_real_target_mass": real_mass.detach(),
                "ot_dummy_mass": dummy_mass.detach(),
                "ot_dummy_active": cost.new_tensor(1.0 if real_target_count < target_mass.numel() else 0.0, dtype=torch.float32),
                "ot_cost_mean": cost.mean().detach(),
                "ot_cost_var": cost.var(unbiased=False).detach(),
                **plan_debug,
            }
        )
        return matched, expected_cost, entropy, barycentric_entropy, cost_metrics

    def _ot_match_targets(
        self,
        model: TimeConditionedLANCETBridge,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        matched = torch.empty_like(target_style)
        total_cost = content.new_tensor(0.0, dtype=torch.float32)
        total_entropy = content.new_tensor(0.0, dtype=torch.float32)
        total_barycentric_entropy = content.new_tensor(0.0, dtype=torch.float32)
        debug_totals: dict[str, torch.Tensor] = {}

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
                matched_group, group_cost, group_entropy, group_barycentric_entropy, group_debug = self._solve_group_coupling(
                    model,
                    content_group.index_select(0, cross_indices),
                    target_group.index_select(0, cross_indices),
                    style_id=target_style_id.index_select(0, indices).index_select(0, cross_indices),
                )
                matched.index_copy_(0, indices.index_select(0, cross_indices), matched_group)
                total_cost = total_cost + group_cost * float(cross_indices.numel())
                total_entropy = total_entropy + group_entropy * float(cross_indices.numel())
                total_barycentric_entropy = total_barycentric_entropy + group_barycentric_entropy * float(cross_indices.numel())
                for key, value in group_debug.items():
                    weight = float(cross_indices.numel())
                    prev = debug_totals.get(key)
                    scaled = value.detach() * weight
                    debug_totals[key] = scaled if prev is None else prev + scaled

            if same_style_mask.any():
                same_indices = indices.index_select(0, torch.nonzero(same_style_mask, as_tuple=False).squeeze(1))
                matched.index_copy_(0, same_indices, content.index_select(0, same_indices))

        denom = max(int(content.shape[0]), 1)
        self._update_target_teacher(matched, target_style_id)
        debug_means = {
            key: value / float(denom)
            for key, value in debug_totals.items()
        }
        return (
            matched,
            total_cost / float(denom),
            total_entropy / float(denom),
            total_barycentric_entropy / float(denom),
            debug_means,
        )

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

    def _vertical_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Strided avg-pool + bilinear upsample for the vertical flow base manifold.

        Shared by :meth:`_bridge_state_and_velocity` (velocity fiber computation)
        and :meth:`_bridge_path_state` (path state base/fiber split). Centralising
        this avoids divergent lowpass implementations across vertical-flow call
        sites.
        """
        k = self.bridge_vertical_base_stride
        d = F.avg_pool2d(x.float(), kernel_size=k, stride=k)
        return F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def _bridge_state_and_velocity(
        self,
        *,
        content: torch.Tensor,
        matched_target: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        t4 = t.view(-1, 1, 1, 1)
        base = self._bridge_path_state(content=content, matched_target=matched_target, t=t)
        velocity = matched_target - content
        if self.bridge_path_mode == "vertical":
            velocity = (matched_target - self._vertical_lowpass(matched_target).to(content.dtype)) - (content - self._vertical_lowpass(content).to(content.dtype))
        endpoint_mode = str(getattr(self, "transport_prediction_mode", "velocity")).strip().lower() == "endpoint"
        noise_projection_metrics = self._training_bridge_noise_projection_metrics_template(content)
        if self.bridge_sigma <= 0.0:
            return base, (matched_target if endpoint_mode else velocity), noise_projection_metrics

        noise_schedule = self.bridge_noise_schedule
        if noise_schedule == "auto":
            # True I2SB should use the exact Brownian bridge by default.
            # Keep the old delayed window only for non-I2SB objectives so
            # existing velocity/heuristic lanes do not silently change.
            noise_schedule = "exact_brownian" if self.objective_mode == "i2sb_endpoint" else "delayed_window"

        if noise_schedule == "exact_brownian":
            noise_gate = torch.ones_like(t4)
        else:
            # Historical heuristic variant: only inject noise in [t_start, t_end]
            # to reduce instability near t≈0 and t≈1.
            t_start = max(0.0, float(getattr(self, "bridge_noise_window_start", 0.18)))
            t_end = min(1.0, float(getattr(self, "bridge_noise_window_end", 0.82)))
            t_flat = t.float()
            gate_flat = torch.zeros_like(t_flat)
            mid_mask = (t_flat >= t_start) & (t_flat <= t_end)
            t_mid = t_flat[mid_mask]
            if t_mid.numel() > 0:
                t_norm = (t_mid - t_start) / max(1e-6, t_end - t_start)
                gate_flat[mid_mask] = torch.sin(t_norm * math.pi) ** 2
            noise_gate = gate_flat.view(-1, 1, 1, 1)

        bridge_var = (t * (1.0 - t)).clamp_min(self.eps)
        bridge_std = torch.sqrt(bridge_var).view(-1, 1, 1, 1)
        noise, noise_projection_metrics = self._project_training_bridge_noise(
            self._style_bridge_noise(content, matched_target)
        )
        x_t = base + (self.bridge_sigma * bridge_std * noise_gate) * noise
        if endpoint_mode:
            return x_t, matched_target, noise_projection_metrics
        d_std_dt = ((1.0 - 2.0 * t) / (2.0 * torch.sqrt(bridge_var))).view(-1, 1, 1, 1)
        return x_t, velocity + (self.bridge_sigma * d_std_dt * noise_gate) * noise, noise_projection_metrics

    def _bridge_path_state(
        self,
        *,
        content: torch.Tensor,
        matched_target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t4 = t.view(-1, 1, 1, 1)
        linear = (1.0 - t4) * content + t4 * matched_target
        if self.bridge_path_mode == "linear":
            return linear

        if self.bridge_path_mode == "vertical":
            # Pure Vertical Flow Matching (616/design.md):
            # mu_base = const (structure frozen)
            # mu_fiber = (1-t)*fiber_content + t*fiber_matched_target
            base_c = self._vertical_lowpass(content).to(content.dtype)
            fiber_c = content - base_c
            fiber_t = matched_target - self._vertical_lowpass(matched_target).to(content.dtype)
            return base_c + (1.0 - t4) * fiber_c + t4 * fiber_t

        if self.bridge_path_mode != "latent_slerp":
            return linear

        b = int(content.shape[0])
        a_flat = content.float().reshape(b, -1)
        z_flat = matched_target.float().reshape(b, -1)
        eps = float(getattr(self, "bridge_path_slerp_eps", 1e-4))
        a_norm = a_flat.norm(dim=1, keepdim=True).clamp_min(eps)
        z_norm = z_flat.norm(dim=1, keepdim=True).clamp_min(eps)
        a_unit = a_flat / a_norm
        z_unit = z_flat / z_norm
        dot = (a_unit * z_unit).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp_min(eps)
        t_flat = t.float().view(-1, 1)
        s0 = torch.sin((1.0 - t_flat) * omega) / sin_omega
        s1 = torch.sin(t_flat * omega) / sin_omega
        direction = s0 * a_unit + s1 * z_unit
        radius = (1.0 - t_flat) * a_norm + t_flat * z_norm
        slerp = (direction * radius).reshape_as(content).to(dtype=content.dtype)
        near_linear = (omega <= 4.0 * eps).view(-1, 1, 1, 1)
        return torch.where(near_linear, linear, slerp)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, beta=0.5)
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _kinetic_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kinetic_lowpass_kernel // 2
        return F.avg_pool2d(x.float(), kernel_size=self.kinetic_lowpass_kernel, stride=1, padding=pad)

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.anchor_pool_size // 2
        return F.avg_pool2d(x.float(), kernel_size=self.anchor_pool_size, stride=1, padding=pad)

    def _row_normalized_plan(self, plan: torch.Tensor) -> torch.Tensor:
        return plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _sample_or_project_from_plan(self, plan: torch.Tensor, target_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_probs = self._row_normalized_plan(plan)
        entropy = -(row_probs * row_probs.clamp_min(1e-12).log()).sum(dim=1).mean()
        if self.coupling_target_mode == "sample":
            sampled_cols = torch.multinomial(row_probs, num_samples=1, replacement=True).squeeze(1)
            matched = target_group.index_select(0, sampled_cols)
            return matched, entropy, target_group.new_tensor(0.0, dtype=torch.float32)

        if self.coupling_target_mode == "barycentric_topk":
            topk = self.coupling_barycentric_topk if self.coupling_barycentric_topk > 0 else int(row_probs.shape[1])
            topk = max(1, min(int(topk), int(row_probs.shape[1])))
            vals, idx = torch.topk(row_probs, k=topk, dim=1)
            vals = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-12)
            flat_targets = target_group.float().flatten(1)
            gathered = flat_targets.index_select(0, idx.reshape(-1)).view(idx.shape[0], idx.shape[1], -1)
            matched_flat = (vals.unsqueeze(-1) * gathered).sum(dim=1)
            bary_entropy = -(vals * vals.clamp_min(1e-12).log()).sum(dim=1).mean()
        else:
            flat_targets = target_group.float().flatten(1)
            matched_flat = row_probs @ flat_targets
            bary_entropy = entropy

        matched = matched_flat.view(int(plan.shape[0]), *target_group.shape[1:]).to(dtype=target_group.dtype)
        return matched, entropy, bary_entropy

    def _teacher_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_teacher_mode == "style_lowfreq_ema":
            return self._kinetic_lowpass(x)
        return x.float()

    def _update_target_teacher(self, target_tensor: torch.Tensor, target_style_id: torch.Tensor) -> None:
        if self.target_teacher_mode == "off" or self.target_teacher_weight <= 0.0:
            return
        reduced = self._teacher_reduce(target_tensor.detach())
        for style_id in torch.unique(target_style_id.long(), sorted=True).tolist():
            mask = target_style_id.long() == int(style_id)
            if not bool(mask.any().item()):
                continue
            style_mean = reduced.index_select(0, torch.nonzero(mask, as_tuple=False).flatten()).mean(dim=0, keepdim=True).cpu()
            old = self.target_teacher_state.get(int(style_id))
            if old is None:
                self.target_teacher_state[int(style_id)] = style_mean.clone()
            else:
                self.target_teacher_state[int(style_id)] = old * self.target_teacher_decay + style_mean * (1.0 - self.target_teacher_decay)

    def _teacher_target(self, target_tensor: torch.Tensor, target_style_id: torch.Tensor) -> torch.Tensor:
        reduced = self._teacher_reduce(target_tensor)
        if self.target_teacher_mode == "off" or self.target_teacher_weight <= 0.0:
            return reduced.detach()
        teacher = []
        for style_id in target_style_id.long().tolist():
            state = self.target_teacher_state.get(int(style_id))
            if state is None:
                teacher.append(reduced.new_zeros((1, *reduced.shape[1:])))
            else:
                teacher.append(state.to(device=reduced.device, dtype=reduced.dtype))
        teacher_tensor = torch.cat(teacher, dim=0)
        return teacher_tensor

    def _teacher_alignment_loss(self, pred_endpoint: torch.Tensor, target_tensor: torch.Tensor, target_style_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero = pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        if self.target_teacher_mode == "off" or self.target_teacher_weight <= 0.0:
            return zero, zero
        teacher = self._teacher_target(target_tensor, target_style_id)
        pred_reduced = self._teacher_reduce(pred_endpoint).to(dtype=teacher.dtype)
        loss = F.mse_loss(pred_reduced, teacher) * self.target_teacher_weight
        return loss, teacher.abs().mean().detach()

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
        normal_field = 0.5 * ((v * nx).square() + (v * ny).square())
        tangent_field = 0.5 * ((v * tx).square() + (v * ty).square())
        if self.structure_penalty_mode in {"edge_gated_anisotropic", "edge_gated_anisotropic_plus_stokes"} and self.anisotropic_edge_gate_gamma > 0.0:
            edge_strength = self._gradient_magnitude(self._lowpass(content.float())).mean(dim=1, keepdim=True)
            edge_gate = 1.0 - torch.exp(-self.anisotropic_edge_gate_gamma * edge_strength)
            normal = (normal_field * edge_gate).mean()
        elif self.structure_penalty_mode in {"quantile_edge_gated_anisotropic", "quantile_edge_gated_anisotropic_plus_stokes"}:
            edge_strength = self._gradient_magnitude(self._lowpass(content.float())).mean(dim=1, keepdim=True)
            flat = edge_strength.flatten(start_dim=2)
            q = torch.quantile(flat, q=self.anisotropic_edge_gate_quantile, dim=2, keepdim=True).view(-1, 1, 1, 1)
            denom = (edge_strength.amax(dim=(2, 3), keepdim=True) - q).clamp_min(self.normalize_eps)
            edge_gate = ((edge_strength - q).clamp_min(0.0) / denom).pow(self.anisotropic_edge_gate_power)
            normal = (normal_field * edge_gate).mean()
        else:
            normal = normal_field.mean()
        tangent = tangent_field.mean()
        return (normal * self.anisotropic_normal_weight + tangent * self.anisotropic_tangent_weight) * self.w_anisotropic_kinetic

    def _stokes_viscous_loss(self, pred_velocity: torch.Tensor) -> torch.Tensor:
        if self.w_stokes_viscous <= 0.0:
            return pred_velocity.new_tensor(0.0, dtype=torch.float32)
        channels = int(pred_velocity.shape[1])
        kernel = pred_velocity.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        kernel = kernel.expand(channels, 1, 3, 3).contiguous()
        lap = F.conv2d(pred_velocity.float(), kernel, padding=1, groups=channels)
        return lap.square().mean() * self.w_stokes_viscous

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
        return self._lowpass(pred_velocity).square().mean() * self.w_lowfreq_velocity

    def _content_topology_anchor_loss(self, pred_endpoint: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero = pred_endpoint.new_tensor(0.0, dtype=torch.float32)
        if self.w_content_lowpass_anchor <= 0.0 and self.w_content_edge_anchor <= 0.0:
            return zero, zero
        pad = self.content_anchor_lowpass_kernel // 2
        pred_low = F.avg_pool2d(pred_endpoint.float(), kernel_size=self.content_anchor_lowpass_kernel, stride=1, padding=pad)
        content_low = F.avg_pool2d(content.float(), kernel_size=self.content_anchor_lowpass_kernel, stride=1, padding=pad)
        lowpass_loss = zero
        edge_loss = zero
        if self.w_content_lowpass_anchor > 0.0:
            lowpass_loss = F.l1_loss(pred_low, content_low) * self.w_content_lowpass_anchor
        if self.w_content_edge_anchor > 0.0:
            pred_edge = self._gradient_magnitude(pred_low)
            content_edge = self._gradient_magnitude(content_low).detach()
            edge_loss = F.l1_loss(pred_edge, content_edge) * self.w_content_edge_anchor
        return lowpass_loss, edge_loss

    def _spectral_split_kinetic_loss(self, pred_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_fft = torch.fft.rfft2(pred_velocity.float(), norm="ortho")
        _, _, h_dim, w_dim = pred_velocity.shape
        freq_y = torch.fft.fftfreq(h_dim, device=pred_velocity.device).view(-1, 1)
        freq_x = torch.fft.rfftfreq(w_dim, device=pred_velocity.device).view(1, -1)
        rho = torch.sqrt(freq_x.square() + freq_y.square()) * float(h_dim)
        low_mask = (rho < self.kinetic_spectral_cutoff).view(1, 1, h_dim, rho.shape[1])
        high_mask = ~low_mask
        low_loss = (torch.abs(v_fft * low_mask) ** 2).mean()
        high_loss = (torch.abs(v_fft * high_mask) ** 2).mean()
        return low_loss, high_loss

    def _manifold_adaptive_kinetic_loss(self, pred_velocity: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_low = self._kinetic_lowpass(pred_velocity)
        v_high = pred_velocity.float() - v_low
        dx = self._diff_x(content.float())
        dy = self._diff_y(content.float())
        grad_mag = (dx.square() + dy.square()).mean(dim=1, keepdim=True)
        flat_mask = torch.exp(-self.kinetic_manifold_gamma * grad_mag)
        edge_penalty = (1.0 - flat_mask) * v_high.square()
        tex_penalty = flat_mask * v_high.square()
        return v_low.square().mean(), edge_penalty.mean() + self.kinetic_lambda_high * tex_penalty.mean()

    def _kinetic_penalty_loss(self, pred_velocity: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = pred_velocity.new_tensor(0.0, dtype=torch.float32)
        mode = self.kinetic_penalty_mode
        if mode == "global_l2":
            base = pred_velocity.float().square().mean() * self.w_kinetic if self.w_kinetic > 0.0 else zero
            return base, zero, zero
        if mode == "spatial_laplacian_split":
            v_low = self._kinetic_lowpass(pred_velocity)
            v_high = pred_velocity.float() - v_low
            low_loss = v_low.square().mean() * self.kinetic_lambda_low
            high_loss = v_high.square().mean() * self.kinetic_lambda_high
            return (low_loss + high_loss) * self.w_kinetic, low_loss.detach(), high_loss.detach()
        if mode == "spectral_orthogonal_split":
            low_loss, high_loss = self._spectral_split_kinetic_loss(pred_velocity)
            low_loss = low_loss * self.kinetic_lambda_low
            high_loss = high_loss * self.kinetic_lambda_high
            return (low_loss + high_loss) * self.w_kinetic, low_loss.detach(), high_loss.detach()
        low_loss, high_penalty = self._manifold_adaptive_kinetic_loss(pred_velocity, content)
        low_loss = low_loss * self.kinetic_lambda_low
        return (low_loss + high_penalty) * self.w_kinetic, low_loss.detach(), high_penalty.detach()

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

    def _generated_delta_diversity_loss(
        self,
        pred_velocity: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = pred_velocity.new_tensor(0.0, dtype=torch.float32)
        if self.w_generated_delta_diversity <= 0.0 or pred_velocity.shape[0] < 2:
            return zero, zero, zero

        target_ids = target_style_id.long().view(-1)
        active_mask = torch.ones_like(target_ids, dtype=torch.bool)
        if source_style_id is not None:
            active_mask = source_style_id.long().view(-1) != target_ids
        active_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_idx.numel() < 2:
            return zero, zero, zero

        active_targets = target_ids.index_select(0, active_idx)
        unique_targets, inverse = torch.unique(active_targets, sorted=True, return_inverse=True)
        num_targets = int(unique_targets.numel())
        if num_targets < 2:
            return zero, zero, pred_velocity.new_tensor(float(num_targets), dtype=torch.float32)

        deltas = pred_velocity.float().flatten(1).index_select(0, active_idx)
        means = deltas.new_zeros((num_targets, deltas.shape[1]))
        means.index_add_(0, inverse, deltas)
        counts = torch.bincount(inverse, minlength=num_targets).to(device=deltas.device, dtype=deltas.dtype).clamp_min(1.0)
        means = means / counts.unsqueeze(1)
        means = F.normalize(means, p=2, dim=1, eps=self.normalize_eps)

        cosine = means @ means.t()
        offdiag = ~torch.eye(num_targets, dtype=torch.bool, device=cosine.device)
        offdiag_cos = cosine[offdiag]
        if offdiag_cos.numel() == 0:
            return zero, zero, pred_velocity.new_tensor(float(num_targets), dtype=torch.float32)
        margin = offdiag_cos.new_tensor(self.generated_delta_diversity_margin)
        loss = F.relu(offdiag_cos - margin).square().mean() * self.w_generated_delta_diversity
        return loss, offdiag_cos.mean().detach(), pred_velocity.new_tensor(float(num_targets), dtype=torch.float32)

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

    def _dino_masked_swd(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        dino_patches: torch.Tensor,
        dino_hw: torch.Tensor | None,
    ) -> torch.Tensor:
        if dino_patches.ndim != 3:
            return self.transport_cost.aligned_cost(pred, target)
        if torch.is_tensor(dino_hw) and dino_hw.numel() >= 2:
            h_dim = max(1, int(dino_hw.view(-1)[0].item()))
            w_dim = max(1, int(dino_hw.view(-1)[1].item()))
        else:
            side = int(round(max(1, dino_patches.shape[1]) ** 0.5))
            h_dim, w_dim = side, side
        if h_dim * w_dim != dino_patches.shape[1]:
            h_dim, w_dim = 1, int(dino_patches.shape[1])
        anchors = min(4, int(dino_patches.shape[1]))
        idx = torch.linspace(0, dino_patches.shape[1] - 1, steps=anchors, device=dino_patches.device).long()
        anchor_tokens = F.normalize(dino_patches.index_select(1, idx).float(), dim=-1, eps=self.normalize_eps)
        patches = F.normalize(dino_patches.float(), dim=-1, eps=self.normalize_eps)
        masks = torch.softmax(torch.bmm(patches, anchor_tokens.transpose(1, 2)), dim=-1)
        masks = masks.transpose(1, 2).contiguous().view(pred.shape[0], anchors, h_dim, w_dim)
        masks = F.interpolate(masks, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        losses: list[torch.Tensor] = []
        for chan in range(anchors):
            weight = masks[:, chan : chan + 1]
            losses.append(self.transport_cost.aligned_cost(pred * weight, target * weight))
        if not losses:
            return self.transport_cost.aligned_cost(pred, target)
        return torch.stack(losses).mean()

    def _fiber_routing_attention_from_model(self, model: TimeConditionedLANCETBridge) -> torch.Tensor | None:
        cached = getattr(model, "last_output_style_context", None)
        if not isinstance(cached, dict):
            return None
        style_maps = cached.get("style_maps")
        masks = getattr(style_maps, "aux_16", None)
        return masks if torch.is_tensor(masks) else None

    def _fiberwise_swd(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        fiber_attn: torch.Tensor,
    ) -> torch.Tensor:
        if fiber_attn.ndim != 4:
            raise ValueError(
                "semantic_supervision_family='fiberwise_swd' requires tokenizer routing attention "
                f"as a 4D aux_16 map, got shape={tuple(fiber_attn.shape)}."
            )
        masks = fiber_attn.to(device=pred.device).float().clamp(0.0, 1.0)
        if masks.shape[-2:] != pred.shape[-2:]:
            masks = F.interpolate(masks, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        if masks.shape[0] != pred.shape[0]:
            raise ValueError(
                "fiberwise_swd mask batch mismatch: "
                f"pred batch={pred.shape[0]} mask batch={masks.shape[0]}."
            )
        masses = masks.mean(dim=(0, 2, 3))
        active = torch.nonzero(masses > 1e-4, as_tuple=False).flatten()
        cluster_mass = masks.mean(dim=(2, 3))
        prob = cluster_mass / cluster_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mask_entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1).mean()
        losses: list[torch.Tensor] = []
        for idx in active.tolist():
            weight = masks[:, idx : idx + 1]
            losses.append(self.transport_cost.aligned_cost(pred * weight, target * weight))
        if not losses:
            loss = self.transport_cost.aligned_cost(pred, target)
            active_count = 0.0
        else:
            loss = torch.stack(losses).mean()
            active_count = float(active.numel())
        self.last_fiberwise_swd_debug = {
            "active_clusters": active_count,
            "loss_mean": float(loss.detach().float().mean().cpu().item()),
            "mask_entropy": float(mask_entropy.detach().float().cpu().item()),
        }
        return loss

    def _calc_terminal_swd_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_style: torch.Tensor,
        source_style_id: torch.Tensor | None,
        target_style_id: torch.Tensor,
        semantic_k: torch.Tensor | None = None,
        content: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        dino_patches: torch.Tensor | None = None,
        dino_hw: torch.Tensor | None = None,
        fiber_attn: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        active = self._terminal_active_indices(pred_endpoint, source_style_id, target_style_id)
        if active_mask is not None and active.numel() > 0:
            mask = active_mask.to(device=pred_endpoint.device).view(-1).bool()
            active = active.index_select(0, torch.nonzero(mask.index_select(0, active), as_tuple=False).flatten())
        if active.numel() == 0:
            self.last_fiberwise_swd_debug = {
                "active_clusters": 0.0,
                "loss_mean": 0.0,
                "mask_entropy": 0.0,
            }
            return None
        pred_active = pred_endpoint.index_select(0, active)
        target_active = target_style.index_select(0, active)
        if self.semantic_supervision_family == "fiberwise_swd":
            if fiber_attn is None:
                raise ValueError(
                    "semantic_supervision_family='fiberwise_swd' requires structured tokenizer routing aux_16."
                )
            fiber_active = fiber_attn.index_select(0, active)
            return self._fiberwise_swd(pred_active, target_active, fiber_attn=fiber_active)
        if self.semantic_supervision_family == "dino_masked_swd" and dino_patches is not None and self.dino_masked_swd_weight > 0.0:
            dino_active = dino_patches.index_select(0, active)
            return self._dino_masked_swd(
                pred_active,
                target_active,
                dino_patches=dino_active,
                dino_hw=dino_hw,
            ) * self.dino_masked_swd_weight
        if (
            self.terminal_swd_mode == "standard"
            and self.terminal_swd_axis_source == "semantic"
            and semantic_k is not None
        ):
            semantic_active = semantic_k.index_select(0, active)
            return self._semantic_guided_swd(pred_active, target_active, semantic_active)
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
        runtime_payload = getattr(model, "runtime_conditioning", {}) if hasattr(model, "runtime_conditioning") else {}
        dino_patches = runtime_payload.get("content_dino_patches") if isinstance(runtime_payload, dict) else None
        dino_hw = runtime_payload.get("content_dino_hw") if isinstance(runtime_payload, dict) else None
        if self.semantic_supervision_family == "fiberwise_swd":
            setattr(model, "force_output_style_context_cache", True)

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
        fiber_attn = (
            self._fiber_routing_attention_from_model(model)
            if self.semantic_supervision_family == "fiberwise_swd"
            else None
        )
        term = self._calc_terminal_swd_loss(
            endpoint,
            matched_target,
            source_style_id,
            target_style_id,
            dino_patches=dino_patches if torch.is_tensor(dino_patches) else None,
            dino_hw=dino_hw if torch.is_tensor(dino_hw) else None,
            fiber_attn=fiber_attn,
        )
        return term, endpoint.abs().mean().detach()

    def _cycle_consistency_loss(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cycle_consistency_weight <= 0.0 or source_style_id is None:
            return content.new_tensor(0.0, dtype=torch.float32)
        forward = model.integrate(
            content,
            style_id=target_style_id,
            num_steps=self.cycle_consistency_num_steps,
            step_size=1.0,
            style_strength=1.0,
        )
        recon = model.integrate(
            forward,
            style_id=source_style_id,
            num_steps=self.cycle_consistency_num_steps,
            step_size=1.0,
            style_strength=1.0,
        )
        return F.l1_loss(recon, content) * self.cycle_consistency_weight

    def _compute_omf_details(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        aux_target_style: torch.Tensor | None = None,
        aux_target_valid: torch.Tensor | None = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor | None]]:
        self.last_profile = {}
        if self.semantic_supervision_family == "fiberwise_swd":
            setattr(model, "force_output_style_context_cache", True)
        t_fixed = content.new_ones(content.shape[0])
        content_for_model = content
        target_for_loss = self._retinex_target(content, target_style)
        matched_target = target_for_loss
        need_ot_target = (
            self.w_flow > 0.0
            or self.coupling_target_mode != "sample"
            or self.coupling_feature_mode not in {"latent", "raw", ""}
        )
        if need_ot_target:
            t_profile = self._profile_start(content)
            if content.device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda", enabled=False)
            else:
                autocast_ctx = torch.autocast("cpu", enabled=False)
            with torch.no_grad():
                with autocast_ctx:
                    matched_target, ot_cost, plan_entropy, barycentric_entropy, ot_debug = self._ot_match_targets(
                        model,
                        content,
                        target_style,
                        target_style_id,
                        source_style_id,
                    )
            self._profile_end("ot_match", t_profile, content)
            target_for_loss = self._retinex_target(content, matched_target)
        t_profile = self._profile_start(content)
        pred_endpoint_base = self._sanitize_tensor(
            model.predict_transport_base(
                content_for_model,
                t=t_fixed,
                style_id=target_style_id,
            ),
            clamp_value=self.endpoint_clamp,
        )
        self._profile_end("model_forward", t_profile, content)
        pred_velocity = self._sanitize_tensor(pred_endpoint_base - content_for_model, clamp_value=self.velocity_clamp)
        pred_endpoint_final = self._sanitize_tensor(
            model.refine_endpoint(
                pred_endpoint_base,
                style_id=target_style_id,
                source_latent=content_for_model,
            ),
            clamp_value=self.endpoint_clamp,
        )
        endpoint_for_losses = pred_endpoint_final if bool(getattr(model, "proximal_bind_terminal_losses", True)) else pred_endpoint_base
        attn_plan = model.last_semantic_attn
        semantic_k = model.last_semantic_k
        topology_attn = getattr(model, "last_semantic_topology_attn", None)

        total_loss = content.new_tensor(0.0, dtype=torch.float32)
        flow_loss = content.new_tensor(0.0, dtype=torch.float32)
        ot_cost = content.new_tensor(0.0, dtype=torch.float32)
        plan_entropy = content.new_tensor(0.0, dtype=torch.float32)
        barycentric_entropy = content.new_tensor(0.0, dtype=torch.float32)

        if self.w_flow > 0.0:
            flow_loss = self._loss(pred_endpoint_base, matched_target) * self.w_flow
            total_loss = total_loss + flow_loss

        t_profile = self._profile_start(content)
        zero = content.new_tensor(0.0, dtype=torch.float32)
        kinetic_loss, kinetic_low_band, kinetic_high_band = self._kinetic_penalty_loss(pred_velocity, content)
        use_anisotropic = self.structure_penalty_mode in {
            "anisotropic",
            "anisotropic_plus_stokes",
            "edge_gated_anisotropic",
            "edge_gated_anisotropic_plus_stokes",
            "quantile_edge_gated_anisotropic",
            "quantile_edge_gated_anisotropic_plus_stokes",
        }
        use_stokes = self.structure_penalty_mode in {
            "stokes",
            "anisotropic_plus_stokes",
            "edge_gated_anisotropic_plus_stokes",
            "quantile_edge_gated_anisotropic_plus_stokes",
        }
        anisotropic_kinetic = self._anisotropic_kinetic_loss(pred_velocity, content) if use_anisotropic and self.w_anisotropic_kinetic > 0.0 else zero
        stokes_viscous = self._stokes_viscous_loss(pred_velocity) if use_stokes and self.w_stokes_viscous > 0.0 else zero
        total_loss = total_loss + kinetic_loss
        curvature_loss = zero
        if self.w_curvature > 0.0:
            dt = self.curvature_dt
            t1 = content.new_full((content.shape[0],), max(self.t_min, min(self.t_max, 1.0 - dt)))
            t2 = (t1 + dt).clamp(max=self.t_max)
            pred_v1 = self._sanitize_tensor(model(content_for_model, t=t1, style_id=target_style_id), clamp_value=self.velocity_clamp)
            pred_v2 = self._sanitize_tensor(model(content_for_model + pred_v1 * dt, t=t2, style_id=target_style_id), clamp_value=self.velocity_clamp)
            curvature_loss = self._loss(pred_v2, pred_v1) * self.w_curvature
            total_loss = total_loss + curvature_loss

        style_energy_floor = self._style_energy_floor_loss(endpoint_for_losses, target_for_loss) if self.w_style_energy_floor > 0.0 else zero
        lowfreq_velocity = self._lowfreq_velocity_loss(pred_velocity) if self.w_lowfreq_velocity > 0.0 else zero
        content_lowpass_anchor, content_edge_anchor = self._content_topology_anchor_loss(endpoint_for_losses, content)
        style_contrastive = (
            self._style_contrastive_loss(endpoint_for_losses, target_for_loss, target_style_id)
            if self.w_style_contrastive > 0.0
            else zero
        )
        residual_style_direction = (
            self._residual_style_direction_loss(endpoint_for_losses, content, target_for_loss)
            if self.w_residual_style_direction > 0.0
            else zero
        )
        generated_delta_diversity, generated_delta_mean_offdiag_cos, generated_delta_active_styles = (
            self._generated_delta_diversity_loss(pred_velocity, target_style_id, source_style_id)
            if self.w_generated_delta_diversity > 0.0
            else (zero, zero, zero)
        )
        spectral_amplitude = (
            self._spectral_amplitude_loss(endpoint_for_losses, target_for_loss)
            if self.w_spectral_amplitude > 0.0
            else zero
        )
        teacher_alignment, teacher_abs = self._teacher_alignment_loss(endpoint_for_losses, matched_target, target_style_id)
        proximal_residual = getattr(model, "last_proximal_residual", None)
        proximal_clamp_scale = getattr(model, "last_proximal_clamp_scale", None)
        proximal_residual_abs = proximal_residual.abs().mean().detach() if torch.is_tensor(proximal_residual) else zero
        proximal_residual_energy = (
            proximal_residual.float().square().mean() * float(getattr(model, "proximal_residual_energy_weight", 0.0))
            if torch.is_tensor(proximal_residual) and float(getattr(model, "proximal_residual_energy_weight", 0.0)) > 0.0
            else zero
        )
        proximal_target = (
            self._loss(pred_endpoint_final, target_for_loss) * self.proximal_target_weight
            if self.proximal_target_weight > 0.0
            else zero
        )
        base_endpoint = getattr(model, "last_base_endpoint", None)
        base_transport_abs = (
            (base_endpoint - content).abs().mean().detach()
            if torch.is_tensor(base_endpoint)
            else zero
        )
        proximal_to_transport_ratio = zero
        proximal_trust_penalty = zero
        proximal_trust_weight = float(getattr(model, "proximal_trust_weight", 0.0))
        proximal_trust_ratio = float(getattr(model, "proximal_trust_ratio", 0.0))
        if (
            torch.is_tensor(proximal_residual)
            and torch.is_tensor(base_endpoint)
            and proximal_trust_weight > 0.0
            and proximal_trust_ratio > 0.0
        ):
            prox_rms = proximal_residual.float().square().mean().sqrt()
            base_transport = (base_endpoint - content).float()
            base_rms = base_transport.square().mean().sqrt().detach()
            proximal_to_transport_ratio = prox_rms.detach() / base_rms.clamp_min(self.eps)
            allowed_rms = base_rms * proximal_trust_ratio
            proximal_trust_penalty = F.relu(prox_rms - allowed_rms).square() * proximal_trust_weight
        self._profile_end("aux_loss", t_profile, content)
        total_loss = (
            total_loss
            + style_energy_floor
            + lowfreq_velocity
            + content_lowpass_anchor
            + content_edge_anchor
            + style_contrastive
            + residual_style_direction
            + generated_delta_diversity
            + spectral_amplitude
            + teacher_alignment
            + anisotropic_kinetic
            + stokes_viscous
            + proximal_residual_energy
            + proximal_trust_penalty
            + proximal_target
        )

        runtime_payload = getattr(model, "runtime_conditioning", {}) if hasattr(model, "runtime_conditioning") else {}
        dino_patches = runtime_payload.get("content_dino_patches") if isinstance(runtime_payload, dict) else None
        dino_hw = runtime_payload.get("content_dino_hw") if isinstance(runtime_payload, dict) else None
        fiber_attn = (
            self._fiber_routing_attention_from_model(model)
            if self.semantic_supervision_family == "fiberwise_swd"
            else None
        )
        terminal_swd = None
        if self.terminal_swd_weight > 0.0:
            t_profile = self._profile_start(content)
            terminal_swd = self._calc_terminal_swd_loss(
                endpoint_for_losses,
                target_for_loss,
                source_style_id,
                target_style_id,
                semantic_k=semantic_k,
                content=content,
                dino_patches=dino_patches if torch.is_tensor(dino_patches) else None,
                dino_hw=dino_hw if torch.is_tensor(dino_hw) else None,
                fiber_attn=fiber_attn,
            )
            self._profile_end("terminal_swd", t_profile, content)
        terminal_loss = content.new_tensor(0.0, dtype=torch.float32)
        if terminal_swd is not None:
            terminal_loss = terminal_swd * self.terminal_swd_weight
            total_loss = total_loss + terminal_loss
        terminal_aux_loss = content.new_tensor(0.0, dtype=torch.float32)
        aux_target_ratio = content.new_tensor(0.0, dtype=torch.float32)
        if self.terminal_swd_aux_weight > 0.0 and aux_target_style is not None:
            aux_target_for_loss = self._retinex_target(content, aux_target_style)
            aux_mask = None
            if aux_target_valid is not None:
                aux_mask = aux_target_valid.to(device=content.device).view(-1) > 0.5
                aux_target_ratio = aux_mask.float().mean().detach()
            else:
                aux_target_ratio = content.new_tensor(1.0, dtype=torch.float32)
            terminal_aux_swd = self._calc_terminal_swd_loss(
                endpoint_for_losses,
                aux_target_for_loss,
                source_style_id,
                target_style_id,
                semantic_k=semantic_k,
                content=content,
                active_mask=aux_mask,
                dino_patches=dino_patches if torch.is_tensor(dino_patches) else None,
                dino_hw=dino_hw if torch.is_tensor(dino_hw) else None,
                fiber_attn=fiber_attn,
            )
            if terminal_aux_swd is not None:
                terminal_aux_loss = terminal_aux_swd * self.terminal_swd_aux_weight
                total_loss = total_loss + terminal_aux_loss
        cycle_consistency = self._cycle_consistency_loss(
            model,
            content=content,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        total_loss = total_loss + cycle_consistency

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "kinetic_energy": kinetic_loss.detach(),
            "kinetic_low_band": kinetic_low_band.detach(),
            "kinetic_high_band": kinetic_high_band.detach(),
            "curvature": curvature_loss.detach(),
            "anisotropic_kinetic": anisotropic_kinetic.detach(),
            "stokes_viscous": stokes_viscous.detach(),
            "proximal_residual_energy": proximal_residual_energy.detach() if torch.is_tensor(proximal_residual_energy) else zero,
            "proximal_trust_penalty": proximal_trust_penalty.detach() if torch.is_tensor(proximal_trust_penalty) else zero,
            "proximal_target": proximal_target.detach() if torch.is_tensor(proximal_target) else zero,
            "terminal_swd": terminal_loss.detach(),
            "terminal_swd_aux": terminal_aux_loss.detach(),
            "aux_target_ratio": aux_target_ratio.detach(),
            "cycle_consistency": cycle_consistency.detach(),
            "style_energy_floor": style_energy_floor.detach(),
            "lowfreq_velocity": lowfreq_velocity.detach(),
            "content_lowpass_anchor": content_lowpass_anchor.detach(),
            "content_edge_anchor": content_edge_anchor.detach(),
            "style_contrastive": style_contrastive.detach(),
            "residual_style_direction": residual_style_direction.detach(),
            "generated_delta_diversity": generated_delta_diversity.detach(),
            "generated_delta_mean_offdiag_cos": generated_delta_mean_offdiag_cos.detach(),
            "generated_delta_active_styles": generated_delta_active_styles.detach(),
            "spectral_amplitude": spectral_amplitude.detach(),
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "ot_plan_entropy": plan_entropy.detach(),
            "barycentric_entropy": barycentric_entropy.detach(),
            "ot_barycentric_entropy": barycentric_entropy.detach(),
            "teacher_alignment": teacher_alignment.detach(),
            "teacher_abs": teacher_abs.detach(),
            "bridge_sigma": content.new_tensor(0.0, dtype=torch.float32),
            "t_mean": t_fixed.mean().detach(),
            "velocity_abs": pred_velocity.abs().mean().detach(),
            "velocity_max": pred_velocity.abs().amax().detach(),
            "endpoint_abs": endpoint_for_losses.abs().mean().detach(),
            "endpoint_max": endpoint_for_losses.abs().amax().detach(),
            "base_endpoint_abs": pred_endpoint_base.abs().mean().detach(),
            "base_endpoint_max": pred_endpoint_base.abs().amax().detach(),
            "final_endpoint_abs": pred_endpoint_final.abs().mean().detach(),
            "final_endpoint_max": pred_endpoint_final.abs().amax().detach(),
            "proximal_residual_abs": proximal_residual_abs.detach(),
            "proximal_clamp_scale": proximal_clamp_scale.detach() if torch.is_tensor(proximal_clamp_scale) else content.new_tensor(1.0),
            "base_transport_abs": base_transport_abs.detach() if torch.is_tensor(base_transport_abs) else zero,
            "proximal_to_transport_ratio": proximal_to_transport_ratio.detach() if torch.is_tensor(proximal_to_transport_ratio) else zero,
            "kinetic_penalty_mode_id": content.new_tensor(float(hash(self.kinetic_penalty_mode) % 1000000), dtype=torch.float32),
            "semantic_attn_mean": attn_plan.mean().detach() if attn_plan is not None else content.new_tensor(0.0),
            "semantic_k_abs": semantic_k.abs().mean().detach() if semantic_k is not None else content.new_tensor(0.0),
            "semantic_topology_attn_entropy": self._attention_entropy(topology_attn, content).detach(),
            "semantic_topology_attn_active": content.new_tensor(1.0 if topology_attn is not None else 0.0, dtype=torch.float32),
        }
        metrics.update(
            self._fiber_probe_metrics(
                pred_endpoint=pred_endpoint_base,
                target_endpoint=target_for_loss,
                content=content,
                mode=self.training_target_projection_mode,
                kernel=self.training_target_projection_kernel,
            )
        )
        metrics.update(ot_debug)
        metrics.update(self._profile_metrics(content))
        metrics.update(self._model_profile_metrics(model, content))
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()

        components = {
            "flow": flow_loss,
            "kinetic_energy": kinetic_loss,
            "curvature": curvature_loss,
            "anisotropic_kinetic": anisotropic_kinetic,
            "stokes_viscous": stokes_viscous,
            "proximal_residual_energy": proximal_residual_energy,
            "proximal_trust_penalty": proximal_trust_penalty,
            "proximal_target": proximal_target,
            "terminal_swd": terminal_loss,
            "terminal_swd_aux": terminal_aux_loss,
            "style_energy_floor": style_energy_floor,
            "lowfreq_velocity": lowfreq_velocity,
            "content_lowpass_anchor": content_lowpass_anchor,
            "content_edge_anchor": content_edge_anchor,
            "style_contrastive": style_contrastive,
            "residual_style_direction": residual_style_direction,
            "generated_delta_diversity": generated_delta_diversity,
            "spectral_amplitude": spectral_amplitude,
            "teacher_alignment": teacher_alignment,
        }
        debug_state: Dict[str, torch.Tensor | None] = {
            "pred_velocity": pred_velocity.detach(),
            "pred_endpoint_base": pred_endpoint_base.detach(),
            "pred_endpoint_final": pred_endpoint_final.detach(),
            "semantic_attn": attn_plan.detach() if attn_plan is not None else None,
            "semantic_k": semantic_k.detach() if semantic_k is not None else None,
            "semantic_topology_attn": topology_attn.detach() if topology_attn is not None else None,
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
        aux_target_style: torch.Tensor | None = None,
        aux_target_valid: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        metrics, _, _ = self._compute_omf_details(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
            aux_target_style=aux_target_style,
            aux_target_valid=aux_target_valid,
        )
        return metrics

    def _compute_sampled_bridge_details(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        enforce_endpoint: bool = False,
        require_flow_weight: bool = False,
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor | None],
    ]:
        self.last_profile = {}
        t_profile = self._profile_start(content)
        if content.device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = torch.autocast("cpu", enabled=False)
        with torch.no_grad():
            with autocast_ctx:
                matched_target, ot_cost, plan_entropy, _, ot_debug = self._ot_match_targets(
                    model,
                    content,
                    target_style,
                    target_style_id,
                    source_style_id,
                )
                if self.retinex_target_blend > 0.0:
                    matched_target = self._retinex_target(content, matched_target)
                objective_target, target_projection_metrics = self._project_training_target(
                    content=content,
                    matched_target=matched_target,
                )
        self._profile_end("ot_match", t_profile, content)

        flow_content = content
        flow_objective_target = objective_target
        transport_stats_metrics: dict[str, torch.Tensor] = {}
        prepare_transport_pair = getattr(model, "prepare_transport_training_pair", None)
        if callable(prepare_transport_pair):
            flow_content, flow_objective_target, transport_stats_metrics = prepare_transport_pair(
                content=content,
                target=objective_target,
                style_id=target_style_id,
            )
        t = self._sample_t(content)
        x_t, target_velocity, bridge_noise_metrics = self._bridge_state_and_velocity(
            content=flow_content,
            matched_target=flow_objective_target,
            t=t,
        )
        if self.sb_noise_epsilon > 0.0:
            bridge_gate = torch.sqrt((t.float() * (1.0 - t.float())).clamp_min(0.0)).view(-1, 1, 1, 1)
            x_t = x_t + torch.randn_like(x_t) * (self.sb_noise_epsilon ** 0.5) * bridge_gate
        t_profile = self._profile_start(content)
        transport_mode = str(getattr(model, "transport_prediction_mode", "velocity")).strip().lower()
        pred_endpoint: torch.Tensor | None = None
        if enforce_endpoint and transport_mode != "endpoint":
            raise ValueError("objective_mode='i2sb_endpoint' requires transport_prediction_mode='endpoint'.")
        if transport_mode == "endpoint":
            pred_endpoint = self._sanitize_tensor(
                model.predict_transport_base(
                    x_t,
                    t=t,
                    style_id=target_style_id,
                ),
                clamp_value=self.endpoint_clamp,
            )
            denom = (1.0 - t).clamp_min(self.eps).view(-1, 1, 1, 1)
            pred_velocity = self._sanitize_tensor((pred_endpoint - x_t) / denom, clamp_value=self.velocity_clamp)
            raw_flow_loss = self._loss(pred_endpoint, flow_objective_target)
        else:
            pred_velocity = model(x_t, t=t, style_id=target_style_id)
            raw_flow_loss = self._loss(pred_velocity, target_velocity)
        self._profile_end("model_forward", t_profile, content)
        if self.w_flow > 0.0:
            flow_loss = raw_flow_loss * self.w_flow
        elif require_flow_weight:
            raise ValueError("objective_mode='i2sb_endpoint' requires bridge.w_flow > 0.")
        else:
            flow_loss = raw_flow_loss
        total_loss = flow_loss

        t_profile = self._profile_start(content)
        zero = content.new_tensor(0.0, dtype=torch.float32)
        kinetic_loss = zero
        if self.w_kinetic > 0.0 and self.kinetic_mode in {"path", "time_gated"}:
            v_sq = pred_velocity.float() ** 2
            if self.kinetic_mode == "time_gated":
                gate = t.view(-1, 1, 1, 1).float() ** self.kinetic_gate_exponent
                kinetic_loss = (gate * v_sq).mean()
            else:
                kinetic_loss = v_sq.mean()
            total_loss = total_loss + kinetic_loss * self.w_kinetic
        use_anisotropic = self.structure_penalty_mode in {
            "anisotropic",
            "anisotropic_plus_stokes",
            "edge_gated_anisotropic",
            "edge_gated_anisotropic_plus_stokes",
            "quantile_edge_gated_anisotropic",
            "quantile_edge_gated_anisotropic_plus_stokes",
        }
        use_stokes = self.structure_penalty_mode in {
            "stokes",
            "anisotropic_plus_stokes",
            "edge_gated_anisotropic_plus_stokes",
            "quantile_edge_gated_anisotropic_plus_stokes",
        }
        anisotropic_kinetic = self._anisotropic_kinetic_loss(pred_velocity, content) if use_anisotropic and self.w_anisotropic_kinetic > 0.0 else zero
        stokes_viscous = self._stokes_viscous_loss(pred_velocity) if use_stokes and self.w_stokes_viscous > 0.0 else zero
        total_loss = total_loss + anisotropic_kinetic + stokes_viscous
        self._profile_end("aux_loss", t_profile, content)

        curvature_loss = content.new_tensor(0.0, dtype=torch.float32)
        if self.w_curvature > 0.0:
            dt = self.curvature_dt
            t2 = (t + dt).clamp(max=self.t_max)
            pred_v2 = model(x_t + pred_velocity * dt, t=t2, style_id=target_style_id)
            curvature_loss = self._loss(pred_v2, pred_velocity)
            total_loss = total_loss + curvature_loss * self.w_curvature
        content_lowpass_anchor, content_edge_anchor = self._content_topology_anchor_loss(pred_endpoint if pred_endpoint is not None else x_t, content)

        t_profile = self._profile_start(content)
        terminal_swd, endpoint_abs = self._terminal_swd(
            model,
            content=content,
            matched_target=objective_target,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        self._profile_end("terminal_swd", t_profile, content)
        if terminal_swd is not None:
            total_loss = total_loss + terminal_swd * self.terminal_swd_weight
        proximal_target = zero
        pred_endpoint_base = None
        pred_endpoint_final = None
        if self.proximal_target_weight > 0.0:
            pred_endpoint_base = self._sanitize_tensor(
                model.predict_transport_base(
                    content,
                    t=content.new_ones(content.shape[0]),
                    style_id=target_style_id,
                ),
                clamp_value=self.endpoint_clamp,
            )
            pred_endpoint_final = self._sanitize_tensor(
                model.refine_endpoint(
                    pred_endpoint_base,
                    style_id=target_style_id,
                    source_latent=content,
                ),
                clamp_value=self.endpoint_clamp,
            )
            proximal_target = self._loss(pred_endpoint_final, objective_target) * self.proximal_target_weight
            total_loss = total_loss + proximal_target
        proximal_residual = getattr(model, "last_proximal_residual", None)
        proximal_clamp_scale = getattr(model, "last_proximal_clamp_scale", None)
        proximal_residual_abs = proximal_residual.abs().mean().detach() if torch.is_tensor(proximal_residual) else zero
        proximal_residual_energy = (
            proximal_residual.float().square().mean() * float(getattr(model, "proximal_residual_energy_weight", 0.0))
            if torch.is_tensor(proximal_residual) and float(getattr(model, "proximal_residual_energy_weight", 0.0)) > 0.0
            else zero
        )
        if torch.is_tensor(proximal_residual_energy) and self.proximal_target_weight > 0.0:
            total_loss = total_loss + proximal_residual_energy
        generated_delta_diversity, generated_delta_mean_offdiag_cos, generated_delta_active_styles = (
            self._generated_delta_diversity_loss(pred_velocity, target_style_id, source_style_id)
            if self.w_generated_delta_diversity > 0.0
            else (zero, zero, zero)
        )
        total_loss = total_loss + generated_delta_diversity
        cycle_consistency = self._cycle_consistency_loss(
            model,
            content=content,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
        total_loss = total_loss + content_lowpass_anchor + content_edge_anchor + cycle_consistency
        attn_plan = model.last_semantic_attn
        semantic_k = model.last_semantic_k
        topology_attn = getattr(model, "last_semantic_topology_attn", None)

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "kinetic_energy": (kinetic_loss * self.w_kinetic).detach(),
            "anisotropic_kinetic": anisotropic_kinetic.detach(),
            "stokes_viscous": stokes_viscous.detach(),
            "curvature": (curvature_loss * self.w_curvature).detach(),
            "ot_cost": ot_cost.detach(),
            "plan_entropy": plan_entropy.detach(),
            "ot_plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(self.bridge_sigma, dtype=torch.float32),
            "bridge_noise_schedule_exact": content.new_tensor(
                1.0 if resolves_exact_brownian_schedule(
                    bridge_noise_schedule=self.bridge_noise_schedule,
                    objective_mode=self.objective_mode,
                ) else 0.0,
                dtype=torch.float32,
            ),
            "bridge_path_slerp_active": content.new_tensor(
                1.0 if self.bridge_path_mode == "latent_slerp" else 0.0,
                dtype=torch.float32,
            ),
            "t_mean": t.mean().detach(),
            "velocity_abs": pred_velocity.abs().mean().detach(),
            "target_velocity_abs": target_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
            "content_lowpass_anchor": content_lowpass_anchor.detach(),
            "content_edge_anchor": content_edge_anchor.detach(),
            "terminal_swd": terminal_swd.detach() if terminal_swd is not None else content.new_tensor(0.0),
            "proximal_target": proximal_target.detach() if torch.is_tensor(proximal_target) else zero,
            "proximal_residual_abs": proximal_residual_abs.detach() if torch.is_tensor(proximal_residual_abs) else zero,
            "proximal_clamp_scale": proximal_clamp_scale.detach() if torch.is_tensor(proximal_clamp_scale) else content.new_tensor(1.0),
            "proximal_residual_energy": proximal_residual_energy.detach() if torch.is_tensor(proximal_residual_energy) else zero,
            "base_endpoint_abs": pred_endpoint_base.abs().mean().detach() if torch.is_tensor(pred_endpoint_base) else content.new_tensor(0.0),
            "final_endpoint_abs": pred_endpoint_final.abs().mean().detach() if torch.is_tensor(pred_endpoint_final) else content.new_tensor(0.0),
            "generated_delta_diversity": generated_delta_diversity.detach(),
            "generated_delta_mean_offdiag_cos": generated_delta_mean_offdiag_cos.detach(),
            "generated_delta_active_styles": generated_delta_active_styles.detach(),
            "cycle_consistency": cycle_consistency.detach(),
            "semantic_attn_mean": attn_plan.mean().detach() if attn_plan is not None else content.new_tensor(0.0),
            "semantic_k_abs": semantic_k.abs().mean().detach() if semantic_k is not None else content.new_tensor(0.0),
            "semantic_topology_attn_entropy": self._attention_entropy(topology_attn, content).detach(),
            "semantic_topology_attn_active": content.new_tensor(1.0 if topology_attn is not None else 0.0, dtype=torch.float32),
        }
        pred_probe_endpoint = pred_endpoint if pred_endpoint is not None else (content + pred_velocity)
        if bool(getattr(model, "transport_stats_mode", "none") == "normalized_solver"):
            restore_transport_output = getattr(model, "restore_transport_output", None)
            if callable(restore_transport_output):
                probe_latent = pred_endpoint if pred_endpoint is not None else (flow_content + pred_velocity)
                pred_probe_endpoint = restore_transport_output(probe_latent, style_id=target_style_id)
        metrics.update(
            self._fiber_probe_metrics(
                pred_endpoint=pred_probe_endpoint,
                target_endpoint=objective_target,
                content=content,
                mode=self.training_target_projection_mode,
                kernel=self.training_target_projection_kernel,
            )
        )
        metrics.update(ot_debug)
        metrics.update(target_projection_metrics)
        metrics.update(bridge_noise_metrics)
        metrics.update(transport_stats_metrics)
        metrics.update(self._profile_metrics(content))
        metrics.update(self._model_profile_metrics(model, content))
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()
        components = {
            "flow": flow_loss,
            "kinetic_energy": kinetic_loss * self.w_kinetic,
            "anisotropic_kinetic": anisotropic_kinetic,
            "stokes_viscous": stokes_viscous,
            "curvature": curvature_loss * self.w_curvature,
            "content_lowpass_anchor": content_lowpass_anchor,
            "content_edge_anchor": content_edge_anchor,
            "terminal_swd": terminal_swd * self.terminal_swd_weight if terminal_swd is not None else zero,
            "proximal_target": proximal_target,
            "proximal_residual_energy": proximal_residual_energy,
            "generated_delta_diversity": generated_delta_diversity,
            "cycle_consistency": cycle_consistency,
        }
        components.update(target_projection_metrics)
        debug_state: Dict[str, torch.Tensor | None] = {
            "content": content.detach(),
            "target_style": target_style.detach(),
            "matched_target": matched_target.detach(),
            "objective_target": objective_target.detach(),
            "x_t": x_t.detach(),
            "t": t.detach(),
            "target_velocity": target_velocity.detach(),
            "pred_velocity": pred_velocity.detach(),
            "pred_endpoint": pred_endpoint.detach() if pred_endpoint is not None else None,
            "pred_endpoint_base": pred_endpoint_base.detach() if torch.is_tensor(pred_endpoint_base) else None,
            "pred_endpoint_final": pred_endpoint_final.detach() if torch.is_tensor(pred_endpoint_final) else None,
            "semantic_attn": getattr(model, "last_semantic_attn", None).detach() if getattr(model, "last_semantic_attn", None) is not None else None,
            "semantic_k": getattr(model, "last_semantic_k", None).detach() if getattr(model, "last_semantic_k", None) is not None else None,
            "semantic_topology_attn": topology_attn.detach() if topology_attn is not None else None,
        }
        return metrics, components, debug_state

    def _compute_sampled_bridge(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        enforce_endpoint: bool = False,
        require_flow_weight: bool = False,
    ) -> Dict[str, torch.Tensor]:
        metrics, _, _ = self._compute_sampled_bridge_details(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
            enforce_endpoint=enforce_endpoint,
            require_flow_weight=require_flow_weight,
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
        aux_target_style: torch.Tensor | None = None,
        aux_target_valid: torch.Tensor | None = None,
        conditioning: dict | None = None,
    ) -> Dict[str, torch.Tensor]:
        self._set_model_conditioning(model, conditioning)
        if self.objective_mode == "omf":
            return self._compute_omf(
                model,
                content=content,
                target_style=target_style,
                target_style_id=target_style_id,
                source_style_id=source_style_id,
                aux_target_style=aux_target_style,
                aux_target_valid=aux_target_valid,
            )
        if self.objective_mode == "i2sb_endpoint":
            return self._compute_sampled_bridge(
                model,
                content=content,
                target_style=target_style,
                target_style_id=target_style_id,
                source_style_id=source_style_id,
                enforce_endpoint=True,
                require_flow_weight=True,
            )

        return self._compute_sampled_bridge(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )

    def compute_debug(
        self,
        model: TimeConditionedLANCETBridge,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        aux_target_style: torch.Tensor | None = None,
        aux_target_valid: torch.Tensor | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor] | Dict[str, torch.Tensor | None]]:
        if self.objective_mode == "omf":
            metrics, components, state = self._compute_omf_details(
                model,
                content=content,
                target_style=target_style,
                target_style_id=target_style_id,
                source_style_id=source_style_id,
                aux_target_style=aux_target_style,
                aux_target_valid=aux_target_valid,
            )
            return {"metrics": metrics, "components": components, "state": state}
        metrics, components, state = self._compute_sampled_bridge_details(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
            enforce_endpoint=self.objective_mode == "i2sb_endpoint",
            require_flow_weight=self.objective_mode == "i2sb_endpoint",
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
        conditioning: dict | None = None,
    ) -> Dict[str, torch.Tensor]:
        self._set_model_conditioning(model, conditioning)
        self._set_model_conditioning(teacher_model, conditioning)
        if content.device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = torch.autocast("cpu", enabled=False)
        with torch.no_grad():
            with autocast_ctx:
                matched_target, ot_cost, plan_entropy, _, ot_debug = self._ot_match_targets(
                    model,
                    content,
                    target_style,
                    target_style_id,
                    source_style_id,
                )
            t = self._sample_t(content)
            x_t, _, _ = self._bridge_state_and_velocity(content=content, matched_target=matched_target, t=t)
            if str(getattr(teacher_model, "transport_prediction_mode", "velocity")).strip().lower() == "endpoint":
                teacher_endpoint_base = teacher_model.predict_transport_base(x_t, t=t, style_id=target_style_id)
                teacher_velocity = (teacher_endpoint_base - x_t) / (1.0 - t).clamp_min(self.eps).view(-1, 1, 1, 1)
            else:
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
            "ot_plan_entropy": plan_entropy.detach(),
            "bridge_sigma": content.new_tensor(self.bridge_sigma, dtype=torch.float32),
            "t_mean": t.mean().detach(),
            "velocity_abs": teacher_velocity.abs().mean().detach(),
            "endpoint_abs": endpoint_abs.detach(),
        }
        metrics.update(ot_debug)
        if source_style_id is not None:
            id_mask = source_style_id.long() == target_style_id.long()
            metrics["identity_ratio"] = id_mask.float().mean().detach()
        return metrics
