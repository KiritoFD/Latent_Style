import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch._inductor.config as inductor_config

# Enable persistent FX graph cache to reduce repeated compile overhead.
inductor_config.fx_graph_cache = True

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from utils.dataset import LatentDataset
from utils.inference import decode_latent, encode_image, load_vae
from losses import (
    GeometricFreeEnergyLoss,
    MultiScaleSWDLoss,
    PyramidStructuralLoss,
    VelocityRegularizationLoss,
)
from utils.style_classifier import StyleClassifier
from model import LGTUNetLite, count_parameters
from physics import generate_latent, get_dynamic_epsilon, invert_latent

logger = logging.getLogger(__name__)

def _log_vram(tag: str, reset_peak: bool = False):
    if not torch.cuda.is_available():
        return
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"[VRAM] {tag} | alloc={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB")


def _log_vram_detail(tag: str):
    if not torch.cuda.is_available():
        return
    stats = torch.cuda.memory_stats()
    def _mb(x): return x / (1024 ** 2)
    detail = {
        "alloc_mb": _mb(stats.get("allocated_bytes.all.current", 0)),
        "reserved_mb": _mb(stats.get("reserved_bytes.all.current", 0)),
        "active_mb": _mb(stats.get("active_bytes.all.current", 0)),
        "inactive_split_mb": _mb(stats.get("inactive_split_bytes.all.current", 0)),
        "alloc_peak_mb": _mb(stats.get("allocated_bytes.all.peak", 0)),
        "reserved_peak_mb": _mb(stats.get("reserved_bytes.all.peak", 0)),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "num_segments": stats.get("segment.all.current", 0),
        "num_active_allocs": stats.get("active.all.current", 0),
    }
    logger.info(
        "[VRAM-DETAIL] %s | alloc=%.1fMB reserved=%.1fMB active=%.1fMB "
        "inactive_split=%.1fMB alloc_peak=%.1fMB reserved_peak=%.1fMB "
        "segments=%s active_allocs=%s retries=%s ooms=%s",
        tag,
        detail["alloc_mb"],
        detail["reserved_mb"],
        detail["active_mb"],
        detail["inactive_split_mb"],
        detail["alloc_peak_mb"],
        detail["reserved_peak_mb"],
        detail["num_segments"],
        detail["num_active_allocs"],
        detail["num_alloc_retries"],
        detail["num_ooms"],
    )


class RunningMean:
    """Simple running mean for adaptive loss normalization."""
    def __init__(self, momentum=0.99, device='cuda'):
        self.mean = torch.tensor(1.0, device=device)
        self.momentum = momentum

    def update(self, val):
        # Detach to prevent gradient tracking on the normalizer itself
        val_detached = val.detach()
        self.mean = self.momentum * self.mean + (1 - self.momentum) * val_detached
        # Return tensor to avoid CPU sync
        return torch.maximum(self.mean, torch.tensor(1e-4, device=self.mean.device))

    def state_dict(self):
        return {'mean': self.mean}

    def load_state_dict(self, state_dict):
        if 'mean' in state_dict:
            self.mean = state_dict['mean'].to(self.mean.device)

class LGTTrainer:
    """Trainer orchestrating optimization, losses, and evaluation."""

    def __init__(self, config: Dict, device: torch.device = torch.device('cuda'), config_path: Optional[str] = None):
        self.config = config
        self.device = device
        self.config_path = Path(config_path) if config_path is not None else None
        self.vram_debug = bool(config.get('training', {}).get('vram_debug', True))
        self.vram_debug_interval = int(config.get('training', {}).get('vram_debug_interval', 50))
        self.vram_debug_reset_peak = bool(config.get('training', {}).get('vram_debug_reset_peak', False))
        self.vram_debug_detail = bool(config.get('training', {}).get('vram_debug_detail', True))
        self.vram_debug_skip_steps = int(config.get('training', {}).get('vram_debug_skip_steps', 2))
        self._vram_log_this_step = False
        if self.vram_debug:
            _log_vram("init/start", reset_peak=self.vram_debug_reset_peak)
            if self.vram_debug_detail:
                _log_vram_detail("init/start")

        # Infra optimization: enable Tensor Cores (Ampere+).
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        self.allow_tf32 = bool(config.get('training', {}).get('allow_tf32', True))
        if self.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        use_checkpointing = config['training'].get('use_gradient_checkpointing', False)
        model_cfg = config['model']
        self.model = LGTUNetLite(
            latent_channels=model_cfg['latent_channels'],
            base_channels=model_cfg.get('base_channels', 64),
            style_dim=model_cfg['style_dim'],
            time_dim=model_cfg.get('time_dim', 64),
            num_styles=model_cfg['num_styles'],
            num_encoder_blocks=model_cfg.get('num_encoder_blocks', 1),
            num_decoder_blocks=model_cfg.get('num_decoder_blocks', 1),
            dropout=model_cfg.get('dropout', 0.0),
            v_max=model_cfg.get('v_max', 2.0),
            use_checkpointing=use_checkpointing,
        )
        self.model = self.model.to(device, memory_format=torch.channels_last)  # Channels Last
        logger.info("Model architecture: unet_lite")
        self.model.compute_avg_style_embedding()
        if self.vram_debug:
            _log_vram("after model init", reset_peak=self.vram_debug_reset_peak)

        use_compile = config['training'].get('use_compile', False)
        if self.vram_debug and use_compile:
            logger.warning("VRAM debug enabled: disabling torch.compile to avoid Dynamo errors")
            use_compile = False
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode='default', fullgraph=False)
                logger.info("Model compiled with torch.compile")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(f"torch.compile failed: {exc}")

        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        self.energy_loss = GeometricFreeEnergyLoss(
            num_styles=config['model']['num_styles'],
            w_style=config['loss']['w_style'],
            swd_scales=config['loss'].get('swd_scales', [1, 3, 5]),  # Avoid large costly scales by default.
            swd_scale_weights=config['loss'].get('swd_scale_weights', [1.0, 1.0, 0.5]),
            num_projections=config['loss'].get('num_projections', 64),
            max_samples=config['loss'].get('max_samples', 4096),
            moment_lowpass_size=config['loss'].get('moment_lowpass_size', 8),
            moment_weight=float(config['loss'].get('swd_moment_weight', 10.0)),
            fixed_sample_indices=config['loss'].get('fixed_swd_sample_indices', True),
            swd_sample_seed=config['loss'].get('swd_sample_seed', 1234),
        ).to(device)
        if self.vram_debug:
            _log_vram("after energy_loss init", reset_peak=self.vram_debug_reset_peak)

        # Style classifier guidance (to stabilize style direction)
        self.use_style_classifier = config['loss'].get('use_style_classifier', True)
        self.style_cls_weight = float(config['loss'].get('w_style_classifier', 0.1))
        self.style_cls_transfer_weight = float(
            config['loss'].get('w_style_classifier_transfer', self.style_cls_weight)
        )
        self.style_cls_identity_weight = float(
            config['loss'].get('w_style_classifier_identity', self.style_cls_weight)
        )
        self.style_cls_target_only = bool(config['loss'].get('style_cls_target_only', True))
        self.style_cls_use_gating = bool(config['loss'].get('style_cls_use_gating', True))
        self.style_cls_conf_threshold = float(config['loss'].get('style_cls_conf_threshold', 0.75))
        self.style_cls_agree_threshold = float(config['loss'].get('style_cls_agree_threshold', 0.70))
        self.style_cls_warmup_steps = int(config['loss'].get('style_cls_warmup_steps', 2000))
        self.style_cls_temperature = float(config['loss'].get('style_cls_temperature', 1.0))
        self.style_classifier = None
        self.style_classifier_ckpt = config['loss'].get('style_classifier_ckpt', None)
        self.style_classifier_strict = bool(config['loss'].get('style_classifier_strict', False))
        self.style_classifier_trainable = bool(config['loss'].get('style_classifier_trainable', False))
        self.use_style_swd = bool(config['loss'].get('use_style_swd', True))
        self.swd_warmup_ratio = float(config['loss'].get('swd_warmup_ratio', 0.2))
        self.swd_warmup_ratio = min(max(self.swd_warmup_ratio, 0.0), 1.0)
        self.style_swd_interval = max(1, int(config['loss'].get('style_swd_interval', 1)))
        self.style_cls_interval = max(1, int(config['loss'].get('style_cls_interval', 1)))
        self.swd_input_size = int(config['loss'].get('swd_input_size', 0))
        self.swd_band_mode = str(config['loss'].get('swd_band_mode', 'highpass')).lower()
        if self.swd_band_mode not in {'highpass', 'lowpass', 'both'}:
            logger.warning(f"Invalid swd_band_mode={self.swd_band_mode}, fallback to highpass")
            self.swd_band_mode = 'highpass'
        self.swd_high_band_weight = float(config['loss'].get('swd_high_band_weight', 1.0))
        self.swd_low_band_weight = float(config['loss'].get('swd_low_band_weight', 1.0))
        raw_feature_levels = config['loss'].get('swd_feature_levels', ['latent'])
        if isinstance(raw_feature_levels, str):
            raw_feature_levels = [raw_feature_levels]
        raw_feature_levels = list(raw_feature_levels)
        valid_levels = {'latent', 'early', 'mid', 'late'}
        self.swd_feature_levels = [lvl for lvl in raw_feature_levels if lvl in valid_levels]
        if not self.swd_feature_levels:
            self.swd_feature_levels = ['latent']
        self.swd_feature_reduce_channels = int(config['loss'].get('swd_feature_reduce_channels', 16))
        self.swd_feature_reduce_channels = max(1, self.swd_feature_reduce_channels)
        self.swd_feature_t = float(config['loss'].get('swd_feature_t', 1.0))
        self.swd_feature_t = min(max(self.swd_feature_t, 0.0), 1.0)
        self._feature_swd_losses: Dict[int, MultiScaleSWDLoss] = {}
        self.swd_warmup_steps = 0
        self.total_train_steps_estimate = 0
        logger.info(
            "SWD setup | enabled=%s warmup_ratio=%.3f band=%s levels=%s "
            "high_w=%.3f low_w=%.3f moment_w=%.3f",
            self.use_style_swd,
            self.swd_warmup_ratio,
            self.swd_band_mode,
            self.swd_feature_levels,
            self.swd_high_band_weight,
            self.swd_low_band_weight,
            float(config['loss'].get('swd_moment_weight', 10.0)),
        )
        self.identity_pair_ratio = float(config['loss'].get('identity_pair_ratio', 0.25))
        self.identity_pair_ratio = max(0.0, min(1.0, self.identity_pair_ratio))
        self.use_identity_consistency = bool(config['loss'].get('use_identity_consistency', True))
        self.identity_weight = float(config['loss'].get('w_identity', 1.0))

        # Optional per-target-style SWD mask/weight (default: symmetric 1.0 for all styles)
        swd_target_weights = config['loss'].get('style_swd_target_weights', None)
        if swd_target_weights is None:
            swd_target_weights = [1.0] * config['model']['num_styles']
        if len(swd_target_weights) < config['model']['num_styles']:
            swd_target_weights = list(swd_target_weights) + [1.0] * (config['model']['num_styles'] - len(swd_target_weights))
        self.style_swd_target_weights = torch.tensor(
            swd_target_weights[:config['model']['num_styles']],
            device=device,
            dtype=torch.float32,
        )

        if self.use_style_classifier and max(self.style_cls_transfer_weight, self.style_cls_identity_weight) > 0.0:
            input_size_train = int(config['loss'].get('style_classifier_input_size_train', 8))
            input_size_infer = int(config['loss'].get('style_classifier_input_size_infer', input_size_train))
            self.style_classifier = StyleClassifier(
                in_channels=config['model']['latent_channels'],
                num_classes=config['model']['num_styles'],
                use_stats=bool(config['loss'].get('style_classifier_use_stats', True)),
                use_gram=bool(config['loss'].get('style_classifier_use_gram', True)),
                use_lowpass_stats=bool(config['loss'].get('style_classifier_use_lowpass_stats', True)),
                spatial_shuffle=bool(config['loss'].get('style_classifier_spatial_shuffle', True)),
                input_size_train=input_size_train,
                input_size_infer=input_size_infer,
                lowpass_size=int(config['loss'].get('style_classifier_lowpass_size', 8)),
            ).to(device)
            ckpt_path = None
            if self.style_classifier_ckpt:
                ckpt_path = Path(self.style_classifier_ckpt)
                if not ckpt_path.is_absolute():
                    ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, map_location=device)
                    state_dict = state.get('model_state_dict', state)
                    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                    self.style_classifier.load_state_dict(state_dict, strict=self.style_classifier_strict)
                    # Prefer calibrated temperature from classifier checkpoint when available.
                    meta = state.get('meta', {}) if isinstance(state, dict) else {}
                    if isinstance(meta, dict) and 'temperature' in meta:
                        self.style_cls_temperature = float(meta['temperature'])
                    logger.info(f"Loaded style classifier weights from {ckpt_path}")
                else:
                    msg = f"Style classifier checkpoint not found: {ckpt_path}"
                    if self.style_classifier_strict:
                        raise FileNotFoundError(msg)
                    logger.warning(msg + " (training from scratch)")
            elif self.style_classifier_strict:
                msg = "style_classifier_ckpt is required when style_classifier_strict=true"
                raise FileNotFoundError(msg)
            # Freeze classifier params but keep grad flow to main model
            if self.style_classifier_trainable is False:
                for p in self.style_classifier.parameters():
                    p.requires_grad = False
                self.style_classifier.eval()
            logger.info(
                f"Style classifier guidance enabled "
                f"(w_transfer={self.style_cls_transfer_weight}, w_identity={self.style_cls_identity_weight}, "
                f"T={self.style_cls_temperature:.3f}, gating={self.style_cls_use_gating})"
            )
        else:
            logger.info("Style classifier guidance disabled")
        if self.use_identity_consistency:
            logger.info(
                f"Identity consistency enabled (pair_ratio={self.identity_pair_ratio}, w_identity={self.identity_weight})"
            )
        else:
            logger.info("Identity consistency disabled")
        if self.vram_debug:
            _log_vram("after style_classifier init", reset_peak=self.vram_debug_reset_peak)

        # Pyramid Structural Loss (frequency-separated structure lock).
        pyramid_weights = config['loss'].get('pyramid_weights', {'low': 5.0, 'mid': 1.0, 'high': 0.1})
        self.pyramid_loss = PyramidStructuralLoss(weights=pyramid_weights).to(device)
        self.w_pyramid = config['loss'].get('w_pyramid', 1.0)
        logger.info(f"Pyramid Structural Loss enabled (w={self.w_pyramid}, weights={pyramid_weights})")
        if self.vram_debug:
            _log_vram("after pyramid_loss init", reset_peak=self.vram_debug_reset_peak)

        self.use_velocity_reg = config['loss'].get('use_velocity_reg', False)
        self.vel_reg_loss = None
        if self.use_velocity_reg:
            vel_reg_weight = config['loss'].get('vel_reg_weight', 0.1)
            self.vel_reg_loss = VelocityRegularizationLoss(weight=vel_reg_weight).to(device)
            logger.info(f"Velocity Regularization enabled (weight={vel_reg_weight})")
        else:
            logger.info("Velocity Regularization disabled")
        if self.vram_debug:
            _log_vram("after vel_reg init", reset_peak=self.vram_debug_reset_peak)

        self.num_epochs = config['training']['num_epochs']

        optim_params = list(self.model.parameters())
        if self.style_classifier is not None and self.style_classifier_trainable:
            optim_params += list(self.style_classifier.parameters())

        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            fused=False,  # Fused AdamW conflicts with GradScaler.unscale_ in this setup.
        )
        if self.vram_debug:
            _log_vram("after optimizer init", reset_peak=self.vram_debug_reset_peak)

        # Initialize scheduler immediately so it can be loaded from checkpoint
        self._setup_scheduler()

        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        if self.vram_debug:
            _log_vram("after scaler init", reset_peak=self.vram_debug_reset_peak)

        # Adaptive loss normalization (running means).
        self.use_adaptive_norm = config['training'].get('use_adaptive_loss_norm', False)
        if self.use_adaptive_norm:
            self.norm_mse = RunningMean(device=device)
            self.norm_style = RunningMean(device=device)
            logger.info("Adaptive Loss Normalization enabled (Running Mean Scaling)")
            
        self.label_drop_prob = config['training'].get('label_drop_prob', 0.1)
        self.use_avg_for_uncond = config['training'].get('use_avg_style_for_uncond', True)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        self.effective_batch_size = config['training']['batch_size'] * self.accumulation_steps

        if self.accumulation_steps > 1:
            logger.info(
                f"Gradient accumulation: {self.accumulation_steps} steps | "
                f"Effective batch size: {self.effective_batch_size}"
            )

        self.global_step = 0
        self.alpha_warmup_steps = config['training'].get('alpha_warmup_steps', 1000)
        logger.info(f"Alpha warmup schedule: 0 -> 1.0 over {self.alpha_warmup_steps} steps")

        self.epsilon = config['training'].get('epsilon', 0.01)
        self.ode_steps = config['training'].get('ode_integration_steps', 5)
        self.model_input_size_train = int(config['training'].get('model_input_size_train', 0))
        self.model_input_size_infer = int(config['training'].get('model_input_size_infer', self.model_input_size_train))

        self.style_weights = torch.tensor(
            config['loss'].get('style_weights', [1.0] * config['model']['num_styles']),
            device=device,
            dtype=torch.float32,
        )
        logger.info(f"Per-style loss weights: {self.style_weights.tolist()}")

        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config['training'].get('save_interval', 5)

        self.eval_interval = config['training'].get('eval_interval', 5)
        self.full_eval_interval = config['training'].get('full_eval_interval', self.eval_interval)
        logger.info(f"Full external evaluation interval: {self.full_eval_interval}")
        self.test_image_dir = Path(config['training'].get('test_image_dir', 'test_images'))
        self.inference_dir = self.checkpoint_dir / 'inference'
        self.inference_dir.mkdir(exist_ok=True)

        self.vae = None
        try:
            self.vae = load_vae(device)
        except Exception as exc:
            logger.warning(f"VAE load failed; inference evaluation will be skipped. reason={exc}")
        if self.vram_debug:
            _log_vram("after VAE load", reset_peak=self.vram_debug_reset_peak)

        self.style_indices_cache = None
        self.style_indices_tensor_cache = None
        self.style_reference_pool = None
        self.dataset_ref: Optional[LatentDataset] = None
        self.use_fixed_style_reference = self.config['training'].get('use_fixed_style_reference', True)
        self.fixed_style_reference_size = int(self.config['training'].get('fixed_style_reference_size', 128))

        self.log_dir = self.checkpoint_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, 'w') as f:
            f.write(
                "epoch,loss_total,loss_style_swd,loss_mse,loss_identity,loss_style_cls,"
                "style_cls_transfer_acc,style_cls_identity_acc,style_cls_transfer_pass_rate,"
                "style_cls_identity_pass_rate,style_cls_conf_mean,style_cls_agree_mean,style_cls_weight_mean,"
                "loss_velocity_reg,learning_rate,epoch_time\n"
            )

        self.start_epoch = 1
        self._maybe_resume(config['training'].get('resume_checkpoint'))

        try:
            from diffusers import DDPMScheduler

            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
            )
        except Exception as exc:
            logger.warning(f"diffusers not available, using simple noise scheduler fallback. reason={exc}")

            class _FallbackNoiseScheduler:
                class _Cfg:
                    num_train_timesteps = 1000

                config = _Cfg()

            self.noise_scheduler = _FallbackNoiseScheduler()
        self.iter_step = 0

    # ------------------------------------------------------------------
    # Initialization utilities
    # ------------------------------------------------------------------
    def _maybe_resume(self, resume_checkpoint: Optional[str]) -> None:
        latest_ckpt = None
        if resume_checkpoint:
            latest_ckpt = Path(resume_checkpoint)
            logger.info(f"Overriding resume checkpoint: {resume_checkpoint}")
        else:
            latest_ckpt = find_latest_checkpoint(self.checkpoint_dir)

        if latest_ckpt is None:
            logger.info("No checkpoint found; starting fresh")
            return

        try:
            resume_info = load_checkpoint(
                checkpoint_path=latest_ckpt,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                current_config=self.config,
                device=self.device,
            )
            self.start_epoch = resume_info.get('start_epoch', 1)
            self.global_step = resume_info.get('global_step', 0)
            
            # Restore Adaptive Normalization State
            if self.use_adaptive_norm and 'adaptive_norm_state' in resume_info:
                norm_state = resume_info['adaptive_norm_state']
                if norm_state:
                    if 'mse' in norm_state: self.norm_mse.load_state_dict(norm_state['mse'])
                    if 'style' in norm_state: self.norm_style.load_state_dict(norm_state['style'])
                    logger.info("Adaptive loss normalization state restored")
            
            # Restore LoRA Alpha immediately
            self._update_lora_alpha(self.global_step)
            
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(f"Failed to resume from {latest_ckpt}: {exc}")

        # Note: We rely on the loaded scheduler state to determine the correct LR.
        # Forcing LR to config['learning_rate'] here would reset it to max_lr,
        # which is incorrect if resuming in the middle of training.
        logger.info(f"Resumed training state. Current LR will be determined by scheduler.")

    def build_style_indices_cache(self, dataset: LatentDataset) -> None:
        logger.info("Building style indices cache...")
        self.style_indices_cache = dataset.style_indices
        self.style_indices_tensor_cache = {
            style_id: torch.tensor(indices, dtype=torch.long)
            for style_id, indices in self.style_indices_cache.items()
        }
        for style_id, indices in self.style_indices_cache.items():
            logger.info(f"  Style {style_id}: {len(indices)} samples")
        self.dataset_ref = dataset
        logger.info("Style indices cache built")

    def sample_style_batch(self, target_style_ids: torch.Tensor) -> torch.Tensor:
        if self.dataset_ref is None:
            raise RuntimeError("Dataset reference is not set; call build_style_indices_cache first")

        device = target_style_ids.device
        b = target_style_ids.shape[0]
        style_latents = torch.empty((b, *self.dataset_ref.latents_tensor.shape[1:]), device=device)
        target_cpu = target_style_ids.detach().cpu()

        for style_id in range(self.config['model']['num_styles']):
            if style_id not in self.style_indices_cache:
                continue
            indices_tensor = self.style_indices_tensor_cache[style_id]
            if self.use_fixed_style_reference and self.style_reference_pool is not None:
                pool = self.style_reference_pool[style_id]
                pool_size = pool.shape[0]
            mask = target_cpu == style_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            if self.use_fixed_style_reference and self.style_reference_pool is not None:
                rand_idx = torch.randint(pool_size, (count,))
                selected = pool[rand_idx]
            else:
                rand_indices = indices_tensor[torch.randint(indices_tensor.numel(), (count,))]
                selected = self.dataset_ref.latents_tensor[rand_indices]
            style_latents[mask.to(device)] = selected.to(device, non_blocking=True)

        return style_latents

    def sample_target_styles(self, source_style_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build target style IDs with a controlled identity ratio.

        Returns:
            target_style_ids: [B]
            identity_mask: [B] bool, where target_style == source_style
        """
        b = source_style_ids.shape[0]
        if b <= 1:
            target_style_ids = source_style_ids.clone()
            identity_mask = torch.ones_like(source_style_ids, dtype=torch.bool)
            return target_style_ids, identity_mask

        shuffled = source_style_ids[torch.randperm(b, device=source_style_ids.device)]
        if self.identity_pair_ratio <= 0.0:
            target_style_ids = shuffled
            return target_style_ids, target_style_ids == source_style_ids
        if self.identity_pair_ratio >= 1.0:
            target_style_ids = source_style_ids.clone()
            return target_style_ids, torch.ones_like(source_style_ids, dtype=torch.bool)

        identity_selector = torch.rand(b, device=source_style_ids.device) < self.identity_pair_ratio
        if not identity_selector.any():
            identity_selector[torch.randint(0, b, (1,), device=source_style_ids.device)] = True
        if identity_selector.all():
            identity_selector[torch.randint(0, b, (1,), device=source_style_ids.device)] = False

        target_style_ids = torch.where(identity_selector, source_style_ids, shuffled)
        identity_mask = target_style_ids == source_style_ids
        return target_style_ids, identity_mask

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def _get_current_alpha(self) -> float:
        return min(self.global_step / max(self.alpha_warmup_steps, 1), 1.0)

    def _update_lora_alpha(self, global_step: int) -> None:
        current_alpha = min(global_step / max(self.alpha_warmup_steps, 1), 1.0)
        for module in self.model.modules():
            if hasattr(module, 'alpha'):
                module.alpha = current_alpha
        if global_step % 250 == 0 and global_step < self.alpha_warmup_steps:
            logger.debug(f"Alpha warmup: step {global_step}/{self.alpha_warmup_steps}, alpha={current_alpha:.4f}")

    def _get_thermodynamic_schedule(self, epoch: int):
        """
        Placeholder hook for future thermodynamic schedule extensions.
        """
        return 1.0, 1.0, 1.0

    def update_loss_weights(self, epoch, total_epochs):
        """
        Piecewise schedule for structure/style loss balance.
        """
        w_style_target = self.config['loss'].get('w_style', 40.0)
        w_mse_target = self.config['loss'].get('w_mse', 0.5)
        
        progress = epoch / max(total_epochs, 1)
        mse_start_ratio = 0.2
        
        current_w_style = w_style_target
        
        if progress < 0.15:
            current_w_mse = w_mse_target * mse_start_ratio
        elif progress < 0.9:
            p = (progress - 0.15) / 0.75
            ratio = mse_start_ratio + (1.0 - mse_start_ratio) * p
            current_w_mse = w_mse_target * ratio
        else:
            current_w_mse = w_mse_target
            
        return current_w_style, current_w_mse

    def _get_swd_warmup_scale(self) -> float:
        if not self.use_style_swd:
            return 0.0
        if self.swd_warmup_steps <= 0:
            return 1.0
        return min(float(self.global_step) / float(max(self.swd_warmup_steps, 1)), 1.0)

    @staticmethod
    def _lowpass_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
        h, w = x.shape[-2:]
        target_size = max(1, min(size, h, w))
        x_lp = F.interpolate(x, size=(target_size, target_size), mode="area")
        x_lp = F.interpolate(x_lp, size=(h, w), mode="bilinear", align_corners=False)
        return x_lp

    @staticmethod
    def _reduce_feature_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
        c = x.shape[1]
        if c == target_channels:
            return x
        if target_channels > c:
            repeat = (target_channels + c - 1) // c
            return x.repeat(1, repeat, 1, 1)[:, :target_channels]
        chunks = torch.tensor_split(x, target_channels, dim=1)
        return torch.cat([chunk.mean(dim=1, keepdim=True) for chunk in chunks], dim=1)

    def _get_feature_swd_loss(self, in_channels: int) -> MultiScaleSWDLoss:
        module = self._feature_swd_losses.get(in_channels)
        if module is None:
            module = MultiScaleSWDLoss(
                scales=self.config['loss'].get('swd_scales', [1, 3, 5]),
                scale_weights=self.config['loss'].get('swd_scale_weights', [1.0, 1.0, 0.5]),
                num_projections=self.config['loss'].get('num_projections', 64),
                max_samples=self.config['loss'].get('max_samples', 4096),
                use_fp32=True,
                in_channels=in_channels,
            ).to(self.device)
            self._feature_swd_losses[in_channels] = module
        return module

    def _compute_direct_swd(self, x_pred: torch.Tensor, x_style: torch.Tensor) -> torch.Tensor:
        module = self._get_feature_swd_loss(int(x_pred.shape[1]))
        loss_val, _ = module(x_pred, x_style)
        return loss_val

    def compute_energy_loss(self, batch: Dict, epoch: int, multipliers: tuple = (1.0, 1.0, 1.0)) -> Dict[str, torch.Tensor]:
        device = self.device
        m_mse, m_layout, m_style = multipliers

        #  Infra Optimization: Channels Last for faster Convolutions
        latent = batch['latent'].to(device, non_blocking=True, memory_format=torch.channels_last)
        style_id = batch['style_id'].to(device, non_blocking=True)
        latent_deformed = batch.get('latent_deformed')
        if latent_deformed is not None:
            latent_deformed = latent_deformed.to(device, non_blocking=True, memory_format=torch.channels_last)
        if self._vram_log_this_step and self.vram_debug:
            _log_vram("compute_energy_loss/after_batch_to_device", reset_peak=self.vram_debug_reset_peak)

        b = latent.shape[0]
        style_id_tgt, identity_mask = self.sample_target_styles(style_id)
        transfer_mask = ~identity_mask
        style_latents = self.sample_style_batch(style_id_tgt)
        if self._vram_log_this_step and self.vram_debug:
            _log_vram("compute_energy_loss/after_style_latents", reset_peak=self.vram_debug_reset_peak)

        use_elastic = self.config['training'].get('use_elastic_deform', False)
        elastic_styles = self.config['training'].get('elastic_styles', [1])
        if use_elastic and latent_deformed is not None:
            mask = torch.zeros_like(style_id_tgt, dtype=torch.bool)
            for s in elastic_styles:
                mask |= style_id_tgt == s
            x_src_target = torch.where(mask.view(-1, 1, 1, 1), latent_deformed, latent)
        else:
            x_src_target = latent

        x0 = torch.randn_like(latent)

        sigma_noise = float(self.config['training'].get('noise_injection_sigma', 0.1))
        if sigma_noise > 0.0:
            noise_injection = torch.randn_like(x_src_target) * sigma_noise
            x_target_noisy = x_src_target + noise_injection
        else:
            x_target_noisy = x_src_target

        epsilon_dynamic = get_dynamic_epsilon(
            epoch=epoch,
            target_epsilon=self.epsilon,
            warmup_epochs=self.config['training'].get('epsilon_warmup_epochs', 10),
        )
        t = torch.rand(b, device=device) * (1 - epsilon_dynamic) + epsilon_dynamic

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        _ = (t * (num_train_timesteps - 1)).long()  # retained for schedule parity
        t_expand = t.view(-1, 1, 1, 1)

        x_t = (1 - t_expand) * x0 + t_expand * x_target_noisy
        if self._vram_log_this_step and self.vram_debug:
            _log_vram("compute_energy_loss/after_x_t", reset_peak=self.vram_debug_reset_peak)

        drop_label = False
        if self.model.training and torch.rand(1).item() < self.label_drop_prob:
            drop_label = True

        target_model_size = self.model_input_size_train if self.model.training else self.model_input_size_infer
        def _model_forward(x_in: torch.Tensor, t_in: torch.Tensor, style_in: torch.Tensor, use_avg: bool) -> torch.Tensor:
            x_model = x_in
            if target_model_size and (x_in.shape[-1] != target_model_size or x_in.shape[-2] != target_model_size):
                x_model = F.interpolate(x_in, size=(target_model_size, target_model_size), mode="area")
            v_out = self.model(x_model, t_in, style_in, use_avg_style=use_avg)
            if x_model.shape[-1] != x_in.shape[-1] or x_model.shape[-2] != x_in.shape[-2]:
                v_out = F.interpolate(v_out, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
            return v_out

        if self._vram_log_this_step and self.vram_debug:
            self.model._vram_debug = True
            self.model._vram_debug_fn = lambda tag: _log_vram(f"model/{tag}", reset_peak=self.vram_debug_reset_peak)
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            v_pred = _model_forward(x_t, t, style_id_tgt, use_avg=drop_label)
        if self._vram_log_this_step and self.vram_debug:
            self.model._vram_debug = False
        if self._vram_log_this_step and self.vram_debug:
            _log_vram("compute_energy_loss/after_v_pred", reset_peak=self.vram_debug_reset_peak)
            if self.vram_debug_detail:
                _log_vram_detail("compute_energy_loss/after_v_pred")

        # [Dynamic Anchoring] Update weights based on epoch
        cur_w_style, cur_w_mse = self.update_loss_weights(epoch, self.num_epochs)

        # Symmetric style supervision by default; can be tuned via style_swd_target_weights
        w_style_batch = self.style_swd_target_weights[style_id_tgt]
        w_mse_batch = torch.ones((b,), device=device)

        do_style_swd = self.use_style_swd and (
            (self.style_swd_interval <= 1) or (self.iter_step % self.style_swd_interval == 0)
        )
        do_style_cls = (self.style_cls_interval <= 1) or (self.iter_step % self.style_cls_interval == 0)

        # Switch to FP32 for numerically sensitive ODE and loss computations.
        with torch.amp.autocast('cuda', enabled=False):
            v_pred_f32 = v_pred.float()
            x_t_f32 = x_t.float()
            t_f32 = t.view(-1, 1, 1, 1).float()
            if self._vram_log_this_step and self.vram_debug:
                _log_vram("compute_energy_loss/after_cast_f32", reset_peak=self.vram_debug_reset_peak)
                if self.vram_debug_detail:
                    _log_vram_detail("compute_energy_loss/after_cast_f32")
            
            # --- ODE integration in FP32 ---
            if self.ode_steps == 1:
                # Use already-computed v_pred to avoid a second forward pass (saves VRAM)
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/ode1/pre", reset_peak=self.vram_debug_reset_peak)
                dt = (1.0 - t_f32)
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/ode1/after_dt", reset_peak=self.vram_debug_reset_peak)
                x_1_pred = x_t_f32 + v_pred_f32 * dt
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/ode1/after_update", reset_peak=self.vram_debug_reset_peak)
            else:
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/ode/pre", reset_peak=self.vram_debug_reset_peak)

                x = x_t_f32.clone()
                t_cur = t_f32.view(-1).clone()
                num_steps = max(self.ode_steps, 1)
                use_ckpt = self.config['training'].get('use_gradient_checkpointing', True)

                def step_fn(x_in, t_in, style_in):
                    with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                        return _model_forward(x_in, t_in, style_in, use_avg=False)

                for i in range(num_steps):
                    t_remaining = 1.0 - t_cur
                    dt = t_remaining / num_steps
                    if use_ckpt and self.model.training:
                        v = torch.utils.checkpoint.checkpoint(step_fn, x, t_cur, style_id_tgt, use_reentrant=False)
                    else:
                        v = step_fn(x, t_cur, style_id_tgt)
                    x = x + v * dt.view(-1, 1, 1, 1)
                    t_cur = t_cur + dt
                    if self._vram_log_this_step and self.vram_debug:
                        _log_vram(f"compute_energy_loss/ode/step{i+1}", reset_peak=self.vram_debug_reset_peak)

                x_1_pred = x
            if self._vram_log_this_step and self.vram_debug:
                _log_vram("compute_energy_loss/after_ode", reset_peak=self.vram_debug_reset_peak)
                if self.vram_debug_detail:
                    _log_vram_detail("compute_energy_loss/after_ode")

            # --- Target velocity (Flow Matching) ---
            v_target_f32 = (x_target_noisy.float() - x0.float())
            if self._vram_log_this_step or self.iter_step <= 5:
                logger.info(
                    f"v_pred_norm={v_pred_f32.square().mean().sqrt().item():.3f} "
                    f"v_tgt_norm={v_target_f32.square().mean().sqrt().item():.3f}"
                )

            # --- Losses in FP32 ---
            # 1) Structural loss (MSE pyramid)
            loss_struc, struc_metrics = self.pyramid_loss(
                v_pred_f32, 
                v_target_f32, 
                sample_weights=w_mse_batch
            )

            # 2) Style loss (SWD) with configurable band/feature routing.
            lowpass_size = int(self.config['loss'].get('moment_lowpass_size', 8))
            style_fp32 = style_latents.float()
            x_lp = self._lowpass_tensor(x_1_pred, lowpass_size)
            style_lp = self._lowpass_tensor(style_fp32, lowpass_size)
            x_hp = x_1_pred - x_lp
            style_hp = style_fp32 - style_lp

            loss_style = torch.tensor(0.0, device=device)
            loss_dict = {'style_swd': loss_style}
            swd_warmup_scale = self._get_swd_warmup_scale()

            if do_style_swd and swd_warmup_scale > 0.0:
                style_terms = []
                feature_levels = [lvl for lvl in self.swd_feature_levels if lvl != 'latent']
                use_latent = 'latent' in self.swd_feature_levels

                if use_latent:
                    latent_term = torch.tensor(0.0, device=device)

                    if self.swd_band_mode in {'highpass', 'both'} and self.swd_high_band_weight > 0.0:
                        x_hp_swd = x_hp
                        style_hp_swd = style_hp
                        if self.swd_input_size and (
                            x_hp.shape[-1] != self.swd_input_size or x_hp.shape[-2] != self.swd_input_size
                        ):
                            x_hp_swd = F.interpolate(x_hp, size=(self.swd_input_size, self.swd_input_size), mode="area")
                            style_hp_swd = F.interpolate(
                                style_hp, size=(self.swd_input_size, self.swd_input_size), mode="area"
                            )

                        hp_dict = self.energy_loss(
                            x_hp_swd,
                            style_hp_swd,
                            style_ids=style_id_tgt,
                            sample_weights=w_style_batch,
                            moments_pred=(x_lp.mean(dim=(2, 3)), x_lp.std(dim=(2, 3))),
                        )
                        latent_term = latent_term + self.swd_high_band_weight * hp_dict['style_swd']
                        loss_dict['latent_highpass_swd'] = hp_dict['style_swd'].detach()
                        if 'moments' in hp_dict:
                            loss_dict['moments'] = hp_dict['moments'].detach()

                    if self.swd_band_mode in {'lowpass', 'both'} and self.swd_low_band_weight > 0.0:
                        x_lp_swd = x_lp
                        style_lp_swd = style_lp
                        if self.swd_input_size and (
                            x_lp.shape[-1] != self.swd_input_size or x_lp.shape[-2] != self.swd_input_size
                        ):
                            x_lp_swd = F.interpolate(x_lp, size=(self.swd_input_size, self.swd_input_size), mode="area")
                            style_lp_swd = F.interpolate(
                                style_lp, size=(self.swd_input_size, self.swd_input_size), mode="area"
                            )
                        lp_term = self._compute_direct_swd(x_lp_swd, style_lp_swd)
                        if self.swd_band_mode == 'lowpass' and self.energy_loss.moment_weight > 0.0:
                            lp_moments = (
                                F.mse_loss(x_lp.mean(dim=(2, 3)), style_lp.mean(dim=(2, 3)))
                                + F.mse_loss(x_lp.std(dim=(2, 3)), style_lp.std(dim=(2, 3)))
                            )
                            lp_term = lp_term + self.energy_loss.moment_weight * lp_moments
                            loss_dict['moments'] = lp_moments.detach()
                        latent_term = latent_term + self.swd_low_band_weight * lp_term
                        loss_dict['latent_lowpass_swd'] = lp_term.detach()

                    style_terms.append(latent_term)

                if feature_levels:
                    t_feat = torch.full((b,), self.swd_feature_t, device=device, dtype=x_1_pred.dtype)
                    pred_feat_input = x_1_pred
                    style_feat_input = style_fp32
                    if target_model_size and (
                        pred_feat_input.shape[-1] != target_model_size
                        or pred_feat_input.shape[-2] != target_model_size
                    ):
                        pred_feat_input = F.interpolate(
                            pred_feat_input, size=(target_model_size, target_model_size), mode="area"
                        )
                        style_feat_input = F.interpolate(
                            style_feat_input, size=(target_model_size, target_model_size), mode="area"
                        )
                    _, pred_feats = self.model(
                        pred_feat_input,
                        t_feat,
                        style_id_tgt,
                        use_avg_style=False,
                        return_features=True,
                        feature_levels=tuple(feature_levels),
                    )
                    with torch.no_grad():
                        _, style_feats = self.model(
                            style_feat_input,
                            t_feat,
                            style_id_tgt,
                            use_avg_style=False,
                            return_features=True,
                            feature_levels=tuple(feature_levels),
                        )

                    for level in feature_levels:
                        if level not in pred_feats or level not in style_feats:
                            continue
                        pred_level = self._reduce_feature_channels(
                            pred_feats[level], self.swd_feature_reduce_channels
                        )
                        style_level = self._reduce_feature_channels(
                            style_feats[level], self.swd_feature_reduce_channels
                        )
                        if self.swd_input_size and (
                            pred_level.shape[-1] != self.swd_input_size or pred_level.shape[-2] != self.swd_input_size
                        ):
                            pred_level = F.interpolate(
                                pred_level, size=(self.swd_input_size, self.swd_input_size), mode="area"
                            )
                            style_level = F.interpolate(
                                style_level, size=(self.swd_input_size, self.swd_input_size), mode="area"
                            )

                        pred_level_lp = self._lowpass_tensor(pred_level, lowpass_size)
                        style_level_lp = self._lowpass_tensor(style_level, lowpass_size)
                        pred_level_hp = pred_level - pred_level_lp
                        style_level_hp = style_level - style_level_lp

                        level_term = torch.tensor(0.0, device=device)
                        if self.swd_band_mode in {'highpass', 'both'} and self.swd_high_band_weight > 0.0:
                            level_term = level_term + self.swd_high_band_weight * self._compute_direct_swd(
                                pred_level_hp, style_level_hp
                            )
                        if self.swd_band_mode in {'lowpass', 'both'} and self.swd_low_band_weight > 0.0:
                            level_term = level_term + self.swd_low_band_weight * self._compute_direct_swd(
                                pred_level_lp, style_level_lp
                            )
                        style_terms.append(level_term)
                        loss_dict[f'feature_{level}_swd'] = level_term.detach()

                if style_terms:
                    loss_style = torch.stack(style_terms).mean() * swd_warmup_scale
                    loss_dict['style_swd'] = loss_style
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/after_swd", reset_peak=self.vram_debug_reset_peak)
                    if self.vram_debug_detail:
                        _log_vram_detail("compute_energy_loss/after_swd")
            loss_dict['swd_warmup_scale'] = torch.tensor(swd_warmup_scale, device=device)

            # Identity consistency to prevent domain collapse.
            loss_identity = torch.tensor(0.0, device=device)
            if self.use_identity_consistency and identity_mask.any():
                loss_identity = F.mse_loss(x_1_pred[identity_mask], latent.float()[identity_mask])
                loss_dict['identity'] = loss_identity.detach()

            # Style classifier guidance with explicit transfer/identity split.
            loss_cls = None
            if self.style_classifier is not None and do_style_cls:
                cls_inputs = x_1_pred
                cls_input_size = int(self.config['loss'].get('style_classifier_input_size_infer', 0))
                if cls_input_size and (
                    cls_inputs.shape[-1] != cls_input_size or cls_inputs.shape[-2] != cls_input_size
                ):
                    cls_inputs = F.interpolate(cls_inputs, size=(cls_input_size, cls_input_size), mode='area')

                logits = self.style_classifier(cls_inputs)
                logits = logits / max(self.style_cls_temperature, 1e-4)
                probs = F.softmax(logits, dim=1)
                conf = probs.max(dim=1).values

                # Agreement score from a cheap view-augmentation pass.
                with torch.no_grad():
                    cls_aug = torch.roll(cls_inputs.detach(), shifts=(1, -1), dims=(2, 3))
                    logits_aug = self.style_classifier(cls_aug) / max(self.style_cls_temperature, 1e-4)
                    probs_aug = F.softmax(logits_aug, dim=1)
                    agree_score = (probs * probs_aug).sum(dim=1)

                if self.style_cls_use_gating:
                    conf_w = torch.clamp(
                        (conf - self.style_cls_conf_threshold) / max(1.0 - self.style_cls_conf_threshold, 1e-6),
                        min=0.0,
                        max=1.0,
                    )
                    agree_w = torch.clamp(
                        (agree_score - self.style_cls_agree_threshold) / max(1.0 - self.style_cls_agree_threshold, 1e-6),
                        min=0.0,
                        max=1.0,
                    )
                    sample_w = (conf_w * agree_w).detach()
                else:
                    sample_w = torch.ones_like(conf)

                warmup_scale = min(float(self.global_step) / max(self.style_cls_warmup_steps, 1), 1.0)
                cls_terms = []

                if transfer_mask.any() and self.style_cls_transfer_weight > 0.0:
                    idx = transfer_mask
                    if self.style_cls_target_only:
                        p_target = probs[idx].gather(1, style_id_tgt[idx].unsqueeze(1)).squeeze(1)
                        per_sample = -torch.log(p_target.clamp_min(1e-8))
                    else:
                        per_sample = F.cross_entropy(logits[idx], style_id_tgt[idx], reduction='none')
                    w = sample_w[idx]
                    denom = w.sum().clamp_min(1e-6)
                    loss_cls_transfer = (per_sample * w).sum() / denom
                    cls_terms.append(warmup_scale * self.style_cls_transfer_weight * loss_cls_transfer)
                    loss_dict['style_cls_transfer'] = loss_cls_transfer.detach()
                    pred_transfer = logits[transfer_mask].argmax(dim=1)
                    loss_dict['style_cls_transfer_acc'] = (pred_transfer == style_id_tgt[transfer_mask]).float().mean().detach()
                    loss_dict['style_cls_transfer_pass_rate'] = (w > 0).float().mean().detach()

                if identity_mask.any() and self.style_cls_identity_weight > 0.0:
                    idx = identity_mask
                    if self.style_cls_target_only:
                        p_target = probs[idx].gather(1, style_id[idx].unsqueeze(1)).squeeze(1)
                        per_sample = -torch.log(p_target.clamp_min(1e-8))
                    else:
                        per_sample = F.cross_entropy(logits[idx], style_id[idx], reduction='none')
                    w = sample_w[idx]
                    denom = w.sum().clamp_min(1e-6)
                    loss_cls_identity = (per_sample * w).sum() / denom
                    cls_terms.append(warmup_scale * self.style_cls_identity_weight * loss_cls_identity)
                    loss_dict['style_cls_identity'] = loss_cls_identity.detach()
                    pred_identity = logits[identity_mask].argmax(dim=1)
                    loss_dict['style_cls_identity_acc'] = (pred_identity == style_id[identity_mask]).float().mean().detach()
                    loss_dict['style_cls_identity_pass_rate'] = (w > 0).float().mean().detach()

                if cls_terms:
                    loss_cls = torch.stack(cls_terms).sum()
                    loss_dict['style_cls'] = loss_cls.detach()
                loss_dict['style_cls_conf_mean'] = conf.mean().detach()
                loss_dict['style_cls_agree_mean'] = agree_score.mean().detach()
                loss_dict['style_cls_weight_mean'] = sample_w.mean().detach()
                loss_dict['style_cls_warmup_scale'] = torch.tensor(warmup_scale, device=device)
                if self._vram_log_this_step and self.vram_debug:
                    _log_vram("compute_energy_loss/after_style_cls", reset_peak=self.vram_debug_reset_peak)
                    if self.vram_debug_detail:
                        _log_vram_detail("compute_energy_loss/after_style_cls")

        style_weight_batch = self.style_weights[style_id_tgt].mean()

        # Adaptive Normalization for Style & Structure
        if self.use_adaptive_norm:
            norm_factor_mse = self.norm_mse.update(loss_struc)
            norm_factor_style = self.norm_style.update(loss_style)
        else:
            norm_factor_mse = 1.0
            norm_factor_style = 1.0
        
        #  Clean Loss Assembly: Pyramid (structure) + SWD (texture)
        # Note: sample_weights are masks only; all scalars are applied here
        term_struc = (cur_w_mse * m_mse * loss_struc / norm_factor_mse)
        term_style = (style_weight_batch * cur_w_style * m_style * loss_style / norm_factor_style)
        total = term_struc + term_style
        if 'identity' in loss_dict:
            total = total + self.identity_weight * loss_identity
        if 'style_cls' in loss_dict and loss_cls is not None:
            total = total + loss_cls
        
        if self.use_velocity_reg and self.vel_reg_loss is not None:
            loss_reg = self.vel_reg_loss(v_pred)
            loss_dict['velocity_reg'] = loss_reg
            total = total + loss_reg

        loss_dict['mse'] = loss_struc # Log pyramid loss as 'mse' for compatibility
        loss_dict.update(struc_metrics)
        loss_dict['total'] = total
        loss_dict['style_id_tgt'] = style_id_tgt

        return loss_dict

    def step_scheduler(self):
        """Step the epoch-based scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()

        if self.style_indices_cache is None:
            self.build_style_indices_cache(dataloader.dataset)
            if self.vram_debug:
                _log_vram(f"epoch {epoch} after style_indices_cache", reset_peak=self.vram_debug_reset_peak)

        if self.vram_debug:
            _log_vram(f"epoch {epoch} start", reset_peak=self.vram_debug_reset_peak)

        # Pre-calculate multipliers for the epoch (constant per epoch)
        multipliers = self._get_thermodynamic_schedule(epoch)
        m_mse, _, m_style = multipliers

        total_loss = 0.0
        total_style_swd = 0.0
        total_style_swd_weighted = 0.0
        total_mse = 0.0
        total_mse_weighted = 0.0
        total_layout = 0.0
        total_mse_raw = 0.0
        total_vel_reg = 0.0
        total_identity = 0.0
        total_style_cls = 0.0
        total_style_cls_transfer_acc = 0.0
        total_style_cls_identity_acc = 0.0
        total_style_cls_transfer_pass = 0.0
        total_style_cls_identity_pass = 0.0
        total_style_cls_conf = 0.0
        total_style_cls_agree = 0.0
        total_style_cls_weight = 0.0
        num_style_cls_transfer = 0
        num_style_cls_identity = 0
        num_style_cls_stats = 0
        num_batches = 0
        accum_counter = 0

        import sys as _sys

        use_tqdm = _sys.stderr.isatty()
        from tqdm import tqdm

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", disable=not use_tqdm, leave=use_tqdm)
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(pbar, start=1):
            torch.compiler.cudagraph_mark_step_begin()
            self.iter_step += 1

            self._vram_log_this_step = self.vram_debug and (
                step_idx > self.vram_debug_skip_steps and
                (step_idx % max(self.vram_debug_interval, 1) == 0)
            )
            if self._vram_log_this_step:
                _log_vram(f"epoch {epoch} step {step_idx} start", reset_peak=self.vram_debug_reset_peak)
                if self.vram_debug_detail:
                    _log_vram_detail(f"epoch {epoch} step {step_idx} start")
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                ld = self.compute_energy_loss(batch, epoch, multipliers=multipliers)

            if self._vram_log_this_step:
                _log_vram(f"epoch {epoch} step {step_idx} after loss", reset_peak=self.vram_debug_reset_peak)
                if self.vram_debug_detail:
                    _log_vram_detail(f"epoch {epoch} step {step_idx} after loss")
                
            loss = ld['total'] / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if self._vram_log_this_step:
                _log_vram(f"epoch {epoch} step {step_idx} after backward", reset_peak=self.vram_debug_reset_peak)
                if self.vram_debug_detail:
                    _log_vram_detail(f"epoch {epoch} step {step_idx} after backward")

            accum_counter += 1

            if accum_counter >= self.accumulation_steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                self._update_lora_alpha(self.global_step)
                self.global_step += 1
                if self._vram_log_this_step:
                    _log_vram(f"epoch {epoch} step {step_idx} after opt", reset_peak=self.vram_debug_reset_peak)
                    if self.vram_debug_detail:
                        _log_vram_detail(f"epoch {epoch} step {step_idx} after opt")

            total_loss += ld['total'].item()
            total_style_swd += ld['style_swd'].item()
            total_mse += ld['mse'].item()
            if 'velocity_reg' in ld:
                total_vel_reg += ld['velocity_reg'].item()
            if 'identity' in ld:
                total_identity += ld['identity'].item()
            if 'style_cls' in ld:
                total_style_cls += ld['style_cls'].item()
            if 'style_cls_transfer_acc' in ld:
                total_style_cls_transfer_acc += ld['style_cls_transfer_acc'].item()
                num_style_cls_transfer += 1
            if 'style_cls_transfer_pass_rate' in ld:
                total_style_cls_transfer_pass += ld['style_cls_transfer_pass_rate'].item()
            if 'style_cls_identity_acc' in ld:
                total_style_cls_identity_acc += ld['style_cls_identity_acc'].item()
                num_style_cls_identity += 1
            if 'style_cls_identity_pass_rate' in ld:
                total_style_cls_identity_pass += ld['style_cls_identity_pass_rate'].item()
            if 'style_cls_conf_mean' in ld:
                total_style_cls_conf += ld['style_cls_conf_mean'].item()
                total_style_cls_agree += ld.get('style_cls_agree_mean', torch.tensor(0.0)).item()
                total_style_cls_weight += ld.get('style_cls_weight_mean', torch.tensor(0.0)).item()
                num_style_cls_stats += 1
            num_batches += 1

            if use_tqdm:
                postfix_dict = {
                    'loss': f"{ld['total'].item():.4f}",
                    '8x8': f"{ld.get('l_8x8', 0):.4f}",
                    '32x32': f"{ld.get('l_32x32', 0):.4f}",
                    'w_str': f"{m_mse:.2f}",
                    'w_sty': f"{m_style:.2f}",
                    'alpha': f"{self._get_current_alpha():.3f}",
                }
                pbar.set_postfix(postfix_dict)

        avg_loss = total_loss / max(num_batches, 1)
        avg_style_swd = total_style_swd / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_vel_reg = total_vel_reg / max(num_batches, 1) if self.use_velocity_reg else 0.0
        avg_identity = total_identity / max(num_batches, 1)
        avg_style_cls = total_style_cls / max(num_batches, 1)
        avg_style_cls_transfer_acc = total_style_cls_transfer_acc / max(num_style_cls_transfer, 1)
        avg_style_cls_identity_acc = total_style_cls_identity_acc / max(num_style_cls_identity, 1)
        avg_style_cls_transfer_pass = total_style_cls_transfer_pass / max(num_style_cls_transfer, 1)
        avg_style_cls_identity_pass = total_style_cls_identity_pass / max(num_style_cls_identity, 1)
        avg_style_cls_conf = total_style_cls_conf / max(num_style_cls_stats, 1)
        avg_style_cls_agree = total_style_cls_agree / max(num_style_cls_stats, 1)
        avg_style_cls_weight = total_style_cls_weight / max(num_style_cls_stats, 1)

        metrics = {
            'loss': avg_loss,
            'style_swd': avg_style_swd,
            'mse': avg_mse,
            'identity': avg_identity,
            'style_cls': avg_style_cls,
            'style_cls_transfer_acc': avg_style_cls_transfer_acc,
            'style_cls_identity_acc': avg_style_cls_identity_acc,
            'style_cls_transfer_pass_rate': avg_style_cls_transfer_pass,
            'style_cls_identity_pass_rate': avg_style_cls_identity_pass,
            'style_cls_conf_mean': avg_style_cls_conf,
            'style_cls_agree_mean': avg_style_cls_agree,
            'style_cls_weight_mean': avg_style_cls_weight,
            'num_batches': num_batches,
            'm_mse': m_mse,
            'm_style': m_style,
        }
        if self.use_velocity_reg:
            metrics['velocity_reg'] = avg_vel_reg

        return metrics

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------
    def get_test_images_by_style(self) -> Dict[int, tuple]:
        test_dir = Path(self.test_image_dir)
        if not test_dir.exists():
            logger.warning(f"Test image directory not found: {test_dir}")
            return {}

        style_images = {}
        style_subdirs = self.config['data'].get('style_subdirs', [])

        for style_id, style_name in enumerate(style_subdirs):
            style_dir = test_dir / style_name
            if not style_dir.exists():
                logger.warning(f"Style directory not found: {style_dir}")
                continue

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            images = []
            for ext in image_extensions:
                images.extend(style_dir.glob(f"*{ext}"))
                images.extend(style_dir.glob(f"*{ext.upper()}"))

            if images:
                test_image = sorted(images)[0]
                style_images[style_id] = (style_name, test_image)
                logger.info(f"  Test image for {style_name}: {test_image.name}")

        return style_images

    @torch.no_grad()
    def evaluate_and_infer(self, epoch: int) -> None:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Inference Evaluation (Epoch {epoch})")
        logger.info(f"{'='*80}")

        if self.vae is None:
            logger.warning("VAE is unavailable. Skipping inference evaluation.")
            return

        epoch_inference_dir = self.inference_dir / ('epoch_-1' if epoch == -1 else f"epoch_{epoch:04d}")
        epoch_inference_dir.mkdir(parents=True, exist_ok=True)

        test_images = self.get_test_images_by_style()
        if not test_images:
            logger.warning("No test images found. Skipping inference.")
            return

        self.model.eval()
        num_styles = self.config['model']['num_styles']

        temp_ckpt = self.checkpoint_dir / "temp_eval.pt"
        torch.save(self.model.state_dict(), temp_ckpt)

        try:
            for src_style_id, (src_style_name, src_image_path) in test_images.items():
                logger.info(f"\nProcessing source: {src_style_name}")
                try:
                    image = Image.open(src_image_path).convert('RGB')
                    image = image.resize((256, 256))
                    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                    image_tensor = image_tensor * 2.0 - 1.0
                    image_tensor = image_tensor.to(self.device)
                except Exception as exc:  # pragma: no cover - IO heavy
                    logger.error(f"Failed to load {src_image_path}: {exc}")
                    continue

                latent_src = encode_image(self.vae, image_tensor, self.device)

                if epoch == -1:
                    try:
                        image_orig = decode_latent(self.vae, latent_src, self.device)
                        output_filename = f"{src_style_name}_original.jpg"
                        output_path = epoch_inference_dir / output_filename
                        from torchvision.utils import save_image

                        save_image(image_orig, output_path)
                        logger.info(f"    Saved original: {output_filename}")
                    except Exception as exc:  # pragma: no cover - IO heavy
                        logger.error(f"    Failed to save original: {exc}")
                    continue

                # Batched inference for all target styles from one inversion.
                try:
                    eval_steps = self.config.get('inference', {}).get('num_steps', 20)
                    
                    # 1) Invert once from source latent.
                    latent_x0 = invert_latent(self.model, latent_src, src_style_id, num_steps=eval_steps)
                    
                    # 2) Build batched target style IDs.
                    tgt_style_ids = torch.arange(num_styles, device=self.device)
                    # Expand source noise to batch size = number of styles.
                    latent_x0_batch = latent_x0.repeat(num_styles, 1, 1, 1)
                    
                    # Single batched pass through the style velocity model
                    latent_tgt_batch = generate_latent(self.model, latent_x0_batch, tgt_style_ids, num_steps=eval_steps)
                    
                    # 3) Decode in batch with VAE.
                    images_out = decode_latent(self.vae, latent_tgt_batch, self.device)
                    
                    # 4) Save per-style outputs.
                    from torchvision.utils import save_image
                    for tgt_id in range(num_styles):
                        tgt_style_name = self.config['data'].get('style_subdirs', [])[tgt_id]
                        output_filename = f"{src_style_name}_to_{tgt_style_name}.jpg"
                        output_path = epoch_inference_dir / output_filename
                        save_image(images_out[tgt_id], output_path)
                        logger.info(f"    Saved: {output_filename}")
                except Exception as exc:
                    logger.error(f"    Batch inference failed for {src_style_name}: {exc}")
                    continue
        finally:
            if temp_ckpt.exists():
                temp_ckpt.unlink()

        logger.info(f"\n{'='*80}")
        logger.info(f"Inference completed. Results saved to: {epoch_inference_dir}")
        logger.info(f"{'='*80}\n")

    def run_full_evaluation(self, epoch: int, timeout: int = 3600) -> None:
        logger.info(f"Starting full external evaluation for epoch {epoch}")

        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        temp_ckpt = None
        if not ckpt_path.exists():
            temp_ckpt = self.checkpoint_dir / f"epoch_{epoch:04d}_eval_temp.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'metrics': {},
            }
            try:
                torch.save(checkpoint, temp_ckpt)
                ckpt_to_use = temp_ckpt
                logger.info(f"Saved temporary checkpoint for evaluation: {temp_ckpt}")
            except Exception as exc:  # pragma: no cover - IO heavy
                logger.error(f"Failed to write temporary checkpoint for evaluation: {exc}")
                return
        else:
            ckpt_to_use = ckpt_path

        eval_out_dir = self.checkpoint_dir / 'full_eval' / f'epoch_{epoch:04d}'
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        run_script = Path(__file__).resolve().parent / 'utils' / 'run_evaluation.py'
        cmd = [sys.executable, str(run_script), '--checkpoint', str(ckpt_to_use), '--output', str(eval_out_dir)]
        
        num_steps = self.config.get('inference', {}).get('num_steps', None)
        if num_steps is not None:
            cmd += ['--num_steps', str(num_steps)]

        logger.info(f"Running external evaluation command: {' '.join(cmd)}")

        proc = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                logger.error(f"External evaluation exited with code {proc.returncode}")
                logger.error(proc.stdout)
                logger.error(proc.stderr)
            else:
                logger.info(f"External evaluation completed for epoch {epoch}; output at {eval_out_dir}")
        except subprocess.TimeoutExpired:
            logger.error(f"External evaluation timed out after {timeout} seconds for epoch {epoch}")
        except Exception as exc:  # pragma: no cover - external call
            logger.error(f"Failed to run external evaluation for epoch {epoch}: {exc}")

        summary_src = eval_out_dir / 'summary.json'
        if summary_src.exists():
            dst = self.log_dir / f'eval_epoch_{epoch:04d}.json'
            try:
                shutil.copy(summary_src, dst)
                logger.info(f"Saved evaluation summary to {dst}")
            except Exception as exc:  # pragma: no cover - IO heavy
                logger.error(f"Failed to copy evaluation summary: {exc}")

        out_log = self.log_dir / f'eval_epoch_{epoch:04d}.log'
        try:
            with open(out_log, 'w', encoding='utf-8') as f:
                if proc is not None:
                    f.write('STDOUT\n')
                    f.write(proc.stdout or '')
                    f.write('\n\nSTDERR\n')
                    f.write(proc.stderr or '')
            logger.info(f"Saved external eval logs to {out_log}")
        except Exception as exc:  # pragma: no cover - IO heavy
            logger.error(f"Failed to write external eval logs: {exc}")

        if temp_ckpt is not None and temp_ckpt.exists():
            try:
                temp_ckpt.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Checkpointing/logging
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        adaptive_norm_state = {}
        if self.use_adaptive_norm:
            adaptive_norm_state['mse'] = self.norm_mse.state_dict()
            adaptive_norm_state['style'] = self.norm_style.state_dict()

        save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            metrics=metrics,
            global_step=self.global_step,
            adaptive_norm_state=adaptive_norm_state if adaptive_norm_state else None
        )

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        current_lr = self.optimizer.param_groups[0]['lr']
        with open(self.log_file, 'a') as f:
            f.write(
                f"{epoch},{metrics['loss']:.6f},{metrics['style_swd']:.6f},{metrics['mse']:.6f},"
                f"{metrics.get('identity', 0.0):.6f},{metrics.get('style_cls', 0.0):.6f},"
                f"{metrics.get('style_cls_transfer_acc', 0.0):.6f},"
                f"{metrics.get('style_cls_identity_acc', 0.0):.6f},"
                f"{metrics.get('style_cls_transfer_pass_rate', 0.0):.6f},"
                f"{metrics.get('style_cls_identity_pass_rate', 0.0):.6f},"
                f"{metrics.get('style_cls_conf_mean', 0.0):.6f},"
                f"{metrics.get('style_cls_agree_mean', 0.0):.6f},"
                f"{metrics.get('style_cls_weight_mean', 0.0):.6f},"
                f"{metrics.get('velocity_reg', 0.0):.6f},{current_lr:.2e},"
                f"{metrics.get('epoch_time', 0.0):.2f}\n"
            )

        epoch_log_path = self.log_dir / 'epoch_logs.jsonl'
        epoch_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'loss': float(metrics['loss']),
            'style_swd': float(metrics['style_swd']),
            'mse': float(metrics['mse']),
            'identity': float(metrics.get('identity', 0.0)),
            'style_cls': float(metrics.get('style_cls', 0.0)),
            'style_cls_transfer_acc': float(metrics.get('style_cls_transfer_acc', 0.0)),
            'style_cls_identity_acc': float(metrics.get('style_cls_identity_acc', 0.0)),
            'style_cls_transfer_pass_rate': float(metrics.get('style_cls_transfer_pass_rate', 0.0)),
            'style_cls_identity_pass_rate': float(metrics.get('style_cls_identity_pass_rate', 0.0)),
            'style_cls_conf_mean': float(metrics.get('style_cls_conf_mean', 0.0)),
            'style_cls_agree_mean': float(metrics.get('style_cls_agree_mean', 0.0)),
            'style_cls_weight_mean': float(metrics.get('style_cls_weight_mean', 0.0)),
            'velocity_reg': float(metrics.get('velocity_reg', 0.0)),
            'm_mse': float(metrics.get('m_mse', 0.0)),
            'm_style': float(metrics.get('m_style', 0.0)),
            'learning_rate': current_lr,
            'epoch_time': metrics.get('epoch_time', 0.0),
            'num_batches': metrics.get('num_batches'),
        }
        with open(epoch_log_path, 'a', encoding='utf-8') as ef:
            ef.write(json.dumps(epoch_entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Convenience helpers for main loop
    # ------------------------------------------------------------------
    def _setup_scheduler(self) -> None:
        """Initialize the epoch-based scheduler."""
        # [Infra Optimized] Structure-Aware Cosine Scheduler (Epoch-based)
        base_lr = float(self.config['training']['learning_rate'])
        min_lr = float(self.config['training'].get('min_learning_rate', 1e-6))
        warmup_epochs = self.config['training'].get('warmup_epochs', 10)

        # 1. Warmup Phase
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )

        # 2. Main Phase (Step or Cosine)
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_type == 'step' or scheduler_type == 'multistep':
            milestones = self.config['training'].get('scheduler_milestones', [50, 100])
            gamma = self.config['training'].get('gamma', 0.1)
            logger.info(f"Scheduler: Linear Warmup ({warmup_epochs}) -> MultiStepLR (milestones={milestones}, gamma={gamma})")
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[m - warmup_epochs for m in milestones if m > warmup_epochs],
                gamma=gamma
            )
        else:
            logger.info(f"Scheduler: Linear Warmup ({warmup_epochs}) -> Cosine Annealing (to {min_lr})")
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - warmup_epochs,
                eta_min=min_lr
            )

        # 3. Combine
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )

    def on_training_start(self, dataloader: DataLoader) -> None:
        if self.style_indices_cache is None:
            self.build_style_indices_cache(dataloader.dataset)

        steps_per_epoch = max((len(dataloader) + self.accumulation_steps - 1) // self.accumulation_steps, 1)
        self.total_train_steps_estimate = int(self.num_epochs * steps_per_epoch)
        self.swd_warmup_steps = int(self.total_train_steps_estimate * self.swd_warmup_ratio)
        logger.info(
            "Estimated train steps=%s, SWD warmup steps=%s (ratio=%.3f)",
            self.total_train_steps_estimate,
            self.swd_warmup_steps,
            self.swd_warmup_ratio,
        )
        
        if self.use_style_swd and 'latent' in self.swd_feature_levels:
            # Initialize Style LUT cache for latent-level SWD.
            logger.info("Initializing Style LUT cache for multi-style SWD...")
            style_prototypes = {}

            # Samples per style used to estimate target style distributions.
            samples_for_distribution = self.fixed_style_reference_size

            # Collect representative latent for each style
            for style_id in range(self.config['model']['num_styles']):
                if hasattr(dataloader.dataset, 'style_indices') and style_id in self.style_indices_cache:
                    indices = self.style_indices_cache[style_id]

                    # Sample without replacement where possible.
                    count = min(len(indices), samples_for_distribution)
                    selected_indices = np.random.choice(indices, count, replace=False)

                    # Pull cached latents directly from dataset tensor.
                    latents_batch = dataloader.dataset.latents_tensor[selected_indices]

                    # Store style prototype batch for LUT initialization.
                    style_prototypes[style_id] = latents_batch
                    if self.use_fixed_style_reference:
                        self.style_reference_pool = self.style_reference_pool or {}
                        self.style_reference_pool[style_id] = latents_batch

                    logger.info(f"  Style {style_id}: Sampled {count} images for distribution statistics.")
                else:
                    # Fallback: random initialization
                    logger.warning(f"  No style {style_id} samples found, using random init")
                    style_prototypes[style_id] = torch.randn(1, 4, 32, 32)
                    if self.use_fixed_style_reference:
                        self.style_reference_pool = self.style_reference_pool or {}
                        self.style_reference_pool[style_id] = style_prototypes[style_id]

            # Initialize LUT cache.
            self.energy_loss.initialize_cache(style_prototypes, self.device)
        else:
            logger.info("Skipping latent SWD LUT initialization (latent SWD disabled for this run).")
        
        save_initial_inference = bool(self.config.get('training', {}).get('save_initial_inference', True))
        if save_initial_inference:
            orig_dir = self.inference_dir / 'epoch_-1'
            if not orig_dir.exists() or not any(orig_dir.iterdir()):
                logger.info("Saving original test images to inference/epoch_-1")
                self.evaluate_and_infer(-1)
            else:
                logger.info("Original test images already saved in inference/epoch_-1; skipping.")
        else:
            logger.info("Skipping initial inference snapshot (training.save_initial_inference=false)")

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()
