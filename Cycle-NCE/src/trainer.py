from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .losses import AdaCUTObjective
    from .model import LatentAdaCUT, count_parameters
    from .utils.style_classifier import StyleClassifier as LatentStyleClassifier
except ImportError:  # pragma: no cover
    from losses import AdaCUTObjective
    from model import LatentAdaCUT, count_parameters
    from utils.style_classifier import StyleClassifier as LatentStyleClassifier

logger = logging.getLogger(__name__)


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _add_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict
    return {f"_orig_mod.{k}": v for k, v in state_dict.items()}


class _CudaPrefetchLoader:
    """
    Asynchronous host->device prefetcher to reduce GPU idle bubbles.
    """

    def __init__(self, dataloader: DataLoader, move_fn, enabled: bool) -> None:
        self.dataloader = dataloader
        self.move_fn = move_fn
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.stream = torch.cuda.Stream() if self.enabled else None

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if not self.enabled:
            for batch in self.dataloader:
                yield self.move_fn(batch)
            return

        loader_iter = iter(self.dataloader)
        next_batch = None

        def _prefetch():
            nonlocal next_batch
            try:
                raw = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = self.move_fn(raw)

        _prefetch()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            _prefetch()
            yield batch


class AdaCUTTrainer:
    def __init__(self, config: Dict, device: torch.device, config_path: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.config_path = config_path

        train_cfg = config.get("training", {})
        torch.set_float32_matmul_precision("high")
        self.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.benchmark = True

        self.channels_last = bool(train_cfg.get("channels_last", True) and device.type == "cuda")
        self.skip_oom_batches = bool(train_cfg.get("skip_oom_batches", True))
        self.empty_cache_interval = max(0, int(train_cfg.get("empty_cache_interval", 0)))
        self.log_vram_interval = max(0, int(train_cfg.get("log_vram_interval", 0)))
        self.cuda_prefetch = bool(train_cfg.get("cuda_prefetch", True))

        model_cfg = config.get("model", {})
        self.model = LatentAdaCUT(
            latent_channels=int(model_cfg.get("latent_channels", 4)),
            num_styles=int(model_cfg.get("num_styles", 3)),
            style_dim=int(model_cfg.get("style_dim", 256)),
            base_dim=int(model_cfg.get("base_dim", 64)),
            lift_channels=int(model_cfg.get("lift_channels", model_cfg.get("base_dim", 64))),
            num_hires_blocks=int(model_cfg.get("num_hires_blocks", 2)),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 4)),
            num_groups=int(model_cfg.get("num_groups", 8)),
            projector_dim=int(model_cfg.get("projector_dim", 256)),
            use_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),
            latent_scale_factor=float(model_cfg.get("latent_scale_factor", 0.18215)),
            residual_gain=float(model_cfg.get("residual_gain", 0.1)),
            style_ref_gain=float(model_cfg.get("style_ref_gain", 1.0)),
            style_spatial_pre_gain_32=float(model_cfg.get("style_spatial_pre_gain_32", 0.25)),
            style_spatial_block_gain_32=float(model_cfg.get("style_spatial_block_gain_32", 0.10)),
            style_spatial_pre_gain_16=float(model_cfg.get("style_spatial_pre_gain_16", 0.35)),
            style_spatial_block_gain_16=float(model_cfg.get("style_spatial_block_gain_16", 0.15)),
            use_decoder_spatial_inject=bool(model_cfg.get("use_decoder_spatial_inject", True)),
            style_spatial_dec_gain_32=float(model_cfg.get("style_spatial_dec_gain_32", 0.18)),
            style_spatial_dec_gain_out=float(model_cfg.get("style_spatial_dec_gain_out", 0.08)),
            use_style_texture_head=bool(model_cfg.get("use_style_texture_head", True)),
            style_texture_gain=float(model_cfg.get("style_texture_gain", 0.12)),
            use_style_delta_gate=bool(model_cfg.get("use_style_delta_gate", True)),
            use_decoder_adagn=bool(model_cfg.get("use_decoder_adagn", True)),
            use_delta_highpass_bias=bool(model_cfg.get("use_delta_highpass_bias", True)),
            style_delta_lowfreq_gain=float(model_cfg.get("style_delta_lowfreq_gain", 0.35)),
            use_style_spatial_highpass=bool(model_cfg.get("use_style_spatial_highpass", False)),
            use_content_skip_fusion=bool(model_cfg.get("use_content_skip_fusion", True)),
            content_skip_gain=float(model_cfg.get("content_skip_gain", 0.6)),
            use_style_skip_gate=bool(model_cfg.get("use_style_skip_gate", True)),
        )
        if self.channels_last:
            self.model = self.model.to(device, memory_format=torch.channels_last)
        else:
            self.model = self.model.to(device)

        self.compile_cache_info = self._configure_compile_cache(train_cfg)
        self.use_compile = bool(train_cfg.get("use_compile", False))
        self.compile_mode = str(train_cfg.get("compile_mode", "reduce-overhead"))
        self.compile_backend = str(train_cfg.get("compile_backend", "inductor")).strip() or "inductor"
        self.compile_dynamic = bool(train_cfg.get("compile_dynamic", False))
        if self.use_compile:
            try:
                try:
                    self.model = torch.compile(
                        self.model,
                        backend=self.compile_backend,
                        mode=self.compile_mode,
                        fullgraph=False,
                        dynamic=self.compile_dynamic,
                    )
                except TypeError:
                    # Older torch versions may not support `dynamic`.
                    self.model = torch.compile(
                        self.model,
                        backend=self.compile_backend,
                        mode=self.compile_mode,
                        fullgraph=False,
                    )
                logger.info(
                    "torch.compile enabled (backend=%s mode=%s dynamic=%s)",
                    self.compile_backend,
                    self.compile_mode,
                    self.compile_dynamic,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("torch.compile failed, fallback to eager. reason=%s", exc)
                self.use_compile = False

        logger.info(
            "Compile cache | enabled=%s dir=%s fx_graph_cache=%s dynamo_cache_limit=%s",
            bool(self.compile_cache_info.get("enabled", False)),
            str(self.compile_cache_info.get("dir", "")),
            str(self.compile_cache_info.get("fx_graph_cache", "")),
            str(self.compile_cache_info.get("dynamo_cache_limit", "")),
        )

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")
        logger.info(
            "Infra | channels_last=%s tf32=%s grad_ckpt=%s compile=%s backend=%s mode=%s prefetch=%s",
            self.channels_last,
            self.allow_tf32,
            bool(train_cfg.get("use_gradient_checkpointing", False)),
            self.use_compile,
            self.compile_backend if self.use_compile else "eager",
            self.compile_mode if self.use_compile else "",
            self.cuda_prefetch and (self.device.type == "cuda"),
        )

        self.use_amp = bool(train_cfg.get("use_amp", True) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "bf16")).lower()
        if amp_dtype_cfg in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        if device.type != "cuda":
            self.use_amp = False
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        use_fused_adamw = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
                fused=use_fused_adamw,
            )
        except TypeError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
            )
            use_fused_adamw = False
        logger.info(
            "Precision | amp=%s amp_dtype=%s grad_scaler=%s fused_adamw=%s",
            self.use_amp,
            "fp16" if self.amp_dtype == torch.float16 else "bf16",
            self.use_grad_scaler,
            use_fused_adamw,
        )

        self.scheduler = None
        scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg.get("num_epochs", 100))),
                eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
            )

        self.style_classifier = self._build_style_classifier()
        self.loss_fn = AdaCUTObjective(config, style_classifier=self.style_classifier)

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(1, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.tqdm_mininterval = float(train_cfg.get("tqdm_mininterval", 0.5))
        self.num_epochs = int(train_cfg.get("num_epochs", 100))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))
        self.full_eval_interval = max(0, int(train_cfg.get("full_eval_interval", 50)))
        self.run_full_eval_on_last_epoch = bool(train_cfg.get("full_eval_on_last_epoch", True))

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "../adacut_ckpt"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.full_eval_root = self.checkpoint_dir / "full_eval"
        self.full_eval_root.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "loss",
                    "distill",
                    "code",
                    "code_ref",
                    "code_proto",
                    "proto",
                    "ref_sep",
                    "ref_dist",
                    "ref_cos",
                    "style_ce",
                    "prob",
                    "prob_margin",
                    "prob_weight_mean",
                    "cls_target_prob",
                    "cls_hard_ratio",
                    "dir",
                    "proto_sep",
                    "style_pred_acc",
                    "xfer_margin",
                    "proto_cos_max",
                    "proto_cos_mean",
                    "same_id",
                    "cycle",
                    "gram",
                    "gram_hf",
                    "moment",
                    "moment_hf",
                    "featmatch",
                    "featmatch_hf",
                    "spatial_proto",
                    "push",
                    "nce",
                    "idt",
                    "w_cycle_eff",
                    "w_nce_eff",
                    "w_idt_eff",
                    "style_ref_alpha",
                    "transfer_ratio",
                    "transfer_weight_mean",
                    "data_wait_sec",
                    "step_wall_sec",
                    "lr",
                    "epoch_time_sec",
                ]
            )

        self.global_step = 0
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))

    def _configure_compile_cache(self, train_cfg: Dict) -> Dict[str, object]:
        info: Dict[str, object] = {
            "enabled": bool(train_cfg.get("compile_cache_enabled", True)),
            "dir": "",
            "fx_graph_cache": "",
            "dynamo_cache_limit": "",
        }
        if not bool(info["enabled"]):
            return info

        cache_dir_raw = str(train_cfg.get("compile_cache_dir", "")).strip()
        cache_dir: Optional[Path] = None
        if cache_dir_raw:
            cache_dir = self._resolve_config_relative_path(cache_dir_raw)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        elif os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).expanduser()
        if cache_dir is not None:
            info["dir"] = str(cache_dir)

        fx_graph_cache = bool(train_cfg.get("compile_fx_graph_cache", True))
        info["fx_graph_cache"] = fx_graph_cache
        try:
            import torch._inductor.config as inductor_config

            if hasattr(inductor_config, "fx_graph_cache"):
                inductor_config.fx_graph_cache = fx_graph_cache
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not set torch._inductor.config.fx_graph_cache: %s", exc)

        dynamo_cache_limit = int(train_cfg.get("compile_dynamo_cache_limit", 256))
        info["dynamo_cache_limit"] = dynamo_cache_limit
        try:
            import torch._dynamo.config as dynamo_config

            if dynamo_cache_limit > 0:
                dynamo_config.cache_size_limit = dynamo_cache_limit
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not set torch._dynamo.config.cache_size_limit: %s", exc)
        return info

    def _find_latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if not ckpts:
            return None
        return ckpts[-1]

    def _maybe_resume(self, resume_checkpoint: str) -> None:
        if resume_checkpoint:
            ckpt_path = Path(resume_checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
        else:
            ckpt_path = self._find_latest_checkpoint()

        if ckpt_path is None or not ckpt_path.exists():
            logger.info("No checkpoint found, start from scratch.")
            return

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_state = state["model_state_dict"]
        model_expects_compile_prefix = any(k.startswith("_orig_mod.") for k in self.model.state_dict().keys())
        ckpt_has_compile_prefix = any(k.startswith("_orig_mod.") for k in model_state.keys())
        if model_expects_compile_prefix and not ckpt_has_compile_prefix:
            model_state = _add_compile_prefix(model_state)
        elif (not model_expects_compile_prefix) and ckpt_has_compile_prefix:
            model_state = _strip_compile_prefix(model_state)
        self.model.load_state_dict(model_state, strict=True)

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.global_step = int(state.get("global_step", 0))
        self.start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch=%d global_step=%d", ckpt_path, self.start_epoch, self.global_step)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if self.channels_last and v.is_floating_point() and v.ndim == 4:
                    out[k] = v.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @staticmethod
    def _is_oom_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg

    def _resolve_config_relative_path(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if p.is_absolute():
            return p
        if self.config_path:
            return (Path(self.config_path).resolve().parent / p).resolve()
        return (Path.cwd() / p).resolve()

    def _build_style_classifier(self) -> Optional[nn.Module]:
        cfg_loss = self.config.get("loss", {})
        w_cls = float(cfg_loss.get("w_cls", cfg_loss.get("w_style_ce", 0.0)))
        w_prob = float(cfg_loss.get("w_prob", 0.0))
        w_prob_margin = float(cfg_loss.get("w_prob_margin", 0.0))
        w_dir = float(cfg_loss.get("w_dir", 0.0))
        need_classifier = any(w > 0.0 for w in (w_cls, w_prob, w_prob_margin, w_dir))
        if not need_classifier:
            return None

        ckpt_raw = str(cfg_loss.get("style_classifier_ckpt", "")).strip()
        if not ckpt_raw:
            raise ValueError(
                "Classifier-guided losses enabled (w_cls/w_prob/w_prob_margin/w_dir) "
                "require loss.style_classifier_ckpt"
            )

        ckpt_path = self._resolve_config_relative_path(ckpt_raw)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Style classifier checkpoint not found: {ckpt_path}")

        model_cfg = self.config.get("model", {})
        in_channels = int(model_cfg.get("latent_channels", 4))
        num_classes = int(model_cfg.get("num_styles", 2))

        classifier = LatentStyleClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            use_stats=bool(cfg_loss.get("style_classifier_use_stats", True)),
            use_gram=bool(cfg_loss.get("style_classifier_use_gram", True)),
            use_lowpass_stats=bool(cfg_loss.get("style_classifier_use_lowpass_stats", True)),
            spatial_shuffle=False,
            input_size_train=int(cfg_loss.get("style_classifier_input_size_train", 8)),
            input_size_infer=int(cfg_loss.get("style_classifier_input_size_infer", 8)),
            lowpass_size=int(cfg_loss.get("style_classifier_lowpass_size", 8)),
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        classifier.load_state_dict(state_dict, strict=True)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

        logger.info(
            "Loaded frozen style classifier: %s | in_channels=%d num_classes=%d",
            ckpt_path,
            in_channels,
            num_classes,
        )
        return classifier

    def _maybe_log_vram(self, epoch: int, step_idx: int) -> None:
        if self.device.type != "cuda" or self.log_vram_interval <= 0:
            return
        if step_idx % self.log_vram_interval != 0:
            return
        alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(
            "VRAM epoch=%d step=%d | alloc=%.1fMB reserved=%.1fMB peak=%.1fMB",
            epoch,
            step_idx,
            alloc_mb,
            reserved_mb,
            max_alloc_mb,
        )

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()
        self.loss_fn.set_progress(epoch=epoch, total_epochs=self.num_epochs)

        device = self.device
        metric_alias = {
            "loss": "loss",
            "distill": "distill",
            "code": "code",
            "code_ref": "code_ref",
            "code_proto": "code_proto",
            "proto": "proto",
            "ref_sep": "ref_sep",
            "ref_dist": "ref_dist",
            "ref_cos": "ref_cos",
            "style_ce": "style_ce",
            "prob": "prob",
            "prob_margin": "prob_margin",
            "prob_weight_mean": "prob_weight_mean",
            "cls_target_prob": "cls_target_prob",
            "cls_hard_ratio": "cls_hard_ratio",
            "dir": "dir",
            "proto_sep": "proto_sep",
            "style_pred_acc": "style_pred_acc",
            "xfer_margin": "xfer_margin",
            "proto_cos_max": "proto_cos_max",
            "proto_cos_mean": "proto_cos_mean",
            "same_id": "same_id",
            "cycle": "cycle",
            "gram": "gram",
            "gram_hf": "gram_hf",
            "moment": "moment",
            "moment_hf": "moment_hf",
            "featmatch": "featmatch",
            "featmatch_hf": "featmatch_hf",
            "spatial_proto": "spatial_proto",
            "push": "push",
            "nce": "nce",
            "idt": "idt",
            "w_cycle_eff": "w_cycle_eff",
            "w_nce_eff": "w_nce_eff",
            "w_idt_eff": "w_idt_eff",
            "style_ref_alpha": "style_ref_alpha",
            "transfer_ratio": "transfer_ratio",
            "transfer_weight_mean": "transfer_weight_mean",
        }
        metric_sums = {
            k: torch.zeros((), device=device, dtype=torch.float32) for k in metric_alias.keys()
        }
        sum_data_wait_sec = 0.0
        sum_step_wall_sec = 0.0
        num_batches = 0

        iter_loader = _CudaPrefetchLoader(
            dataloader=dataloader,
            move_fn=self._move_batch,
            enabled=self.cuda_prefetch and (self.device.type == "cuda"),
        )
        total_steps = len(iter_loader)
        progress = tqdm(
            iter_loader,
            total=total_steps,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=self.tqdm_mininterval,
            disable=not self.use_tqdm,
        )

        self.optimizer.zero_grad(set_to_none=True)
        prev_step_end = epoch_start
        for step_idx, raw_batch in enumerate(progress, start=1):
            step_start = time.time()
            sum_data_wait_sec += max(0.0, step_start - prev_step_end)
            try:
                batch = raw_batch
                content = batch["content"]
                target_style = batch["target_style"]
                target_style_id = batch["target_style_id"]
                content_style_id = batch["content_style_id"]

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_dict = self.loss_fn.compute(
                        self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        content_style_id=content_style_id,
                    )
                    loss = loss_dict["loss"]

                loss_to_backward = loss / self.accumulation_steps
                if self.use_grad_scaler:
                    self.scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                should_step = (step_idx % self.accumulation_steps == 0)
                if should_step:
                    if self.grad_clip_norm > 0:
                        if self.use_grad_scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    if self.use_grad_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.empty_cache_interval > 0 and self.device.type == "cuda":
                        if self.global_step % self.empty_cache_interval == 0:
                            torch.cuda.empty_cache()

                for out_key, loss_key in metric_alias.items():
                    if loss_key == "loss":
                        val = loss.detach()
                    else:
                        val = loss_dict.get(loss_key)
                    if val is None:
                        continue
                    metric_sums[out_key] += val.detach().float()
                num_batches += 1

                if step_idx % self.log_interval == 0:
                    elapsed = time.time() - epoch_start
                    step_per_sec = step_idx / max(elapsed, 1e-6)
                    eta = (total_steps - step_idx) / max(step_per_sec, 1e-6)
                    inv_batches = 1.0 / max(num_batches, 1)
                    avg_data_wait = sum_data_wait_sec * inv_batches
                    avg_step_wall = sum_step_wall_sec * inv_batches
                    avg_loss = float((metric_sums["loss"] * inv_batches).item())
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        cls=f"{float((metric_sums['style_ce'] * inv_batches).item()):.3f}",
                        p=f"{float((metric_sums['prob'] * inv_batches).item()):.3f}",
                        pm=f"{float((metric_sums['prob_margin'] * inv_batches).item()):.3f}",
                        pw=f"{float((metric_sums['prob_weight_mean'] * inv_batches).item()):.2f}",
                        p_t=f"{float((metric_sums['cls_target_prob'] * inv_batches).item()):.2f}",
                        hard=f"{float((metric_sums['cls_hard_ratio'] * inv_batches).item()):.2f}",
                        mrg=f"{float((metric_sums['xfer_margin'] * inv_batches).item()):.3f}",
                        pcos=f"{float((metric_sums['proto_cos_max'] * inv_batches).item()):.3f}",
                        fm=f"{float((metric_sums['featmatch'] * inv_batches).item()):.3f}",
                        fmh=f"{float((metric_sums['featmatch_hf'] * inv_batches).item()):.3f}",
                        sp=f"{float((metric_sums['spatial_proto'] * inv_batches).item()):.3f}",
                        dw=f"{avg_data_wait*1000:.1f}ms",
                        sw=f"{avg_step_wall*1000:.1f}ms",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                    if not self.use_tqdm:
                        logger.info(
                            "epoch %d step %d/%d | loss=%.4f cls=%.4f p=%.4f pm=%.4f pw=%.3f p_t=%.3f hard=%.3f margin=%.4f proto_cos=%.4f feat=%.4f feat_hf=%.4f spatial=%.4f data_wait=%.1fms step_wall=%.1fms | %.2f it/s eta %.1fs",
                            epoch,
                            step_idx,
                            total_steps,
                            avg_loss,
                            float((metric_sums["style_ce"] * inv_batches).item()),
                            float((metric_sums["prob"] * inv_batches).item()),
                            float((metric_sums["prob_margin"] * inv_batches).item()),
                            float((metric_sums["prob_weight_mean"] * inv_batches).item()),
                            float((metric_sums["cls_target_prob"] * inv_batches).item()),
                            float((metric_sums["cls_hard_ratio"] * inv_batches).item()),
                            float((metric_sums["xfer_margin"] * inv_batches).item()),
                            float((metric_sums["proto_cos_max"] * inv_batches).item()),
                            float((metric_sums["featmatch"] * inv_batches).item()),
                            float((metric_sums["featmatch_hf"] * inv_batches).item()),
                            float((metric_sums["spatial_proto"] * inv_batches).item()),
                            avg_data_wait * 1000.0,
                            avg_step_wall * 1000.0,
                            step_per_sec,
                            eta,
                        )
                self._maybe_log_vram(epoch, step_idx)
            except RuntimeError as exc:
                if self.device.type == "cuda" and self._is_oom_error(exc) and self.skip_oom_batches:
                    logger.warning(
                        "CUDA OOM at epoch=%d step=%d, skipping batch (reduce batch_size if frequent).",
                        epoch,
                        step_idx,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise
            finally:
                prev_step_end = time.time()
                sum_step_wall_sec += max(0.0, prev_step_end - step_start)

        progress.close()

        # Flush leftover gradients when last batch is not divisible by accumulation_steps.
        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            if self.grad_clip_norm > 0:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        epoch_time = time.time() - epoch_start
        lr = float(self.optimizer.param_groups[0]["lr"])
        inv_batches = 1.0 / max(num_batches, 1)
        metrics = {k: float((v * inv_batches).item()) for k, v in metric_sums.items()}
        metrics["data_wait_sec"] = sum_data_wait_sec * inv_batches
        metrics["step_wall_sec"] = sum_step_wall_sec * inv_batches
        metrics["lr"] = lr
        metrics["epoch_time_sec"] = epoch_time
        if self.use_tqdm:
            tqdm.write(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"loss={metrics['loss']:.4f} "
                f"cls={metrics['style_ce']:.4f} p={metrics['prob']:.4f} pm={metrics['prob_margin']:.4f} pw={metrics['prob_weight_mean']:.3f} "
                f"p_t={metrics['cls_target_prob']:.3f} hard={metrics['cls_hard_ratio']:.3f} "
                f"margin={metrics['xfer_margin']:.4f} proto_cos={metrics['proto_cos_max']:.4f} "
                f"ref_dist={metrics.get('ref_dist', 0.0):.4f} ref_cos={metrics.get('ref_cos', 0.0):.4f} "
                f"feat={metrics.get('featmatch', 0.0):.4f} feat_hf={metrics.get('featmatch_hf', 0.0):.4f} spatial={metrics.get('spatial_proto', 0.0):.4f} "
                f"cycle={metrics['cycle']:.4f} idt={metrics['idt']:.4f} "
                f"data_wait={metrics['data_wait_sec']*1000.0:.1f}ms step_wall={metrics['step_wall_sec']*1000.0:.1f}ms | time={epoch_time:.1f}s"
            )
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("distill", 0.0)),
                    float(metrics.get("code", 0.0)),
                    float(metrics.get("code_ref", 0.0)),
                    float(metrics.get("code_proto", 0.0)),
                    float(metrics.get("proto", 0.0)),
                    float(metrics.get("ref_sep", 0.0)),
                    float(metrics.get("ref_dist", 0.0)),
                    float(metrics.get("ref_cos", 0.0)),
                    float(metrics.get("style_ce", 0.0)),
                    float(metrics.get("prob", 0.0)),
                    float(metrics.get("prob_margin", 0.0)),
                    float(metrics.get("prob_weight_mean", 0.0)),
                    float(metrics.get("cls_target_prob", 0.0)),
                    float(metrics.get("cls_hard_ratio", 0.0)),
                    float(metrics.get("dir", 0.0)),
                    float(metrics.get("proto_sep", 0.0)),
                    float(metrics.get("style_pred_acc", 0.0)),
                    float(metrics.get("xfer_margin", 0.0)),
                    float(metrics.get("proto_cos_max", 0.0)),
                    float(metrics.get("proto_cos_mean", 0.0)),
                    float(metrics.get("same_id", 0.0)),
                    float(metrics.get("cycle", 0.0)),
                    float(metrics.get("gram", 0.0)),
                    float(metrics.get("gram_hf", 0.0)),
                    float(metrics.get("moment", 0.0)),
                    float(metrics.get("moment_hf", 0.0)),
                    float(metrics.get("featmatch", 0.0)),
                    float(metrics.get("featmatch_hf", 0.0)),
                    float(metrics.get("spatial_proto", 0.0)),
                    float(metrics.get("push", 0.0)),
                    float(metrics.get("nce", 0.0)),
                    float(metrics.get("idt", 0.0)),
                    float(metrics.get("w_cycle_eff", 0.0)),
                    float(metrics.get("w_nce_eff", 0.0)),
                    float(metrics.get("w_idt_eff", 0.0)),
                    float(metrics.get("style_ref_alpha", 0.0)),
                    float(metrics.get("transfer_ratio", 0.0)),
                    float(metrics.get("transfer_weight_mean", 0.0)),
                    float(metrics.get("data_wait_sec", 0.0)),
                    float(metrics.get("step_wall_sec", 0.0)),
                    float(metrics.get("lr", 0.0)),
                    float(metrics.get("epoch_time_sec", 0.0)),
                ]
            )

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)
        return path

    def run_full_evaluation(self, epoch: int, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Launch external full evaluation script.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        if not checkpoint_path.exists():
            logger.warning("Skip full eval: checkpoint missing: %s", checkpoint_path)
            return False

        utils_script = Path(__file__).resolve().parent / "utils" / "run_evaluation.py"
        if not utils_script.exists():
            logger.warning("Skip full eval: script not found: %s", utils_script)
            return False

        out_dir = self.full_eval_root / f"epoch_{epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg_train = self.config.get("training", {})
        cfg_infer = self.config.get("inference", {})
        cfg_loss = self.config.get("loss", {})

        cmd = [
            sys.executable,
            str(utils_script),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(out_dir),
            "--num_steps",
            str(int(cfg_infer.get("num_steps", 1))),
            "--batch_size",
            str(int(cfg_train.get("full_eval_batch_size", 8))),
            "--max_src_samples",
            str(int(cfg_train.get("full_eval_max_src_samples", 30))),
            "--max_ref_compare",
            str(int(cfg_train.get("full_eval_max_ref_compare", 50))),
            "--max_ref_cache",
            str(int(cfg_train.get("full_eval_max_ref_cache", 256))),
            "--ref_feature_batch_size",
            str(int(cfg_train.get("full_eval_ref_feature_batch_size", 64))),
            "--style_ref_mode",
            str(cfg_train.get("full_eval_style_ref_mode", "prototype")),
            "--style_ref_count",
            str(int(cfg_train.get("full_eval_style_ref_count", 8))),
            "--style_ref_seed",
            str(int(cfg_train.get("full_eval_style_ref_seed", 2026))),
        ]

        test_dir = cfg_train.get("test_image_dir", "")
        if test_dir:
            cmd += ["--test_dir", str(test_dir)]
        cache_dir = cfg_train.get("full_eval_cache_dir", "")
        if cache_dir:
            cmd += ["--cache_dir", str(cache_dir)]
        classifier_path = cfg_train.get("full_eval_classifier_path", "")
        if not classifier_path:
            classifier_path = cfg_loss.get("style_classifier_ckpt", "")
        if classifier_path:
            resolved_classifier = self._resolve_config_relative_path(str(classifier_path))
            cmd += ["--classifier_path", str(resolved_classifier)]
        if bool(cfg_train.get("full_eval_classifier_only", False)):
            cmd += ["--eval_classifier_only"]
        if bool(cfg_train.get("full_eval_disable_lpips", False)):
            cmd += ["--eval_disable_lpips"]
        if "full_eval_save_images" in cfg_train and (not bool(cfg_train.get("full_eval_save_images", True))):
            cmd += ["--no_save_images"]
        if "full_eval_disable_xformers" in cfg_train:
            if bool(cfg_train.get("full_eval_disable_xformers", False)):
                cmd += ["--disable_xformers"]
            else:
                cmd += ["--enable_xformers"]

        log_path = self.log_dir / f"full_eval_epoch_{epoch:04d}.log"
        logger.info("Running full eval for epoch %d -> %s", epoch, out_dir)
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            logger.error("Full eval failed for epoch %d (code=%d). See %s", epoch, proc.returncode, log_path)
            return False
        logger.info("Full eval completed for epoch %d. Log: %s", epoch, log_path)
        return True
