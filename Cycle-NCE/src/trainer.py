from __future__ import annotations

import csv
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .losses import AdaCUTObjective
    from .model import LatentAdaCUT, count_parameters
except ImportError:  # pragma: no cover
    from losses import AdaCUTObjective
    from model import LatentAdaCUT, count_parameters

logger = logging.getLogger(__name__)


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


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
        )
        if self.channels_last:
            self.model = self.model.to(device, memory_format=torch.channels_last)
        else:
            self.model = self.model.to(device)

        self.use_compile = bool(train_cfg.get("use_compile", False))
        if self.use_compile:
            try:
                compile_mode = str(train_cfg.get("compile_mode", "reduce-overhead"))
                self.model = torch.compile(self.model, mode=compile_mode, fullgraph=False)
                logger.info("torch.compile enabled (mode=%s)", compile_mode)
            except Exception as exc:  # pragma: no cover
                logger.warning("torch.compile failed, fallback to eager. reason=%s", exc)

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")
        logger.info(
            "Infra | channels_last=%s tf32=%s grad_ckpt=%s compile=%s",
            self.channels_last,
            self.allow_tf32,
            bool(train_cfg.get("use_gradient_checkpointing", False)),
            self.use_compile,
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

        self.loss_fn = AdaCUTObjective(config)

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(1, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
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
                    "gram",
                    "moment",
                    "code",
                    "push",
                    "cycle",
                    "nce",
                    "idt",
                    "w_cycle_eff",
                    "w_idt_eff",
                    "lr",
                    "epoch_time_sec",
                ]
            )

        self.global_step = 0
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))

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
        model_state = _strip_compile_prefix(state["model_state_dict"])
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

        sum_loss = 0.0
        sum_gram = 0.0
        sum_moment = 0.0
        sum_code = 0.0
        sum_push = 0.0
        sum_cycle = 0.0
        sum_nce = 0.0
        sum_idt = 0.0
        sum_w_cycle_eff = 0.0
        sum_w_idt_eff = 0.0
        num_batches = 0

        total_steps = len(dataloader)
        progress = tqdm(
            dataloader,
            total=total_steps,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=not self.use_tqdm,
        )

        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, raw_batch in enumerate(progress, start=1):
            try:
                batch = self._move_batch(raw_batch)
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

                sum_loss += float(loss.detach().item())
                sum_gram += float(loss_dict["gram"].item())
                sum_moment += float(loss_dict["moment"].item())
                sum_code += float(loss_dict["code"].item())
                sum_push += float(loss_dict["push"].item())
                sum_cycle += float(loss_dict["cycle"].item())
                sum_nce += float(loss_dict["nce"].item())
                sum_idt += float(loss_dict["idt"].item())
                sum_w_cycle_eff += float(loss_dict["w_cycle_eff"].item())
                sum_w_idt_eff += float(loss_dict["w_idt_eff"].item())
                num_batches += 1

                if step_idx % self.log_interval == 0:
                    elapsed = time.time() - epoch_start
                    step_per_sec = step_idx / max(elapsed, 1e-6)
                    eta = (total_steps - step_idx) / max(step_per_sec, 1e-6)
                    avg_loss = sum_loss / num_batches
                    avg_gram = sum_gram / num_batches
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        gram=f"{avg_gram:.4f}",
                        cyc=f"{(sum_cycle / num_batches):.4f}",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                    if not self.use_tqdm:
                        logger.info(
                            "epoch %d step %d/%d | loss=%.4f gram=%.4f moment=%.4f code=%.4f cycle=%.4f nce=%.4f idt=%.4f | %.2f it/s eta %.1fs",
                            epoch,
                            step_idx,
                            total_steps,
                            avg_loss,
                            avg_gram,
                            sum_moment / num_batches,
                            sum_code / num_batches,
                            sum_cycle / num_batches,
                            sum_nce / num_batches,
                            sum_idt / num_batches,
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
        metrics = {
            "loss": sum_loss / max(num_batches, 1),
            "gram": sum_gram / max(num_batches, 1),
            "moment": sum_moment / max(num_batches, 1),
            "code": sum_code / max(num_batches, 1),
            "push": sum_push / max(num_batches, 1),
            "cycle": sum_cycle / max(num_batches, 1),
            "nce": sum_nce / max(num_batches, 1),
            "idt": sum_idt / max(num_batches, 1),
            "w_cycle_eff": sum_w_cycle_eff / max(num_batches, 1),
            "w_idt_eff": sum_w_idt_eff / max(num_batches, 1),
            "lr": lr,
            "epoch_time_sec": epoch_time,
        }
        if self.use_tqdm:
            tqdm.write(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"loss={metrics['loss']:.4f} gram={metrics['gram']:.4f} moment={metrics['moment']:.4f} "
                f"code={metrics['code']:.4f} push={metrics['push']:.4f} "
                f"cycle={metrics['cycle']:.4f} "
                f"nce={metrics['nce']:.4f} "
                f"idt={metrics['idt']:.4f} "
                f"wcyc={metrics['w_cycle_eff']:.2f} widt={metrics['w_idt_eff']:.2f} | time={epoch_time:.1f}s"
            )
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("gram", 0.0)),
                    float(metrics.get("moment", 0.0)),
                    float(metrics.get("code", 0.0)),
                    float(metrics.get("push", 0.0)),
                    float(metrics.get("cycle", 0.0)),
                    float(metrics.get("nce", 0.0)),
                    float(metrics.get("idt", 0.0)),
                    float(metrics.get("w_cycle_eff", 0.0)),
                    float(metrics.get("w_idt_eff", 0.0)),
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
        ]

        test_dir = cfg_train.get("test_image_dir", "")
        if test_dir:
            cmd += ["--test_dir", str(test_dir)]
        cache_dir = cfg_train.get("full_eval_cache_dir", "")
        if cache_dir:
            cmd += ["--cache_dir", str(cache_dir)]
        classifier_path = cfg_loss.get("style_classifier_ckpt", "")
        if classifier_path:
            cmd += ["--classifier_path", str(classifier_path)]
        if bool(cfg_train.get("full_eval_classifier_only", False)):
            cmd += ["--eval_classifier_only"]
        if bool(cfg_train.get("full_eval_disable_lpips", False)):
            cmd += ["--eval_disable_lpips"]

        log_path = self.log_dir / f"full_eval_epoch_{epoch:04d}.log"
        logger.info("Running full eval for epoch %d -> %s", epoch, out_dir)
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            logger.error("Full eval failed for epoch %d (code=%d). See %s", epoch, proc.returncode, log_path)
            return False
        logger.info("Full eval completed for epoch %d. Log: %s", epoch, log_path)
        return True
