from __future__ import annotations

import csv
import json
import logging
import os
import shlex
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
    from .model import build_model_from_config, count_parameters
    from .utils.checkpoint import (
        find_latest_checkpoint,
        load_checkpoint as load_training_checkpoint,
        save_checkpoint as save_training_checkpoint,
    )
except ImportError:  # pragma: no cover
    from losses import AdaCUTObjective
    from model import build_model_from_config, count_parameters
    from utils.checkpoint import (
        find_latest_checkpoint,
        load_checkpoint as load_training_checkpoint,
        save_checkpoint as save_training_checkpoint,
    )

logger = logging.getLogger(__name__)

class AdaCUTTrainer:
    def __init__(self, config: Dict, device: torch.device, config_path: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.config_path = config_path

        train_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        ckpt_cfg = config.get("checkpoint", {})

        self.model = build_model_from_config(
            model_cfg,
            use_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),
        ).to(device)
        logger.info("Model params: %s", f"{count_parameters(self.model):,}")

        self.use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "fp16")).lower()
        self.amp_dtype = torch.float16 if amp_dtype_cfg in {"fp16", "float16", "half"} else torch.bfloat16
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        # 1. 先初始化 Loss Function (这样才能拿到里面的 MLP)
        self.loss_fn = AdaCUTObjective(config)
        # 确保模块被移动到了 GPU (因为 init 里只是定义了结构，compute 时才 to_device，
        # 但我们需要参数列表，所以这里先强制初始化一下)
        self.loss_fn._ensure_modules(self.device)

        # 2. 收集所有需要训练的参数
        # 包括：ResNet 主干 + NCE Loss 里的 MLP
        params_to_optimize = list(self.model.parameters())

        # 检查并添加 NCE MLP 参数
        if hasattr(self.loss_fn, "_content_loss_module"):
            nce_module = self.loss_fn._content_loss_module
            if isinstance(nce_module, torch.nn.Module):
                mlp_params = list(nce_module.parameters())
                logger.info("Adding NCE Projector params to optimizer: %d tensors", len(mlp_params))
                params_to_optimize.extend(mlp_params)

        # 3. 初始化 Optimizer
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,  # 使用合并后的参数列表
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            betas=(0.9, 0.999),
        )

        self.scheduler = None
        if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
            num_epochs = max(1, int(train_cfg.get("num_epochs", 100)))
            warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
            min_lr = float(train_cfg.get("min_learning_rate", 1e-5))
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, num_epochs - warmup_epochs),
                    eta_min=min_lr,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=min_lr,
                )

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.num_epochs = int(train_cfg.get("num_epochs", 100))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))
        self.full_eval_interval = max(0, int(train_cfg.get("full_eval_interval", 50)))
        self.run_full_eval_on_last_epoch = bool(train_cfg.get("full_eval_on_last_epoch", True))
        self.debug_grad_enabled = bool(train_cfg.get("debug_grad_enabled", False))
        self.debug_grad_interval = max(1, int(train_cfg.get("debug_grad_interval", 100)))
        self.debug_grad_loss_threshold = float(train_cfg.get("debug_grad_loss_threshold", 100.0))

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
                    "style_swd",
                    "style_moment",
                    "structure",
                    "tv",
                    "identity",
                    "train_num_steps",
                    "train_step_size",
                    "train_style_strength",
                    "lr",
                    "data_time_sec",
                    "compute_time_sec",
                    "epoch_time_sec",
                    "samples_seen",
                    "samples_per_sec",
                ]
            )

        self.global_step = 0
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))

    def _find_latest_checkpoint(self) -> Optional[Path]:
        return find_latest_checkpoint(self.checkpoint_dir)

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

        state = load_training_checkpoint(
            checkpoint_path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=str(self.device),
        )
        self.start_epoch = int(state.get("start_epoch", 1))
        self.global_step = int(state.get("global_step", 0))
        logger.info("Resumed from %s (start_epoch=%d)", ckpt_path, self.start_epoch)

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    # =========================================================================
    # DEBUG TOOLS: gradient microscope
    # =========================================================================
    def _log_gradient_norms(self, epoch: int, step: int) -> None:
        """
        Gradient magnitude monitor for vanishing/exploding gradient checks.
        """
        header = f"\n[Epoch {epoch} Step {step}] --- Gradient Magnitude Radar ---"
        cols = f"{ 'Layer':<30} | {'Grad':<10} | {'Weight':<10} | {'Ratio(1e-3)':<10}"
        divider = "-" * 70
        try:
            tqdm.write(header)
        except Exception:
            pass
        logger.info(header)
        print(header)
        try:
            tqdm.write(cols)
        except Exception:
            pass
        print(cols)
        try:
            tqdm.write(divider)
        except Exception:
            pass
        print(divider)

        watch_list = [
            "style_emb",
            "enc_in",
            "hires_body.0",
            "dec_out",
        ]

        lr = float(self.optimizer.param_groups[0]["lr"])
        has_data = False

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            is_watched = any(k in name for k in watch_list)
            g_norm = float(p.grad.detach().norm().item())
            w_norm = float(p.detach().norm().item())
            ratio = (g_norm * lr) / (w_norm + 1e-9) * 1e3

            if is_watched or g_norm > 5.0 or (g_norm == 0.0 and w_norm > 0.0):
                line = f"{name:<30} | {g_norm:.2e}   | {w_norm:.2e}   | {ratio:.2f}"
                try:
                    tqdm.write(line)
                except Exception:
                    pass
                logger.info(line)
                print(line)
                has_data = True

        if not has_data:
            try:
                tqdm.write("No significant gradients found.")
            except Exception:
                pass
            logger.info("No significant gradients found.")
            print("No significant gradients found.")
        try:
            tqdm.write(divider + "\n")
        except Exception:
            pass
        print(divider + "\n")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()
        data_wait_start = time.perf_counter()

        data_time_total = 0.0
        compute_time_total = 0.0
        metric_sums: Dict[str, float] = {}
        num_batches = 0
        batch_size = int(getattr(dataloader, "batch_size", 0) or 0)

        self.optimizer.zero_grad(set_to_none=True)
        total_steps = len(dataloader)
        progress = tqdm(dataloader, total=total_steps, disable=not self.use_tqdm, desc=f"Epoch {epoch}")

        for step_idx, batch in enumerate(progress, start=1):
            data_time_total += max(0.0, time.perf_counter() - data_wait_start)
            step_start = time.perf_counter()

            content = batch["content"].to(self.device, non_blocking=True)
            target_style = batch["target_style"].to(self.device, non_blocking=True)
            target_style_id = batch["target_style_id"].to(self.device, non_blocking=True).long()
            source_style_id = batch["source_style_id"].to(self.device, non_blocking=True).long()

            try:
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_dict = self.loss_fn.compute(
                        model=self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        source_style_id=source_style_id,
                        debug_timing=False,
                    )
                    loss = loss_dict["loss"]
            except RuntimeError as exc:
                msg = str(exc)
                if "illegal memory access" in msg.lower():
                    logger.error(
                        "CUDA illegal memory access at epoch=%d step=%d. "
                        "Likely from async kernel failure in previous op. "
                        "Try CUDA_LAUNCH_BLOCKING=1 for exact op, and lower batch size if near VRAM limit.",
                        epoch,
                        step_idx,
                    )
                raise

            loss_to_backward = loss / self.accumulation_steps
            if self.use_grad_scaler:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = (step_idx % self.accumulation_steps == 0)
            if should_step:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)

                should_debug = False
                if self.debug_grad_enabled:
                    should_debug = (
                        (step_idx % self.debug_grad_interval == 0)
                        or (float(loss.detach().item()) > self.debug_grad_loss_threshold)
                    )
                if should_debug:
                    self._log_gradient_norms(epoch, step_idx)

                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.use_grad_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v.detach().item())
            num_batches += 1
            compute_time_total += max(0.0, time.perf_counter() - step_start)

            if self.log_interval > 0 and (step_idx % self.log_interval == 0):
                denom = max(1, num_batches)
                avg_loss = metric_sums.get("loss", 0.0) / denom
                avg_style = metric_sums.get("style_swd", 0.0) / denom
                avg_moment = metric_sums.get("style_moment", 0.0) / denom
                avg_struct = metric_sums.get("structure", 0.0) / denom
                avg_tv = metric_sums.get("tv", 0.0) / denom
                avg_id = metric_sums.get("identity", 0.0) / denom
                step_per_sec = step_idx / max(time.time() - epoch_start, 1e-6)
                eta = (total_steps - step_idx) / max(step_per_sec, 1e-6)
                if self.use_tqdm:
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        style_swd=f"{avg_style:.4f}",
                        moment=f"{avg_moment:.4f}",
                        struct=f"{avg_struct:.4f}",
                        tv=f"{avg_tv:.4f}",
                        idt=f"{avg_id:.4f}",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                else:
                    logger.info(
                        "[Epoch %d Step %d/%d] loss=%.4f style=%.4f mom=%.4f struct=%.4f tv=%.4f idt=%.4f it/s=%.2f eta=%.1fs",
                        epoch,
                        step_idx,
                        total_steps,
                        avg_loss,
                        avg_style,
                        avg_moment,
                        avg_struct,
                        avg_tv,
                        avg_id,
                        step_per_sec,
                        eta,
                    )

            data_wait_start = time.perf_counter()

        progress.close()

        # Flush remaining gradients for uneven accumulation.
        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            if self.grad_clip_norm > 0.0:
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
        samples_seen = int(num_batches * batch_size) if batch_size > 0 else 0
        samples_per_sec = float(samples_seen / max(epoch_time, 1e-6)) if samples_seen > 0 else 0.0

        metrics: Dict[str, float] = {}
        denom = max(1, num_batches)
        for k, v in metric_sums.items():
            metrics[k] = float(v / denom)

        metrics.setdefault("loss", 0.0)
        metrics.setdefault("style_swd", 0.0)
        metrics.setdefault("style_moment", 0.0)
        metrics.setdefault("structure", 0.0)
        metrics.setdefault("tv", 0.0)
        metrics.setdefault("identity", 0.0)
        metrics.setdefault("train_num_steps", 0.0)
        metrics.setdefault("train_step_size", 0.0)
        metrics.setdefault("train_style_strength", 0.0)

        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["data_time_sec"] = data_time_total
        metrics["compute_time_sec"] = compute_time_total
        metrics["epoch_time_sec"] = epoch_time
        metrics["samples_seen"] = float(samples_seen)
        metrics["samples_per_sec"] = samples_per_sec

        if self.use_tqdm:
            tqdm.write(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"loss={metrics['loss']:.4f} style_swd={metrics['style_swd']:.4f} "
                f"mom={metrics['style_moment']:.4f} struct={metrics['structure']:.4f} tv={metrics['tv']:.4f} "
                f"idt={metrics['identity']:.4f} "
                f"steps={metrics['train_num_steps']:.1f} h={metrics['train_step_size']:.2f} s={metrics['train_style_strength']:.2f} "
                f"| data={metrics['data_time_sec']:.1f}s compute={metrics['compute_time_sec']:.1f}s "
                f"total={metrics['epoch_time_sec']:.1f}s sps={metrics['samples_per_sec']:.1f}"
            )
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("style_swd", 0.0)),
                    float(metrics.get("style_moment", 0.0)),
                    float(metrics.get("structure", 0.0)),
                    float(metrics.get("tv", 0.0)),
                    float(metrics.get("identity", 0.0)),
                    float(metrics.get("train_num_steps", 0.0)),
                    float(metrics.get("train_step_size", 0.0)),
                    float(metrics.get("train_style_strength", 0.0)),
                    float(metrics.get("lr", 0.0)),
                    float(metrics.get("data_time_sec", 0.0)),
                    float(metrics.get("compute_time_sec", 0.0)),
                    float(metrics.get("epoch_time_sec", 0.0)),
                    int(float(metrics.get("samples_seen", 0.0))),
                    float(metrics.get("samples_per_sec", 0.0)),
                ]
            )

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        return save_training_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            epoch=int(epoch),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            metrics=metrics,
            global_step=int(self.global_step),
        )

    def run_full_evaluation(self, epoch: int, checkpoint_path: Optional[Path] = None) -> bool:
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
        full_eval_mode = str(cfg_train.get("full_eval_mode", "full")).strip().lower()
        if full_eval_mode not in {"full", "gen", "ana"}:
            full_eval_mode = "full"

        cmd = [
            sys.executable,
            str(utils_script),
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(out_dir),
            "--mode",
            full_eval_mode,
        ]
        if bool(cfg_train.get("full_eval_clip_allow_network", False)):
            cmd += ["--clip_allow_network"]
        test_dir = cfg_train.get("test_image_dir", "")
        if test_dir:
            cmd += ["--test_dir", str(test_dir)]
        cache_dir = cfg_train.get("full_eval_cache_dir", "")
        if cache_dir:
            cmd += ["--cache_dir", str(cache_dir)]
        if bool(cfg_train.get("full_eval_disable_lpips", False)):
            cmd += ["--eval_disable_lpips"]
        if bool(cfg_train.get("full_eval_reuse_generated", True)):
            cmd += ["--reuse_generated"]
        if bool(cfg_train.get("full_eval_generation_only", False)):
            cmd += ["--generation_only"]

        log_path = self.log_dir / f"full_eval_epoch_{epoch:04d}.log"
        if os.name == "nt":
            cmd_text = subprocess.list2cmdline(cmd)
        else:
            cmd_text = shlex.join(cmd)
        logger.info("Running full eval for epoch %d -> %s", epoch, out_dir)
        logger.info("Full eval command: %s", cmd_text)
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# Full eval command:\n{cmd_text}\n\n")
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), stdout=logf, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            logger.error("Full eval failed for epoch %d (code=%d). See %s", epoch, proc.returncode, log_path)
            return False

        self._write_full_eval_history()
        logger.info("Full eval completed for epoch %d. Log: %s", epoch, log_path)
        return True

    def _write_full_eval_history(self) -> None:
        rounds = []
        for epoch_dir in sorted(self.full_eval_root.glob("epoch_*")):
            summary_path = epoch_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                continue
            try:
                epoch = int(epoch_dir.name.split("_")[-1])
            except Exception:
                continue

            analysis = summary.get("analysis", {})
            transfer = analysis.get("style_transfer_ability", {})
            p2a = analysis.get("photo_to_art_performance", {})
            rounds.append(
                {
                    "epoch": epoch,
                    "summary_path": str(summary_path),
                    "transfer_style": float(transfer.get("style", transfer.get("clip_style_sim", transfer.get("style_swd", 0.0)))),
                    "transfer_style_swd": float(transfer.get("style_swd", 0.0)),
                    "transfer_clip_style_sim": float(transfer.get("clip_style_sim", 0.0)),
                    "transfer_content_lf_ssim": float(transfer.get("content_lf_ssim", transfer.get("content_ssim", 0.0))),
                    "transfer_content_ssim": float(transfer.get("content_lf_ssim", transfer.get("content_ssim", 0.0))),
                    "photo_to_art_style": float(p2a.get("style", p2a.get("clip_style_sim", p2a.get("style_swd", 0.0)))),
                    "photo_to_art_style_swd": float(p2a.get("style_swd", 0.0)),
                    "photo_to_art_clip_style_sim": float(p2a.get("clip_style_sim", 0.0)),
                }
            )

        if not rounds:
            return
        rounds.sort(key=lambda x: x["epoch"])

        payload = {
            "num_rounds": len(rounds),
            "latest": rounds[-1],
            "rounds": rounds,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        history_path = self.full_eval_root / "summary_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
