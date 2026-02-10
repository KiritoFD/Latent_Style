from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:  # pragma: no cover
    from model import LatentAdaCUT


def calc_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Channel-correlation style descriptor.
    """
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    feat = feat - feat.mean(dim=2, keepdim=True)
    feat = feat / (feat.std(dim=2, keepdim=True, unbiased=False) + 1e-6)
    feat_t = feat.transpose(1, 2)
    return feat.bmm(feat_t) / max(h * w, 1)


def calc_gram_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gram_x = calc_gram_matrix(x)
    gram_y = calc_gram_matrix(y)
    return F.mse_loss(gram_x, gram_y)


def calc_moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Global mean/std matching per channel.
    """
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return F.mse_loss(mu_x, mu_y) + F.mse_loss(std_x, std_y)


def _lowpass(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)

def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    gx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    gy = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = gx.repeat(x.shape[1], 1, 1, 1)
    gy = gy.repeat(x.shape[1], 1, 1, 1)
    dx = F.conv2d(x, gx, padding=1, groups=x.shape[1])
    dy = F.conv2d(x, gy, padding=1, groups=x.shape[1])
    return torch.sqrt(dx * dx + dy * dy + 1e-6)


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


def _multiscale_latent_feats(x: torch.Tensor) -> list[torch.Tensor]:
    def patch_expand(f: torch.Tensor, patch: int = 3) -> torch.Tensor:
        if patch <= 1:
            return f
        b, c, h, w = f.shape
        u = F.unfold(f, kernel_size=patch, padding=patch // 2, stride=1)
        return u.view(b, c * patch * patch, h, w)

    def enrich(f: torch.Tensor) -> torch.Tensor:
        f = patch_expand(f, patch=3)
        low = F.interpolate(
            F.avg_pool2d(f, kernel_size=2, stride=2),
            size=f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = f - low
        dx = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([f, high, dx, dy], dim=1)

    return [
        enrich(x),
        enrich(F.avg_pool2d(x, kernel_size=2, stride=2)),
        enrich(F.avg_pool2d(x, kernel_size=4, stride=4)),
    ]


def calc_nce_loss(
    model: LatentAdaCUT,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    temperature: float = 0.1,
    spatial_size: int = 8,
    max_tokens: int = 2048,
) -> torch.Tensor:
    """
    Token-level InfoNCE to preserve content structure.
    """
    if spatial_size > 0 and (x_in.shape[-1] != spatial_size or x_in.shape[-2] != spatial_size):
        x_in = F.interpolate(x_in, size=(spatial_size, spatial_size), mode="area")
        x_out = F.interpolate(x_out, size=(spatial_size, spatial_size), mode="area")

    q = F.normalize(model.project_tokens(x_out), dim=1)
    k = F.normalize(model.project_tokens(x_in), dim=1)

    token_count = q.shape[0]
    if token_count > max_tokens:
        idx = torch.randperm(token_count, device=q.device)[:max_tokens]
        q = q.index_select(0, idx)
        k = k.index_select(0, idx)
        token_count = max_tokens

    logits = (q @ k.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(token_count, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


class AdaCUTObjective:
    """
    Weighted objective wrapper.
    """

    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        self.w_distill = float(loss_cfg.get("w_distill", 10.0))
        self.distill_low_only = bool(loss_cfg.get("distill_low_only", False))
        self.distill_cross_domain_only = bool(loss_cfg.get("distill_cross_domain_only", True))
        self.w_code = float(loss_cfg.get("w_code", 10.0))
        self.w_struct = float(loss_cfg.get("w_struct", 0.0))
        self.w_edge = float(loss_cfg.get("w_edge", 0.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 0.0))
        self.cycle_lowpass_strength = float(loss_cfg.get("cycle_lowpass_strength", 1.0))
        self.cycle_lowpass_strength = max(0.0, min(1.0, self.cycle_lowpass_strength))
        self.struct_lowpass_strength = float(loss_cfg.get("struct_lowpass_strength", 1.0))
        self.struct_lowpass_strength = max(0.0, min(1.0, self.struct_lowpass_strength))
        self.cycle_loss_type = str(loss_cfg.get("cycle_loss_type", "l1")).lower()
        if self.cycle_loss_type not in {"l1", "mse"}:
            self.cycle_loss_type = "l1"
        self.struct_loss_type = str(loss_cfg.get("struct_loss_type", "l1")).lower()
        if self.struct_loss_type not in {"l1", "mse"}:
            self.struct_loss_type = "l1"
        self.cycle_edge_strength = float(loss_cfg.get("cycle_edge_strength", 0.0))
        self.cycle_edge_strength = max(0.0, min(1.0, self.cycle_edge_strength))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))

        # Optional auxiliary losses (default-off in the new scheme)
        self.w_gram = float(loss_cfg.get("w_gram", 0.0))
        self.w_moment = float(loss_cfg.get("w_moment", 0.0))
        self.w_nce = float(loss_cfg.get("w_nce", 0.0))
        # Hard-disable idt for overfit experiments (avoid identity lock-in).
        self.w_idt = 0.0
        self.w_push = float(loss_cfg.get("w_push", 0.0))
        self.push_margin = float(loss_cfg.get("push_margin", 0.2))

        self.cycle_warmup_epochs = int(loss_cfg.get("cycle_warmup_epochs", 0))
        self.cycle_ramp_epochs = int(loss_cfg.get("cycle_ramp_epochs", 1))
        self.struct_warmup_epochs = int(loss_cfg.get("struct_warmup_epochs", 0))
        self.struct_ramp_epochs = int(loss_cfg.get("struct_ramp_epochs", 1))
        self.edge_warmup_epochs = int(loss_cfg.get("edge_warmup_epochs", 0))
        self.edge_ramp_epochs = int(loss_cfg.get("edge_ramp_epochs", 1))
        self.idt_warmup_epochs = 0
        self.idt_ramp_epochs = 0
        self.nce_temperature = float(loss_cfg.get("nce_temperature", 0.1))
        self.nce_spatial_size = int(loss_cfg.get("nce_spatial_size", 16))
        self.nce_max_tokens = int(loss_cfg.get("nce_max_tokens", 2048))
        self.nce_warmup_epochs = int(loss_cfg.get("nce_warmup_epochs", 0))
        self.nce_ramp_epochs = int(loss_cfg.get("nce_ramp_epochs", 1))
        self.current_epoch = 1
        self.total_epochs = 1

    def set_progress(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.total_epochs = max(1, int(total_epochs))

    @staticmethod
    def _ramp_weight(base: float, epoch: int, warmup: int, ramp: int) -> float:
        if base <= 0.0:
            return 0.0
        if epoch <= warmup:
            return 0.0
        if ramp <= 0:
            return base
        ratio = min(1.0, max(0.0, (epoch - warmup) / float(ramp)))
        return float(base * ratio)

    @staticmethod
    def _per_sample_alignment(
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        loss_type: str,
        lowpass_strength: float,
    ) -> torch.Tensor:
        """
        Compute per-sample alignment with optional low-pass blending.
        """
        if loss_type == "mse":
            diff_raw = (x.float() - y.float()).pow(2).mean(dim=(1, 2, 3))
        else:
            diff_raw = (x.float() - y.float()).abs().mean(dim=(1, 2, 3))

        if lowpass_strength <= 0.0:
            return diff_raw

        x_lp = _lowpass(x.float())
        y_lp = _lowpass(y.float())
        if loss_type == "mse":
            diff_lp = (x_lp - y_lp).pow(2).mean(dim=(1, 2, 3))
        else:
            diff_lp = (x_lp - y_lp).abs().mean(dim=(1, 2, 3))
        return (1.0 - lowpass_strength) * diff_raw + lowpass_strength * diff_lp

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        content_style_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Teacher path uses reference style for stronger supervision.
        pred_teacher = model(
            content,
            style_id=target_style_id,
            style_ref=target_style,
            style_mix_alpha=1.0,
        )
        # Student path matches deployment: style_id only (no reference).
        pred_student = model(
            content,
            style_id=target_style_id,
            style_ref=None,
            style_mix_alpha=0.0,
        )
        transfer_mask = (target_style_id.long() != content_style_id.long()).float()

        # Distill: student should mimic teacher output (stop grad on teacher).
        # Optionally do LP-only distill and/or cross-domain-only aggregation.
        if self.distill_low_only:
            pred_s_distill = _lowpass(pred_student.float())
            pred_t_distill = _lowpass(pred_teacher.detach().float())
        else:
            pred_s_distill = pred_student.float()
            pred_t_distill = pred_teacher.detach().float()
        per_sample_distill = (pred_s_distill - pred_t_distill).abs().mean(dim=(1, 2, 3))
        if self.distill_cross_domain_only and float(transfer_mask.sum().item()) > 0.0:
            loss_distill = (per_sample_distill * transfer_mask).sum() / transfer_mask.sum().clamp_min(1.0)
        else:
            loss_distill = per_sample_distill.mean()

        # Code closure: teacher should match reference, student should match style_id.
        s_ref = model.encode_style(target_style.float()).detach()
        code_teacher = model.encode_style(pred_teacher.float())
        code_student = model.encode_style(pred_student.float())
        s_id = model.encode_style_id(target_style_id)
        loss_code = F.l1_loss(code_teacher, s_ref) + F.l1_loss(code_student, s_id)
        code_pred_norm = code_student.norm(dim=1).mean()
        code_ref_norm = s_ref.norm(dim=1).mean()

        w_cycle_eff = self._ramp_weight(
            self.w_cycle,
            epoch=self.current_epoch,
            warmup=self.cycle_warmup_epochs,
            ramp=self.cycle_ramp_epochs,
        )
        if w_cycle_eff > 0.0 and float(transfer_mask.sum().item()) > 0.0:
            # Cross-domain cycle with configurable pixel/low-pass blend.
            rec = model(
                pred_student,
                style_id=content_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
            )
            per_sample_cycle = self._per_sample_alignment(
                rec,
                content,
                loss_type=self.cycle_loss_type,
                lowpass_strength=self.cycle_lowpass_strength,
            )
            if self.cycle_edge_strength > 0.0:
                edge_rec = _sobel_magnitude(rec.float())
                edge_content = _sobel_magnitude(content.float())
                per_sample_cycle_edge = (edge_rec - edge_content).abs().mean(dim=(1, 2, 3))
                per_sample_cycle = (
                    (1.0 - self.cycle_edge_strength) * per_sample_cycle
                    + self.cycle_edge_strength * per_sample_cycle_edge
                )
            loss_cycle = (per_sample_cycle * transfer_mask).sum() / transfer_mask.sum().clamp_min(1.0)
        else:
            loss_cycle = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_struct_eff = self._ramp_weight(
            self.w_struct,
            epoch=self.current_epoch,
            warmup=self.struct_warmup_epochs,
            ramp=self.struct_ramp_epochs,
        )
        if w_struct_eff > 0.0:
            per_sample_struct = self._per_sample_alignment(
                pred_student,
                content,
                loss_type=self.struct_loss_type,
                lowpass_strength=self.struct_lowpass_strength,
            )
            loss_struct = per_sample_struct.mean()
        else:
            loss_struct = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_edge_eff = self._ramp_weight(
            self.w_edge,
            epoch=self.current_epoch,
            warmup=self.edge_warmup_epochs,
            ramp=self.edge_ramp_epochs,
        )
        if w_edge_eff > 0.0:
            edge_pred = _sobel_magnitude(pred_student.float())
            edge_content = _sobel_magnitude(content.float())
            loss_edge = F.l1_loss(edge_pred, edge_content)
        else:
            loss_edge = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Optional extras: style statistics on output.
        loss_gram = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        loss_moment = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        if self.w_gram > 0.0 or self.w_moment > 0.0:
            pred_feats = _multiscale_latent_feats(pred_teacher.float())
            tgt_feats = _multiscale_latent_feats(target_style.float())
            for a, b in zip(pred_feats, tgt_feats):
                loss_gram = loss_gram + calc_gram_loss(a, b)
                loss_moment = loss_moment + calc_moment_loss(a, b)
            scale = 1.0 / float(len(pred_feats))
            loss_gram = loss_gram * scale
            loss_moment = loss_moment * scale
        w_nce_eff = self._ramp_weight(
            self.w_nce,
            epoch=self.current_epoch,
            warmup=self.nce_warmup_epochs,
            ramp=self.nce_ramp_epochs,
        )
        if w_nce_eff > 0.0:
            loss_nce = calc_nce_loss(
                model,
                x_in=content.float(),
                x_out=pred_student.float(),
                temperature=self.nce_temperature,
                spatial_size=self.nce_spatial_size,
                max_tokens=self.nce_max_tokens,
            )
        else:
            loss_nce = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_idt_eff = 0.0
        loss_idt = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_push > 0.0:
            p_src = model.encode_style_id(content_style_id)
            dist_to_src = (code_student - p_src).abs().mean(dim=1)
            push_term = F.relu(self.push_margin - dist_to_src) * transfer_mask
            denom = transfer_mask.sum().clamp_min(1.0)
            loss_push = push_term.sum() / denom
        else:
            loss_push = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_delta_tv > 0.0:
            delta = pred_student.float() - content.float()
            loss_delta_tv = _tv_per_sample(delta).mean()
        else:
            loss_delta_tv = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        total = (
            self.w_distill * loss_distill
            + self.w_code * loss_code
            + w_struct_eff * loss_struct
            + w_edge_eff * loss_edge
            + w_cycle_eff * loss_cycle
            + self.w_gram * loss_gram
            + self.w_moment * loss_moment
            + w_nce_eff * loss_nce
            + self.w_push * loss_push
            + self.w_delta_tv * loss_delta_tv
        )

        return {
            "loss": total,
            "distill": loss_distill.detach(),
            "gram": loss_gram.detach(),
            "gram_w": (loss_gram.detach() * float(self.w_gram)),
            "moment": loss_moment.detach(),
            "code": loss_code.detach(),
            "code_pred_norm": code_pred_norm.detach(),
            "code_ref_norm": code_ref_norm.detach(),
            "push": loss_push.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "nce": loss_nce.detach(),
            "idt": loss_idt.detach(),
            "cycle": loss_cycle.detach(),
            "struct": loss_struct.detach(),
            "edge": loss_edge.detach(),
            "w_cycle_eff": torch.tensor(w_cycle_eff, device=content.device),
            "w_struct_eff": torch.tensor(w_struct_eff, device=content.device),
            "w_edge_eff": torch.tensor(w_edge_eff, device=content.device),
            "w_nce_eff": torch.tensor(w_nce_eff, device=content.device),
            "w_idt_eff": torch.tensor(w_idt_eff, device=content.device),
            "style_ref_alpha": torch.tensor(0.0, device=content.device),
            "transfer_ratio": transfer_mask.mean().detach(),
        }
