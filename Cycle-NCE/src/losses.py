from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:  # pragma: no cover
    from model import LatentAdaCUT


def calc_moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Global mean/std matching per channel.
    """
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return F.mse_loss(mu_x, mu_y) + F.mse_loss(std_x, std_y)


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


def _lowpass(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


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
        # Core loss set for "train with reference, infer with style_id only".
        # Keep backward-compatible fallback names for old configs.
        self.w_distill = float(loss_cfg.get("w_distill", loss_cfg.get("w_token", 20.0)))
        self.w_code = float(loss_cfg.get("w_code", 10.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 10.0))

        # Optional auxiliary losses (default-off in the new scheme)
        self.w_gram = float(loss_cfg.get("w_gram", 0.0))
        self.w_moment = float(loss_cfg.get("w_moment", 0.0))
        self.w_nce = float(loss_cfg.get("w_nce", 0.0))
        self.w_idt = float(loss_cfg.get("w_idt", 0.0))
        self.w_push = float(loss_cfg.get("w_push", 0.0))
        self.push_margin = float(loss_cfg.get("push_margin", 0.2))

        self.cycle_warmup_epochs = int(loss_cfg.get("cycle_warmup_epochs", 0))
        self.cycle_ramp_epochs = int(loss_cfg.get("cycle_ramp_epochs", 1))
        self.idt_warmup_epochs = int(loss_cfg.get("idt_warmup_epochs", 0))
        self.idt_ramp_epochs = int(loss_cfg.get("idt_ramp_epochs", 1))

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

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        content_style_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Teacher path (with reference style); this is not used at inference.
        pred_teacher = model(
            content,
            style_id=target_style_id,
            style_ref=target_style,
            style_mix_alpha=1.0,
        )
        # Student path (deployment path, style_id only).
        pred_student = model(
            content,
            style_id=target_style_id,
            style_ref=None,
            style_mix_alpha=0.0,
        )
        transfer_mask = (target_style_id.long() != content_style_id.long()).float()

        if self.w_distill > 0.0:
            loss_distill = F.l1_loss(pred_student.float(), pred_teacher.detach().float())
        else:
            loss_distill = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Code closure:
        # - teacher output should map back to reference style code.
        # - student output should map back to target style prototype.
        s_ref = model.encode_style(target_style.float()).detach()
        p_tgt = model.encode_style_id(target_style_id)
        code_teacher = model.encode_style(pred_teacher.float())
        code_student = model.encode_style(pred_student.float())
        loss_code_ref = F.l1_loss(code_teacher, s_ref)
        loss_code_proto = F.l1_loss(code_student, p_tgt)
        loss_code = loss_code_ref + loss_code_proto

        w_cycle_eff = self._ramp_weight(
            self.w_cycle,
            epoch=self.current_epoch,
            warmup=self.cycle_warmup_epochs,
            ramp=self.cycle_ramp_epochs,
        )
        if w_cycle_eff > 0.0 and float(transfer_mask.sum().item()) > 0.0:
            # Low-pass cycle only for cross-domain samples:
            # x -> teacher(target) -> student(source prototype).
            rec = model(
                pred_teacher,
                style_id=content_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
            )
            rec_lp = _lowpass(rec.float())
            content_lp = _lowpass(content.float())
            per_sample_cycle = (rec_lp - content_lp).abs().mean(dim=(1, 2, 3))
            loss_cycle = (per_sample_cycle * transfer_mask).sum() / transfer_mask.sum().clamp_min(1.0)
        else:
            loss_cycle = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Optional extras (off by default): style statistics on teacher output.
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

        w_idt_eff = self._ramp_weight(
            self.w_idt,
            epoch=self.current_epoch,
            warmup=self.idt_warmup_epochs,
            ramp=self.idt_ramp_epochs,
        )
        if w_idt_eff > 0.0:
            idt_pred = model(
                content,
                style_id=content_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
            )
            loss_idt = F.l1_loss(idt_pred.float(), content.float())
        else:
            loss_idt = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_push > 0.0:
            p_src = model.encode_style_id(content_style_id)
            dist_to_src = (code_student - p_src).abs().mean(dim=1)
            push_term = F.relu(self.push_margin - dist_to_src) * transfer_mask
            denom = transfer_mask.sum().clamp_min(1.0)
            loss_push = push_term.sum() / denom
        else:
            loss_push = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        total = (
            self.w_distill * loss_distill
            + self.w_code * loss_code
            + w_cycle_eff * loss_cycle
            + self.w_gram * loss_gram
            + self.w_moment * loss_moment
            + w_nce_eff * loss_nce
            + w_idt_eff * loss_idt
            + self.w_push * loss_push
        )

        return {
            "loss": total,
            "distill": loss_distill.detach(),
            "gram": loss_gram.detach(),
            "moment": loss_moment.detach(),
            "code": loss_code.detach(),
            "code_ref": loss_code_ref.detach(),
            "code_proto": loss_code_proto.detach(),
            "push": loss_push.detach(),
            "nce": loss_nce.detach(),
            "idt": loss_idt.detach(),
            "cycle": loss_cycle.detach(),
            "w_cycle_eff": torch.tensor(w_cycle_eff, device=content.device),
            "w_nce_eff": torch.tensor(w_nce_eff, device=content.device),
            "w_idt_eff": torch.tensor(w_idt_eff, device=content.device),
            "style_ref_alpha": torch.tensor(1.0, device=content.device),
            "transfer_ratio": transfer_mask.mean().detach(),
            # Backward-compatible aliases for old logs.
            "token": loss_distill.detach(),
            "token_spatial": torch.tensor(0.0, device=content.device),
        }
