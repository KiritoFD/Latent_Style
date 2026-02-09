from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
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


def _highpass_same_size(x: torch.Tensor) -> torch.Tensor:
    low = F.interpolate(
        _lowpass(x),
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return x - low


def _augment_for_classifier(
    x: torch.Tensor,
    prob: float,
    brightness: float,
    contrast: float,
    noise_std: float,
    blur_prob: float,
) -> torch.Tensor:
    """
    Lightweight stochastic augmentation to reduce classifier cheating by pure
    brightness/contrast shifts.
    """
    if prob <= 0.0:
        return x
    if float(torch.rand((), device=x.device).item()) > prob:
        return x

    y = x
    bsz = y.shape[0]

    if brightness > 0.0:
        shift = (torch.rand((bsz, 1, 1, 1), device=y.device, dtype=y.dtype) * 2.0 - 1.0) * float(brightness)
        y = y + shift

    if contrast > 0.0:
        scale = 1.0 + (torch.rand((bsz, 1, 1, 1), device=y.device, dtype=y.dtype) * 2.0 - 1.0) * float(contrast)
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * scale + mean

    if noise_std > 0.0:
        y = y + torch.randn_like(y) * float(noise_std)

    if blur_prob > 0.0 and float(torch.rand((), device=y.device).item()) < blur_prob:
        y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

    return y


def _norm_spatial_feat(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=(2, 3), keepdim=True)
    x = x / (x.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
    return x


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

    def __init__(self, config: Dict, style_classifier: Optional[nn.Module] = None) -> None:
        loss_cfg = config.get("loss", {})
        self.style_classifier = style_classifier
        # Core loss set for "train with reference, infer with style_id only".
        # Keep backward-compatible fallback names for old configs.
        self.w_distill = float(loss_cfg.get("w_distill", loss_cfg.get("w_token", 20.0)))
        self.w_code = float(loss_cfg.get("w_code", 10.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 10.0))
        self.w_proto = float(loss_cfg.get("w_proto", 5.0))
        self.w_same_id = float(loss_cfg.get("w_same_id", 1.0))
        # Classifier CE is disabled by default. We keep probability-based guidance
        # (w_prob / w_prob_margin / w_dir) and treat CE as optional legacy behavior.
        self.disable_cls_ce = bool(loss_cfg.get("disable_cls_ce", True))
        self.w_cls = 0.0 if self.disable_cls_ce else float(loss_cfg.get("w_cls", loss_cfg.get("w_style_ce", 0.0)))
        self.w_prob = float(loss_cfg.get("w_prob", 0.0))
        self.w_prob_margin = float(loss_cfg.get("w_prob_margin", 0.0))
        self.prob_focal_gamma = float(loss_cfg.get("prob_focal_gamma", 0.0))
        raw_prob_target_weights = loss_cfg.get("prob_target_weights", [])
        if isinstance(raw_prob_target_weights, (list, tuple)):
            self.prob_target_weights = [float(v) for v in raw_prob_target_weights]
        else:
            self.prob_target_weights = []
        self.w_dir = float(loss_cfg.get("w_dir", 0.0))
        self.w_proto_sep = float(loss_cfg.get("w_proto_sep", 0.0))
        self.cls_temp = float(loss_cfg.get("cls_temperature", loss_cfg.get("style_ce_temp", 1.0)))
        self.cls_aug_views = max(1, int(loss_cfg.get("cls_aug_views", 1)))
        self.cls_aug_prob = float(loss_cfg.get("cls_aug_prob", 0.0))
        self.cls_aug_brightness = float(loss_cfg.get("cls_aug_brightness", 0.08))
        self.cls_aug_contrast = float(loss_cfg.get("cls_aug_contrast", 0.08))
        self.cls_aug_noise_std = float(loss_cfg.get("cls_aug_noise_std", 0.01))
        self.cls_aug_blur_prob = float(loss_cfg.get("cls_aug_blur_prob", 0.25))
        self.cls_label_smoothing = float(loss_cfg.get("cls_label_smoothing", 0.0))
        self.cls_stop_conf = float(loss_cfg.get("cls_stop_conf", 1.0))
        self.cls_hard_min_weight = float(loss_cfg.get("cls_hard_min_weight", 0.0))
        self.dir_margin = float(loss_cfg.get("dir_margin", 0.10))
        self.prob_margin = float(loss_cfg.get("prob_margin", self.dir_margin))
        self.proto_cos_max = float(loss_cfg.get("proto_cos_max", 0.10))

        # Teacher style anchors: keep teacher stylized, then distill to student.
        self.w_gram = float(loss_cfg.get("w_gram", 80.0))
        self.w_moment = float(loss_cfg.get("w_moment", 5.0))
        self.w_featmatch = float(loss_cfg.get("w_featmatch", 0.0))
        self.w_featmatch_teacher = float(loss_cfg.get("w_featmatch_teacher", 0.0))
        self.w_featmatch_hf = float(loss_cfg.get("w_featmatch_hf", 0.0))
        self.w_gram_hf = float(loss_cfg.get("w_gram_hf", 0.0))
        self.w_moment_hf = float(loss_cfg.get("w_moment_hf", 0.0))
        self.style_feat_min_level = int(loss_cfg.get("style_feat_min_level", 1))
        self.style_feat_use_highpass = bool(loss_cfg.get("style_feat_use_highpass", True))
        self.prob_gate_enabled = bool(loss_cfg.get("prob_gate_enabled", True))
        self.prob_gate_min = float(loss_cfg.get("prob_gate_min", 0.15))
        self.prob_gate_power = float(loss_cfg.get("prob_gate_power", 1.0))
        self.prob_gate_detach = bool(loss_cfg.get("prob_gate_detach", True))
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
        def _zero() -> torch.Tensor:
            return torch.tensor(0.0, device=content.device, dtype=content.dtype)

        target_ids = target_style_id.long().view(-1)
        source_ids = content_style_id.long().view(-1)
        transfer_mask = (target_ids != source_ids).float()
        transfer_weight = transfer_mask
        transfer_denom = transfer_weight.sum().clamp_min(1.0)
        has_transfer = float(transfer_mask.sum().item()) > 0.0

        # Deployment path: style_id only.
        pred_student = model(
            content,
            style_id=target_style_id,
            style_ref=None,
            style_mix_alpha=0.0,
        )

        # Optional teacher path (training-only, with reference style).
        need_teacher = any(
            w > 0.0
            for w in (
                self.w_distill,
                self.w_code,
                self.w_proto,
            )
        )
        pred_teacher = None
        if need_teacher:
            pred_teacher = model(
                content,
                style_id=target_style_id,
                style_ref=target_style,
                style_mix_alpha=1.0,
            )

        teacher_for_code = pred_teacher if pred_teacher is not None else pred_student

        # Monitor prototype collapse directly from style_id embeddings.
        proto_bank = F.normalize(model.style_emb.weight.float(), dim=1)
        if proto_bank.shape[0] > 1:
            sim_mat = proto_bank @ proto_bank.t()
            offdiag = sim_mat[~torch.eye(sim_mat.shape[0], device=sim_mat.device, dtype=torch.bool)]
            proto_cos_max_metric = offdiag.max()
            proto_cos_mean_metric = offdiag.mean()
        else:
            offdiag = torch.tensor([], device=content.device, dtype=torch.float32)
            proto_cos_max_metric = _zero()
            proto_cos_mean_metric = _zero()

        # 1) Frozen classifier guidance (hard cross-domain signal).
        need_classifier_guidance = any(
            w > 0.0 for w in (self.w_cls, self.w_prob, self.w_prob_margin, self.w_dir)
        )
        if need_classifier_guidance:
            if self.style_classifier is None:
                raise RuntimeError(
                    "Classifier-guided losses are enabled but no frozen style classifier was provided."
                )

            logits_views = []
            num_views = max(1, self.cls_aug_views)
            for view_idx in range(num_views):
                cls_in = pred_student.float()
                if (view_idx > 0) or (num_views == 1):
                    cls_in = _augment_for_classifier(
                        cls_in,
                        prob=self.cls_aug_prob,
                        brightness=self.cls_aug_brightness,
                        contrast=self.cls_aug_contrast,
                        noise_std=self.cls_aug_noise_std,
                        blur_prob=self.cls_aug_blur_prob,
                    )
                logits = self.style_classifier(cls_in)
                logits = logits / max(float(self.cls_temp), 1e-6)
                logits_views.append(logits)

            logits_main = logits_views[0]
            pred_ids = logits_main.argmax(dim=1)
            probs = torch.softmax(logits_main, dim=1)
            prob_tgt = probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            prob_src = probs.gather(1, source_ids.unsqueeze(1)).squeeze(1)
            cls_target_prob = (prob_tgt * transfer_mask).sum() / transfer_denom if has_transfer else prob_tgt.mean()

            sample_prob_weight = torch.ones_like(prob_tgt)
            if self.prob_target_weights:
                w_vec = torch.tensor(
                    self.prob_target_weights,
                    device=prob_tgt.device,
                    dtype=prob_tgt.dtype,
                )
                if int(target_ids.max().item()) < int(w_vec.numel()):
                    sample_prob_weight = w_vec[target_ids]
                else:
                    max_idx = int(w_vec.numel() - 1)
                    safe_ids = target_ids.clamp(min=0, max=max_idx)
                    sample_prob_weight = w_vec[safe_ids]

            stop_conf = float(min(max(self.cls_stop_conf, 0.0), 1.0))
            min_weight = float(min(max(self.cls_hard_min_weight, 0.0), 1.0))
            if stop_conf < 1.0:
                hard_weight = ((stop_conf - prob_tgt).clamp_min(0.0) / max(stop_conf, 1e-6)).detach()
                if min_weight > 0.0:
                    hard_weight = min_weight + (1.0 - min_weight) * hard_weight
                if has_transfer:
                    cls_hard_ratio = ((prob_tgt < stop_conf).float() * transfer_mask).sum() / transfer_denom
                else:
                    cls_hard_ratio = (prob_tgt < stop_conf).float().mean()
            else:
                hard_weight = torch.ones_like(prob_tgt)
                cls_hard_ratio = torch.tensor(1.0, device=content.device, dtype=content.dtype)

            if self.w_cls > 0.0:
                ce_per = torch.zeros_like(transfer_mask)
                label_smoothing = float(min(max(self.cls_label_smoothing, 0.0), 1.0))
                for logits in logits_views:
                    ce_per = ce_per + F.cross_entropy(
                        logits,
                        target_ids,
                        reduction="none",
                        label_smoothing=label_smoothing,
                    )
                ce_per = ce_per / float(len(logits_views))

                if has_transfer:
                    ce_weight = hard_weight * transfer_mask
                    loss_style_ce = (ce_per * ce_weight).sum() / ce_weight.sum().clamp_min(1.0)
                else:
                    ce_weight = hard_weight
                    loss_style_ce = (ce_per * ce_weight).sum() / ce_weight.sum().clamp_min(1.0)
            else:
                loss_style_ce = _zero()

            if has_transfer:
                style_pred_acc = ((pred_ids == target_ids).float() * transfer_mask).sum() / transfer_denom
                xfer_margin = ((prob_tgt - prob_src) * transfer_mask).sum() / transfer_denom
            else:
                style_pred_acc = (pred_ids == target_ids).float().mean()
                xfer_margin = (prob_tgt - prob_src).mean()

            if self.prob_gate_enabled:
                gate = prob_tgt
                if self.prob_gate_power != 1.0:
                    gate = torch.pow(gate.clamp_min(1e-6), self.prob_gate_power)
                if self.prob_gate_min > 0.0:
                    gate = self.prob_gate_min + (1.0 - self.prob_gate_min) * gate
                if self.prob_gate_detach:
                    gate = gate.detach()
                transfer_weight = transfer_mask * gate
                transfer_denom = transfer_weight.sum().clamp_min(1.0)

            # Probability-level guidance (continuous signal before argmax).
            # pull: directly maximize target-domain probability.
            prob_pull = 1.0 - prob_tgt
            if self.prob_focal_gamma > 0.0:
                prob_pull = prob_pull * torch.pow(prob_pull.clamp_min(0.0), self.prob_focal_gamma)
            if has_transfer:
                prob_weight = sample_prob_weight * transfer_mask
                prob_weight_sum = prob_weight.sum().clamp_min(1.0)
                loss_prob = (prob_pull * prob_weight).sum() / prob_weight_sum
            else:
                prob_weight = sample_prob_weight
                prob_weight_sum = prob_weight.sum().clamp_min(1.0)
                loss_prob = (prob_pull * prob_weight).sum() / prob_weight_sum

            # margin: enforce p(target) - p(source) > margin.
            margin = float(max(self.prob_margin, 0.0))
            if has_transfer:
                pm_term = F.relu(margin - (prob_tgt - prob_src))
                prob_weight = sample_prob_weight * transfer_mask
                prob_weight_sum = prob_weight.sum().clamp_min(1.0)
                loss_prob_margin = (pm_term * prob_weight).sum() / prob_weight_sum
            else:
                pm_term = F.relu(margin - (prob_tgt - prob_src))
                prob_weight = sample_prob_weight
                prob_weight_sum = prob_weight.sum().clamp_min(1.0)
                loss_prob_margin = (pm_term * prob_weight).sum() / prob_weight_sum
            prob_weight_mean = prob_weight.mean()
        else:
            loss_style_ce = _zero()
            loss_prob = _zero()
            loss_prob_margin = _zero()
            style_pred_acc = _zero()
            xfer_margin = _zero()
            cls_target_prob = _zero()
            cls_hard_ratio = _zero()
            prob_weight_mean = _zero()
            prob_tgt = None
            prob_src = None
            transfer_weight = transfer_mask
            transfer_denom = transfer_weight.sum().clamp_min(1.0)

        # Optional margin push from classifier confidence difference.
        if self.w_dir > 0.0 and has_transfer and (prob_tgt is not None) and (prob_src is not None):
            dir_term = F.relu(prob_src - prob_tgt + self.dir_margin) * transfer_weight
            loss_dir = dir_term.sum() / transfer_denom
        else:
            loss_dir = _zero()

        if self.w_proto_sep > 0.0 and proto_bank.shape[0] > 1:
            loss_proto_sep = F.relu(offdiag - self.proto_cos_max).mean()
        else:
            loss_proto_sep = _zero()

        # 2) Low-frequency cycle (cross-domain only).
        w_cycle_eff = self._ramp_weight(
            self.w_cycle,
            epoch=self.current_epoch,
            warmup=self.cycle_warmup_epochs,
            ramp=self.cycle_ramp_epochs,
        )
        if w_cycle_eff > 0.0 and has_transfer:
            rec = model(
                pred_student,
                style_id=content_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
            )
            rec_lp = _lowpass(rec.float())
            content_lp = _lowpass(content.float())
            per_sample_cycle = (rec_lp - content_lp).abs().mean(dim=(1, 2, 3))
            loss_cycle = (per_sample_cycle * transfer_weight).sum() / transfer_denom
        else:
            loss_cycle = _zero()

        # 3) Light identity (optional).
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
            loss_idt = _zero()

        if self.w_same_id > 0.0:
            same_mask = 1.0 - transfer_mask
            if float(same_mask.sum().item()) > 0.0:
                per_sample_same = (pred_student.float() - content.float()).abs().mean(dim=(1, 2, 3))
                loss_same_id = (per_sample_same * same_mask).sum() / same_mask.sum().clamp_min(1.0)
            else:
                loss_same_id = _zero()
        else:
            loss_same_id = _zero()

        # Optional reference-to-student distillation/code losses.
        if self.w_distill > 0.0 and pred_teacher is not None:
            loss_distill = F.l1_loss(pred_student.float(), pred_teacher.detach().float())
        else:
            loss_distill = _zero()

        need_code = (self.w_code > 0.0) or (self.w_proto > 0.0) or (self.w_push > 0.0)
        code_teacher = None
        code_student = None
        p_tgt = None
        s_ref = None
        if need_code:
            code_student = model.encode_style(pred_student.float())
            p_tgt = model.encode_style_id(target_style_id)
            s_ref = model.encode_style(target_style.float()).detach()
            code_teacher = model.encode_style(teacher_for_code.float())

        if self.w_code > 0.0 and (code_teacher is not None) and (code_student is not None) and (p_tgt is not None) and (s_ref is not None):
            loss_code_ref = F.l1_loss(code_teacher, s_ref)
            loss_code_proto = F.l1_loss(code_student, p_tgt)
            loss_code = loss_code_ref + loss_code_proto
        else:
            loss_code_ref = _zero()
            loss_code_proto = _zero()
            loss_code = _zero()

        if self.w_proto > 0.0 and (p_tgt is not None) and (s_ref is not None):
            loss_proto = F.l1_loss(p_tgt, s_ref)
        else:
            loss_proto = _zero()

        if self.w_push > 0.0 and (code_student is not None):
            p_src = model.encode_style_id(content_style_id)
            dist_to_src = (code_student - p_src).abs().mean(dim=1)
            push_term = F.relu(self.push_margin - dist_to_src) * transfer_weight
            loss_push = push_term.sum() / transfer_denom
        else:
            loss_push = _zero()

        # Optional style anchors on student outputs.
        # Step-B change: supervise style in encoder multi-scale feature space
        # (higher-dimensional, spatially aware) instead of low-dim latent stats.
        loss_gram = _zero()
        loss_moment = _zero()
        loss_gram_hf = _zero()
        loss_moment_hf = _zero()
        loss_featmatch = _zero()
        loss_featmatch_hf = _zero()
        if (
            self.w_gram > 0.0
            or self.w_moment > 0.0
            or self.w_gram_hf > 0.0
            or self.w_moment_hf > 0.0
        ) and has_transfer:
            xfer_idx = transfer_mask > 0.5
            pred_feats = model.encode_style_feats(pred_student.float()[xfer_idx])
            with torch.no_grad():
                tgt_feats = model.encode_style_feats(target_style.float()[xfer_idx])
            start_level = max(0, min(self.style_feat_min_level, len(pred_feats)))
            pred_feats_sel = pred_feats[start_level:] if len(pred_feats) > start_level else pred_feats
            tgt_feats_sel = tgt_feats[start_level:] if len(tgt_feats) > start_level else tgt_feats
            for a, b in zip(pred_feats_sel, tgt_feats_sel):
                a_n = _norm_spatial_feat(a)
                b_n = _norm_spatial_feat(b)
                if self.w_gram > 0.0:
                    loss_gram = loss_gram + calc_gram_loss(a_n, b_n)
                if self.w_moment > 0.0:
                    loss_moment = loss_moment + calc_moment_loss(a_n, b_n)
                if self.style_feat_use_highpass and (self.w_gram_hf > 0.0 or self.w_moment_hf > 0.0):
                    a_hf = _norm_spatial_feat(_highpass_same_size(a_n))
                    b_hf = _norm_spatial_feat(_highpass_same_size(b_n))
                    if self.w_gram_hf > 0.0:
                        loss_gram_hf = loss_gram_hf + calc_gram_loss(a_hf, b_hf)
                    if self.w_moment_hf > 0.0:
                        loss_moment_hf = loss_moment_hf + calc_moment_loss(a_hf, b_hf)
            scale = 1.0 / float(max(len(pred_feats_sel), 1))
            loss_gram = loss_gram * scale
            loss_moment = loss_moment * scale
            loss_gram_hf = loss_gram_hf * scale
            loss_moment_hf = loss_moment_hf * scale

        # Direct multi-scale feature matching against target-style reference.
        # This improves texture/style strength without relying on classifier shortcuts.
        if (self.w_featmatch > 0.0 or self.w_featmatch_hf > 0.0) and has_transfer:
            xfer_idx = transfer_mask > 0.5
            pred_style_feats = model.encode_style_feats(pred_student.float()[xfer_idx])
            with torch.no_grad():
                tgt_style_feats = model.encode_style_feats(target_style.float()[xfer_idx])
            start_level = max(0, min(self.style_feat_min_level, len(pred_style_feats)))
            pred_feats_sel = pred_style_feats[start_level:] if len(pred_style_feats) > start_level else pred_style_feats
            tgt_feats_sel = tgt_style_feats[start_level:] if len(tgt_style_feats) > start_level else tgt_style_feats
            if len(pred_feats_sel) > 0:
                fm = _zero()
                for p_f, t_f in zip(pred_feats_sel, tgt_feats_sel):
                    p_n = _norm_spatial_feat(p_f)
                    t_n = _norm_spatial_feat(t_f.detach())
                    if self.w_featmatch > 0.0:
                        fm = fm + F.l1_loss(p_n, t_n)
                    if self.w_featmatch_hf > 0.0 and self.style_feat_use_highpass:
                        p_hf = _norm_spatial_feat(_highpass_same_size(p_n))
                        t_hf = _norm_spatial_feat(_highpass_same_size(t_n))
                        loss_featmatch_hf = loss_featmatch_hf + F.l1_loss(p_hf, t_hf)
                loss_featmatch = fm / float(len(pred_feats_sel))
                loss_featmatch_hf = loss_featmatch_hf / float(len(pred_feats_sel))

        loss_featmatch_teacher = _zero()
        if self.w_featmatch_teacher > 0.0 and has_transfer and pred_teacher is not None:
            xfer_idx = transfer_mask > 0.5
            pred_teacher_feats = model.encode_style_feats(pred_teacher.float()[xfer_idx])
            with torch.no_grad():
                tgt_style_feats = model.encode_style_feats(target_style.float()[xfer_idx])
            if len(pred_teacher_feats) > 0:
                fm_t = _zero()
                for p_f, t_f in zip(pred_teacher_feats, tgt_style_feats):
                    fm_t = fm_t + F.l1_loss(_norm_spatial_feat(p_f), _norm_spatial_feat(t_f.detach()))
                loss_featmatch_teacher = fm_t / float(len(pred_teacher_feats))

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
            loss_nce = _zero()

        total = (
            self.w_cls * loss_style_ce
            + self.w_prob * loss_prob
            + self.w_prob_margin * loss_prob_margin
            + w_cycle_eff * loss_cycle
            + w_idt_eff * loss_idt
            + self.w_distill * loss_distill
            + self.w_code * loss_code
            + self.w_proto * loss_proto
            + self.w_same_id * loss_same_id
            + self.w_dir * loss_dir
            + self.w_proto_sep * loss_proto_sep
            + self.w_gram * loss_gram
            + self.w_gram_hf * loss_gram_hf
            + self.w_moment * loss_moment
            + self.w_moment_hf * loss_moment_hf
            + self.w_featmatch * loss_featmatch
            + self.w_featmatch_hf * loss_featmatch_hf
            + self.w_featmatch_teacher * loss_featmatch_teacher
            + w_nce_eff * loss_nce
            + self.w_push * loss_push
        )

        return {
            "loss": total,
            "distill": loss_distill.detach(),
            "gram": loss_gram.detach(),
            "gram_hf": loss_gram_hf.detach(),
            "moment": loss_moment.detach(),
            "moment_hf": loss_moment_hf.detach(),
            "featmatch": loss_featmatch.detach(),
            "featmatch_hf": loss_featmatch_hf.detach(),
            "featmatch_teacher": loss_featmatch_teacher.detach(),
            "code": loss_code.detach(),
            "code_ref": loss_code_ref.detach(),
            "code_proto": loss_code_proto.detach(),
            "proto": loss_proto.detach(),
            "style_ce": loss_style_ce.detach(),
            "prob": loss_prob.detach(),
            "prob_margin": loss_prob_margin.detach(),
            "prob_weight_mean": prob_weight_mean.detach(),
            "cls_target_prob": cls_target_prob.detach(),
            "cls_hard_ratio": cls_hard_ratio.detach(),
            "dir": loss_dir.detach(),
            "proto_sep": loss_proto_sep.detach(),
            "style_pred_acc": style_pred_acc.detach(),
            "xfer_margin": xfer_margin.detach(),
            "proto_cos_max": proto_cos_max_metric.detach(),
            "proto_cos_mean": proto_cos_mean_metric.detach(),
            "same_id": loss_same_id.detach(),
            "push": loss_push.detach(),
            "nce": loss_nce.detach(),
            "idt": loss_idt.detach(),
            "cycle": loss_cycle.detach(),
            "w_cycle_eff": torch.tensor(w_cycle_eff, device=content.device),
            "w_nce_eff": torch.tensor(w_nce_eff, device=content.device),
            "w_idt_eff": torch.tensor(w_idt_eff, device=content.device),
            "style_ref_alpha": torch.tensor(0.0, device=content.device),
            "transfer_ratio": transfer_mask.mean().detach(),
            "transfer_weight_mean": transfer_weight.mean().detach(),
            # Backward-compatible aliases for old logs.
            "token": loss_distill.detach(),
            "token_spatial": torch.tensor(0.0, device=content.device),
        }
