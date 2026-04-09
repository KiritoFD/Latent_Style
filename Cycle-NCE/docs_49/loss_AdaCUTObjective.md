# AdaCUTObjective 模块文档

## 基本信息
- **名称**: AdaCUTObjective
- **行号**: 第 261 行
- **类型**: class
- **文件**: losses.py

## 代码
```python
class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        legacy_w_swd = float(loss_cfg.get("w_swd", 0.0))
        self.w_swd_unified = float(loss_cfg.get("w_swd_unified", 0.0))
        self.w_swd_micro = float(loss_cfg.get("w_swd_micro", 1.0 if legacy_w_swd <= 0.0 else 1.0))
        self.w_swd_macro = float(loss_cfg.get("w_swd_macro", 10.0 if legacy_w_swd <= 0.0 else legacy_w_swd))
        self.swd_use_high_freq = bool(loss_cfg.get("swd_use_high_freq", False))
        self.swd_hf_weight_ratio = max(0.0, float(loss_cfg.get("swd_hf_weight_ratio", 1.0)))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 256))
        self.swd_projection_chunk_size = int(loss_cfg.get("swd_projection_chunk_size", 64))
        self.swd_distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
        self.swd_cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 64))
        self.swd_cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
        self.swd_cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
        self.swd_cdf_bin_chunk_size = int(loss_cfg.get("swd_cdf_bin_chunk_size", 16))
        self.swd_cdf_sample_chunk_size = int(loss_cfg.get("swd_cdf_sample_chunk_size", 128))
        self.swd_tree_num_trees = int(loss_cfg.get("swd_tree_num_trees", 16))
        self.swd_tree_max_depth = int(loss_cfg.get("swd_tree_max_depth", 8))
        self.swd_batch_size = int(loss_cfg.get("swd_batch_size", 0))
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.idt_mode = str(loss_cfg.get("idt_mode", "topology")).strip().lower()
        self.w_repulsive = float(loss_cfg.get("w_repulsive", 0.0))
        self.repulsive_margin = float(loss_cfg.get("repulsive_margin", 0.5))
        self.repulsive_temperature = float(loss_cfg.get("repulsive_temperature", 0.1))
        self.repulsive_mode = str(loss_cfg.get("repulsive_mode", "l1")).strip().lower()
        self.w_color = float(loss_cfg.get("w_color", 0.0))
        self.w_aux_delta_variance = float(loss_cfg.get("w_aux_delta_variance", 0.0))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        data_cfg = config.get("data", {})
        self.group_by_content = bool(data_cfg.get("group_by_content", False))
        self.grouped_style_count = max(1, int(data_cfg.get("grouped_style_count", 1)))
        self._projection_cache: Dict[tuple[int, int, int, str, str, str], torch.Tensor] = {}
        self._sobel_kernel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_projection_bank(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mask_mode: str = "none",
    ) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        mode = str(mask_mode).strip().lower()
        for p in self.swd_patch_sizes:
            key = (int(channels), int(p), int(self.swd_num_projections), str(device), str(dtype), mode)
            w = self._projection_cache.get(key)
            if w is None:
                with torch.no_grad():
                    w = torch.randn(self.swd_num_projections, channels, p, p, device=device, dtype=dtype)
                    if mode == "luma_chroma_masked" and channels >= 2:
                        luma_count = max(1, min(self.swd_num_projections - 1, int(self.swd_num_projections * 0.6)))
                        w[:luma_count, 1:, :, :] = 0.0
                        w[luma_count:, 0:1, :, :] = 0.0
                    w = F.normalize(w.view(self.swd_num_projections, -1), p=2, dim=1).view_as(w)
                self._projection_cache[key] = w
            bank[p] = w
        return bank

    def _get_sobel_kernels(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(channels), str(device), str(dtype))
        cached = self._sobel_kernel_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        wx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        wy = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_kernel_cache[key] = (wx, wy)
        return wx, wy

    def _compute_fused_hf_feature(self, z: torch.Tensor) -> torch.Tensor:
        wx, wy = self._get_sobel_kernels(int(z.shape[1]), device=z.device, dtype=z.dtype)
        gx = F.conv2d(z, wx, padding=1, groups=int(z.shape[1]))
        gy = F.conv2d(z, wy, padding=1, groups=int(z.shape[1]))
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        return mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-5)

    def _select_xid_indices(self, xid_mask: torch.Tensor) -> torch.Tensor:
        valid = torch.nonzero(xid_mask, as_tuple=False).squeeze(1)
        if valid.numel() == 0 or self.swd_batch_size <= 0 or valid.numel() == self.swd_batch_size:
            return valid
        if valid.numel() > self.swd_batch_size:
            pick = torch.randint(0, valid.numel(), (self.swd_batch_size,), device=valid.device)
            return valid.index_select(0, pick)
        pad = torch.randint(0, valid.numel(), (self.swd_batch_size - valid.numel(),), device=valid.device)
        return torch.cat([valid, valid.index_select(0, pad)], dim=0)

    @contextmanager
    def _nvtx_range(self, name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def _compute_swd_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0:
            return None

        swd_x = pred.index_select(0, xid_idx)
        swd_y = target_style.index_select(0, xid_idx)
        indexed_style_ids = target_style_id.index_select(0, xid_idx)

        if self.w_swd_unified > 0.0:
            bank_unified = self._get_projection_bank(
                int(swd_x.shape[1]),
                device=pred.device,
                dtype=pred.dtype,
                mask_mode="luma_chroma_masked",
            )
            loss_unified = calc_swd_loss(
                swd_x,
                swd_y,
                indexed_style_ids,
                self.swd_patch_sizes,
                self.swd_num_projections,
                projection_chunk_size=self.swd_projection_chunk_size,
                distance_mode=self.swd_distance_mode,
                cdf_num_bins=self.swd_cdf_num_bins,
                cdf_tau=self.swd_cdf_tau,
                cdf_sample_size=self.swd_cdf_sample_size,
                tree_num_trees=self.swd_tree_num_trees,
                tree_max_depth=self.swd_tree_max_depth,
                projection_bank=bank_unified,
            )
            return loss_unified * self.w_swd_unified

        if self.w_swd_micro <= 0.0 and self.w_swd_macro <= 0.0:
            return None

        if swd_x.shape[1] >= 2:
            x_struct = swd_x[:, :2, :, :]
            y_struct = swd_y[:, :2, :, :]
        else:
            x_struct = swd_x
            y_struct = swd_y

        loss_micro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_micro > 0.0:
            micro_patches = [p for p in self.swd_patch_sizes if p <= 3]
            x_hp = x_struct - F.avg_pool2d(x_struct, kernel_size=5, stride=1, padding=2)
            y_hp = y_struct - F.avg_pool2d(y_struct, kernel_size=5, stride=1, padding=2)
            # Keep raw high-pass energy so SWD can see true local mean/variance shifts.
            x_micro_base = x_hp
            y_micro_base = y_hp
            if self.swd_use_high_freq:
                hf_x = self._compute_fused_hf_feature(x_micro_base)
                with torch.no_grad():
                    hf_y = self._compute_fused_hf_feature(y_micro_base)
                ratio = max(0.0, float(self.swd_hf_weight_ratio))
                x_micro = torch.cat([x_micro_base, hf_x * ratio], dim=1)
                y_micro = torch.cat([y_micro_base, hf_y * ratio], dim=1)
            else:
                x_micro = x_micro_base
                y_micro = y_micro_base

            if micro_patches:
                bank_micro = self._get_projection_bank(
                    int(x_micro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_micro = calc_swd_loss(
                    x_micro,
                    y_micro,
                    indexed_style_ids,
                    micro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    tree_num_trees=self.swd_tree_num_trees,
                    tree_max_depth=self.swd_tree_max_depth,
                    projection_bank=bank_micro,
                )

        loss_macro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_macro > 0.0:
            macro_patches = [p for p in self.swd_patch_sizes if p >= 11]
            if macro_patches:
                x_color_lp = F.avg_pool2d(swd_x, kernel_size=5, stride=1, padding=2)
                y_color_lp = F.avg_pool2d(swd_y, kernel_size=5, stride=1, padding=2)
                x_macro = x_color_lp
                y_macro = y_color_lp
                bank_macro = self._get_projection_bank(
                    int(x_macro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_macro = calc_swd_loss(
                    x_macro,
                    y_macro,
                    indexed_style_ids,
                    macro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    tree_num_trees=self.swd_tree_num_trees,
                    tree_max_depth=self.swd_tree_max_depth,
                    projection_bank=bank_macro,
                )

        return loss_micro * self.w_swd_micro + loss_macro * self.w_swd_macro

    def _compute_color_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0 or self.w_color <= 0.0:
            return None
        pred_color = pred.float().index_select(0, xid_idx)
        target_color = target_style.float().index_select(0, xid_idx)
        return calc_spatial_agnostic_color_loss(
            pred_color,
            target_color,
        )

    def _compute_identity_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        del id_mask
        if self.w_identity <= 0.0:
            return None
        return F.l1_loss(pred, content)

    def _compute_topology_alignment(
        self,
        pred_xid: torch.Tensor,
        content_xid: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.idt_mode
        if mode == "energy":
            p_energy = F.avg_pool2d(pred_xid.pow(2).mean(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
            c_energy = F.avg_pool2d(content_xid.pow(2).mean(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
            p_ref = F.instance_norm(p_energy, eps=1e-3)
            c_ref = F.instance_norm(c_energy, eps=1e-3)
            return F.l1_loss(p_ref, c_ref)

        p_ref = F.instance_norm(F.avg_pool2d(pred_xid, kernel_size=3, stride=1, padding=1), eps=1e-3)
        c_ref = F.instance_norm(F.avg_pool2d(content_xid, kernel_size=3, stride=1, padding=1), eps=1e-3)
        return F.l1_loss(p_ref, c_ref)

    def _compute_repulsive_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        xid_mask: torch.Tensor,
        target_style_id: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.w_repulsive <= 0.0 or not xid_mask.any():
            return None
        if self.group_by_content and self.grouped_style_count > 1:
            group_size = self.grouped_style_count
            num_groups = int(pred.shape[0]) // group_size
            if num_groups > 0:
                usable = num_groups * group_size
                pred_g = pred[:usable].reshape(num_groups, group_size, *pred.shape[1:])
                rolled_g = torch.roll(pred_g, shifts=1, dims=1)
                repulsive_per_pair = soft_repulsive_loss(
                    pred_g.reshape(-1, *pred.shape[1:]),
                    rolled_g.reshape(-1, *pred.shape[1:]),
                    margin=self.repulsive_margin,
                    temperature=self.repulsive_temperature,
                    dist_mode=self.repulsive_mode,
                )
                if target_style_id is not None:
                    ids_g = target_style_id[:usable].reshape(num_groups, group_size)
                    valid_mask = (ids_g != torch.roll(ids_g, shifts=1, dims=1)).reshape(-1)
                    if valid_mask.any():
                        return (repulsive_per_pair * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
                return repulsive_per_pair.mean()
        repulsive_per_sample = soft_repulsive_loss(
            pred,
            content,
            margin=self.repulsive_margin,
            temperature=self.repulsive_temperature,
            dist_mode=self.repulsive_mode,
        )
        return (repulsive_per_sample * xid_mask.float()).sum() / xid_mask.float().sum().clamp_min(1.0)

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        pred_override: torch.Tensor | None = None,
        epoch: int = 1,
        num_epochs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()
        del epoch, num_epochs

        if pred_override is None:
            with self._nvtx_range("loss/pred", nvtx_enabled):
                pred = model(
                    content,
                    style_id=target_style_id,
                    step_size=1.0,
                    style_strength=1.0,
                    target_style_latent=target_style,
                )
        else:
            pred = pred_override

        content_cast = content.to(dtype=pred.dtype)
        target_cast = target_style.to(dtype=pred.dtype)
        total = torch.tensor(0.0, device=content.device)
        metrics = {
            "identity_ratio": id_ratio.detach(),
            "sched_factor": pred.new_tensor(1.0).detach(),
        }
        xid_idx = self._select_xid_indices(xid_mask) if xid_mask.any() else None

        ls = self._compute_swd_term(pred, target_cast, target_style_id, xid_idx)
        if ls is not None:
            total = total + ls
            metrics["swd"] = ls.detach()
            metrics["_swd_raw"] = ls

        lcol = self._compute_color_term(pred, target_cast, xid_idx)
        if lcol is not None:
            lcol_weighted = self.w_color * lcol
            total = total + lcol_weighted
            metrics["color"] = lcol_weighted.detach()
            metrics["_color_raw"] = lcol

        identity_total_weighted = torch.tensor(0.0, device=content.device, dtype=pred.dtype)
        identity_total_raw = torch.tensor(0.0, device=content.device, dtype=pred.dtype)

        if self.w_identity > 0.0 and id_mask.any():
            l_idt_anchor = F.l1_loss(pred[id_mask], content_cast[id_mask])
            l_idt_anchor_weighted = l_idt_anchor * (self.w_identity * 5.0)
            total = total + l_idt_anchor_weighted
            identity_total_weighted = identity_total_weighted + l_idt_anchor_weighted
            identity_total_raw = identity_total_raw + l_idt_anchor
            metrics["idt_anchor"] = l_idt_anchor_weighted.detach()
            metrics["_idt_anchor_raw"] = l_idt_anchor

        if self.w_identity > 0.0 and xid_mask.any():
            pred_xid = pred[xid_mask]
            content_xid = content_cast[xid_mask]
            l_topo = self._compute_topology_alignment(pred_xid, content_xid)
            l_topo_weighted = l_topo * self.w_identity
            total = total + l_topo_weighted
            identity_total_weighted = identity_total_weighted + l_topo_weighted
            identity_total_raw = identity_total_raw + l_topo
            metrics["topo_align"] = l_topo_weighted.detach()
            metrics["_topo_align_raw"] = l_topo

            dist_per_sample = (pred_xid - content_xid).abs().mean(dim=(1, 2, 3))
            l_idt_repel = F.relu(pred.new_tensor(0.5) - dist_per_sample).mean()
            l_idt_repel_weighted = l_idt_repel * (self.w_identity * 2.0)
            total = total + l_idt_repel_weighted
            identity_total_weighted = identity_total_weighted + l_idt_repel_weighted
            identity_total_raw = identity_total_raw + l_idt_repel
            metrics["idt_repel"] = l_idt_repel_weighted.detach()
            metrics["_idt_repel_raw"] = l_idt_repel

        if float(identity_total_weighted.detach().item()) > 0.0:
            metrics["identity"] = identity_total_weighted.detach()
            metrics["_identity_raw"] = identity_total_raw

        lrepel = self._compute_repulsive_term(pred, content_cast, xid_mask, target_style_id=target_style_id)
        if lrepel is not None:
            lrepel_weighted = self.w_repulsive * lrepel
            total = total + lrepel_weighted
            metrics["repulsive"] = lrepel_weighted.detach()
            metrics["_repulsive_raw"] = lrepel

        last_delta = getattr(model, "last_delta", None)
        if self.w_aux_delta_variance > 0.0 and torch.is_tensor(last_delta):
            delta_variance = torch.var(last_delta.float(), dim=(2, 3), unbiased=False).mean()
            l_aux = pred.new_tensor(self.w_aux_delta_variance) * (1.0 / (delta_variance + 1e-6))
            total = total + l_aux
            metrics["aux_delta"] = l_aux.detach()
            metrics["_aux_delta_raw"] = delta_variance.detach()

        metrics["loss"] = total
        return metrics

```

## 输入输出
待补充

## 对应实验
待补充


## 完整代码实现

```python
class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        legacy_w_swd = float(loss_cfg.get("w_swd", 0.0))
        self.w_swd_unified = float(loss_cfg.get("w_swd_unified", 0.0))
        self.w_swd_micro = float(loss_cfg.get("w_swd_micro", 1.0 if legacy_w_swd <= 0.0 else 1.0))
        self.w_swd_macro = float(loss_cfg.get("w_swd_macro", 10.0 if legacy_w_swd <= 0.0 else legacy_w_swd))
        self.swd_use_high_freq = bool(loss_cfg.get("swd_use_high_freq", False))
        self.swd_hf_weight_ratio = max(0.0, float(loss_cfg.get("swd_hf_weight_ratio", 1.0)))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 256))
        self.swd_projection_chunk_size = int(loss_cfg.get("swd_projection_chunk_size", 64))
        self.swd_distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
        self.swd_cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 64))
        self.swd_cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
        self.swd_cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
        self.swd_cdf_bin_chunk_size = int(loss_cfg.get("swd_cdf_bin_chunk_size", 16))
        self.swd_cdf_sample_chunk_size = int(loss_cfg.get("swd_cdf_sample_chunk_size", 128))
        self.swd_tree_num_trees = int(loss_cfg.get("swd_tree_num_trees", 16))
        self.swd_tree_max_depth = int(loss_cfg.get("swd_tree_max_depth", 8))
        self.swd_batch_size = int(loss_cfg.get("swd_batch_size", 0))
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.idt_mode = str(loss_cfg.get("idt_mode", "topology")).strip().lower()
        self.w_repulsive = float(loss_cfg.get("w_repulsive", 0.0))
        self.repulsive_margin = float(loss_cfg.get("repulsive_margin", 0.5))
        self.repulsive_temperature = float(loss_cfg.get("repulsive_temperature", 0.1))
        self.repulsive_mode = str(loss_cfg.get("repulsive_mode", "l1")).strip().lower()
        self.w_color = float(loss_cfg.get("w_color", 0.0))
        self.w_aux_delta_variance = float(loss_cfg.get("w_aux_delta_variance", 0.0))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        data_cfg = config.get("data", {})
        self.group_by_content = bool(data_cfg.get("group_by_content", False))
        self.grouped_style_count = max(1, int(data_cfg.get("grouped_style_count", 1)))
        self._projection_cache: Dict[tuple[int, int, int, str, str, str], torch.Tensor] = {}
        self._sobel_kernel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_projection_bank(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mask_mode: str = "none",
    ) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        mode = str(mask_mode).strip().lower()
        for p in self.swd_patch_sizes:
            key = (int(channels), int(p), int(self.swd_num_projections), str(device), str(dtype), mode)
            w = self._projection_cache.get(key)
            if w is None:
                with torch.no_grad():
                    w = torch.randn(self.swd_num_projections, channels, p, p, device=device, dtype=dtype)
                    if mode == "luma_chroma_masked" and channels >= 2:
                        luma_count = max(1, min(self.swd_num_projections - 1, int(self.swd_num_projections * 0.6)))
                        w[:luma_count, 1:, :, :] = 0.0
                        w[luma_count:, 0:1, :, :] = 0.0
                    w = F.normalize(w.view(self.swd_num_projections, -1), p=2, dim=1).view_as(w)
                self._projection_cache[key] = w
            bank[p] = w
        return bank

    def _get_sobel_kernels(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(channels), str(device), str(dtype))
        cached = self._sobel_kernel_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        wx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        wy = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_kernel_cache[key] = (wx, wy)
        return wx, wy

    def _compute_fused_hf_feature(self, z: torch.Tensor) -> torch.Tensor:
        wx, wy = self._get_sobel_kernels(int(z.shape[1]), device=z.device, dtype=z.dtype)
        gx = F.conv2d(z, wx, padding=1, groups=int(z.shape[1]))
        gy = F.conv2d(z, wy, padding=1, groups=int(z.shape[1]))
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        return mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-5)

    def _select_xid_indices(self, xid_mask: torch.Tensor) -> torch.Tensor:
        valid = torch.nonzero(xid_mask, as_tuple=False).squeeze(1)
        if valid.numel() == 0 or self.swd_batch_size <= 0 or valid.numel() == self.swd_batch_size:
            return valid
        if valid.numel() > self.swd_batch_size:
            pick = torch.randint(0, valid.numel(), (self.swd_batch_size,), device=valid.device)
            return valid.index_select(0, pick)
        pad = torch.randint(0, valid.numel(), (self.swd_batch_size - valid.numel(),), device=valid.device)
        return torch.cat([valid, valid.index_select(0, pad)], dim=0)

    @contextmanager
    def _nvtx_range(self, name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def _compute_swd_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0:
            return None

        swd_x = pred.index_select(0, xid_idx)
        swd_y = target_style.index_select(0, xid_idx)
        indexed_style_ids = target_style_id.index_select(0, xid_idx)

        if self.w_swd_unified > 0.0:
            bank_unified = self._get_projection_bank(
                int(swd_x.shape[1]),
                device=pred.device,
                dtype=pred.dtype,
                mask_mode="luma_chroma_masked",
            )
            loss_unified = calc_swd_loss(
                swd_x,
                swd_y,
                indexed_style_ids,
                self.swd_patch_sizes,
                self.swd_num_projections,
                projection_chunk_size=self.swd_projection_chunk_size,
                distance_mode=self.swd_distance_mode,
                cdf_num_bins=self.swd_cdf_num_bins,
                cdf_tau=self.swd_cdf_tau,
                cdf_sample_size=self.swd_cdf_sample_size,
                tree_num_trees=self.swd_tree_num_trees,
                tree_max_depth=self.swd_tree_max_depth,
                projection_bank=bank_unified,
            )
            return loss_unified * self.w_swd_unified

        if self.w_swd_micro <= 0.0 and self.w_swd_macro <= 0.0:
            return None

        if swd_x.shape[1] >= 2:
            x_struct = swd_x[:, :2, :, :]
            y_struct = swd_y[:, :2, :, :]
        else:
            x_struct = swd_x
            y_struct = swd_y

        loss_micro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_micro > 0.0:
            micro_patches = [p for p in self.swd_patch_sizes if p <= 3]
            x_hp = x_struct - F.avg_pool2d(x_struct, kernel_size=5, stride=1, padding=2)
            y_hp = y_struct - F.avg_pool2d(y_struct, kernel_size=5, stride=1, padding=2)
            # Keep raw high-pass energy so SWD can see true local mean/variance shifts.
            x_micro_base = x_hp
            y_micro_base = y_hp
            if self.swd_use_high_freq:
                hf_x = self._compute_fused_hf_feature(x_micro_base)
                with torch.no_grad():
                    hf_y = self._compute_fused_hf_feature(y_micro_base)
                ratio = max(0.0, float(self.swd_hf_weight_ratio))
                x_micro = torch.cat([x_micro_base, hf_x * ratio], dim=1)
                y_micro = torch.cat([y_micro_base, hf_y * ratio], dim=1)
            else:
                x_micro = x_micro_base
                y_micro = y_micro_base

            if micro_patches:
                bank_micro = self._get_projection_bank(
                    int(x_micro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_micro = calc_swd_loss(
                    x_micro,
                    y_micro,
                    indexed_style_ids,
                    micro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    tree_num_trees=self.swd_tree_num_trees,
                    tree_max_depth=self.swd_tree_max_depth,
                    projection_bank=bank_micro,
                )

        loss_macro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_macro > 0.0:
            macro_patches = [p for p in self.swd_patch_sizes if p >= 11]
            if macro_patches:
                x_color_lp = F.avg_pool2d(swd_x, kernel_size=5, stride=1, padding=2)
                y_color_lp = F.avg_pool2d(swd_y, kernel_size=5, stride=1, padding=2)
                x_macro = x_color_lp
                y_macro = y_color_lp
                bank_macro = self._get_projection_bank(
                    int(x_macro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_macro = calc_swd_loss(
                    x_macro,
                    y_macro,
                    indexed_style_ids,
                    macro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    tree_num_trees=self.swd_tree_num_trees,
                    tree_max_depth=self.swd_tree_max_depth,
                    projection_bank=bank_macro,
                )

        return loss_micro * self.w_swd_micro + loss_macro * self.w_swd_macro

    def _compute_color_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0 or self.w_color <= 0.0:
            return None
        pred_color = pred.float().index_select(0, xid_idx)
        target_color = target_style.float().index_select(0, xid_idx)
        return calc_spatial_agnostic_color_loss(
            pred_color,
            target_color,
        )

    def _compute_identity_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        del id_mask
        if self.w_identity <= 0.0:
            return None
        return F.l1_loss(pred, content)

    def _compute_topology_alignment(
        self,
        pred_xid: torch.Tensor,
        content_xid: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.idt_mode
        if mode == "energy":
            p_energy = F.avg_pool2d(pred_xid.pow(2).mean(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
            c_energy = F.avg_pool2d(content_xid.pow(2).mean(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
            p_ref = F.instance_norm(p_energy, eps=1e-3)
            c_ref = F.instance_norm(c_energy, eps=1e-3)
            return F.l1_loss(p_ref, c_ref)

        p_ref = F.instance_norm(F.avg_pool2d(pred_xid, kernel_size=3, stride=1, padding=1), eps=1e-3)
        c_ref = F.instance_norm(F.avg_pool2d(content_xid, kernel_size=3, stride=1, padding=1), eps=1e-3)
        return F.l1_loss(p_ref, c_ref)

    def _compute_repulsive_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        xid_mask: torch.Tensor,
        target_style_id: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.w_repulsive <= 0.0 or not xid_mask.any():
            return None
        if self.group_by_content and self.grouped_style_count > 1:
            group_size = self.grouped_style_count
            num_groups = int(pred.shape[0]) // group_size
            if num_groups > 0:
                usable = num_groups * group_size
                pred_g = pred[:usable].reshape(num_groups, group_size, *pred.shape[1:])
                rolled_g = torch.roll(pred_g, shifts=1, dims=1)
                repulsive_per_pair = soft_repulsive_loss(
                    pred_g.reshape(-1, *pred.shape[1:]),
                    rolled_g.reshape(-1, *pred.shape[1:]),
                    margin=self.repulsive_margin,
                    temperature=self.repulsive_temperature,
                    dist_mode=self.repulsive_mode,
                )
                if target_style_id is not None:
                    ids_g = target_style_id[:usable].reshape(num_groups, group_size)
                    valid_mask = (ids_g != torch.roll(ids_g, shifts=1, dims=1)).reshape(-1)
                    if valid_mask.any():
                        return (repulsive_per_pair * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
                return repulsive_per_pair.mean()
        repulsive_per_sample = soft_repulsive_loss(
            pred,
            content,
            margin=self.repulsive_margin,
            temperature=self.repulsive_temperature,
            dist_mode=self.repulsive_mode,
        )
        return (repulsive_per_sample * xid_mask.float()).sum() / xid_mask.float().sum().clamp_min(1.0)

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        pred_override: torch.Tensor | None = None,
        epoch: int = 1,
        num_epochs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()
        del epoch, num_epochs

        if pred_override is None:
            with self._nvtx_range("loss/pred", nvtx_enabled):
                pred = model(
                    content,
                    style_id=target_style_id,
                    step_size=1.0,
                    style_strength=1.0,
                    target_style_latent=target_style,
                )
        else:
            pred = pred_override

        content_cast = content.to(dtype=pred.dtype)
        target_cast = target_style.to(dtype=pred.dtype)
        total = torch.tensor(0.0, device=content.device)
        metrics = {
            "identity_ratio": id_ratio.detach(),
            "sched_factor": pred.new_tensor(1.0).detach(),
        }
        xid_idx = self._select_xid_indices(xid_mask) if xid_mask.any() else None

        ls = self._compute_swd_term(pred, target_cast, target_style_id, xid_idx)
        if ls is not None:
            total = total + ls
            metrics["swd"] = ls.detach()
            metrics["_swd_raw"] = ls

        lcol = self._compute_color_term(pred, target_cast, xid_idx)
        if lcol is not None:
            lcol_weighted = self.w_color * lcol
            total = total + lcol_weighted
            metrics["color"] = lcol_weighted.detach()
            metrics["_color_raw"] = lcol

        identity_total_weighted = torch.tensor(0.0, device=content.device, dtype=pred.dtype)
        identity_total_raw = torch.tensor(0.0, device=content.device, dtype=pred.dtype)

        if self.w_identity > 0.0 and id_mask.any():
            l_idt_anchor = F.l1_loss(pred[id_mask], content_cast[id_mask])
            l_idt_anchor_weighted = l_idt_anchor * (self.w_identity * 5.0)
            total = total + l_idt_anchor_weighted
            identity_total_weighted = identity_total_weighted + l_idt_anchor_weighted
            identity_total_raw = identity_total_raw + l_idt_anchor
            metrics["idt_anchor"] = l_idt_anchor_weighted.detach()
            metrics["_idt_anchor_raw"] = l_idt_anchor

        if self.w_identity > 0.0 and xid_mask.any():
            pred_xid = pred[xid_mask]
            content_xid = content_cast[xid_mask]
            l_topo = self._compute_topology_alignment(pred_xid, content_xid)
            l_topo_weighted = l_topo * self.w_identity
            total = total + l_topo_weighted
            identity_total_weighted = identity_total_weighted + l_topo_weighted
            identity_total_raw = identity_total_raw + l_topo
            metrics["topo_align"] = l_topo_weighted.detach()
            metrics["_topo_align_raw"] = l_topo

            dist_per_sample = (pred_xid - content_xid).abs().mean(dim=(1, 2, 3))
            l_idt_repel = F.relu(pred.new_tensor(0.5) - dist_per_sample).mean()
            l_idt_repel_weighted = l_idt_repel * (self.w_identity * 2.0)
            total = total + l_idt_repel_weighted
            identity_total_weighted = identity_total_weighted + l_idt_repel_weighted
            identity_total_raw = identity_total_raw + l_idt_repel
            metrics["idt_repel"] = l_idt_repel_weighted.detach()
            metrics["_idt_repel_raw"] = l_idt_repel

        if float(identity_total_weighted.detach().item()) > 0.0:
            metrics["identity"] = identity_total_weighted.detach()
            metrics["_identity_raw"] = identity_total_raw

        lrepel = self._compute_repulsive_term(pred, content_cast, xid_mask, target_style_id=target_style_id)
        if lrepel is not None:
            lrepel_weighted = self.w_repulsive * lrepel
            total = total + lrepel_weighted
            metrics["repulsive"] = lrepel_weighted.detach()
            metrics["_repulsive_raw"] = lrepel

        last_delta = getattr(model, "last_delta", None)
        if self.w_aux_delta_variance > 0.0 and torch.is_tensor(last_delta):
            delta_variance = torch.var(last_delta.float(), dim=(2, 3), unbiased=False).mean()
            l_aux = pred.new_tensor(self.w_aux_delta_variance) * (1.0 / (delta_variance + 1e-6))
            total = total + l_aux
            metrics["aux_delta"] = l_aux.detach()
            metrics["_aux_delta_raw"] = delta_variance.detach()

        metrics["loss"] = total
        return metrics

```
