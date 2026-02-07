import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import lpips
except ImportError:
    lpips = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_latent(path: Path, device: str) -> torch.Tensor:
    latent = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(latent, dict):
        latent = latent.get("latent", latent)
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    return latent.float().to(device)


def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _sobel_edges(images: torch.Tensor) -> torch.Tensor:
    # images: [B,3,H,W] in [0,1]
    gray = images.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def edge_sobel_l1(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    e_pred = _sobel_edges(pred)
    e_ref = _sobel_edges(ref)
    return (e_pred - e_ref).abs().mean(dim=(1, 2, 3))


def compute_hf_ratio(images: torch.Tensor, radius_ratio: float = 0.35) -> torch.Tensor:
    # images: [B,C,H,W]
    b, c, h, w = images.shape
    x = images.float()
    fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    power = fft.abs().square().mean(dim=1)  # [B,H,W]

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device),
        torch.linspace(-1.0, 1.0, w, device=x.device),
        indexing="ij",
    )
    rr = torch.sqrt(xx.square() + yy.square())
    hf_mask = rr > float(radius_ratio)
    lf_mask = ~hf_mask

    hf = power[:, hf_mask].mean(dim=1)
    lf = power[:, lf_mask].mean(dim=1)
    return hf / (lf + 1e-8)


def compute_patch_consistency_var(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    patch_size: int = 4,
) -> torch.Tensor:
    # x_*: [B,C,H,W] (latent or image)
    diff = (x_a.float() - x_b.float()).square().mean(dim=1, keepdim=True)  # [B,1,H,W]
    h, w = diff.shape[-2:]
    k = max(1, min(patch_size, h, w))
    patches = F.unfold(diff, kernel_size=k, stride=k)  # [B, k*k, N]
    patch_means = patches.mean(dim=1)  # [B, N]
    return patch_means.var(dim=1, unbiased=False)


def compute_margin_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.float(), dim=1)
    p_t = probs.gather(1, target_ids.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask.scatter_(1, target_ids.view(-1, 1), False)
    p_not = probs.masked_fill(~mask, -1.0).max(dim=1).values
    margin = p_t - p_not
    return margin, p_t, p_not


def _pairwise_lpips_mean(loss_fn, images: torch.Tensor) -> float:
    # images: [S,3,H,W]
    s = images.shape[0]
    if s <= 1:
        return 0.0
    pairs = []
    for i in range(s):
        for j in range(i + 1, s):
            d = loss_fn(_to_lpips_input(images[i : i + 1]), _to_lpips_input(images[j : j + 1]))
            pairs.append(float(d.mean().item()))
    return float(sum(pairs) / max(len(pairs), 1))


def _fallback_l2_mean(images: torch.Tensor) -> float:
    s = images.shape[0]
    if s <= 1:
        return 0.0
    pairs = []
    for i in range(s):
        for j in range(i + 1, s):
            pairs.append(float((images[i] - images[j]).square().mean().item()))
    return float(sum(pairs) / max(len(pairs), 1))


def _load_classifier(
    config: Dict,
    classifier_path: Path,
    device: str,
) -> "StyleClassifier":
    from utils.style_classifier import StyleClassifier

    num_classes = int(config.get("model", {}).get("num_styles", 4))
    in_channels = int(config.get("model", {}).get("latent_channels", 4))
    classifier = StyleClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        use_stats=bool(config.get("loss", {}).get("style_classifier_use_stats", True)),
        use_gram=bool(config.get("loss", {}).get("style_classifier_use_gram", True)),
        use_lowpass_stats=bool(config.get("loss", {}).get("style_classifier_use_lowpass_stats", True)),
        spatial_shuffle=bool(config.get("loss", {}).get("style_classifier_spatial_shuffle", True)),
        input_size_train=int(config.get("loss", {}).get("style_classifier_input_size_train", 8)),
        input_size_infer=int(config.get("loss", {}).get("style_classifier_input_size_infer", 8)),
        lowpass_size=int(config.get("loss", {}).get("style_classifier_lowpass_size", 8)),
    ).to(device)

    state = torch.load(classifier_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    return classifier


def _make_decision_flags(summary: Dict, baseline: Optional[Dict]) -> Dict:
    if baseline is None:
        return {"has_baseline": False}

    def _safe_delta(cur: float, base: float) -> float:
        return (cur - base) / (base + 1e-8)

    delta_margin = float(summary["margin_mean"] - baseline["margin_mean"])
    identity_deg = float(_safe_delta(summary["identity_mse_latent_mean"], baseline["identity_mse_latent_mean"]))
    delta_hf = float(_safe_delta(summary["hf_ratio_mean"], baseline["hf_ratio_mean"]))
    delta_div = float(summary["diversity_lpips_across_styles_mean"] - baseline["diversity_lpips_across_styles_mean"])

    is_effective = (
        delta_margin >= 0.05
        and identity_deg <= 0.10
        and delta_hf <= 0.05
        and delta_div > 0.0
    )
    noise_shortcut = delta_margin >= 0.05 and delta_hf > 0.05
    structure_break = identity_deg > 0.10

    return {
        "has_baseline": True,
        "delta_margin_mean": delta_margin,
        "identity_degradation_ratio": identity_deg,
        "delta_hf_ratio_pct": delta_hf,
        "delta_diversity_lpips": delta_div,
        "is_effective": bool(is_effective),
        "noise_shortcut": bool(noise_shortcut),
        "structure_break": bool(structure_break),
    }


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWD validation metrics on a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--classifier_path", type=str, default=None)
    parser.add_argument("--baseline_summary", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_radius_ratio", type=float, default=0.35)
    parser.add_argument("--consistency_patch_size", type=int, default=4)
    parser.add_argument("--max_control", type=int, default=0)
    parser.add_argument("--max_identity", type=int, default=0)
    args = parser.parse_args()

    from utils.inference import LGTInference, decode_latent, load_vae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = _repo_root()

    checkpoint_path = _resolve_path(args.checkpoint, Path.cwd())
    manifest_path = _resolve_path(args.manifest, Path.cwd())
    output_dir = _resolve_path(args.output, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    styles = list(manifest["styles"])
    style_to_id = {k: int(v) for k, v in manifest["style_to_id"].items()}
    num_styles = len(styles)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = int(config.get("inference", {}).get("num_steps", 20))

    lgt = LGTInference(
        str(checkpoint_path),
        device=device,
        temperature_lambda=float(config.get("inference", {}).get("temperature_lambda", 0.3)),
        temperature_threshold=float(config.get("inference", {}).get("temperature_threshold", 0.5)),
        use_cfg=bool(config.get("inference", {}).get("use_cfg", True)),
        cfg_scale=float(config.get("inference", {}).get("cfg_scale", 5.0)),
        num_steps=num_steps,
        use_source_repulsion=bool(config.get("inference", {}).get("use_source_repulsion", False)),
        repulsion_strength=float(config.get("inference", {}).get("repulsion_strength", 0.7)),
    )
    decode_available = True
    vae = None
    try:
        vae = load_vae(device)
    except Exception as exc:
        decode_available = False
        print(f"WARNING: failed to load VAE decoder, fallback to latent-proxy metrics. reason={exc}")

    def _decode_proxy(latent: torch.Tensor) -> torch.Tensor:
        if decode_available and vae is not None:
            return decode_latent(vae, latent, device).float()
        x = latent.float()
        if x.shape[1] < 3:
            repeat = (3 + x.shape[1] - 1) // x.shape[1]
            x = x.repeat(1, repeat, 1, 1)
        img = x[:, :3]
        lo = img.amin(dim=(1, 2, 3), keepdim=True)
        hi = img.amax(dim=(1, 2, 3), keepdim=True)
        return (img - lo) / (hi - lo + 1e-6)

    classifier_path = args.classifier_path
    if classifier_path is None:
        classifier_path = config.get("loss", {}).get("style_classifier_ckpt", "../style_classifier.pt")
    classifier_ckpt = _resolve_path(str(classifier_path), Path(__file__).resolve().parent)
    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Style classifier checkpoint not found: {classifier_ckpt}")
    classifier = _load_classifier(config, classifier_ckpt, device)
    cls_input_size = int(config.get("loss", {}).get("style_classifier_input_size_infer", 0))

    if lpips is not None and decode_available:
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
    else:
        lpips_fn = None
        print("WARNING: LPIPS unavailable (or decode disabled). Falling back to L2 for diversity/content metrics.")

    rows: List[Dict] = []
    diversity_values: List[float] = []
    margins: List[float] = []
    margin_pos: List[float] = []
    content_vals: List[float] = []
    hf_vals: List[float] = []
    consistency_vals: List[float] = []
    identity_mse_vals: List[float] = []
    identity_content_vals: List[float] = []
    edge_vals: List[float] = []

    control_subset = list(manifest.get("control_subset", []))
    identity_subset = list(manifest.get("identity_subset", []))
    if args.max_control > 0:
        control_subset = control_subset[: args.max_control]
    if args.max_identity > 0:
        identity_subset = identity_subset[: args.max_identity]

    # Control + consistency pass.
    for idx, item in enumerate(control_subset):
        latent_path = _resolve_path(item["latent_path"], repo_root)
        src_style = item["source_style"]
        src_style_id = style_to_id[src_style]
        latent_src = _load_latent(latent_path, device)

        with torch.no_grad():
            latent_x0 = lgt.inversion(latent_src, src_style_id, num_steps=num_steps)
            tgt_ids = torch.arange(num_styles, device=device, dtype=torch.long)
            src_ids = torch.full((num_styles,), src_style_id, device=device, dtype=torch.long)
            x0_batch = latent_x0.repeat(num_styles, 1, 1, 1)

            _set_seed(args.seed + idx * 2 + 1)
            gen_latent_1 = lgt.generation(
                x0_batch,
                tgt_ids,
                num_steps=num_steps,
                source_style_id=src_ids if lgt.sampler.use_source_repulsion else None,
            )
            _set_seed(args.seed + idx * 2 + 2)
            gen_latent_2 = lgt.generation(
                x0_batch,
                tgt_ids,
                num_steps=num_steps,
                source_style_id=src_ids if lgt.sampler.use_source_repulsion else None,
            )

            gen_img_1 = _decode_proxy(gen_latent_1)
            src_img = _decode_proxy(latent_src).repeat(num_styles, 1, 1, 1)

            cls_inputs = gen_latent_1
            if cls_input_size and (
                cls_inputs.shape[-1] != cls_input_size or cls_inputs.shape[-2] != cls_input_size
            ):
                cls_inputs = F.interpolate(cls_inputs, size=(cls_input_size, cls_input_size), mode="area")
            logits = classifier(cls_inputs)
            margin, p_t, p_not = compute_margin_from_logits(logits, tgt_ids)

            if lpips_fn is not None:
                content_lpips = lpips_fn(_to_lpips_input(gen_img_1), _to_lpips_input(src_img)).view(-1)
                diversity = _pairwise_lpips_mean(lpips_fn, gen_img_1)
            else:
                content_lpips = (gen_img_1 - src_img).square().mean(dim=(1, 2, 3))
                diversity = _fallback_l2_mean(gen_img_1)

            hf_ratio = compute_hf_ratio(gen_img_1, radius_ratio=args.hf_radius_ratio)
            consistency_var = compute_patch_consistency_var(
                gen_latent_1, gen_latent_2, patch_size=args.consistency_patch_size
            )
            edge_l1 = edge_sobel_l1(gen_img_1, src_img)

            diversity_values.append(float(diversity))
            for t_idx, tgt_style in enumerate(styles):
                m = float(margin[t_idx].item())
                cp = float(content_lpips[t_idx].item())
                hf = float(hf_ratio[t_idx].item())
                cv = float(consistency_var[t_idx].item())
                edge = float(edge_l1[t_idx].item())
                margins.append(m)
                margin_pos.append(1.0 if m > 0.0 else 0.0)
                content_vals.append(cp)
                hf_vals.append(hf)
                consistency_vals.append(cv)
                edge_vals.append(edge)

                rows.append(
                    {
                        "subset": "control",
                        "source_style": src_style,
                        "target_style": tgt_style,
                        "latent_path": item["latent_path"],
                        "margin": m,
                        "margin_positive": 1 if m > 0.0 else 0,
                        "p_target": float(p_t[t_idx].item()),
                        "p_not_target": float(p_not[t_idx].item()),
                        "content_lpips": cp,
                        "diversity_lpips_across_styles": float(diversity),
                        "identity_mse_latent": "",
                        "edge_sobel_l1": edge,
                        "hf_ratio": hf,
                        "patch_consistency_var": cv,
                    }
                )

    # Identity pass.
    for idx, item in enumerate(identity_subset):
        latent_path = _resolve_path(item["latent_path"], repo_root)
        src_style = item["source_style"]
        src_style_id = style_to_id[src_style]
        latent_src = _load_latent(latent_path, device)

        with torch.no_grad():
            latent_x0 = lgt.inversion(latent_src, src_style_id, num_steps=num_steps)
            tgt_ids = torch.tensor([src_style_id], device=device, dtype=torch.long)
            src_ids = torch.tensor([src_style_id], device=device, dtype=torch.long)
            _set_seed(args.seed + 100000 + idx)
            gen_latent = lgt.generation(
                latent_x0,
                tgt_ids,
                num_steps=num_steps,
                source_style_id=src_ids if lgt.sampler.use_source_repulsion else None,
            )
            gen_img = _decode_proxy(gen_latent)
            src_img = _decode_proxy(latent_src)

            mse_latent = F.mse_loss(gen_latent.float(), latent_src.float()).item()
            if lpips_fn is not None:
                content_lpips = float(lpips_fn(_to_lpips_input(gen_img), _to_lpips_input(src_img)).mean().item())
            else:
                content_lpips = float((gen_img - src_img).square().mean().item())
            edge_l1 = float(edge_sobel_l1(gen_img, src_img).mean().item())
            hf_ratio = float(compute_hf_ratio(gen_img, radius_ratio=args.hf_radius_ratio).mean().item())

            identity_mse_vals.append(mse_latent)
            identity_content_vals.append(content_lpips)
            edge_vals.append(edge_l1)
            hf_vals.append(hf_ratio)

            rows.append(
                {
                    "subset": "identity",
                    "source_style": src_style,
                    "target_style": src_style,
                    "latent_path": item["latent_path"],
                    "margin": "",
                    "margin_positive": "",
                    "p_target": "",
                    "p_not_target": "",
                    "content_lpips": content_lpips,
                    "diversity_lpips_across_styles": "",
                    "identity_mse_latent": mse_latent,
                    "edge_sobel_l1": edge_l1,
                    "hf_ratio": hf_ratio,
                    "patch_consistency_var": "",
                }
            )

    csv_path = output_dir / "metrics_per_sample.csv"
    fieldnames = [
        "subset",
        "source_style",
        "target_style",
        "latent_path",
        "margin",
        "margin_positive",
        "p_target",
        "p_not_target",
        "content_lpips",
        "diversity_lpips_across_styles",
        "identity_mse_latent",
        "edge_sobel_l1",
        "hf_ratio",
        "patch_consistency_var",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "control_subset": len(control_subset),
            "identity_subset": len(identity_subset),
            "rows": len(rows),
        },
        "margin_mean": _mean(margins),
        "margin_pos_rate": _mean(margin_pos),
        "diversity_lpips_across_styles_mean": _mean(diversity_values),
        "identity_mse_latent_mean": _mean(identity_mse_vals),
        "content_lpips_mean": _mean(content_vals),
        "identity_content_lpips_mean": _mean(identity_content_vals),
        "edge_sobel_l1_mean": _mean(edge_vals),
        "hf_ratio_mean": _mean(hf_vals),
        "patch_consistency_var_mean": _mean(consistency_vals),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    baseline = None
    if args.baseline_summary:
        baseline_path = _resolve_path(args.baseline_summary, Path.cwd())
        if baseline_path.exists():
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)

    decision = _make_decision_flags(summary, baseline)
    decision_path = output_dir / "decision_flags.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(f"Wrote per-sample metrics: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote decision flags: {decision_path}")


if __name__ == "__main__":
    main()
