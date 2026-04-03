from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from prob import StyleTokenizer
from utils.inference import encode_image, load_vae


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _find_style_emb_key(state_dict: dict[str, torch.Tensor]) -> str:
    keys = [k for k in state_dict.keys() if k.endswith("style_emb.weight")]
    if not keys:
        raise KeyError("style_emb.weight not found in checkpoint")
    return keys[0]


def _find_style_spatial_key(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = [k for k in state_dict.keys() if k.endswith("style_spatial_id_16")]
    if not keys:
        return None
    return keys[0]


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    return img.resize((size, size), Image.Resampling.BICUBIC)


def _resize_image(img: Image.Image, size: int, mode: str) -> Image.Image:
    mode = str(mode).strip().lower()
    if mode == "stretch":
        return img.resize((size, size), Image.Resampling.BICUBIC)
    if mode == "center_crop":
        return _center_crop_resize(img, size)
    raise ValueError(f"Unsupported resize_mode: {mode}")


def _pil_to_nchw_unit(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0)
    return x * 2.0 - 1.0


def extract_latents(
    *,
    image_dir: Path,
    out_dir: Path,
    image_size: int,
    resize_mode: str,
    max_images: int,
    vae_model_id: str,
    model_latent_scale: float,
    device: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(_iter_images(image_dir))
    if max_images > 0:
        images = images[: max_images]
    if not images:
        raise RuntimeError(f"No images found under: {image_dir}")

    vae = load_vae(device=device, model_id=vae_model_id)
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_latent_scale))
    scale_in = float(model_latent_scale) / max(vae_scale, 1e-8)
    if abs(scale_in - 1.0) > 1e-4:
        print(f"[latent] scale mismatch: model={model_latent_scale:.6f}, vae={vae_scale:.6f}, apply x{scale_in:.6f}")

    saved: list[Path] = []
    for i, p in enumerate(images, start=1):
        img = Image.open(p).convert("RGB")
        img = _resize_image(img, image_size, resize_mode)
        x = _pil_to_nchw_unit(img)
        z = encode_image(vae, x, device=device).detach().cpu()
        if abs(scale_in - 1.0) > 1e-4:
            z = z * scale_in
        z = z.squeeze(0).contiguous()
        dst = out_dir / f"{i:04d}_{p.stem}.pt"
        torch.save(z, dst)
        saved.append(dst)

    print(f"[latent] saved {len(saved)} latents to: {out_dir}")
    return saved


def _load_tokenizer(tokenizer_ckpt: Path, device: str) -> StyleTokenizer:
    obj = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
    style_dim = int(obj.get("style_dim", 160))
    tk = StyleTokenizer(style_dim=style_dim).to(device)
    tk.load_state_dict(obj["tokenizer_state_dict"], strict=True)
    tk.eval()
    return tk


@torch.no_grad()
def tokenizer_mean_vector(tokenizer: StyleTokenizer, latent_paths: list[Path], device: str, batch_size: int = 16) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for s in range(0, len(latent_paths), max(1, int(batch_size))):
        cur = []
        for p in latent_paths[s : s + max(1, int(batch_size))]:
            z = torch.load(p, map_location="cpu", weights_only=False)
            z = torch.as_tensor(z).float()
            if z.ndim == 3:
                z = z.unsqueeze(0)
            cur.append(z)
        x = torch.cat(cur, dim=0).to(device)
        chunks.append(tokenizer(x).float().detach().cpu())
    feats = torch.cat(chunks, dim=0)
    return feats.mean(dim=0)


def refine_style_vector(
    *,
    tokenizer: StyleTokenizer,
    init_vec: torch.Tensor,
    latent_paths: list[Path],
    style_bank: torch.Tensor,
    steps: int,
    lr: float,
    tune_last_linear: bool,
    ortho_weight: float,
    compact_weight: float,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    if steps <= 0:
        return init_vec.detach().cpu()

    for p in tokenizer.parameters():
        p.requires_grad_(False)
    if tune_last_linear:
        last = tokenizer.encoder[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError("Unexpected tokenizer tail; expected nn.Linear as last layer")
        for p in last.parameters():
            p.requires_grad_(True)

    w = nn.Parameter(init_vec.detach().to(device))
    params = [w]
    if tune_last_linear:
        params += [p for p in tokenizer.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)

    bank = style_bank.to(device).float()
    bank = bank / (bank.norm(dim=-1, keepdim=True) + 1e-8)
    tokenizer.train(tune_last_linear)

    for step in range(1, int(steps) + 1):
        ids = torch.randperm(len(latent_paths))[: max(1, min(int(batch_size), len(latent_paths)))]
        xb = []
        for idx in ids.tolist():
            z = torch.load(latent_paths[idx], map_location="cpu", weights_only=False)
            z = torch.as_tensor(z).float()
            if z.ndim == 3:
                z = z.unsqueeze(0)
            xb.append(z)
        x = torch.cat(xb, dim=0).to(device)
        pred = tokenizer(x).float()
        w_expand = w.unsqueeze(0).expand_as(pred)

        loss_compact = torch.mean((pred - w_expand) ** 2)
        pred_var = pred.var(dim=0, unbiased=False).mean()

        wn = w / (w.norm() + 1e-8)
        cos = torch.matmul(bank, wn)
        loss_ortho = (cos**2).mean()

        loss = loss_compact + float(compact_weight) * pred_var + float(ortho_weight) * loss_ortho
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % max(1, math.ceil(steps / 10)) == 0 or step == 1 or step == steps:
            print(
                f"[refine] step {step:4d}/{steps} "
                f"loss={float(loss.item()):.6f} "
                f"compact={float(loss_compact.item()):.6f} "
                f"var={float(pred_var.item()):.6f} "
                f"ortho={float(loss_ortho.item()):.6f}"
            )

    tokenizer.eval()
    return w.detach().cpu()


def patch_checkpoint_style(
    *,
    checkpoint: Path,
    out_ckpt: Path,
    replace_style_id: int,
    new_style_vec: torch.Tensor,
    append_style_name: str = "",
    append_new_style: bool = False,
) -> tuple[Path, list[str], int]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("checkpoint missing model_state_dict")
    model_sd = ckpt["model_state_dict"]
    clean_sd = _strip_compile_prefix(model_sd)
    style_key = _find_style_emb_key(clean_sd)
    spatial_key = _find_style_spatial_key(clean_sd)
    style_bank = clean_sd[style_key]

    n = int(style_bank.shape[0])
    style_names: list[str] = []
    cfg = ckpt.get("config", {})
    style_names = list(cfg.get("data", {}).get("style_subdirs", []) or [])

    if append_new_style:
        append_name = str(append_style_name).strip()
        if not append_name:
            raise ValueError("append_style_name is required when append_new_style is enabled")

        new_sid = n
        for k in list(model_sd.keys()):
            if k.endswith("style_emb.weight"):
                vec = new_style_vec.to(dtype=model_sd[k].dtype).view(1, -1)
                model_sd[k] = torch.cat([model_sd[k], vec], dim=0)
            elif k.endswith("style_spatial_id_16"):
                extra = torch.zeros_like(model_sd[k][:1])
                model_sd[k] = torch.cat([model_sd[k], extra], dim=0)

        model_cfg = cfg.setdefault("model", {})
        model_cfg["num_styles"] = int(n + 1)
        data_cfg = cfg.setdefault("data", {})
        cur_names = list(data_cfg.get("style_subdirs", []) or [])
        cur_names.append(append_name)
        data_cfg["style_subdirs"] = cur_names
        style_names = cur_names
    else:
        if replace_style_id < 0 or replace_style_id >= n:
            raise ValueError(f"replace_style_id={replace_style_id} out of range [0,{n-1}]")

        new_sid = int(replace_style_id)
        for k in list(model_sd.keys()):
            if k.endswith("style_emb.weight"):
                model_sd[k][replace_style_id] = new_style_vec.to(dtype=model_sd[k].dtype)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_ckpt)
    return out_ckpt, style_names, new_sid


def run_eval_lpips_clip_style(
    *,
    src_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    test_dir: str,
    clip_openai_model: str,
    batch_size: int,
    max_src_samples: int,
    max_ref_compare: int,
    eval_lpips_chunk_size: int,
) -> None:
    script = src_dir / "utils" / "run_evaluation.py"
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--clip_backend",
        "openai",
        "--clip_openai_model",
        str(clip_openai_model),
        "--eval_only_lpips_clip_style",
        "--batch_size",
        str(int(batch_size)),
        "--max_src_samples",
        str(int(max_src_samples)),
        "--max_ref_compare",
        str(int(max_ref_compare)),
        "--eval_lpips_chunk_size",
        str(int(eval_lpips_chunk_size)),
    ]
    if str(test_dir).strip():
        cmd += ["--test_dir", str(test_dir).strip()]

    print("[eval] CMD:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click few-shot pipeline: image->latent, tokenizer mean/refine, patch checkpoint, eval(lpips+clip_style)."
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="Base checkpoint path, e.g. epoch_0120.pt")
    parser.add_argument("--tokenizer_ckpt", required=True, type=str, help="Tokenizer ckpt from prob.py output (tokenizer.pt)")
    parser.add_argument("--fewshot_image_dir", required=True, type=str, help="Few-shot style image root (5-10 images recommended)")
    parser.add_argument("--output_dir", type=str, default="../fewshot_ukiyoe_runs", help="Output root")
    parser.add_argument("--replace_style_id", type=int, default=-1, help="Which style slot to overwrite; -1 means last style id")
    parser.add_argument("--append_new_style", action="store_true", help="Append as a brand new style slot instead of overwriting an existing one")
    parser.add_argument("--new_style_name", type=str, default="", help="Required when --append_new_style is set")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resize_mode", type=str, default="stretch", choices=["stretch", "center_crop"])
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--vae_model_id", type=str, default="sd15")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_subdir_name", type=str, default="ukiyo_e")
    parser.add_argument("--refine_steps", type=int, default=0, help=">0 to run lightweight vector refinement")
    parser.add_argument("--refine_lr", type=float, default=5e-4)
    parser.add_argument("--refine_tune_last_linear", action="store_true")
    parser.add_argument("--ortho_weight", type=float, default=0.2, help="Orthogonality regularization weight")
    parser.add_argument("--compact_weight", type=float, default=0.1, help="Prototype compactness weight")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--test_dir", type=str, default="", help="Optional override test dir for eval")
    parser.add_argument("--clip_openai_model", type=str, default="ViT-B/32")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_max_src_samples", type=int, default=30)
    parser.add_argument("--eval_max_ref_compare", type=int, default=24)
    parser.add_argument("--eval_lpips_chunk_size", type=int, default=8)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    ckpt_path = Path(args.checkpoint).resolve()
    tokenizer_ckpt = Path(args.tokenizer_ckpt).resolve()
    image_dir = Path(args.fewshot_image_dir).resolve()
    out_root = _resolve(args.output_dir, src_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not tokenizer_ckpt.exists():
        raise FileNotFoundError(f"tokenizer_ckpt not found: {tokenizer_ckpt}")
    if not image_dir.exists():
        raise FileNotFoundError(f"fewshot_image_dir not found: {image_dir}")

    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = base_ckpt.get("config", {})
    model_latent_scale = float(cfg.get("model", {}).get("latent_scale_factor", 0.18215))
    model_sd = _strip_compile_prefix(base_ckpt["model_state_dict"])
    style_key = _find_style_emb_key(model_sd)
    style_bank = model_sd[style_key].float()
    num_styles = int(style_bank.shape[0])
    style_names = list(cfg.get("data", {}).get("style_subdirs", []) or [])

    append_new_style = bool(args.append_new_style)
    replace_sid = int(args.replace_style_id)
    if append_new_style:
        replace_sid = num_styles
        if not str(args.new_style_name).strip():
            raise ValueError("--new_style_name is required when --append_new_style is enabled")
    else:
        if replace_sid < 0:
            replace_sid = num_styles - 1
        if replace_sid < 0 or replace_sid >= num_styles:
            raise ValueError(f"Invalid replace_style_id={replace_sid}, num_styles={num_styles}")

    tag = f"fewshot_{args.latent_subdir_name}_sid{replace_sid}"
    work_dir = out_root / tag
    latent_dir = work_dir / "latents" / str(args.latent_subdir_name)
    work_dir.mkdir(parents=True, exist_ok=True)

    latent_paths = extract_latents(
        image_dir=image_dir,
        out_dir=latent_dir,
        image_size=int(args.image_size),
        resize_mode=str(args.resize_mode),
        max_images=int(args.max_images),
        vae_model_id=str(args.vae_model_id),
        model_latent_scale=model_latent_scale,
        device=str(args.device),
    )

    tokenizer = _load_tokenizer(tokenizer_ckpt, device=str(args.device))
    mean_vec = tokenizer_mean_vector(tokenizer, latent_paths, device=str(args.device), batch_size=int(args.batch_size))
    final_vec = mean_vec.clone()
    print(f"[tokenizer] mean vector norm={float(final_vec.norm().item()):.6f}")

    if int(args.refine_steps) > 0:
        final_vec = refine_style_vector(
            tokenizer=tokenizer,
            init_vec=final_vec,
            latent_paths=latent_paths,
            style_bank=style_bank,
            steps=int(args.refine_steps),
            lr=float(args.refine_lr),
            tune_last_linear=bool(args.refine_tune_last_linear),
            ortho_weight=float(args.ortho_weight),
            compact_weight=float(args.compact_weight),
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        print(f"[tokenizer] refined vector norm={float(final_vec.norm().item()):.6f}")

    style_vec_path = work_dir / "style_vector.pt"
    torch.save(
        {
            "style_vector": final_vec,
            "replace_style_id": replace_sid,
            "replace_style_name": style_names[replace_sid] if 0 <= replace_sid < len(style_names) else f"style_{replace_sid}",
            "source_fewshot_dir": str(image_dir),
            "latent_dir": str(latent_dir),
        },
        style_vec_path,
    )
    print(f"[tokenizer] saved style vector: {style_vec_path}")

    patched_ckpt = work_dir / f"{ckpt_path.stem}_fewshot_{args.latent_subdir_name}.pt"
    patched_ckpt, names, final_sid = patch_checkpoint_style(
        checkpoint=ckpt_path,
        out_ckpt=patched_ckpt,
        replace_style_id=replace_sid,
        new_style_vec=final_vec,
        append_style_name=str(args.new_style_name),
        append_new_style=append_new_style,
    )
    target_name = names[final_sid] if 0 <= final_sid < len(names) else f"style_{final_sid}"
    if append_new_style:
        print(f"[patch] appended new style id {final_sid} ({target_name}). checkpoint: {patched_ckpt}")
    else:
        print(f"[patch] style id {final_sid} ({target_name}) replaced. checkpoint: {patched_ckpt}")

    meta_path = work_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_checkpoint": str(ckpt_path),
                "patched_checkpoint": str(patched_ckpt),
                "tokenizer_ckpt": str(tokenizer_ckpt),
                "fewshot_image_dir": str(image_dir),
                "latent_dir": str(latent_dir),
                "replace_style_id": final_sid,
                "replace_style_name": target_name,
                "append_new_style": append_new_style,
                "refine_steps": int(args.refine_steps),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[meta] {meta_path}")

    if bool(args.run_eval):
        eval_out = work_dir / "full_eval_lpips_clip_style"
        run_eval_lpips_clip_style(
            src_dir=src_dir,
            checkpoint=patched_ckpt,
            output_dir=eval_out,
            test_dir=str(args.test_dir),
            clip_openai_model=str(args.clip_openai_model),
            batch_size=int(args.eval_batch_size),
            max_src_samples=int(args.eval_max_src_samples),
            max_ref_compare=int(args.eval_max_ref_compare),
            eval_lpips_chunk_size=int(args.eval_lpips_chunk_size),
        )
        print(f"[done] eval summary: {eval_out / 'summary.json'}")
    else:
        print("[done] skipped eval. use --run_eval to execute lpips+clip_style evaluation.")


if __name__ == "__main__":
    main()
