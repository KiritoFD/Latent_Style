from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from tqdm import tqdm

try:
    import lpips
except ImportError:
    lpips = None

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None

torch.set_float32_matmul_precision("high")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference import LGTInference, decode_latent, encode_image, load_vae
from utils.style_classifier import StyleClassifier as LatentStyleClassifier

_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _to_lpips_input(img: torch.Tensor) -> torch.Tensor:
    return img * 2.0 - 1.0


def _save_image_task(tensor_cpu: torch.Tensor, path: Path) -> None:
    try:
        save_image(tensor_cpu, path)
    except Exception as exc:
        print(f"Error saving {path}: {exc}")


def _extract_clip_embeddings(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, dict):
        if "image_embeds" in output:
            return output["image_embeds"]
        if "pooler_output" in output:
            return output["pooler_output"]
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Cannot extract CLIP embedding from type: {type(output)}")


def _load_eval_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    return T.ToTensor()(Image.open(path).convert("RGB").resize((size, size)))


def _resolve_path(raw: str | Path | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    p = Path(txt)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_auto_config_path(config_arg: str | None) -> Path:
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return p
    candidates = [
        (Path.cwd() / "config.json").resolve(),
        (Path.cwd() / "src" / "config.json").resolve(),
        (_ROOT / "config.json").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No config.json found in current workspace")


def _parse_epoch(path: Path) -> int | None:
    m = re.match(r"epoch_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else None


def _eval_done(out_dir: Path) -> bool:
    return (out_dir / "metrics.csv").exists() and (out_dir / "summary.json").exists()


def _sanitize_ref_cache(cache_obj: Any, style_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    out = {sid: [] for sid in style_ids}
    if not isinstance(cache_obj, dict):
        return out
    for sid in style_ids:
        raw = cache_obj.get(sid, cache_obj.get(str(sid), []))
        if not isinstance(raw, list):
            continue
        clean: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                clean.append({"path": item, "clip": None})
                continue
            if not isinstance(item, dict):
                continue
            p = item.get("path")
            if not p:
                continue
            clip = item.get("clip")
            if torch.is_tensor(clip):
                clip = clip.detach().cpu()
                if clip.ndim == 1:
                    clip = clip.unsqueeze(0)
                elif clip.ndim != 2:
                    clip = clip.reshape(1, -1)
            else:
                clip = None
            clean.append({"path": str(p), "clip": clip})
        out[sid] = clean
    return out


def _pick(arg_value: Any, cfg_value: Any, default: Any) -> Any:
    if arg_value is not None:
        return arg_value
    if cfg_value is not None:
        return cfg_value
    return default


def _resolve_schedule(arg_schedule: str | None, cfg_schedule: Any) -> str | None:
    if arg_schedule is not None:
        txt = str(arg_schedule).strip()
        return None if txt.lower() in {"", "none"} else txt
    if cfg_schedule is None:
        return None
    if isinstance(cfg_schedule, (list, tuple)):
        return ",".join(str(float(v)) for v in cfg_schedule) if cfg_schedule else None
    txt = str(cfg_schedule).strip()
    return None if txt.lower() in {"", "none"} else txt

def _run_single_evaluation(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_base_dir = Path.cwd()
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        config_base_dir = cfg_path.parent

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    cfg_train = cfg.get("training", {})
    cfg_infer = cfg.get("inference", {})

    test_dir_raw = _pick(args.test_dir, cfg_train.get("test_image_dir"), None)
    if not test_dir_raw:
        raise ValueError("Missing test_dir; pass --test_dir or set training.test_image_dir")
    test_dir = _resolve_path(test_dir_raw, config_base_dir)
    if test_dir is None or not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    cache_dir_raw = _pick(args.cache_dir, cfg_train.get("full_eval_cache_dir"), "../eval_cache")
    cache_dir = _resolve_path(cache_dir_raw, config_base_dir)
    if cache_dir is None:
        raise ValueError("Invalid cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)

    num_steps = max(1, int(_pick(args.num_steps, cfg_train.get("full_eval_num_steps", cfg_infer.get("num_steps")), 1)))
    step_size = float(_pick(args.step_size, cfg_train.get("full_eval_step_size", cfg_infer.get("step_size")), 1.0))
    style_strength = _pick(args.style_strength, cfg_train.get("full_eval_style_strength", cfg_infer.get("style_strength")), None)
    if style_strength is not None:
        style_strength = float(style_strength)
    step_schedule = _resolve_schedule(args.step_schedule, cfg_train.get("full_eval_step_schedule", cfg_infer.get("step_schedule")))

    batch_size = max(1, int(_pick(args.batch_size, cfg_train.get("full_eval_batch_size"), 20)))
    max_src_samples = int(_pick(args.max_src_samples, cfg_train.get("full_eval_max_src_samples"), 30))
    max_ref_compare = int(_pick(args.max_ref_compare, cfg_train.get("full_eval_max_ref_compare"), 50))
    max_ref_cache = int(_pick(args.max_ref_cache, cfg_train.get("full_eval_max_ref_cache"), 256))
    ref_feature_batch_size = max(1, int(_pick(args.ref_feature_batch_size, cfg_train.get("full_eval_ref_feature_batch_size"), 64)))
    lpips_chunk_size = max(1, int(_pick(args.lpips_chunk_size, cfg_train.get("full_eval_lpips_chunk_size"), max(8, min(32, batch_size)))))

    classifier_path_raw = _pick(args.classifier_path, cfg_train.get("full_eval_classifier_path"), "../../style_classifier.pt")
    if classifier_path_raw is not None and not str(classifier_path_raw).strip():
        classifier_path_raw = None

    eval_classifier_only = bool(args.eval_classifier_only or cfg_train.get("full_eval_classifier_only", False))
    eval_disable_lpips = bool(args.eval_disable_lpips or cfg_train.get("full_eval_disable_lpips", False))
    style_ref_mode = str(_pick(args.style_ref_mode, cfg_train.get("full_eval_style_ref_mode"), "none")).lower()
    style_ref_count = int(_pick(args.style_ref_count, cfg_train.get("full_eval_style_ref_count"), 8))
    style_ref_seed = int(_pick(args.style_ref_seed, cfg_train.get("full_eval_style_ref_seed"), 2026))

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {out_dir}")
    print(f"Test dir: {test_dir}")
    print(f"Cache dir: {cache_dir}")
    print(
        "Eval params | "
        f"steps={num_steps} step_size={step_size:.3f} style_strength={style_strength} schedule={step_schedule} "
        f"batch={batch_size} src_cap={max_src_samples} ref_cap={max_ref_cache} ref_bs={ref_feature_batch_size} lpips_chunk={lpips_chunk_size}"
    )
    print(f"Style ref mode={style_ref_mode} count={style_ref_count} seed={style_ref_seed} (style_id-only path)")

    style_subdirs = list(cfg.get("data", {}).get("style_subdirs", []))
    if not style_subdirs:
        style_subdirs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    test_images: dict[int, tuple[str, list[Path]]] = {}
    for style_id, style_name in enumerate(style_subdirs):
        style_dir = test_dir / style_name
        if not style_dir.exists():
            continue
        imgs = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in _VALID_IMAGE_EXTS])
        if imgs:
            test_images[style_id] = (style_name, imgs)
    if not test_images:
        raise RuntimeError(f"No test images found in {test_dir}")

    all_src: list[dict[str, Any]] = []
    rng = random.Random(42)
    for sid, (sname, slist) in test_images.items():
        sampled = slist[:]
        rng.shuffle(sampled)
        if max_src_samples > 0:
            sampled = sampled[:max_src_samples]
        for p in sampled:
            all_src.append({"path": p, "style_id": sid, "style_name": sname})

    if not all_src:
        raise RuntimeError("No source images selected")

    print(f"\nPhase 1: generation ({len(all_src)} source images)")
    lgt = LGTInference(
        str(checkpoint_path),
        device=device,
        num_steps=num_steps,
        step_size=step_size,
        style_strength=style_strength,
        step_schedule=step_schedule,
    )
    vae = load_vae(device)

    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    store_latents = bool(classifier_path_raw)
    generated: list[dict[str, Any]] = []
    io_pool = ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 4)))
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    num_src_total = len(all_src)
    num_styles = len(style_subdirs)
    for b_start in range(0, num_src_total, batch_size):
        b_end = min(b_start + batch_size, num_src_total)
        batch_info = all_src[b_start:b_end]

        src_batch = torch.stack([_load_eval_image_tensor(item["path"]) for item in batch_info], dim=0).to(device)
        src_style_ids = torch.tensor([item["style_id"] for item in batch_info], device=device)

        with torch.no_grad(), amp_ctx:
            latents_src = encode_image(vae, src_batch, device)
            if abs(scale_in - 1.0) > 1e-4:
                latents_src = latents_src * scale_in
            latents_x0 = lgt.inversion(latents_src, src_style_ids)

            for tgt_id in range(num_styles):
                tgt_name = style_subdirs[tgt_id]
                tgt_ids = torch.full((len(batch_info),), tgt_id, device=device, dtype=torch.long)
                latents_gen = lgt.generation(latents_x0, tgt_ids)
                if abs(scale_out - 1.0) > 1e-4:
                    latents_gen = latents_gen * scale_out
                imgs_gen = decode_latent(vae, latents_gen, device)

                latents_gen_cpu = latents_gen.detach().to(device="cpu", dtype=torch.float16) if store_latents else None
                imgs_gen_cpu = imgs_gen.detach().cpu()

                for i, src_item in enumerate(batch_info):
                    out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                    out_path = out_dir / out_name
                    io_pool.submit(_save_image_task, imgs_gen_cpu[i], out_path)
                    rec: dict[str, Any] = {
                        "src_path": src_item["path"],
                        "src_style": src_item["style_name"],
                        "tgt_style_name": tgt_name,
                        "tgt_style_id": tgt_id,
                        "gen_name": out_name,
                        "gen_path": str(out_path),
                    }
                    if latents_gen_cpu is not None:
                        rec["gen_latent"] = latents_gen_cpu[i].clone()
                    generated.append(rec)

                del imgs_gen, imgs_gen_cpu, latents_gen, tgt_ids
                if latents_gen_cpu is not None:
                    del latents_gen_cpu

        del src_batch, src_style_ids, latents_src, latents_x0

    io_pool.shutdown(wait=True)
    del lgt, vae
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("  Generation completed")

    if not generated:
        raise RuntimeError("No generated samples")

    print("\nPhase 2: evaluation")
    classifier = None
    classifier_labels = list(style_subdirs)
    if args.classifier_classes:
        names = [c.strip() for c in args.classifier_classes.split(",") if c.strip()]
        if names:
            classifier_labels = names
    if classifier_path_raw:
        classifier_path = _resolve_path(classifier_path_raw, config_base_dir)
        if classifier_path is None:
            classifier_path = Path(classifier_path_raw)
        if not classifier_path.exists():
            print(f"  WARNING: classifier checkpoint not found: {classifier_path}")
        else:
            try:
                num_classes = int(cfg.get("model", {}).get("num_styles", len(style_subdirs)))
                in_channels = int(cfg.get("model", {}).get("latent_channels", 4))
                classifier = LatentStyleClassifier(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    use_stats=bool(cfg.get("loss", {}).get("style_classifier_use_stats", True)),
                    use_gram=bool(cfg.get("loss", {}).get("style_classifier_use_gram", True)),
                    use_lowpass_stats=bool(cfg.get("loss", {}).get("style_classifier_use_lowpass_stats", True)),
                    spatial_shuffle=bool(cfg.get("loss", {}).get("style_classifier_spatial_shuffle", True)),
                    input_size_train=int(cfg.get("loss", {}).get("style_classifier_input_size_train", 8)),
                    input_size_infer=int(cfg.get("loss", {}).get("style_classifier_input_size_infer", 8)),
                    lowpass_size=int(cfg.get("loss", {}).get("style_classifier_lowpass_size", 8)),
                ).to(device)
                state = torch.load(classifier_path, map_location=device, weights_only=False)
                state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
                if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                classifier.load_state_dict(state_dict, strict=True)
                classifier.eval()
                print(f"  Loaded classifier: {classifier_path}")
            except Exception as exc:
                print(f"  WARNING: failed to load classifier: {exc}")
                classifier = None

    run_full_metrics = not eval_classifier_only
    loss_fn = None
    clip_model = None
    clip_processor = None
    has_clip = False
    to_pil = ToPILImage()

    if run_full_metrics:
        if not eval_disable_lpips and lpips is not None:
            try:
                loss_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
                print("  LPIPS loaded")
            except Exception as exc:
                print(f"  WARNING: LPIPS unavailable: {exc}")
        try:
            from transformers import CLIPModel, CLIPProcessor

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
            has_clip = True
            print("  CLIP loaded")
        except Exception as exc:
            print(f"  WARNING: CLIP unavailable: {exc}")

    ref_features: dict[int, list[dict[str, Any]]] = {sid: [] for sid in test_images.keys()}
    dataset_hash = hashlib.md5(str(test_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    cache_tag = "all" if max_ref_cache <= 0 else str(max_ref_cache)
    cache_file = cache_dir / f"ref_feats_{dataset_hash}_m{cache_tag}.pt"

    if run_full_metrics and cache_file.exists() and not args.force_regen:
        try:
            ref_features = _sanitize_ref_cache(torch.load(cache_file, map_location="cpu"), list(test_images.keys()))
            print(f"  Loaded ref cache: {cache_file}")
        except Exception as exc:
            print(f"  WARNING: invalid ref cache ({exc}), rebuilding")
            ref_features = {sid: [] for sid in test_images.keys()}

    if run_full_metrics and any(len(v) == 0 for v in ref_features.values()):
        print("  Building reference cache...")
        ref_features = {sid: [] for sid in test_images.keys()}
        clip_amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

        for sid, (sname, imgs) in test_images.items():
            sampled = imgs[:]
            if max_ref_cache > 0:
                sampled = sampled[: min(len(sampled), max_ref_cache)]
            entries: list[dict[str, Any]] = [{"path": str(p), "clip": None} for p in sampled]

            if has_clip and clip_model is not None and clip_processor is not None and sampled:
                for b_start in tqdm(range(0, len(sampled), ref_feature_batch_size), desc=f"Ref {sname}", leave=False):
                    batch_paths = sampled[b_start:b_start + ref_feature_batch_size]
                    batch_pils = [Image.open(p).convert("RGB").resize((256, 256)) for p in batch_paths]
                    with torch.no_grad(), clip_amp_ctx:
                        inputs = clip_processor(images=batch_pils, return_tensors="pt").to(device)
                        out = clip_model.get_image_features(**inputs)
                        emb = _extract_clip_embeddings(out).float()
                        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        emb = emb.detach().cpu().to(dtype=torch.float16)
                    for i in range(emb.shape[0]):
                        entries[b_start + i]["clip"] = emb[i:i + 1].clone()
                    del batch_pils, inputs, out, emb

            ref_features[sid] = entries

        torch.save(ref_features, cache_file)
        print(f"  Saved ref cache: {cache_file}")

    ref_clip_mats: dict[int, torch.Tensor] = {}
    if run_full_metrics and has_clip:
        for sid, entries in ref_features.items():
            clips = [e.get("clip") for e in entries if torch.is_tensor(e.get("clip"))]
            if not clips:
                continue
            stacked = torch.cat(clips, dim=0).float()
            stacked = stacked / (stacked.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            ref_clip_mats[sid] = stacked.to(device)

    csv_path = out_dir / "metrics.csv"
    csv_mode = "w" if args.force_regen or not csv_path.exists() else "a"
    columns = [
        "src_style", "tgt_style", "src_image", "gen_image",
        "content_lpips", "style_lpips", "clip_style", "clip_content",
        "pred_style", "class_correct"
    ]

    warned_missing_latent = False
    with open(csv_path, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if csv_mode == "w":
            writer.writeheader()

        for b_start in tqdm(range(0, len(generated), batch_size), desc="Eval", dynamic_ncols=True):
            batch_items = generated[b_start:b_start + batch_size]
            gen_imgs_cpu = torch.stack([_load_eval_image_tensor(Path(x["gen_path"])) for x in batch_items], dim=0)
            src_imgs_cpu = torch.stack([_load_eval_image_tensor(x["src_path"]) for x in batch_items], dim=0)
            gen_imgs = gen_imgs_cpu.to(device)
            src_imgs = src_imgs_cpu.to(device)

            gen_latents = None
            if classifier is not None:
                latent_list = [x.get("gen_latent") for x in batch_items]
                if latent_list and all(torch.is_tensor(t) for t in latent_list):
                    gen_latents = torch.stack([t for t in latent_list], dim=0).to(device=device, dtype=torch.float32)
                elif not warned_missing_latent:
                    print("  WARNING: classifier enabled but generated latents are missing")
                    warned_missing_latent = True

            with torch.no_grad():
                if loss_fn is not None:
                    dists = loss_fn(_to_lpips_input(gen_imgs.float()), _to_lpips_input(src_imgs.float()))
                    content_lpips = dists.view(-1).detach().cpu().numpy().astype(np.float32)
                else:
                    content_lpips = np.zeros(len(batch_items), dtype=np.float32)

                gen_clips = None
                clip_content = np.zeros(len(batch_items), dtype=np.float32)
                if has_clip and clip_model is not None and clip_processor is not None:
                    pil_gen = [to_pil(img.float()) for img in gen_imgs_cpu]
                    pil_src = [to_pil(img.float()) for img in src_imgs_cpu]
                    in_gen = clip_processor(images=pil_gen, return_tensors="pt").to(device)
                    in_src = clip_processor(images=pil_src, return_tensors="pt").to(device)
                    out_gen = clip_model.get_image_features(**in_gen)
                    out_src = clip_model.get_image_features(**in_src)
                    gen_clips = _extract_clip_embeddings(out_gen).float()
                    src_clips = _extract_clip_embeddings(out_src).float()
                    if gen_clips.ndim == 1:
                        gen_clips = gen_clips.unsqueeze(0)
                    if src_clips.ndim == 1:
                        src_clips = src_clips.unsqueeze(0)
                    gen_clips = gen_clips / (gen_clips.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                    src_clips = src_clips / (src_clips.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                    clip_content = F.cosine_similarity(gen_clips, src_clips).detach().cpu().numpy().astype(np.float32)

                pred_idx = [-1] * len(batch_items)
                if classifier is not None and gen_latents is not None:
                    cls_inputs = gen_latents
                    cls_sz = int(cfg.get("loss", {}).get("style_classifier_input_size_infer", 0))
                    if cls_sz and (cls_inputs.shape[-1] != cls_sz or cls_inputs.shape[-2] != cls_sz):
                        cls_inputs = F.interpolate(cls_inputs, size=(cls_sz, cls_sz), mode="area")
                    pred_idx = classifier(cls_inputs).argmax(dim=1).detach().cpu().tolist()

                for i, item in enumerate(batch_items):
                    tgt_id = int(item["tgt_style_id"])

                    pred_style_name = "N/A"
                    class_correct: str | int = "N/A"
                    if classifier is not None and pred_idx[i] != -1:
                        if 0 <= pred_idx[i] < len(classifier_labels):
                            pred_style_name = classifier_labels[pred_idx[i]]
                            class_correct = 1 if pred_idx[i] == tgt_id else 0
                        else:
                            pred_style_name = f"Unknown({pred_idx[i]})"
                            class_correct = 0

                    clip_style = 0.0
                    if gen_clips is not None and tgt_id in ref_clip_mats:
                        ref_mat = ref_clip_mats[tgt_id]
                        emb = gen_clips[i:i + 1]
                        if emb.shape[-1] == ref_mat.shape[-1]:
                            clip_style = float(torch.matmul(emb, ref_mat.t()).mean().item())

                    style_lpips = 0.0
                    if loss_fn is not None:
                        refs = ref_features.get(tgt_id, [])
                        if max_ref_compare > 0:
                            refs = refs[: min(len(refs), max_ref_compare)]
                        if refs:
                            dsum = 0.0
                            dcnt = 0
                            for rs in range(0, len(refs), lpips_chunk_size):
                                chunk = refs[rs:rs + lpips_chunk_size]
                                ref_batch = torch.stack([_load_eval_image_tensor(Path(r["path"])) for r in chunk], dim=0).to(device)
                                gen_expand = gen_imgs[i:i + 1].expand(ref_batch.shape[0], -1, -1, -1)
                                cd = loss_fn(_to_lpips_input(gen_expand.float()), _to_lpips_input(ref_batch.float())).view(-1)
                                dsum += float(cd.sum().item())
                                dcnt += int(cd.numel())
                                del ref_batch, gen_expand, cd
                            if dcnt > 0:
                                style_lpips = dsum / float(dcnt)

                    writer.writerow(
                        {
                            "src_style": item["src_style"],
                            "tgt_style": item["tgt_style_name"],
                            "src_image": item["src_path"].name,
                            "gen_image": item["gen_name"],
                            "content_lpips": float(content_lpips[i]),
                            "style_lpips": style_lpips,
                            "clip_style": clip_style,
                            "clip_content": float(clip_content[i]),
                            "pred_style": pred_style_name,
                            "class_correct": class_correct,
                        }
                    )

            f.flush()
            del gen_imgs_cpu, src_imgs_cpu, gen_imgs, src_imgs
            if gen_latents is not None:
                del gen_latents

    _generate_summary_json(csv_path, out_dir, checkpoint_path)
def _generate_summary_json(csv_path: Path, out_dir: Path, ckpt_path: Path) -> None:
    print("\nGenerating summary...")
    if not csv_path.exists():
        print("No metrics.csv found")
        return

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        print("No rows in metrics.csv")
        return

    def to_f(x: str | None) -> float:
        try:
            return float(x) if x not in {None, ""} else 0.0
        except Exception:
            return 0.0

    matrix = defaultdict(lambda: defaultdict(list))
    for r in rows:
        matrix[r["src_style"]][r["tgt_style"]].append(r)

    matrix_json: dict[str, dict[str, dict[str, Any]]] = {}
    transfer_pool: list[dict[str, float]] = []
    photo_transfer_pool: list[dict[str, float]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    for src, targets in matrix.items():
        matrix_json[src] = {}
        for tgt, items in targets.items():
            stats = {
                "count": len(items),
                "clip_style": float(np.mean([to_f(x.get("clip_style")) for x in items])),
                "style_lpips": float(np.mean([to_f(x.get("style_lpips")) for x in items])),
                "content_lpips": float(np.mean([to_f(x.get("content_lpips")) for x in items])),
                "clip_content": float(np.mean([to_f(x.get("clip_content")) for x in items])),
            }
            cls_vals = [x.get("class_correct") for x in items if x.get("class_correct") not in {None, "", "N/A"}]
            if cls_vals:
                stats["classifier_acc"] = float(np.mean([int(v) for v in cls_vals]))
                for x in items:
                    if x.get("class_correct") not in {None, "", "N/A"}:
                        y_true.append(tgt)
                        y_pred.append(x.get("pred_style", ""))
            else:
                stats["classifier_acc"] = None

            matrix_json[src][tgt] = stats
            if src != tgt:
                transfer_pool.append(stats)
                if src == "photo":
                    photo_transfer_pool.append(stats)

    def pool_avg(pool: list[dict[str, Any]], key: str) -> float:
        vals = [float(x[key]) for x in pool if x.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    cls_report = None
    if classification_report is not None and y_true:
        try:
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            cls_report = None

    summary = {
        "checkpoint": str(ckpt_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matrix_breakdown": matrix_json,
        "analysis": {
            "style_transfer_ability": {
                "clip_style": pool_avg(transfer_pool, "clip_style"),
                "content_lpips": pool_avg(transfer_pool, "content_lpips"),
                "classifier_acc": pool_avg(transfer_pool, "classifier_acc"),
            },
            "photo_to_art_performance": {
                "clip_style": pool_avg(photo_transfer_pool, "clip_style"),
                "valid": len(photo_transfer_pool) > 0,
                "classifier_acc": pool_avg(photo_transfer_pool, "classifier_acc"),
            },
        },
        "classification_report": cls_report,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


def _run_auto_mode(args: argparse.Namespace) -> None:
    config_path = _resolve_auto_config_path(args.config)
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpoint", {})

    ckpt_dir = _resolve_path(ckpt_cfg.get("save_dir", "../adacut_ckpt"), config_path.parent)
    if ckpt_dir is None or not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    ckpt_map: dict[int, Path] = {}
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        e = _parse_epoch(p)
        if e is not None:
            ckpt_map[e] = p
    if not ckpt_map:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    interval = int(train_cfg.get("full_eval_interval", 0))
    run_last = bool(train_cfg.get("full_eval_on_last_epoch", False))
    num_epochs = int(train_cfg.get("num_epochs", 0))

    expected: set[int] = set()
    if interval > 0:
        expected.update(e for e in ckpt_map.keys() if e % interval == 0)
    if run_last:
        if num_epochs > 0 and num_epochs in ckpt_map:
            expected.add(num_epochs)
        elif num_epochs <= 0:
            expected.add(max(ckpt_map.keys()))

    if not expected:
        print("No target eval epochs inferred from config")
        return

    full_eval_root = ckpt_dir / "full_eval"
    full_eval_root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, Path, Path]] = []
    print(f"Auto config: {config_path}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"full_eval_interval={interval}, full_eval_on_last_epoch={run_last}, num_epochs={num_epochs}")

    for e in sorted(expected):
        ckpt_path = ckpt_map.get(e)
        if ckpt_path is None:
            print(f"  [missing-ckpt] epoch {e:04d}")
            continue
        out_dir = full_eval_root / f"epoch_{e:04d}"
        done = _eval_done(out_dir)
        if done and not args.force_regen:
            print(f"  [done] epoch {e:04d} -> {out_dir}")
            continue
        print(f"  [{'regen' if done else 'pending'}] epoch {e:04d} -> {out_dir}")
        pending.append((e, ckpt_path, out_dir))

    if not pending:
        print("All expected evaluations are already complete")
        return

    for idx, (e, ckpt_path, out_dir) in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] Evaluating epoch {e:04d}")
        run_args = argparse.Namespace(**vars(args))
        run_args.config = str(config_path)
        run_args.checkpoint = str(ckpt_path)
        run_args.output = str(out_dir)
        _run_single_evaluation(run_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full evaluation utility.\n"
            "Single mode: pass --checkpoint and --output.\n"
            "Auto mode: pass neither; script uses config.json + full_eval_interval."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Config path for auto mode and relative path resolution")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--style_strength", type=float, default=None)
    parser.add_argument("--step_schedule", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_src_samples", type=int, default=None)
    parser.add_argument("--max_ref_compare", type=int, default=None)
    parser.add_argument("--max_ref_cache", type=int, default=None)
    parser.add_argument("--ref_feature_batch_size", type=int, default=None)
    parser.add_argument("--lpips_chunk_size", type=int, default=None)

    parser.add_argument("--force_regen", action="store_true")
    parser.add_argument("--classifier_path", type=str, default=None)
    parser.add_argument("--classifier_classes", type=str, default="")
    parser.add_argument("--eval_classifier_only", action="store_true")
    parser.add_argument("--eval_disable_lpips", action="store_true")
    parser.add_argument("--style_ref_mode", type=str, default=None)
    parser.add_argument("--style_ref_count", type=int, default=None)
    parser.add_argument("--style_ref_seed", type=int, default=None)

    args = parser.parse_args()

    has_ckpt = args.checkpoint is not None
    has_out = args.output is not None
    if has_ckpt != has_out:
        parser.error("--checkpoint and --output must be provided together, or both omitted for auto mode")

    if has_ckpt:
        _run_single_evaluation(args)
    else:
        _run_auto_mode(args)


if __name__ == "__main__":
    main()

