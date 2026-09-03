from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parent
CYCLE_ROOT = SRC_ROOT.parent
REPO_ROOT = CYCLE_ROOT.parent
DEFAULT_CKPT = CYCLE_ROOT / "Ablate43" / "Ablate43_S01_Baseline_Gold" / "epoch_0060.pt"
DEFAULT_MODEL_FILE = CYCLE_ROOT / "Ablate43" / "Ablate43_S01_Baseline_Gold" / "model.py"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


def _load_model_builder(model_py: Path):
    spec = importlib.util.spec_from_file_location("ablate43_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_fn = getattr(mod, "build_model_from_config", None)
    if build_fn is None:
        raise RuntimeError("build_model_from_config not found in model.py")
    return build_fn


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def _load_student(ckpt_path: Path, model_py: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = payload["config"]
    model_cfg = config.get("model", {})
    state = payload["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    build_model_from_config = _load_model_builder(model_py)
    model = build_model_from_config(model_cfg, use_checkpointing=False).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.train()
    return model, config


def _load_vae(vae_id: str, device: torch.device):
    from diffusers import AutoencoderKL

    vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _pil_to_neg1_1(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if image_size > 0:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    return x * 2.0 - 1.0


@dataclass(frozen=True)
class PairItem:
    src_path: Path
    tgt_path: Path
    src_style_id: int
    target_style_id: int


class PairedDistillDataset(Dataset):
    def __init__(self, items: list[PairItem], image_size: int) -> None:
        self.items = items
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        src = _pil_to_neg1_1(item.src_path, self.image_size)
        tgt = _pil_to_neg1_1(item.tgt_path, self.image_size)
        src_sid = torch.tensor(item.src_style_id, dtype=torch.long)
        sid = torch.tensor(item.target_style_id, dtype=torch.long)
        return src, tgt, src_sid, sid


def _find_existing_file(base: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def _iter_pairs(
    generated_dir: Path,
    source_root: Path,
    style_to_id: dict[str, int],
) -> Iterable[PairItem]:
    for p in sorted(generated_dir.glob("*.jpg")):
        stem = p.stem
        if "_to_" not in stem:
            continue
        src_token, tgt_style = stem.rsplit("_to_", 1)
        if tgt_style not in style_to_id:
            continue
        src_style = src_token.split("_", 1)[0]
        if src_style not in style_to_id:
            continue
        suffix = src_token[len(src_style) :]
        suffix = suffix[1:] if suffix.startswith("_") else suffix
        if not suffix:
            continue
        source_candidate = source_root / src_style / suffix
        src_path = _find_existing_file(source_candidate)
        if src_path is None:
            continue
        yield PairItem(
            src_path=src_path,
            tgt_path=p,
            src_style_id=style_to_id[src_style],
            target_style_id=style_to_id[tgt_style],
        )


def _save_ckpt(
    output_path: Path,
    model: torch.nn.Module,
    config: dict,
    train_meta: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "distill_meta": train_meta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))


def _run_full_eval(
    *,
    checkpoint_path: Path,
    tag: str,
    eval_script: Path,
    eval_workdir: Path,
    eval_output_root: Path,
    eval_test_dir: Path,
    eval_num_steps: int,
    eval_step_size: float,
    eval_style_strength: float | None,
) -> Path:
    out_dir = eval_output_root / checkpoint_path.stem
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(out_dir),
        "--test_dir",
        str(eval_test_dir),
        "--num_steps",
        str(int(eval_num_steps)),
        "--step_size",
        str(float(eval_step_size)),
    ]
    if eval_style_strength is not None:
        cmd += ["--style_strength", str(float(eval_style_strength))]

    print(json.dumps({"phase": "auto_eval_start", "tag": tag, "checkpoint": str(checkpoint_path), "out_dir": str(out_dir)}, ensure_ascii=False))
    subprocess.run(cmd, check=True, cwd=str(eval_workdir))
    summary_path = (out_dir / "summary.json").resolve()
    metrics = {"phase": "auto_eval_done", "tag": tag, "summary": str(summary_path)}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            all_pairs = (summary.get("analysis") or {}).get("all_pairs_overview") or {}
            transfer = (summary.get("analysis") or {}).get("style_transfer_ability") or {}
            cls = summary.get("classification_report") or {}
            cls_acc = cls.get("accuracy")
            metrics.update(
                {
                    "all_classifier_acc": all_pairs.get("classifier_acc"),
                    "all_clip_style": all_pairs.get("clip_style"),
                    "all_content_lpips": all_pairs.get("content_lpips"),
                    "xfer_classifier_acc": transfer.get("classifier_acc"),
                    "xfer_clip_style": transfer.get("clip_style"),
                    "xfer_content_lpips": transfer.get("content_lpips"),
                    "cls_report_acc": cls_acc,
                }
            )
        except Exception as exc:
            metrics["summary_parse_error"] = str(exc)
    print(json.dumps(metrics, ensure_ascii=False))
    return out_dir


def _set_trainable_for_stage(model: torch.nn.Module, stage: str) -> int:
    stage = str(stage).strip().lower()

    def is_style_param(name: str) -> bool:
        style_tokens = (
            "style_emb",
            "style_spatial_id",
            "style_tokens_basis",
            "global_proj",
            "style_proj",
            "pos_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "token_norm",
            "query_norm",
            "ffn_norm",
            "gamma",
            "style_scale",
            "style_shift",
            "dec_mod.mapper",
        )
        return any(tok in name for tok in style_tokens)

    def is_decoder_skip_param(name: str) -> bool:
        prefixes = (
            "decoder_blocks.",
            "skip_up_proj.",
            "skip_src_proj.",
            "skip_fusion.",
            "skip_router.",
            "dec_up.",
            "dec_post.",
            "dec_out.",
            "dec_mod.",
        )
        return name.startswith(prefixes)

    for _, p in model.named_parameters():
        p.requires_grad_(False)

    for n, p in model.named_parameters():
        if stage == "stage_a":
            if is_style_param(n):
                p.requires_grad_(True)
        elif stage == "stage_b":
            if is_style_param(n) or is_decoder_skip_param(n):
                p.requires_grad_(True)
        elif stage == "full":
            p.requires_grad_(True)
        elif stage == "style_emb_only":
            if n.startswith("style_emb."):
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_stage(
    *,
    stage_name: str,
    model: torch.nn.Module,
    vae,
    scaling_factor: float,
    dl: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    mse_weight: float,
    l1_weight: float,
    img_l1_weight: float,
    img_l2_weight: float,
    edge_weight: float,
    identity_weight: float,
    img_loss_scale: float,
    grad_clip_norm: float,
    step_size: float,
    style_strength: float,
    output_path: Path,
    config: dict,
    train_meta: dict,
) -> dict:
    if epochs <= 0:
        return {"stage": stage_name, "skipped": True}

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable params for stage {stage_name}")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    stage_log = {"stage": stage_name, "epochs": int(epochs), "lr": float(lr), "weight_decay": float(weight_decay)}
    print(json.dumps({"phase": "stage_init", **stage_log}, ensure_ascii=False))

    last_avg_loss = None
    def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
        kx = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        # RGB -> gray
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for src_img, tgt_img, src_sid, target_sid in dl:
            src_img = src_img.to(device=device, non_blocking=True)
            tgt_img = tgt_img.to(device=device, non_blocking=True)
            src_sid = src_sid.to(device=device, non_blocking=True)
            target_sid = target_sid.to(device=device, non_blocking=True)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    src_lat = vae.encode(src_img).latent_dist.mean * scaling_factor
                    tgt_lat = vae.encode(tgt_img).latent_dist.mean * scaling_factor

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                pred_lat = model.integrate(
                    src_lat,
                    style_id=target_sid,
                    num_steps=1,
                    step_size=float(step_size),
                    style_strength=float(style_strength),
                )
                loss_mse = F.mse_loss(pred_lat, tgt_lat)
                loss_l1 = F.l1_loss(pred_lat, tgt_lat)

                if float(img_loss_scale) > 0:
                    # Decode prediction to image domain for stronger supervision.
                    try:
                        pred_img = vae.decode(pred_lat / scaling_factor).sample
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise RuntimeError(
                                "CUDA OOM in VAE decode branch. Try lower --batch-size (e.g. 2/4), "
                                "or set --stage-a-img-loss-scale 0 --stage-b-img-loss-scale 0.5."
                            ) from exc
                        raise
                    loss_img_l1 = F.l1_loss(pred_img, tgt_img)
                    loss_img_l2 = F.mse_loss(pred_img, tgt_img)

                    # Preserve brush-stroke boundaries / high frequency structure.
                    pred_edge = _sobel_edges(pred_img)
                    tgt_edge = _sobel_edges(tgt_img)
                    loss_edge = F.l1_loss(pred_edge, tgt_edge)
                else:
                    loss_img_l1 = pred_lat.new_zeros(())
                    loss_img_l2 = pred_lat.new_zeros(())
                    loss_edge = pred_lat.new_zeros(())

                # Identity anchor: same-style pass should remain close to source.
                pred_id_lat = model.integrate(
                    src_lat,
                    style_id=src_sid,
                    num_steps=1,
                    step_size=float(step_size),
                    style_strength=float(style_strength),
                )
                loss_identity = F.mse_loss(pred_id_lat, src_lat)

                loss = (
                    float(mse_weight) * loss_mse
                    + float(l1_weight) * loss_l1
                    + float(img_loss_scale) * (
                        float(img_l1_weight) * loss_img_l1
                        + float(img_l2_weight) * loss_img_l2
                        + float(edge_weight) * loss_edge
                    )
                    + float(identity_weight) * loss_identity
                )

            scaler.scale(loss).backward()
            if float(grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            b = int(src_img.shape[0])
            total_loss += float(loss.detach().item()) * b
            total_count += b

        avg_loss = total_loss / max(1, total_count)
        last_avg_loss = avg_loss
        print(
            json.dumps(
                {"phase": "stage_train", "stage": stage_name, "epoch": epoch, "epochs": int(epochs), "avg_loss": avg_loss},
                ensure_ascii=False,
            )
        )

    stage_ckpt = output_path.with_name(f"{output_path.stem}.{stage_name}{output_path.suffix}")
    _save_ckpt(
        stage_ckpt,
        model,
        config,
        {
            **train_meta,
            "stage": stage_name,
            "stage_epochs": int(epochs),
            "stage_lr": float(lr),
            "stage_weight_decay": float(weight_decay),
            "stage_last_avg_loss": last_avg_loss,
        },
    )
    print(json.dumps({"phase": "stage_done", "stage": stage_name, "ckpt": str(stage_ckpt.resolve())}, ensure_ascii=False))
    return {"stage": stage_name, "last_avg_loss": last_avg_loss, "ckpt": str(stage_ckpt)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Ablate43 checkpoint with paired pseudo data")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=REPO_ROOT / "Related_Works" / "runs" / "cut_5x5" / "infer_5x5" / "images",
    )
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "style_data" / "overfit50")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--model-py", type=Path, default=Path(DEFAULT_MODEL_FILE))
    parser.add_argument("--vae", type=str, default=DEFAULT_VAE)
    parser.add_argument("--output", type=Path, default=Path("./distilled_epoch_0060_pairs.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage-a-epochs", type=int, default=10)
    parser.add_argument("--stage-b-epochs", type=int, default=10)
    parser.add_argument("--stage-c-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stage-a-lr", type=float, default=6e-5)
    parser.add_argument("--stage-b-lr", type=float, default=4e-5)
    parser.add_argument("--stage-c-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=0.3)
    parser.add_argument("--img-l1-weight", type=float, default=1.2)
    parser.add_argument("--img-l2-weight", type=float, default=0.5)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--stage-a-img-loss-scale", type=float, default=1.0)
    parser.add_argument("--stage-b-img-loss-scale", type=float, default=1.0)
    parser.add_argument("--stage-c-img-loss-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument(
        "--auto-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After distillation, run full eval for stage ckpts and final ckpt (default: on).",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=SRC_ROOT / "utils" / "run_evaluation.py",
        help="Path to run_evaluation.py",
    )
    parser.add_argument(
        "--eval-workdir",
        type=Path,
        default=SRC_ROOT,
        help="Working directory for evaluation command",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=SRC_ROOT / "full_eval",
        help="Root directory for auto-eval outputs",
    )
    parser.add_argument(
        "--eval-test-dir",
        type=Path,
        default=REPO_ROOT / "style_data" / "overfit50",
        help="Test image directory passed to run_evaluation.py",
    )
    parser.add_argument("--eval-num-steps", type=int, default=1)
    parser.add_argument("--eval-step-size", type=float, default=1.0)
    parser.add_argument("--eval-style-strength", type=float, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    student, config = _load_student(args.checkpoint, args.model_py, device)
    vae = _load_vae(args.vae, device)
    scaling_factor = float(getattr(vae.config, "scaling_factor", config.get("model", {}).get("latent_scale_factor", 0.18215)))

    style_subdirs = config.get("data", {}).get("style_subdirs", ["photo", "Hayao", "monet", "vangogh", "cezanne"])
    style_to_id = {str(s): i for i, s in enumerate(style_subdirs)}

    pairs = list(_iter_pairs(args.generated_dir, args.source_root, style_to_id))
    if not pairs:
        raise RuntimeError("No valid pairs found. Check --generated-dir and --source-root.")

    ds = PairedDistillDataset(pairs, image_size=args.image_size)
    dl = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    train_meta = {
        "generated_dir": str(args.generated_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "num_pairs": len(pairs),
        "stage_a_epochs": int(args.stage_a_epochs),
        "stage_b_epochs": int(args.stage_b_epochs),
        "stage_c_epochs": int(args.stage_c_epochs),
        "batch_size": int(args.batch_size),
        "stage_a_lr": float(args.stage_a_lr),
        "stage_b_lr": float(args.stage_b_lr),
        "stage_c_lr": float(args.stage_c_lr),
        "mse_weight": float(args.mse_weight),
        "l1_weight": float(args.l1_weight),
        "img_l1_weight": float(args.img_l1_weight),
        "img_l2_weight": float(args.img_l2_weight),
        "edge_weight": float(args.edge_weight),
        "identity_weight": float(args.identity_weight),
        "stage_a_img_loss_scale": float(args.stage_a_img_loss_scale),
        "stage_b_img_loss_scale": float(args.stage_b_img_loss_scale),
        "stage_c_img_loss_scale": float(args.stage_c_img_loss_scale),
    }
    print(json.dumps({"phase": "init", **train_meta}, ensure_ascii=False))
    n_a = _set_trainable_for_stage(student, "stage_a")
    print(json.dumps({"phase": "freeze", "stage": "stage_a", "trainable_params": int(n_a)}, ensure_ascii=False))
    res_a = _run_stage(
        stage_name="stage_a",
        model=student,
        vae=vae,
        scaling_factor=scaling_factor,
        dl=dl,
        device=device,
        epochs=int(args.stage_a_epochs),
        lr=float(args.stage_a_lr),
        weight_decay=float(args.weight_decay),
        mse_weight=float(args.mse_weight),
        l1_weight=float(args.l1_weight),
        img_l1_weight=float(args.img_l1_weight),
        img_l2_weight=float(args.img_l2_weight),
        edge_weight=float(args.edge_weight),
        identity_weight=float(args.identity_weight),
        img_loss_scale=float(args.stage_a_img_loss_scale),
        grad_clip_norm=float(args.grad_clip_norm),
        step_size=float(args.step_size),
        style_strength=float(args.style_strength),
        output_path=args.output,
        config=config,
        train_meta=train_meta,
    )

    n_b = _set_trainable_for_stage(student, "stage_b")
    print(json.dumps({"phase": "freeze", "stage": "stage_b", "trainable_params": int(n_b)}, ensure_ascii=False))
    res_b = _run_stage(
        stage_name="stage_b",
        model=student,
        vae=vae,
        scaling_factor=scaling_factor,
        dl=dl,
        device=device,
        epochs=int(args.stage_b_epochs),
        lr=float(args.stage_b_lr),
        weight_decay=float(args.weight_decay),
        mse_weight=float(args.mse_weight),
        l1_weight=float(args.l1_weight),
        img_l1_weight=float(args.img_l1_weight),
        img_l2_weight=float(args.img_l2_weight),
        edge_weight=float(args.edge_weight),
        identity_weight=float(args.identity_weight),
        img_loss_scale=float(args.stage_b_img_loss_scale),
        grad_clip_norm=float(args.grad_clip_norm),
        step_size=float(args.step_size),
        style_strength=float(args.style_strength),
        output_path=args.output,
        config=config,
        train_meta=train_meta,
    )

    n_c = _set_trainable_for_stage(student, "full")
    print(json.dumps({"phase": "freeze", "stage": "stage_c", "trainable_params": int(n_c)}, ensure_ascii=False))
    res_c = _run_stage(
        stage_name="stage_c",
        model=student,
        vae=vae,
        scaling_factor=scaling_factor,
        dl=dl,
        device=device,
        epochs=int(args.stage_c_epochs),
        lr=float(args.stage_c_lr),
        weight_decay=float(args.weight_decay),
        mse_weight=float(args.mse_weight),
        l1_weight=float(args.l1_weight),
        img_l1_weight=float(args.img_l1_weight),
        img_l2_weight=float(args.img_l2_weight),
        edge_weight=float(args.edge_weight),
        identity_weight=float(args.identity_weight),
        img_loss_scale=float(args.stage_c_img_loss_scale),
        grad_clip_norm=float(args.grad_clip_norm),
        step_size=float(args.step_size),
        style_strength=float(args.style_strength),
        output_path=args.output,
        config=config,
        train_meta=train_meta,
    )

    final_ckpt = args.output.resolve()
    _save_ckpt(final_ckpt, student, config, {**train_meta, "stage_a": res_a, "stage_b": res_b, "stage_c": res_c})
    print(json.dumps({"phase": "done", "output": str(final_ckpt)}, ensure_ascii=False))

    if args.auto_eval:
        eval_script = args.eval_script.resolve()
        eval_workdir = args.eval_workdir.resolve()
        eval_output_root = args.eval_output_root.resolve()
        eval_test_dir = args.eval_test_dir.resolve()
        eval_output_root.mkdir(parents=True, exist_ok=True)

        targets: list[tuple[str, Path]] = []
        for tag, res in (("stage_a", res_a), ("stage_b", res_b), ("stage_c", res_c)):
            ck = res.get("ckpt") if isinstance(res, dict) else None
            if ck:
                targets.append((tag, Path(ck).resolve()))
        targets.append(("final", final_ckpt))

        for tag, ck in targets:
            _run_full_eval(
                checkpoint_path=ck,
                tag=tag,
                eval_script=eval_script,
                eval_workdir=eval_workdir,
                eval_output_root=eval_output_root,
                eval_test_dir=eval_test_dir,
                eval_num_steps=int(args.eval_num_steps),
                eval_step_size=float(args.eval_step_size),
                eval_style_strength=args.eval_style_strength,
            )


if __name__ == "__main__":
    main()
