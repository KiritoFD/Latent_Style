from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "Cycle-NCE" / "Ablate43" / "Ablate43_S01_Baseline_Gold" / "epoch_0060.pt"
DEFAULT_MODEL_FILE = ROOT / "Cycle-NCE" / "Ablate43" / "Ablate43_S01_Baseline_Gold" / "model.py"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


class _TorchPipeline(torch.nn.Module):
    def __init__(self, core_model: torch.nn.Module, vae: Any, scaling_factor: float) -> None:
        super().__init__()
        self.core_model = core_model
        self.vae = vae
        self.scaling_factor = float(scaling_factor)

    def forward(
        self,
        image_nchw_neg1_1: torch.Tensor,
        style_id: torch.Tensor,
        step_size: float,
        style_strength: float,
    ) -> torch.Tensor:
        latent = self.vae.encode(image_nchw_neg1_1).latent_dist.mean
        latent = latent * self.scaling_factor
        out_latent = self.core_model.integrate(
            latent,
            style_id=style_id,
            num_steps=1,
            step_size=float(step_size),
            style_strength=float(style_strength),
        )
        out = self.vae.decode(out_latent / self.scaling_factor).sample
        return torch.clamp((out + 1.0) * 0.5, 0.0, 1.0)


def _load_model_builder(model_py: Path):
    spec = importlib.util.spec_from_file_location("ablate43_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    # torch.compile / dynamo may try to import this module by name while tracing.
    # Register it so importlib can resolve "ablate43_model" later.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    build_fn = getattr(mod, "build_model_from_config", None)
    if build_fn is None:
        raise RuntimeError("build_model_from_config not found in model.py")
    return build_fn


def _load_checkpoint(ckpt_path: Path, model_py: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = payload["config"]
    model_cfg = config.get("model", {})
    infer_cfg = config.get("inference", {})
    state = payload["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    build_model_from_config = _load_model_builder(model_py)
    model = build_model_from_config(model_cfg, use_checkpointing=False).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, config, infer_cfg


def _to_nchw_neg1_1(img: Image.Image, force_hw: tuple[int, int] | None = None) -> torch.Tensor:
    if force_hw is not None:
        w, h = force_hw
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten * 2.0 - 1.0


def _to_pil_0_1(nchw: torch.Tensor) -> Image.Image:
    x = nchw.detach().cpu().squeeze(0).clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _round_hw_to_8(width: int, height: int) -> tuple[int, int]:
    w = max(8, (width // 8) * 8)
    h = max(8, (height // 8) * 8)
    return w, h


class Ablate43FastInference:
    def __init__(
        self,
        ckpt: Path = DEFAULT_CKPT,
        model_py: Path = DEFAULT_MODEL_FILE,
        vae_id: str = DEFAULT_VAE,
        device: str = "cuda",
        dtype: str = "fp16",
        use_compile: bool = True,
        compile_mode: str = "max-autotune",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" and self.device.type == "cuda" else torch.float32

        self.model, self.config, self.infer_cfg = _load_checkpoint(Path(ckpt), Path(model_py), self.device)

        try:
            from diffusers import AutoencoderKL
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("diffusers is required for VAE encode/decode") from exc

        vae_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype).to(self.device)
        self.vae.eval()

        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", self.config.get("model", {}).get("latent_scale_factor", 0.18215)))

        self.pipeline = _TorchPipeline(self.model, self.vae, self.scaling_factor).to(self.device)
        self.pipeline.eval()

        self.use_compile = bool(use_compile and self.device.type == "cuda" and hasattr(torch, "compile"))
        if self.use_compile:
            self.pipeline = torch.compile(self.pipeline, mode=compile_mode, fullgraph=False, dynamic=True)

        self.default_step_size = float(self.infer_cfg.get("step_size", 1.0))
        self.default_style_strength = float(self.infer_cfg.get("style_strength", 1.0))

    @torch.no_grad()
    def infer_image(
        self,
        image: Image.Image,
        style_id: int,
        step_size: float | None = None,
        style_strength: float | None = None,
        keep_exact_resolution: bool = False,
    ) -> Image.Image:
        ow, oh = image.size
        tw, th = (ow, oh) if keep_exact_resolution else _round_hw_to_8(ow, oh)
        x = _to_nchw_neg1_1(image, force_hw=(tw, th)).to(device=self.device, dtype=self.dtype)
        sid = torch.tensor([int(style_id)], dtype=torch.long, device=self.device)
        ss = float(self.default_step_size if step_size is None else step_size)
        st = float(self.default_style_strength if style_strength is None else style_strength)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
            y = self.pipeline(x, sid, ss, st)

        out = _to_pil_0_1(y)
        if (tw, th) != (ow, oh):
            out = out.resize((ow, oh), Image.Resampling.LANCZOS)
        return out

    @torch.no_grad()
    def infer_video(
        self,
        input_video: Path,
        output_video: Path,
        style_id: int,
        step_size: float | None = None,
        style_strength: float | None = None,
        batch_size: int = 8,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video inference")

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tw, th = _round_hw_to_8(width, height)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

        sid = torch.tensor([int(style_id)], dtype=torch.long, device=self.device)
        ss = float(self.default_step_size if step_size is None else step_size)
        st = float(self.default_style_strength if style_strength is None else style_strength)

        frames: list[np.ndarray] = []

        def _flush(batch_frames: Iterable[np.ndarray]) -> None:
            arrs = []
            for bgr in batch_frames:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((tw, th), Image.Resampling.LANCZOS)
                ten = _to_nchw_neg1_1(pil).squeeze(0).numpy()
                arrs.append(ten)
            x = torch.from_numpy(np.stack(arrs, axis=0)).to(device=self.device, dtype=self.dtype)
            sid_batch = sid.expand(x.shape[0])
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
                y = self.pipeline(x, sid_batch, ss, st)
            y = y.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
            for out_rgb in y:
                out_u8 = (out_rgb * 255.0).astype(np.uint8)
                if (tw, th) != (width, height):
                    out_u8 = cv2.resize(out_u8, (width, height), interpolation=cv2.INTER_LANCZOS4)
                writer.write(cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                if len(frames) >= max(1, int(batch_size)):
                    _flush(frames)
                    frames.clear()
            if frames:
                _flush(frames)
        finally:
            cap.release()
            writer.release()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablate43 Fast Inference (image/video)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--model-py", type=Path, default=DEFAULT_MODEL_FILE)
    p.add_argument("--vae", type=str, default=DEFAULT_VAE)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--no-compile", action="store_true")

    sub = p.add_subparsers(dest="task", required=True)

    pi = sub.add_parser("image", help="image -> image")
    pi.add_argument("--input", type=Path, required=True)
    pi.add_argument("--output", type=Path, required=True)
    pi.add_argument("--style-id", type=int, required=True)
    pi.add_argument("--step-size", type=float, default=None)
    pi.add_argument("--style-strength", type=float, default=None)
    pi.add_argument("--keep-exact-resolution", action="store_true")

    pv = sub.add_parser("video", help="video -> video")
    pv.add_argument("--input", type=Path, required=True)
    pv.add_argument("--output", type=Path, required=True)
    pv.add_argument("--style-id", type=int, required=True)
    pv.add_argument("--step-size", type=float, default=None)
    pv.add_argument("--style-strength", type=float, default=None)
    pv.add_argument("--batch-size", type=int, default=8)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    engine = Ablate43FastInference(
        ckpt=args.checkpoint,
        model_py=args.model_py,
        vae_id=args.vae,
        device=args.device,
        dtype=args.dtype,
        use_compile=not args.no_compile,
    )

    if args.task == "image":
        img = Image.open(args.input)
        out = engine.infer_image(
            img,
            style_id=args.style_id,
            step_size=args.step_size,
            style_strength=args.style_strength,
            keep_exact_resolution=args.keep_exact_resolution,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.save(args.output)
        print(json.dumps({"ok": True, "task": "image", "output": str(args.output)}, ensure_ascii=False))
        return

    if args.task == "video":
        engine.infer_video(
            input_video=args.input,
            output_video=args.output,
            style_id=args.style_id,
            step_size=args.step_size,
            style_strength=args.style_strength,
            batch_size=args.batch_size,
        )
        print(json.dumps({"ok": True, "task": "video", "output": str(args.output)}, ensure_ascii=False))
        return

    raise RuntimeError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
