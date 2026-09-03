import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


_ACTIVE_TEXT_LOG_PATH: Optional[str] = None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _set_active_text_log_path(path: Optional[str]) -> None:
    global _ACTIVE_TEXT_LOG_PATH
    _ACTIVE_TEXT_LOG_PATH = path


def log_message(message: str, log_path: Optional[str] = None, flush: bool = True) -> None:
    timestamped = f"[{_now_str()}] {message}"
    print(timestamped, flush=flush)

    target_path = log_path if log_path is not None else _ACTIVE_TEXT_LOG_PATH
    if not target_path:
        return

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    with open(target_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(timestamped + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights_normal(m: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, mean=mean, std=std)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        # Only applies if affine=True; ours uses affine=False, so usually no params here.
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, mean=1.0, std=std)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


def zero_init_module(m: nn.Module) -> None:
    if hasattr(m, "weight") and getattr(m, "weight", None) is not None:
        nn.init.constant_(m.weight.data, 0.0)
    if hasattr(m, "bias") and getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias.data, 0.0)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _recover_user_path(text: str) -> str:
    normalized = text.strip()
    if "~" not in normalized:
        return normalized

    parts = Path(normalized).parts
    if "~" not in parts:
        return normalized

    tilde_idx = parts.index("~")
    return str(Path("~").joinpath(*parts[tilde_idx + 1 :]))


def _normalize_model_name_or_path(model_name_or_path: str) -> str:
    raw = _recover_user_path(model_name_or_path)
    if raw.startswith("~"):
        return str(Path(raw).expanduser())
    return raw


def resolve_vae_source(cfg: Dict[str, Any], vis_cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    shared_cfg = cfg.get("shared", {})
    if not isinstance(shared_cfg, dict):
        shared_cfg = {}

    shared_vae_name = shared_cfg.get("vae_model_name_or_path")
    if isinstance(shared_vae_name, str) and shared_vae_name.strip():
        vae_name = shared_vae_name.strip()
    else:
        vae_name = str(vis_cfg.get("vae_model_name_or_path", "") or "").strip()

    if "vae_subfolder" in shared_cfg:
        shared_subfolder = str(shared_cfg.get("vae_subfolder", "") or "").strip()
        vae_subfolder = shared_subfolder if shared_subfolder else None
    else:
        vis_subfolder = str(vis_cfg.get("vae_subfolder", "") or "").strip()
        vae_subfolder = vis_subfolder if vis_subfolder else None

    return vae_name, vae_subfolder


def _default_checkpoint_dir() -> str:
    # As requested: keep checkpoints under ./outputs/model
    return os.path.join("outputs", "model")


def _checkpoint_last_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "last.pt")


def save_checkpoint(
    ckpt_path: str,
    step: int,
    current_kimg: float,
    G: nn.Module,
    F: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sched_G: torch.optim.lr_scheduler._LRScheduler,
    sched_D: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    cfg: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "step": int(step),
        "kimg": float(current_kimg),
        "G": G.state_dict(),
        "F": F.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_D": sched_D.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    _ensure_dir(os.path.dirname(ckpt_path))
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    G: nn.Module,
    F: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sched_G: torch.optim.lr_scheduler._LRScheduler,
    sched_D: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    restore_rng: bool = True,
) -> Tuple[int, float]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

    def _load_discriminator_compat(module: nn.Module, sd: Any, name: str) -> None:
        if not isinstance(sd, dict):
            raise ValueError(f"Invalid discriminator state_dict for {name}")
        try:
            module.load_state_dict(sd)  # type: ignore[arg-type]
            return
        except Exception:
            pass

        # Compatibility across historical discriminator variants:
        # - older layouts used net.5 for the last conv
        # - spectral_norm moves Conv2d weights from *.weight to *.weight_orig
        mapped: Dict[str, Any] = {}
        target_state = module.state_dict()

        def _map_conv(old_prefix: str, new_prefix: str) -> None:
            weight_key = ""
            for candidate in (f"{old_prefix}.weight_orig", f"{old_prefix}.weight"):
                if candidate in sd:
                    weight_key = candidate
                    break
            if not weight_key:
                return

            target_weight_key = f"{new_prefix}.weight"
            if f"{new_prefix}.weight_orig" in target_state:
                target_weight_key = f"{new_prefix}.weight_orig"
            mapped[target_weight_key] = sd[weight_key]

            bias_key = f"{old_prefix}.bias"
            if bias_key in sd and f"{new_prefix}.bias" in target_state:
                mapped[f"{new_prefix}.bias"] = sd[bias_key]

            for aux_name in ("weight_u", "weight_v"):
                src_key = f"{old_prefix}.{aux_name}"
                dst_key = f"{new_prefix}.{aux_name}"
                if src_key in sd and dst_key in target_state:
                    mapped[dst_key] = sd[src_key]

        _map_conv("net.0", "net.0")
        _map_conv("net.2", "net.2")
        _map_conv("net.4", "net.4")
        _map_conv("net.5", "net.4")

        if not mapped:
            raise RuntimeError(f"Failed to compat-load discriminator '{name}' from checkpoint")

        missing, unexpected = module.load_state_dict(mapped, strict=False)
        if missing or unexpected:
            log_message(
                f"[warn] Loaded discriminator '{name}' with compatibility mapping. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    G.load_state_dict(payload["G"])  # type: ignore[arg-type]
    F.load_state_dict(payload["F"])  # type: ignore[arg-type]
    _load_discriminator_compat(D_A, payload["D_A"], "D_A")
    _load_discriminator_compat(D_B, payload["D_B"], "D_B")

    G.to(device)
    F.to(device)
    D_A.to(device)
    D_B.to(device)

    if "opt_G" in payload:
        opt_G.load_state_dict(payload["opt_G"])  # type: ignore[arg-type]
    if "opt_D" in payload:
        opt_D.load_state_dict(payload["opt_D"])  # type: ignore[arg-type]
    if "sched_G" in payload:
        sched_G.load_state_dict(payload["sched_G"])  # type: ignore[arg-type]
    if "sched_D" in payload:
        sched_D.load_state_dict(payload["sched_D"])  # type: ignore[arg-type]
    if "scaler" in payload:
        try:
            scaler.load_state_dict(payload["scaler"])  # type: ignore[arg-type]
        except Exception:
            pass

    # Move optimizer states to device
    for opt in (opt_G, opt_D):
        for state in opt.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if restore_rng and isinstance(payload.get("rng"), dict):
        rng = payload["rng"]
        try:
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass

    loaded_step = int(payload.get("step", 0))
    loaded_kimg = float(payload.get("kimg", 0.0))
    return loaded_step, loaded_kimg


class ReplayBuffer:
    def __init__(self, max_size: int = 50, p_use_history: float = 0.5):
        if max_size < 0:
            raise ValueError("ReplayBuffer max_size must be >= 0")
        if not (0.0 <= p_use_history <= 1.0):
            raise ValueError("ReplayBuffer p_use_history must be in [0,1]")
        self.max_size = int(max_size)
        self.p_use_history = float(p_use_history)
        self.data: List[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        """Return a batch for discriminator training.

        If buffer is enabled (max_size>0):
        - before buffer is full: return current fakes and store them.
        - after full: with probability p_use_history, swap with a random stored fake.
        """
        if self.max_size == 0:
            return batch.detach()

        if batch.ndim < 1:
            raise ValueError(f"ReplayBuffer expects a batch tensor, got shape={tuple(batch.shape)}")

        out: List[torch.Tensor] = []
        for i in range(batch.shape[0]):
            item = batch[i : i + 1].detach()
            if len(self.data) < self.max_size:
                self.data.append(item)
                out.append(item)
            else:
                if random.random() < self.p_use_history:
                    idx = random.randint(0, self.max_size - 1)
                    old = self.data[idx]
                    self.data[idx] = item
                    out.append(old)
                else:
                    out.append(item)
        return torch.cat(out, dim=0)


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'pyyaml'. Install it: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/dict at the top level")
    return cfg


def list_pt_files(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    paths = sorted([p for p in root_path.rglob("*.pt") if p.is_file()])
    if len(paths) == 0:
        raise FileNotFoundError(f"No .pt files found under: {root}")
    return paths


def load_latent_tensor(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict):
        if "latent" in obj and isinstance(obj["latent"], torch.Tensor):
            tensor = obj["latent"]
        else:
            raise ValueError(
                f"Unsupported .pt dict format in {pt_path}. Expected key 'latent' -> Tensor"
            )
    else:
        raise ValueError(
            f"Unsupported .pt content type in {pt_path}: {type(obj)}. Expected Tensor or dict."
        )

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    _require(tensor.ndim == 3, f"Latent tensor must be [C,H,W] or [1,C,H,W], got {tuple(tensor.shape)}")
    # Defensive: some .pt latents may have been saved with autograd history / requires_grad.
    # DataLoader workers on Windows will fail to serialize such tensors across process boundaries.
    tensor = tensor.detach()
    return tensor.to(dtype=torch.float32)


class DomainLatentDataset(Dataset):
    def __init__(self, root_dir: str, latent_divisor: float = 1.0, max_samples: int = -1, seed: int = 42):
        self.paths = list_pt_files(root_dir)
        if max_samples > 0 and max_samples < len(self.paths):
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, max_samples)
            log_message(f"Dataset {root_dir} truncated to {max_samples} samples (seed={seed})")
        if latent_divisor <= 0:
            raise ValueError("latent_divisor must be > 0")
        self.latent_divisor = float(latent_divisor)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = load_latent_tensor(self.paths[idx])
        if self.latent_divisor != 1.0:
            x = x / self.latent_divisor
        return x


def collate_latents(batch: List[torch.Tensor]) -> torch.Tensor:
    # batch: list of [C,H,W]
    return torch.stack(batch, dim=0)


def get_infinite_iterator(dataloader: DataLoader) -> Iterator[torch.Tensor]:
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, use_pointwise_only: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        if use_pointwise_only:
            layers += [
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            ]
        else:
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        ngf: int = 32,
        n_res_blocks: int = 6,
        out_activation: str = "none",
        use_global_residual: bool = False,
        use_pointwise_only: bool = False,
    ):
        super().__init__()
        self.use_global_residual = use_global_residual
        layers: List[nn.Module] = []

        # c7s1
        if use_pointwise_only:
            layers += [
                nn.Conv2d(in_ch, ngf, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
        else:
            layers += [
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_ch, ngf, kernel_size=5, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]

        # res blocks
        for _ in range(n_res_blocks):
            layers += [ResnetBlock(ngf, use_pointwise_only=use_pointwise_only)]

        # output
        if use_pointwise_only:
            layers += [
                nn.Conv2d(ngf, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            ]
        else:
            layers += [
                nn.ReflectionPad2d(2),
                nn.Conv2d(ngf, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            ]

        if out_activation.lower() == "tanh":
            layers += [nn.Tanh()]
        elif out_activation.lower() in ("none", "identity", "linear", ""):
            pass
        else:
            raise ValueError(f"Unsupported out_activation: {out_activation}")

        self.net = nn.Sequential(*layers)
        self.output_conv = next(
            module for module in reversed(self.net) if isinstance(module, nn.Conv2d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_global_residual:
            return x + out
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4, ndf: int = 32, n_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        _ = n_layers  # kept for backward-compatible constructor/config usage

        def maybe_sn(layer: nn.Module) -> nn.Module:
            if not use_spectral_norm:
                return layer
            from torch.nn.utils import spectral_norm

            return spectral_norm(layer)

        seq: List[nn.Module] = [
            maybe_sn(nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(
                nn.Conv2d(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=1)),
        ]

        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _load_vae(model_name_or_path: str, subfolder: Optional[str], device: torch.device) -> nn.Module:
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'diffusers'. Install it: pip install diffusers transformers accelerate safetensors"
        ) from e

    model_name_or_path = _normalize_model_name_or_path(model_name_or_path)
    kwargs: Dict[str, Any] = {}
    if subfolder:
        kwargs["subfolder"] = subfolder

    vae = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents_to_images(
    vae: nn.Module,
    latents: torch.Tensor,
    latents_scaled: bool,
    scaling_factor: float,
    latent_divisor: float = 1.0,
) -> torch.Tensor:
    # latents: [B,4,H,W]
    # Convert from training-space latents back to original latent scale before VAE decode.
    x = latents
    if latent_divisor != 1.0:
        x = x * latent_divisor
    if latents_scaled:
        x = x / scaling_factor

    decoded = vae.decode(x).sample  # type: ignore[attr-defined]
    images = (decoded / 2 + 0.5).clamp(0, 1)
    return images


def write_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(obj)
    payload.setdefault("timestamp", _now_str())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _plot_loss_curves(logs: List[Dict[str, Any]], out_path: str) -> None:
    """Plot loss curves vs kimg/epoch and save to out_path.

    This is intentionally lightweight: it overwrites a single PNG each time metrics are logged.
    """
    if not logs:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # Plotting is optional; training should still work without matplotlib.
        return

    def _series(key: str) -> List[float]:
        vals: List[float] = []
        for d in logs:
            v = d.get(key)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))
        return vals

    use_kimg = any("kimg" in d for d in logs)
    x_vals = [float(d.get("kimg", i + 1)) if use_kimg else float(d.get("epoch", i + 1)) for i, d in enumerate(logs)]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, _series("loss_G"), label="G total")
    plt.plot(x_vals, _series("loss_D"), label="D total")
    plt.plot(x_vals, _series("loss_gan_G"), label="GAN(G)")
    plt.plot(x_vals, _series("loss_gan_F"), label="GAN(F)")
    plt.plot(x_vals, _series("loss_cyc"), label="cycle")
    plt.plot(x_vals, _series("loss_id"), label="identity")

    plt.xlabel("kimg" if use_kimg else "epoch")
    plt.ylabel("loss")
    plt.title("Training loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def train(cfg: Dict[str, Any]) -> None:
    from tqdm import tqdm  # type: ignore

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})
    vis_cfg = cfg.get("visualization", {})

    a_dir = data_cfg.get("a_dir")
    b_dir = data_cfg.get("b_dir")
    _require(isinstance(a_dir, str) and a_dir, "config.data.a_dir is required")
    _require(isinstance(b_dir, str) and b_dir, "config.data.b_dir is required")

    latents_scaled = bool(data_cfg.get("latents_scaled", False))
    latent_divisor = float(data_cfg.get("latent_divisor", 1.0))
    max_samples_a = int(data_cfg.get("max_samples_a", -1))
    max_samples_b = int(data_cfg.get("max_samples_b", -1))
    _require(latent_divisor > 0, "config.data.latent_divisor must be > 0")

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device_str = train_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    total_kimg = int(train_cfg.get("total_kimg", 25000))
    lr = float(train_cfg.get("lr", 2e-4))
    beta1 = float(train_cfg.get("beta1", 0.5))
    beta2 = float(train_cfg.get("beta2", 0.999))

    lambda_cyc = float(train_cfg.get("lambda_cyc", 10.0))
    lambda_id = float(train_cfg.get("lambda_id", 0.5))

    # Paper: "divide the objective by 2 while optimizing D"
    d_loss_scale = float(train_cfg.get("d_loss_scale", 0.5))

    log_every_steps = int(train_cfg.get("log_every_steps", 100))
    ckpt_every_kimg = int(train_cfg.get("ckpt_every_kimg", 1000))
    vis_every_kimg = int(train_cfg.get("vis_every_kimg", 500))

    lr_constant_kimg = int(train_cfg.get("lr_constant_kimg", 12500))
    lr_decay_kimg = int(train_cfg.get("lr_decay_kimg", 12500))

    fake_buffer_size = int(train_cfg.get("fake_buffer_size", 50))
    fake_buffer_prob = float(train_cfg.get("fake_buffer_prob", 0.5))

    _require(batch_size > 0, "config.train.batch_size must be > 0")
    _require(total_kimg > 0, "config.train.total_kimg must be > 0")
    steps_per_kimg = int(1000 / batch_size)
    _require(steps_per_kimg > 0, "config.train.batch_size must be <= 1000 to produce steps_per_kimg >= 1")
    total_steps = int(total_kimg * 1000 / batch_size)
    _require(total_steps > 0, "config.train.total_steps computed from total_kimg and batch_size must be > 0")

    ckpt_every_steps = ckpt_every_kimg * steps_per_kimg if ckpt_every_kimg > 0 else 0
    vis_every_steps = vis_every_kimg * steps_per_kimg if vis_every_kimg > 0 else 0

    tqdm_ncols = int(log_cfg.get("tqdm_ncols", 120))
    tqdm_bar_len = int(log_cfg.get("tqdm_bar_len", 30))

    log_dir = str(log_cfg.get("log_dir", "outputs/logs"))
    log_file = str(log_cfg.get("log_file", "train_log.jsonl"))
    text_log_file = str(log_cfg.get("text_log_file", "train_console.log"))
    log_path = os.path.join(log_dir, log_file)
    text_log_path = os.path.join(log_dir, text_log_file)
    loss_plot_path = os.path.join(log_dir, "loss_curves.png")

    os.makedirs(log_dir, exist_ok=True)
    _set_active_text_log_path(text_log_path)

    vis_dir = str(vis_cfg.get("out_dir", "outputs/vis"))
    vis_num = int(vis_cfg.get("num_samples", 4))

    # checkpoint
    ckpt_dir = str(train_cfg.get("checkpoint_dir", _default_checkpoint_dir()))
    resume = bool(train_cfg.get("resume", False))
    resume_path = train_cfg.get("resume_path")
    resume_path = str(resume_path) if isinstance(resume_path, str) and resume_path else ""
    restore_rng = bool(train_cfg.get("resume_restore_rng", True))

    _ensure_dir(ckpt_dir)

    # data
    dataset_a = DomainLatentDataset(
        a_dir,
        latent_divisor=latent_divisor,
        max_samples=max_samples_a,
        seed=seed,
    )
    dataset_b = DomainLatentDataset(
        b_dir,
        latent_divisor=latent_divisor,
        max_samples=max_samples_b,
        seed=seed,
    )

    loader_a = DataLoader(
        dataset_a,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_latents,
    )
    loader_b = DataLoader(
        dataset_b,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_latents,
    )

    loader_a_iter = get_infinite_iterator(loader_a)
    loader_b_iter = get_infinite_iterator(loader_b)

    in_ch = int(model_cfg.get("in_channels", 4))
    out_ch = int(model_cfg.get("out_channels", 4))
    ngf = int(model_cfg.get("ngf", 32))
    ndf = int(model_cfg.get("ndf", 32))
    n_res_blocks = int(model_cfg.get("n_res_blocks", 6))
    d_layers = int(model_cfg.get("d_layers", 3))
    use_discriminator_sn = bool(model_cfg.get("use_discriminator_sn", True))
    out_activation = str(model_cfg.get("out_activation", "none"))
    use_global_residual = bool(model_cfg.get("use_global_residual", False))
    use_pointwise_only = bool(model_cfg.get("use_pointwise_only", False))

    G = ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
        use_global_residual=use_global_residual,
        use_pointwise_only=use_pointwise_only,
    ).to(device)
    F = ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
        use_global_residual=use_global_residual,
        use_pointwise_only=use_pointwise_only,
    ).to(device)
    D_A = PatchDiscriminator(
        in_ch=in_ch,
        ndf=ndf,
        n_layers=d_layers,
        use_spectral_norm=use_discriminator_sn,
    ).to(device)
    D_B = PatchDiscriminator(
        in_ch=in_ch,
        ndf=ndf,
        n_layers=d_layers,
        use_spectral_norm=use_discriminator_sn,
    ).to(device)

    criterion_gan = nn.MSELoss()
    criterion_cyc = nn.L1Loss()

    opt_G = torch.optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(beta1, beta2))

    def lr_lambda(step: int) -> float:
        current_kimg = step * batch_size / 1000.0
        if lr_decay_kimg <= 0:
            return 1.0
        if current_kimg <= lr_constant_kimg:
            return 1.0
        t = (current_kimg - lr_constant_kimg) / float(lr_decay_kimg)
        return max(0.0, 1.0 - t)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)

    scaler = GradScaler(device.type, enabled=amp)

    start_step = 1
    resumed_kimg = 0.0
    resumed_from_checkpoint = False
    if resume:
        candidate = resume_path if resume_path else _checkpoint_last_path(ckpt_dir)
        if os.path.exists(candidate):
            loaded_step, resumed_kimg = load_checkpoint(
                candidate,
                device=device,
                G=G,
                F=F,
                D_A=D_A,
                D_B=D_B,
                opt_G=opt_G,
                opt_D=opt_D,
                sched_G=sched_G,
                sched_D=sched_D,
                scaler=scaler,
                restore_rng=restore_rng,
            )
            resumed_from_checkpoint = True
            start_step = loaded_step + 1
            log_message(f"Resumed from checkpoint: {candidate} (step={loaded_step}, kimg={resumed_kimg:.3f})")
        else:
            log_message(f"[warn] resume enabled but checkpoint not found: {candidate}. Start from scratch.")

    if not resumed_from_checkpoint:
        # Init weights from N(0, 0.02) only when starting from scratch
        G.apply(init_weights_normal)
        F.apply(init_weights_normal)
        D_A.apply(init_weights_normal)
        D_B.apply(init_weights_normal)
        if use_global_residual:
            zero_init_module(G.output_conv)
            zero_init_module(F.output_conv)

    fake_a_buffer = ReplayBuffer(max_size=max(0, fake_buffer_size), p_use_history=fake_buffer_prob)
    fake_b_buffer = ReplayBuffer(max_size=max(0, fake_buffer_size), p_use_history=fake_buffer_prob)

    # visualization setup (lazy load VAE)
    vae = None
    vae_scaling_factor = 0.18215
    vae_name, vae_subfolder = resolve_vae_source(cfg, vis_cfg)
    if isinstance(vis_cfg.get("vae_scaling_factor"), (float, int)):
        vae_scaling_factor = float(vis_cfg.get("vae_scaling_factor"))

    os.makedirs(vis_dir, exist_ok=True)

    metric_logs_for_plot: List[Dict[str, Any]] = []
    losses_g_total: List[float] = []
    losses_d_total: List[float] = []
    losses_g_gan: List[float] = []
    losses_f_gan: List[float] = []
    losses_cyc: List[float] = []
    losses_id: List[float] = []
    losses_d_a: List[float] = []
    losses_d_b: List[float] = []
    train_start = time.time()
    log_window_start = train_start

    if start_step > total_steps:
        log_message(
            f"Checkpoint already reached or exceeded total_steps: start_step={start_step}, total_steps={total_steps}. "
            f"Nothing to do."
        )
        return

    use_tqdm = sys.stderr.isatty()

    pbar = tqdm(
        total=total_steps,
        initial=start_step - 1,
        desc="training",
        dynamic_ncols=False,
        ncols=tqdm_ncols,
        ascii=True,
        bar_format=f"{{l_bar}}{{bar:{tqdm_bar_len}}}{{r_bar}}",
        disable=not use_tqdm,
    )

    for step in range(start_step, total_steps + 1):
        real_a = next(loader_a_iter)
        real_b = next(loader_b_iter)

        real_a = real_a.to(device, non_blocking=True)
        real_b = real_b.to(device, non_blocking=True)

        ones_a = None
        zeros_a = None
        ones_b = None
        zeros_b = None

        # ------------------ Generators ------------------
        opt_G.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp):
            fake_b = G(real_a)
            fake_a = F(real_b)

            rec_a = F(fake_b)
            rec_b = G(fake_a)

            pred_fake_b = D_B(fake_b)
            pred_fake_a = D_A(fake_a)

            ones_b = torch.ones_like(pred_fake_b)
            ones_a = torch.ones_like(pred_fake_a)

            loss_g_gan = criterion_gan(pred_fake_b, ones_b)
            loss_f_gan = criterion_gan(pred_fake_a, ones_a)

            loss_cycle = criterion_cyc(rec_a, real_a) + criterion_cyc(rec_b, real_b)

            if lambda_id > 0:
                # Identity mapping: G(B)≈B and F(A)≈A
                id_b = G(real_b)
                id_a = F(real_a)
                loss_identity = criterion_cyc(id_b, real_b) + criterion_cyc(id_a, real_a)
            else:
                loss_identity = torch.zeros((), device=device, dtype=fake_a.dtype)

            # Common CycleGAN practice: scale identity by lambda_cyc as well
            loss_g = loss_g_gan + loss_f_gan + lambda_cyc * loss_cycle + (lambda_cyc * lambda_id) * loss_identity

        scaler.scale(loss_g).backward()
        scaler.step(opt_G)

        # ------------------ Discriminators ------------------
        opt_D.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp):
            # D_A
            pred_real_a = D_A(real_a)
            fake_a_for_d = fake_a_buffer.push_and_pop(fake_a)
            pred_fake_a_det = D_A(fake_a_for_d)
            zeros_a = torch.zeros_like(pred_fake_a_det)
            ones_a = torch.ones_like(pred_real_a)
            loss_d_a_val = 0.5 * (criterion_gan(pred_real_a, ones_a) + criterion_gan(pred_fake_a_det, zeros_a))

            # D_B
            pred_real_b = D_B(real_b)
            fake_b_for_d = fake_b_buffer.push_and_pop(fake_b)
            pred_fake_b_det = D_B(fake_b_for_d)
            zeros_b = torch.zeros_like(pred_fake_b_det)
            ones_b = torch.ones_like(pred_real_b)
            loss_d_b_val = 0.5 * (criterion_gan(pred_real_b, ones_b) + criterion_gan(pred_fake_b_det, zeros_b))

            loss_d = (loss_d_a_val + loss_d_b_val) * d_loss_scale

        scaler.scale(loss_d).backward()

        scale_before_update = scaler.get_scale() if amp else 1.0
        scaler.step(opt_D)
        scaler.update()
        scale_after_update = scaler.get_scale() if amp else 1.0
        should_step_scheduler = (not amp) or (scale_after_update >= scale_before_update)
        if should_step_scheduler:
            sched_G.step()
            sched_D.step()

        # logging
        losses_g_total.append(float(loss_g.detach().cpu().item()))
        losses_d_total.append(float(loss_d.detach().cpu().item()))
        losses_g_gan.append(float(loss_g_gan.detach().cpu().item()))
        losses_f_gan.append(float(loss_f_gan.detach().cpu().item()))
        losses_cyc.append(float(loss_cycle.detach().cpu().item()))
        losses_id.append(float(loss_identity.detach().cpu().item()))
        losses_d_a.append(float(loss_d_a_val.detach().cpu().item()))
        losses_d_b.append(float(loss_d_b_val.detach().cpu().item()))

        current_kimg = step * batch_size / 1000.0
        lr_g = float(opt_G.param_groups[0]["lr"])
        lr_d = float(opt_D.param_groups[0]["lr"])

        if use_tqdm:
            pbar.set_postfix_str(
                f"step {step}/{total_steps} | kimg {current_kimg:.3f} | G {losses_g_total[-1]:.3f} | D {losses_d_total[-1]:.3f}",
                refresh=True,
            )
        pbar.update(1)

        should_log = (log_every_steps > 0 and step % log_every_steps == 0) or step == total_steps
        if should_log:
            metric_log = {
                "step": step,
                "total_steps": total_steps,
                "kimg": current_kimg,
                "total_kimg": total_kimg,
                "steps_per_kimg": steps_per_kimg,
                "time_sec": time.time() - log_window_start,
                "elapsed_total_sec": time.time() - train_start,
                "loss_G": _mean(losses_g_total),
                "loss_D": _mean(losses_d_total),
                "loss_gan_G": _mean(losses_g_gan),
                "loss_gan_F": _mean(losses_f_gan),
                "loss_cyc": _mean(losses_cyc),
                "loss_id": _mean(losses_id),
                "loss_D_A": _mean(losses_d_a),
                "loss_D_B": _mean(losses_d_b),
                "lr_G": lr_g,
                "lr_D": lr_d,
                "lambda_cyc": lambda_cyc,
                "lambda_id": lambda_id,
                "d_loss_scale": d_loss_scale,
                "lr_constant_kimg": lr_constant_kimg,
                "lr_decay_kimg": lr_decay_kimg,
                "latents_scaled": latents_scaled,
                "latent_divisor": latent_divisor,
            }
            write_jsonl_line(log_path, metric_log)
            metric_logs_for_plot.append(metric_log)
            _plot_loss_curves(metric_logs_for_plot, loss_plot_path)

            if not use_tqdm:
                log_message(
                    "train "
                    f"step={step}/{total_steps} "
                    f"kimg={current_kimg:.3f} "
                    f"loss_G={metric_log['loss_G']:.4f} "
                    f"loss_D={metric_log['loss_D']:.4f} "
                    f"lr_G={lr_g:.6f} "
                    f"lr_D={lr_d:.6f}",
                    flush=True,
                )

            losses_g_total.clear()
            losses_d_total.clear()
            losses_g_gan.clear()
            losses_f_gan.clear()
            losses_cyc.clear()
            losses_id.clear()
            losses_d_a.clear()
            losses_d_b.clear()
            log_window_start = time.time()

        should_save_ckpt = (ckpt_every_steps > 0 and step % ckpt_every_steps == 0) or step == total_steps
        if should_save_ckpt:
            completed_kimg = step // steps_per_kimg
            ckpt_path = os.path.join(ckpt_dir, f"kimg_{completed_kimg:05d}.pt")
            save_checkpoint(
                ckpt_path=ckpt_path,
                step=step,
                current_kimg=current_kimg,
                G=G,
                F=F,
                D_A=D_A,
                D_B=D_B,
                opt_G=opt_G,
                opt_D=opt_D,
                sched_G=sched_G,
                sched_D=sched_D,
                scaler=scaler,
                cfg=cfg,
            )
            save_checkpoint(
                ckpt_path=_checkpoint_last_path(ckpt_dir),
                step=step,
                current_kimg=current_kimg,
                G=G,
                F=F,
                D_A=D_A,
                D_B=D_B,
                opt_G=opt_G,
                opt_D=opt_D,
                sched_G=sched_G,
                sched_D=sched_D,
                scaler=scaler,
                cfg=cfg,
            )

        should_visualize = vis_every_steps > 0 and step % vis_every_steps == 0
        if should_visualize:
            if not (isinstance(vae_name, str) and vae_name):
                log_message("[warn] visualization.vae_model_name_or_path not set; skip visualization")
            else:
                if vae is None:
                    log_message("Loading VAE for visualization...")
                    vae = _load_vae(vae_name, vae_subfolder, device)

                from torchvision.utils import make_grid, save_image  # type: ignore

                # build a fixed small batch
                a_lat = torch.stack([dataset_a[i] for i in range(min(vis_num, len(dataset_a)))], dim=0).to(device)
                b_lat = torch.stack([dataset_b[i] for i in range(min(vis_num, len(dataset_b)))], dim=0).to(device)

                with torch.no_grad():
                    fake_b_lat = G(a_lat)
                    fake_a_lat = F(b_lat)

                    a_img = decode_latents_to_images(
                        vae,
                        a_lat,
                        latents_scaled=latents_scaled,
                        scaling_factor=vae_scaling_factor,
                        latent_divisor=latent_divisor,
                    )
                    b_img = decode_latents_to_images(
                        vae,
                        b_lat,
                        latents_scaled=latents_scaled,
                        scaling_factor=vae_scaling_factor,
                        latent_divisor=latent_divisor,
                    )
                    fake_b_img = decode_latents_to_images(
                        vae,
                        fake_b_lat,
                        latents_scaled=latents_scaled,
                        scaling_factor=vae_scaling_factor,
                        latent_divisor=latent_divisor,
                    )
                    fake_a_img = decode_latents_to_images(
                        vae,
                        fake_a_lat,
                        latents_scaled=latents_scaled,
                        scaling_factor=vae_scaling_factor,
                        latent_divisor=latent_divisor,
                    )

                # layout: each sample is [A, G(A), B, F(B)]
                tiles: List[torch.Tensor] = []
                for i in range(min(vis_num, a_img.shape[0], b_img.shape[0])):
                    tiles += [a_img[i], fake_b_img[i], b_img[i], fake_a_img[i]]

                grid = make_grid(torch.stack(tiles, dim=0), nrow=4)
                completed_kimg = step // steps_per_kimg
                out_path = os.path.join(vis_dir, f"kimg_{completed_kimg:05d}.png")
                save_image(grid, out_path)
                log_message(f"Saved visualization: {out_path}")

    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CycleGAN training on SD1.5 VAE latents (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.yaml",
        help="Path to YAML config (default: configs/example.yaml)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
