from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms


warnings.filterwarnings("ignore")

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download  # type: ignore
except Exception:  # pragma: no cover - optional dependency / remote-only path
    ms_snapshot_download = None


class _IntroStyleUNet(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep,
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            dtype = torch.float32 if isinstance(timestep, float) and is_mps else (
                torch.float64 if isinstance(timestep, float) else (torch.int32 if is_mps else torch.int64)
            )
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > int(np.max(up_ft_indices)):
                break
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            if i in up_ft_indices:
                up_ft[i] = sample.detach()
        return {"up_ft": up_ft}


class _IntroStyleOneStep:
    def __init__(
        self,
        *,
        vae: AutoencoderKL,
        unet: _IntroStyleUNet,
        scheduler: DDIMScheduler,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: str,
    ) -> None:
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = str(device)

    def to(self, device: str):
        self.device = str(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return self

    def enable_attention_slicing(self) -> None:
        if hasattr(self.unet, "set_attention_slice"):
            self.unet.set_attention_slice("auto")

    def enable_xformers_memory_efficient_attention(self) -> None:
        if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
            self.unet.enable_xformers_memory_efficient_attention()

    @torch.no_grad()
    def encode_prompt(
        self,
        *,
        prompt: list[str],
        device: str,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
    ):
        del do_classifier_free_guidance
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device) if getattr(tokens, "attention_mask", None) is not None else None
        prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        if int(num_images_per_prompt) > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(int(num_images_per_prompt), dim=0)
        return (prompt_embeds, None)

    @torch.no_grad()
    def __call__(
        self,
        img_tensor: torch.Tensor,
        *,
        t: int,
        up_ft_indices: list[int],
        prompt_embeds: torch.FloatTensor,
        cross_attention_kwargs=None,
    ):
        device = img_tensor.device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        timestep = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, timestep)
        return self.unet(
            latents_noisy,
            timestep,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )


class IntroStyleFeatureExtractor:
    def __init__(
        self,
        *,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda",
        t: int = 25,
        up_ft_index: int = 1,
        ensemble_size: int = 4,
    ) -> None:
        self.device = str(device)
        self.t = int(t)
        self.up_ft_index = int(up_ft_index)
        self.ensemble_size = int(ensemble_size)

        unet = _IntroStyleUNet.from_pretrained(model_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = _IntroStyleOneStep(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=self.device,
        )
        pipe = pipe.to(self.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        self.pipe = pipe
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            return self.preprocess(img.convert("RGB"))

    @torch.no_grad()
    def encode_batch(self, paths: list[Path]) -> torch.Tensor:
        tensors = torch.stack([self._load_image(p) for p in paths], dim=0)
        bsz = tensors.shape[0]
        tensors = tensors.repeat_interleave(self.ensemble_size, dim=0).to(self.device)
        prompt_embeds = self.pipe.encode_prompt(
            prompt=["a photo of"],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0].repeat(self.ensemble_size * bsz, 1, 1)
        out = self.pipe(
            img_tensor=tensors,
            t=self.t,
            up_ft_indices=[self.up_ft_index],
            prompt_embeds=prompt_embeds,
        )
        feats = out["up_ft"][self.up_ft_index]
        feats = torch.cat(
            [feats[i * self.ensemble_size:(i + 1) * self.ensemble_size].mean(0, keepdim=True) for i in range(bsz)],
            dim=0,
        )
        return feats

    @torch.no_grad()
    def encode_paths(self, paths: list[Path], *, batch_size: int = 4) -> torch.Tensor:
        outputs = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            batch_paths = paths[start:start + max(1, int(batch_size))]
            outputs.append(self.encode_batch(batch_paths).cpu())
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1, 1, 1))


def resolve_introstyle_model_path(
    *,
    model_id: str,
    modelscope_id: str = "",
    modelscope_cache_dir: str = "",
    allow_network: bool = False,
) -> str:
    raw_model_id = str(model_id or "").strip()
    if raw_model_id:
        candidate = Path(raw_model_id).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        if "/" in raw_model_id and not Path(raw_model_id).suffix:
            return raw_model_id
        return raw_model_id

    ms_id = str(modelscope_id or "").strip()
    if not ms_id:
        raise ValueError("IntroStyle requires either model_id or modelscope_id.")
    if ms_snapshot_download is None:
        raise RuntimeError("ModelScope is not available for IntroStyle model resolution.")

    cache_dir = Path(str(modelscope_cache_dir or "").strip() or ".").expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"cache_dir": str(cache_dir)}
    if not bool(allow_network):
        kwargs["local_files_only"] = True
    try:
        local_path = ms_snapshot_download(ms_id, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        local_path = ms_snapshot_download(ms_id, **kwargs)
    return str(Path(str(local_path)).resolve())


def introstyle_style_vector(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError(f"Expected BCHW features, got shape={tuple(feat.shape)}")
    bsz = feat.shape[0]
    flat = feat.float().flatten(start_dim=1)
    flat = F.normalize(flat, p=2, dim=1)
    return flat.view(bsz, -1)


def cosine_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), p=2, dim=1)
    y = F.normalize(y.float(), p=2, dim=1)
    return x @ y.T


def topk_mean(sim: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), sim.shape[1]))
    vals = torch.topk(sim, k=k, dim=1).values
    return vals.mean(dim=1)


def list_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]) if root.exists() else []


def style_bank_paths(root: Path, *, per_style_limit: int = 0) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        imgs = list_images(sub)
        if per_style_limit > 0:
            imgs = imgs[:per_style_limit]
        if imgs:
            out[sub.name] = imgs
    return out


def mean_pool_scores(vectors: torch.Tensor, bank_vectors: dict[str, torch.Tensor], *, topk: int = 8) -> dict[str, torch.Tensor]:
    scores = {}
    for style, refs in bank_vectors.items():
        sims = cosine_matrix(vectors, refs)
        scores[style] = topk_mean(sims, topk)
    return scores
