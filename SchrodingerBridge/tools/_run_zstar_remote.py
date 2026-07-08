"""Batch Z-STAR (CVPR 2024) inference on D5 / P2A / R5 on remote RTX 3060 12GB.

Z*: Zero-shot Style Transfer via Attention Rearrangement
  Yingying Deng, Xiangyu He, Fan Tang, Weiming Dong (CAS/IAT)
  CVPR 2024

Core algorithm:
  1. DDIM-invert content image → content_latents + null-text embeddings
  2. DDIM-invert style image → style_latents (intermediate latents per step)
  3. Dual-path denoising from x_T (content inversion endpoint):
     - Style path latents: directly take from style_inversion[-1-i]
     - Content path latents: blend content_inversion[-1-i] * 0.01 + current * 0.99
     - Attention rearrangement (ReweightCrossAttentionControl): at specified
       self-attention layers/steps, mix style-content cross-attention maps
       into the content denoising path
  4. VAE decode final latent → stylized image

This script is self-contained: no external zstar/ package needed.
It must be scp'd to remote and run there.

Remote environment:
  - Windows Server, RTX 3060 12GB
  - Python: C:\\Program Files\\Python312\\python.exe
  - PyTorch 2.11.0+cu128, diffusers 0.38.0
  - SD1.5 cached at C:\\Users\\Administrator\\.cache\\huggingface\\
"""
import os, sys, json, time, gc, argparse, pickle
from pathlib import Path
from typing import Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat
from diffusers import StableDiffusionPipeline, DDIMScheduler

# ═══════════════════════════════════════════════════════════════════
#  Z-STAR hyper-parameters (from official demo.py)
# ═══════════════════════════════════════════════════════════════════
TARGET_IMG_SIZE = 384          # internal processing resolution (reduced from 560→512→384 for 12GB VRAM)
TOTAL_STEP = 20               # DDIM steps for inversion + denoising (reduced from 30 for VRAM)
NUM_DDIM_STEPS = TOTAL_STEP
GUIDANCE_SCALE = 7.5
SEED = 9999
START_STEP = 5                # attention rearrangement start step
END_STEP = 20                  # attention rearrangement end step
LAYER_INDEX = [16, 18, 20, 22, 24, 26]  # self-attn layers to rearrange

DEVICE = "cuda"
DEFAULT_SD = r"C:\Users\Administrator\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

OUT_ROOT = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_zstar")
CACHE_ROOT = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_zstar/cache")

# ═══════════════════════════════════════════════════════════════════
#  Dataset definitions (aligned with StyleAligned remote script)
# ═══════════════════════════════════════════════════════════════════
D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
D5_SIZE = 512

P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
P2A_SIZE = 256

R5_HOLDOUT = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
R5_SIZE = 512


# ═══════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════
def list_images(d: Path) -> list[Path]:
    """List all image files in a directory, sorted."""
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def load_image(image_path: str, device: str, reverse: bool = False) -> torch.Tensor:
    """Load an image, resize to TARGET_IMG_SIZE, return tensor in [-1,1]."""
    from torchvision import transforms as T
    totensor = T.ToTensor()
    image = totensor(Image.open(image_path))
    image = image[:3].unsqueeze_(0).float() * 2.0 - 1.0
    image = F.interpolate(image, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    if reverse:
        image = torch.flip(image, dims=[2])
    return image.to(device)


# ═══════════════════════════════════════════════════════════════════
#  ZstarPipeline — extends StableDiffusionPipeline with invert + step
# ═══════════════════════════════════════════════════════════════════
class ZstarPipeline(StableDiffusionPipeline):
    """SD1.5 pipeline with DDIM inversion and dual-path denoising support."""

    def next_step(self, model_output, timestep, x, eta=0.0, verbose=False):
        """Inverse sampling step for DDIM inversion."""
        next_step_ts = timestep
        timestep = min(
            timestep - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = (self.scheduler.alphas_cumprod[timestep]
                        if timestep >= 0 else self.scheduler.final_alpha_cumprod)
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step_ts]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(self, model_output, timestep, x, eta=0.0, verbose=False):
        """DDIM denoising step."""
        prev_timestep = (timestep
                         - self.scheduler.config.num_train_timesteps
                         // self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep]
                             if prev_timestep > 0
                             else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[1] == 4:
            # Already a 4-channel latent tensor
            latents = image
        elif isinstance(image, torch.Tensor) and image.dim() == 4:
            # 4D tensor but not 4-channel → pixel-space image, encode via VAE
            image = image.to(device=self.device, dtype=self.vae.dtype)
            latents = self.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
            latents = self.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents.to(self.vae.dtype))["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def __call__(self, prompt, batch_size=1, height=TARGET_IMG_SIZE,
                 width=TARGET_IMG_SIZE, num_inference_steps=TOTAL_STEP,
                 guidance_scale=GUIDANCE_SCALE, eta=0.0, latents=None,
                 uncond_embeddings=None, neg_prompt=None,
                 ref_intermediate_latents=None, return_intermediates=False,
                 **kwds):
        """Dual-path DDIM denoising with attention rearrangement.

        When ref_intermediate_latents is provided:
          ref_intermediate_latents[0] = content_ddim_latents (from inversion)
          ref_intermediate_latents[1] = style_ddim_latents  (from inversion)
        The batch must be >= 2 (style path + content path).
        """
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels,
                         height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device)
        else:
            assert latents.shape == latents_shape

        # unconditional embedding for classifier free guidance
        uncond_embeddings_ = None
        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            if uncond_embeddings is None:
                uncond_input = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length", max_length=max_length,
                    return_tensors="pt")
                uncond_embeddings_ = self.text_encoder(
                    uncond_input.input_ids.to(self.device))[0]

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(tqdm(self.scheduler.timesteps,
                                    desc="DDIM Sampler", leave=False)):
            if uncond_embeddings_ is None and uncond_embeddings is not None:
                context = torch.cat([
                    uncond_embeddings[i].expand(*text_embeddings.shape),
                    text_embeddings,
                ])
            else:
                if uncond_embeddings_ is not None:
                    context = torch.cat([uncond_embeddings_, text_embeddings])
                else:
                    context = text_embeddings

            if ref_intermediate_latents is not None:
                # Dual-path latent injection
                style_latents_ref = ref_intermediate_latents[1][-1 - i]
                _, content_latents_cur = latents.chunk(2)
                content_latents_cur = (
                    ref_intermediate_latents[0][-1 - i] * 0.01
                    + content_latents_cur * 0.99
                )
                latents = torch.cat([style_latents_ref, content_latents_cur])

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            noise_pred = self.unet(
                model_inputs.half(), t,
                encoder_hidden_states=context.half()).sample
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon)

            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt")
                           for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt")
                           for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(self, image, prompt, num_inference_steps=TOTAL_STEP,
               guidance_scale=GUIDANCE_SCALE, eta=0.0,
               return_intermediates=False, **kwds):
        """DDIM inversion: real image → noise map."""
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]

        latents = self.image2latent(image)
        start_latents = latents

        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length", max_length=77, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0)

        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps),
                                    desc="DDIM Inversion", leave=False)):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            noise_pred = self.unet(
                model_inputs.half(), t,
                encoder_hidden_states=text_embeddings.half()).sample
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents, start_latents


# ═══════════════════════════════════════════════════════════════════
#  Attention editing infrastructure (from zstar_utils.py + zstar.py)
# ═══════════════════════════════════════════════════════════════════
class AttentionBase:
    """Base class for attention editors. Tracks current step/layer."""
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet,
                 num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet,
                           num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet,
                num_heads, **kwargs):
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class ReweightCrossAttentionControl(AttentionBase):
    """Z-STAR core: rearrange self-attention by mixing style↔content maps.

    At specified steps and layers (self-attention only):
    - Compute cross-path attention similarities:
        style→content (sc) and content→style (cs)
    - Reconstruct style-path attn:  softmax([1.5*sc, ss]) × [v_content, v_style]
    - Reconstruct content-path attn: softmax([1.2*cs, cc]) × [v_style, v_content]
    This injects style attention patterns into the content denoising path
    while preserving content structure.
    """

    def __init__(self, start_step=5, end_step=30, layer_idx=None,
                 step_idx=None, total_steps=TOTAL_STEP,
                 content_img_name=None):
        super().__init__()
        self.total_layers = 16
        self.total_steps = total_steps
        self.start_step = max(0, start_step)
        self.end_step = min(end_step, total_steps)
        self.layer_idx = layer_idx if layer_idx is not None else list(
            range(self.start_step, self.end_step))
        self.step_idx = step_idx if step_idx is not None else list(
            range(self.start_step, self.end_step))
        self.content_img_name = content_img_name

    def get_batch_sim(self, q, k, num_heads, **kwargs):
        """Compute Q·K^T similarity for a single batch path."""
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        return sim

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet,
                num_heads, **kwargs):
        """Attention forward: rearrange self-attention if in target steps/layers."""
        if (is_cross
                or self.cur_step not in self.step_idx
                or self.cur_att_layer not in self.layer_idx):
            return super().forward(q, k, v, sim, attn, is_cross,
                                   place_in_unet, num_heads, **kwargs)

        # Split into style (first half) and content (second half) paths
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        # Compute cross-path and same-path attention similarities
        # U = uncondition, C = condition branch; [0:nh] = style, [nh:] = content
        style_style_u = self.get_batch_sim(
            qu[:num_heads], ku[:num_heads], num_heads, **kwargs)
        style_style_c = self.get_batch_sim(
            qc[:num_heads], kc[:num_heads], num_heads, **kwargs)
        content_content_u = self.get_batch_sim(
            qu[-num_heads:], ku[-num_heads:], num_heads, **kwargs)
        content_content_c = self.get_batch_sim(
            qc[-num_heads:], kc[-num_heads:], num_heads, **kwargs)
        style_content_u = self.get_batch_sim(
            qu[:num_heads], ku[-num_heads:], num_heads, **kwargs)
        style_content_c = self.get_batch_sim(
            qc[:num_heads], kc[-num_heads:], num_heads, **kwargs)
        content_style_u = self.get_batch_sim(
            qu[-num_heads:], ku[:num_heads], num_heads, **kwargs)
        content_style_c = self.get_batch_sim(
            qc[-num_heads:], kc[:num_heads], num_heads, **kwargs)

        # Reweighting factors (from official code)
        content_style_u *= 1.2
        content_style_c *= 1.2
        style_content_u *= 1.5
        style_content_c *= 1.5

        b = qu[-num_heads:].shape[0] // num_heads

        # Content path: softmax([cs, cc]) × [v_style, v_content]
        cscc_u = torch.cat((content_style_u, content_content_u), 2)
        cscc_c = torch.cat((content_style_c, content_content_c), 2)
        vu_cscc = torch.cat((vu[:num_heads], vu[-num_heads:]), 1)
        vc_cscc = torch.cat((vc[:num_heads], vc[-num_heads:]), 1)

        cscc_u = cscc_u.softmax(-1)
        cscc_c = cscc_c.softmax(-1)
        mixup_u = torch.einsum("h i j, h j d -> h i d", cscc_u, vu_cscc)
        mixup_u = rearrange(mixup_u, "h (b n) d -> b n (h d)", b=b)
        mixup_c = torch.einsum("h i j, h j d -> h i d", cscc_c, vc_cscc)
        mixup_c = rearrange(mixup_c, "h (b n) d -> b n (h d)", b=b)

        # Style path: softmax([sc, ss]) × [v_content, v_style]
        scss_u = torch.cat((style_content_u, style_style_u), 2)
        scss_c = torch.cat((style_content_c, style_style_c), 2)
        vu_scss = torch.cat((vu[-num_heads:], vu[:num_heads]), 1)
        vc_scss = torch.cat((vc[-num_heads:], vc[:num_heads]), 1)

        scss_u = scss_u.softmax(-1)
        scss_c = scss_c.softmax(-1)
        original_u = torch.einsum("h i j, h j d -> h i d", scss_u, vu_scss)
        original_u = rearrange(original_u, "h (b n) d -> b n (h d)", b=b)
        original_c = torch.einsum("h i j, h j d -> h i d", scss_c, vc_scss)
        original_c = rearrange(original_c, "h (b n) d -> b n (h d)", b=b)

        out = torch.cat([original_u, mixup_u, original_c, mixup_c], dim=0)
        return out


def register_attention_editor(model: ZstarPipeline, editor: AttentionBase):
    """Register an attention editor into the pipeline's UNet (from zstar_utils.py)."""
    def ca_forward(self_attn, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None,
                    context=None, mask=None):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask
            to_out = self_attn.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self_attn.to_out[0]
            h = self_attn.heads
            q = self_attn.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self_attn.to_k(context)
            v = self_attn.to_v(context)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
                          (q, k, v))
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self_attn.scale
            if mask is not None:
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)
            attn = sim.softmax(dim=-1)
            out = editor(q, k, v, sim, attn, is_cross, place_in_unet,
                         self_attn.heads, scale=self_attn.scale)
            return to_out(out)
        return forward

    def _register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, "children"):
                count = _register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += _register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += _register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += _register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def register_attention_control(model, editor):
    """Register null attention editor (passthrough). Same as ptp_utils version."""
    if editor is None:
        # Reset UNet forward to default — just re-register with a base editor
        register_attention_editor(model, AttentionBase())
    else:
        register_attention_editor(model, editor)


# ═══════════════════════════════════════════════════════════════════
#  Null-text inversion (from demo.py)
# ═══════════════════════════════════════════════════════════════════
class NullInversion:
    """Null-text optimization for precise DDIM inversion (Prompt-to-Prompt)."""

    def __init__(self, model: ZstarPipeline):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

    @property
    def scheduler(self):
        return self.model.scheduler

    def prev_step(self, model_output, timestep, sample):
        prev_timestep = (timestep
                         - self.scheduler.config.num_train_timesteps
                         // self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep]
                             if prev_timestep >= 0
                             else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
                                ) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample
                       + pred_sample_direction)
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        ts, next_ts = (
            min(timestep - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps, 999),
            timestep)
        alpha_prod_t = (self.scheduler.alphas_cumprod[ts]
                        if ts >= 0 else self.scheduler.final_alpha_cumprod)
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_ts]
        beta_prod_t = 1 - alpha_prod_t
        next_original = (sample - beta_prod_t ** 0.5 * model_output
                         ) / alpha_prod_t ** 0.5
        next_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original + next_dir
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        unet = self.model.unet
        return unet(latents.to(unet.dtype), t,
                    encoder_hidden_states=context.to(unet.dtype))["sample"]

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        unet = self.model.unet
        noise_pred = unet(
            latents_input.to(unet.dtype), t,
            encoder_hidden_states=context.to(unet.dtype))["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        vae = self.model.vae
        image = vae.decode(latents.to(vae.dtype))["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            return (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[1] == 4:
            latents = image
        elif isinstance(image, torch.Tensor) and image.dim() == 4:
            # 4D tensor but not 4-channel → pixel-space, encode via VAE
            vae = self.model.vae
            image = image.to(device=self.model.device, dtype=vae.dtype)
            latents = vae.encode(image)["latent_dist"].mean * 0.18215
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
            vae = self.model.vae
            image = image.to(dtype=vae.dtype)
            latents = vae.encode(image)["latent_dist"].mean * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt")
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt], padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        """Optimize null-text embeddings to reconstruct the original image.

        NOTE: Uses torch.cuda.amp.autocast so UNet forward runs in fp16
        while gradients are managed in fp32. UNet params are frozen.
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings = uncond_embeddings.float()
        cond_embeddings = cond_embeddings.float()
        latent_cur = latents[-1].float()
        latent_prev_list = [l.float() for l in latents]

        # Freeze UNet to avoid storing param gradients
        unet = self.model.unet
        unet_params_frozen = False
        for p in unet.parameters():
            if p.requires_grad:
                p.requires_grad = False
                unet_params_frozen = True

        uncond_embeddings_list = []
        for i in tqdm(range(NUM_DDIM_STEPS), desc="Null-text opt", leave=False):
            uncond_emb = uncond_embeddings.clone().detach().float()
            uncond_emb.requires_grad = True
            optimizer = torch.optim.Adam(
                [uncond_emb], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latent_prev_list[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=unet.dtype == torch.float16):
                    noise_pred_cond = self.model.unet(
                        latent_cur.half() if unet.dtype == torch.float16 else latent_cur,
                        t,
                        encoder_hidden_states=cond_embeddings.half() if unet.dtype == torch.float16 else cond_embeddings
                    )["sample"].float()
            for j in range(num_inner_steps):
                with torch.cuda.amp.autocast(enabled=unet.dtype == torch.float16):
                    noise_pred_uncond = self.model.unet(
                        latent_cur.half() if unet.dtype == torch.float16 else latent_cur,
                        t,
                        encoder_hidden_states=uncond_emb.half() if unet.dtype == torch.float16 else uncond_emb
                    )["sample"].float()
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                    noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.item() < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_emb[:1].detach())
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=unet.dtype == torch.float16):
                    context = torch.cat([uncond_emb, cond_embeddings])
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.model.unet(
                        latents_input.half() if unet.dtype == torch.float16 else latents_input,
                        t,
                        encoder_hidden_states=context.half() if unet.dtype == torch.float16 else context
                    )["sample"].float()
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_cfg = noise_pred_uncond + GUIDANCE_SCALE * (
                        noise_pred_text - noise_pred_uncond)
                    latent_cur = self.prev_step(noise_pred_cfg, t, latent_cur)
            torch.cuda.empty_cache()

        # Unfreeze UNet
        if unet_params_frozen:
            for p in unet.parameters():
                p.requires_grad = True

        return uncond_embeddings_list

    def invert(self, image_path, prompt, num_inner_steps=5,
               early_stop_epsilon=1e-5, verbose=False, skip_null_opt=False):
        """Full inversion: DDIM inversion + optional null-text optimization."""
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_gt = np.array(Image.open(image_path))[:, :, :3]
        image_gt = np.array(Image.fromarray(image_gt).resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE)))
        if verbose:
            print("  DDIM inversion...", flush=True)
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if skip_null_opt:
            if verbose:
                print("  Skipping null-text optimization (saves VRAM)", flush=True)
            # Use default uncond embeddings without optimization
            uncond_input = self.model.tokenizer(
                [""], padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt")
            uncond_emb = self.model.text_encoder(
                uncond_input.input_ids.to(self.model.device))[0]
            uncond_embeddings = [uncond_emb] * NUM_DDIM_STEPS
        else:
            if verbose:
                print("  Null-text optimization...", flush=True)
            uncond_embeddings = self.null_optimization(
                ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents, ddim_latents[-1], uncond_embeddings


# ═══════════════════════════════════════════════════════════════════
#  Dataset inference
# ═══════════════════════════════════════════════════════════════════
def run_dataset(model: ZstarPipeline, null_inversion: NullInversion,
                name: str, test_dir: Path, styles: list[str], img_size: int,
                fp16: bool, subset: int, cache_dir: Path, skip_null_opt: bool = False):
    """Run Z-STAR inference on all style pairs for one dataset."""
    out_dir = OUT_ROOT / name / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect content images across all style dirs
    content_items: list[tuple[str, str, Path]] = []
    for style in styles:
        d = test_dir / style
        for f in list_images(d):
            content_items.append((style, f.stem, f))
    if subset and subset > 0:
        content_items = content_items[:subset]
    print(f"  [{name}] {len(content_items)} content images, styles={styles}",
          flush=True)

    # ── Precompute style inversions (once per style) ──
    style_latents = {}
    for tgt in styles:
        ref_files = list_images(test_dir / tgt)
        if not ref_files:
            continue
        style_img = load_image(str(ref_files[0]), DEVICE)
        editor = AttentionBase()
        register_attention_editor(model, editor)
        _, style_latent_list = model.invert(
            style_img, "", guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=TOTAL_STEP, return_intermediates=True)
        style_latents[tgt] = style_latent_list
        print(f"    style inversion done: {tgt}", flush=True)

    # ── Generate all pairs ──
    prompts = ["", ""]
    total = len(content_items) * len(style_latents)
    done = 0
    t0 = time.time()
    timing_log: list[tuple[str, float]] = []

    for ci, (src_style, stem, cpath) in enumerate(content_items, 1):
        # Cached content null inversion
        cache_file = cache_dir / f"{src_style}__{stem}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                ddim_latents_data, x_t_data, uncond_data = pickle.load(f)
            ddim_latents = [t.to(DEVICE) for t in ddim_latents_data]
            x_t = x_t_data.to(DEVICE)
            uncond_embeddings = [t.to(DEVICE) for t in uncond_data]
            # Free CPU copies
            del ddim_latents_data, x_t_data, uncond_data
        else:
            (_, _), ddim_latents, x_t, uncond_embeddings = null_inversion.invert(
                str(cpath), prompts, verbose=(ci == 1), skip_null_opt=skip_null_opt)
            with open(cache_file, "wb") as f:
                pickle.dump([
                    [t.cpu() for t in ddim_latents],
                    x_t.cpu(),
                    [t.cpu() for t in uncond_embeddings],
                ], f)

        start_code = x_t.expand(len(prompts), -1, -1, -1)

        for tgt in styles:
            if tgt not in style_latents:
                continue
            out_name = f"{src_style}__{stem}__to__{tgt}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                done += 1
                continue

            editor = ReweightCrossAttentionControl(
                START_STEP, END_STEP, layer_idx=LAYER_INDEX,
                total_steps=TOTAL_STEP, content_img_name=str(cpath))
            register_attention_editor(model, editor)

            t_start = time.time()
            with torch.no_grad():
                image_stylized = model(
                    prompts, latents=start_code, guidance_scale=GUIDANCE_SCALE,
                    uncond_embeddings=uncond_embeddings,
                    num_inference_steps=TOTAL_STEP,
                    ref_intermediate_latents=[
                        ddim_latents, style_latents[tgt]])
            dt = time.time() - t_start

            # image_stylized[-1] is the stylized content image (last batch item)
            out_img = image_stylized[-1]
            out_pil = Image.fromarray(
                (out_img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            if img_size != TARGET_IMG_SIZE:
                out_pil = out_pil.resize((img_size, img_size), Image.LANCZOS)
            out_pil.save(str(out_path))
            timing_log.append((out_name, dt))
            done += 1

            if done % 25 == 0 or done == total:
                el = time.time() - t0
                eta = el / max(done, 1) * (total - done) / 60
                vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                print(f"  [{name}] {done}/{total}  dt={dt:.1f}s  "
                      f"elapsed={el/60:.1f}m  eta={eta:.1f}m  "
                      f"VRAM={vram_mb:.0f}MB", flush=True)

    total_time = time.time() - t0
    print(f"  [{name}] complete. {total_time:.1f}s "
          f"({total_time / max(done, 1):.2f} s/img)", flush=True)

    # ── Save metadata ──
    meta = {
        "method": "zstar_sd15_attention_rearrangement",
        "dataset": name,
        "test_dir": str(test_dir),
        "out_dir": str(out_dir),
        "img_size": img_size,
        "target_img_size": TARGET_IMG_SIZE,
        "style_list": styles,
        "total_step": TOTAL_STEP,
        "start_step": START_STEP,
        "end_step": END_STEP,
        "layer_index": LAYER_INDEX,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "fp16": fp16,
        "total_pairs": total,
        "total_generated": done,
        "total_seconds": round(total_time, 1),
        "seconds_per_image": round(total_time / max(done, 1), 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 0),
    }
    meta_path = out_dir.parent / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to {meta_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Z-STAR remote inference")
    ap.add_argument("--sd_model", default=DEFAULT_SD,
                    help="Path to SD1.5 model snapshot")
    ap.add_argument("--fp16", action="store_true",
                    help="Run in float16 (saves VRAM)")
    ap.add_argument("--subset", type=int, default=0,
                    help="Limit content images per dataset (0=all)")
    ap.add_argument("--datasets", default="D5,P2A,R5",
                    help="Comma-separated dataset names to run")
    ap.add_argument("--skip_null_opt", action="store_true",
                    help="Skip null-text optimization (saves ~5GB VRAM)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 60, flush=True)
    print("Z-STAR Remote Inference — D5 + P2A + R5", flush=True)
    print("=" * 60, flush=True)

    # ── Load model ──
    print("\nLoading ZstarPipeline (SD1.5) ...", flush=True)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False)
    model = ZstarPipeline.from_pretrained(
        args.sd_model, scheduler=scheduler,
        safety_checker=None, requires_safety_checker=False).to(DEVICE)
    # Note: gradient checkpointing only helps with backward, not inference
    if args.fp16:
        # ZstarPipeline is a custom subclass without .half(); convert components.
        # UNet calls use .half() cast on inputs to handle fp32 latents from invert/null_opt.
        model.unet = model.unet.half()
        model.vae = model.vae.half()
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            model.text_encoder = model.text_encoder.half()
        print("  running in fp16 (unet/vae/text_encoder .half()); null_opt stays fp32", flush=True)
    else:
        print("  running in fp32", flush=True)

    null_inversion = NullInversion(model)

    # ── Build dataset specs ──
    specs = []

    if "D5" in args.datasets:
        d5_dir = Path("I:/datasets/wikiarts20_512_test")
        if d5_dir.exists():
            styles = [s for s in D5_STYLES if (d5_dir / s).exists()]
            if styles:
                specs.append(("D5", d5_dir, styles, D5_SIZE))
            else:
                print("  SKIP D5: no valid style dirs found", flush=True)
        else:
            print(f"  SKIP D5: {d5_dir} not found", flush=True)

    if "P2A" in args.datasets:
        p2a_dir = Path("I:/datasets/legacy256_overfit50/test")
        if p2a_dir.exists():
            styles = [s for s in P2A_STYLES if (p2a_dir / s).exists()]
            if styles:
                specs.append(("P2A", p2a_dir, styles, P2A_SIZE))
            else:
                print("  SKIP P2A: no valid style dirs found", flush=True)
        else:
            print(f"  SKIP P2A: {p2a_dir} not found", flush=True)

    if "R5" in args.datasets:
        r5_dir = Path("I:/datasets/wikiarts20_512_test")
        if r5_dir.exists():
            all_dirs = sorted(d.name for d in r5_dir.iterdir()
                              if d.is_dir() and not d.name.startswith('.'))
            distinct5 = set(D5_STYLES)
            styles = [s for s in R5_HOLDOUT if s in all_dirs]
            if len(styles) < 5:
                # Fallback: pick 5 styles not in D5
                styles = sorted(
                    [s for s in all_dirs if s not in distinct5])[:5]
            if styles:
                specs.append(("R5", r5_dir, styles, R5_SIZE))
            else:
                print("  SKIP R5: no valid style dirs found", flush=True)
        else:
            print(f"  SKIP R5: {r5_dir} not found", flush=True)

    # ── Run each dataset ──
    for name, test_dir, styles, img_size in specs:
        print(f"\n{'=' * 40}", flush=True)
        print(f" Z-STAR [{name}]  {len(styles)} styles × ? images", flush=True)
        print(f"{'=' * 40}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        run_dataset(model, null_inversion, name, test_dir, styles,
                    img_size, args.fp16, args.subset, CACHE_ROOT / name,
                    skip_null_opt=args.skip_null_opt)
        # Free style inversion cache between datasets
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cleanup ──
    del model, null_inversion
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60, flush=True)
    print("Z-STAR remote inference complete!", flush=True)


if __name__ == "__main__":
    main()
