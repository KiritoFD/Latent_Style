# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SD 1.5 adaptation of DDIM inversion for StyleAligned transfer."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from tqdm import tqdm

T = torch.Tensor
InversionCallback = Callable[[StableDiffusionPipeline, int, T, dict[str, T]], dict[str, T]]


def _encode_text(model: StableDiffusionPipeline, prompt: str) -> T:
    device = model._execution_device
    text_inputs = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = model.text_encoder(text_inputs.input_ids.to(device))[0]
    return prompt_embeds


def _encode_text_with_negative(model: StableDiffusionPipeline, prompt: str) -> T:
    prompt_embeds = _encode_text(model, prompt)
    uncond_embeds = _encode_text(model, "")
    return torch.cat([uncond_embeds, prompt_embeds])


def _encode_image(model: StableDiffusionPipeline, image: Image.Image | np.ndarray) -> T:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB").resize((512, 512)))
    model.vae.to(dtype=torch.float32)
    image_t = torch.from_numpy(image).float() / 255.0
    image_t = (image_t * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        latent = model.vae.encode(image_t.to(model.vae.device)).latent_dist.mean
    latent = latent * model.vae.config.scaling_factor
    model.vae.to(dtype=model.unet.dtype)
    return latent


def _next_step(model: StableDiffusionPipeline, model_output: T, timestep: int, sample: T) -> T:
    scheduler: DDIMScheduler = model.scheduler
    timestep, next_timestep = (
        min(timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999),
        timestep,
    )
    alpha_prod_t = scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(
    model: StableDiffusionPipeline, latent: T, t: T, context: T, guidance_scale: float
) -> T:
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    return noise_pred


def _ddim_loop(model: StableDiffusionPipeline, z0: T, prompt: str, guidance_scale: float) -> T:
    all_latent = [z0]
    text_embedding = _encode_text_with_negative(model, prompt)
    latent = z0.clone().detach().to(model.unet.dtype)
    for i in tqdm(range(model.scheduler.num_inference_steps), desc="DDIM inversion"):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = _get_noise_pred(model, latent, t, text_embedding, guidance_scale)
        latent = _next_step(model, noise_pred, int(t.item()), latent)
        all_latent.append(latent)
    return torch.cat(all_latent).flip(0)


def make_inversion_callback(zts: T, offset: int = 0) -> tuple[T, InversionCallback]:
    def callback_on_step_end(
        pipeline: StableDiffusionPipeline, i: int, t: T, callback_kwargs: dict[str, T]
    ) -> dict[str, T]:
        latents = callback_kwargs["latents"]
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {"latents": latents}

    return zts[offset], callback_on_step_end


@torch.no_grad()
def ddim_inversion(
    model: StableDiffusionPipeline,
    image: Image.Image | np.ndarray,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
) -> T:
    z0 = _encode_image(model, image)
    if not isinstance(model.scheduler, DDIMScheduler):
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs
