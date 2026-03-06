from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from lbm.models.embedders import ConditionerWrapper
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestLBM:
    @pytest.fixture()
    def denoiser(self):
        return DiffusersUNet2DCondWrapper(
            in_channels=4,  # VAE channels
            out_channels=4,  # VAE channels
            up_block_types=["CrossAttnUpBlock2D"],
            down_block_types=[
                "CrossAttnDownBlock2D",
            ],
            cross_attention_dim=[320],
            block_out_channels=[320],
            transformer_layers_per_block=[1],
            attention_head_dim=[5],
            norm_num_groups=32,
        )

    @pytest.fixture()
    def conditioner(self):
        return ConditionerWrapper([])

    @pytest.fixture()
    def vae(self):
        return AutoencoderKLDiffusers(AutoencoderKLDiffusersConfig())

    @pytest.fixture()
    def sampling_noise_scheduler(self):
        return FlowMatchEulerDiscreteScheduler()

    @pytest.fixture()
    def training_noise_scheduler(self):
        return FlowMatchEulerDiscreteScheduler()

    @pytest.fixture()
    def model_config(self):
        return LBMConfig(
            source_key="source_image",
            target_key="target_image",
        )

    @pytest.fixture()
    def model_input(self):
        return {
            "source_image": torch.randn(2, 3, 256, 256).to(DEVICE),
            "target_image": torch.randn(2, 3, 256, 256).to(DEVICE),
        }

    @pytest.fixture()
    def model(
        self,
        model_config,
        denoiser,
        vae,
        sampling_noise_scheduler,
        training_noise_scheduler,
        conditioner,
    ):
        return LBMModel(
            config=model_config,
            denoiser=denoiser,
            vae=vae,
            sampling_noise_scheduler=sampling_noise_scheduler,
            training_noise_scheduler=training_noise_scheduler,
            conditioner=conditioner,
        ).to(DEVICE)

    @torch.no_grad()
    def test_model_forward(self, model, model_input):
        model_output = model(
            model_input,
        )
        assert model_output["loss"] > 0.0

    def test_optimizers(self, model, model_input):
        optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=1e-4)

        model.train()
        model_init = deepcopy(model)
        optimizer.zero_grad()
        loss = model(model_input)["loss"]
        loss.backward()
        optimizer.step()
        assert not torch.equal(
            torch.cat([p.flatten() for p in model.denoiser.parameters()]),
            torch.cat([p.flatten() for p in model_init.denoiser.parameters()]),
        )
