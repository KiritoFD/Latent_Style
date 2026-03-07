import argparse
import datetime
import json
import os
import random
from pathlib import Path

import braceexpand
import torch
from diffusers import StableDiffusionXLPipeline
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.utils import StateDictAdapter


def _unbrace(urls: list[str]) -> list[str]:
    out: list[str] = []
    for u in urls:
        out.extend(braceexpand.braceexpand(u))
    return out


def _get_filter_mappers(image_size: int):
    return [
        KeyFilter(KeyFilterConfig(keys=["src.jpg", "tgt.jpg"])),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "src.jpg": "source_image",
                            "tgt.jpg": "target_image",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="source_image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (int(image_size), int(image_size)),
                                "interpolation": InterpolationMode.BILINEAR,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="target_image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (int(image_size), int(image_size)),
                                "interpolation": InterpolationMode.BILINEAR,
                            },
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="source_image")),
                RescaleMapper(RescaleMapperConfig(key="target_image")),
            ],
        ),
    ]


def _get_data_module(train_shards: list[str], val_shards: list[str], *, batch_size: int, num_workers: int):
    train_urls = _unbrace(train_shards)
    val_urls = _unbrace(val_shards)
    random.shuffle(train_urls)

    train_cfg = DataModuleConfig(
        shards_path_or_urls=train_urls,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=50,
        shuffle_before_split_by_workers_buffer_size=50,
        shuffle_before_filter_mappers_buffer_size=200,
        shuffle_after_filter_mappers_buffer_size=200,
        per_worker_batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    val_cfg = DataModuleConfig(
        shards_path_or_urls=val_urls,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=50,
        shuffle_after_filter_mappers_buffer_size=50,
        per_worker_batch_size=int(batch_size),
        num_workers=max(1, int(min(num_workers, len(val_urls)))),
    )

    data_module = DataModule(
        train_config=train_cfg,
        train_filters_mappers=None,  # filled by caller
        eval_config=val_cfg,
        eval_filters_mappers=None,  # filled by caller
    )
    return data_module


def _get_model(
    *,
    backbone_signature: str,
    image_size: int,
    enable_minibatch_ot: bool,
    bridge_noise_sigma: float,
):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        attention_head_dim=[5, 10, 20],
        use_linear_projection=True,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()
    for k in [
        "add_embedding.linear_1.weight",
        "add_embedding.linear_1.bias",
        "add_embedding.linear_2.weight",
        "add_embedding.linear_2.bias",
    ]:
        if k in state_dict:
            del state_dict[k]

    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )
    denoiser.load_state_dict(state_dict, strict=True)
    del pipe

    denoiser.enable_gradient_checkpointing()

    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    config = LBMConfig(
        source_key="source_image",
        target_key="target_image",
        mask_key=None,
        latent_loss_weight=1.0,
        latent_loss_type="l2",
        pixel_loss_type="lpips",
        pixel_loss_weight=0.0,
        timestep_sampling="log_normal",
        logit_mean=0.0,
        logit_std=1.0,
        selected_timesteps=None,
        prob=None,
        bridge_noise_sigma=float(bridge_noise_sigma),
        enable_minibatch_ot=bool(enable_minibatch_ot),
        minibatch_ot_cost="latent_l2",
        minibatch_ot_detach=True,
        pixel_loss_max_size=int(image_size),
    )

    from diffusers import FlowMatchEulerDiscreteScheduler

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=None,
    ).to(torch.bfloat16)

    return model


def main():
    ap = argparse.ArgumentParser("Train LBM for unpaired style transfer via minibatch OT (Hungarian matching)")
    ap.add_argument("--train_shards", action="append", required=True, help="WebDataset shards pattern(s)")
    ap.add_argument("--val_shards", action="append", required=True, help="WebDataset shards pattern(s)")
    ap.add_argument("--out_dir", required=True, help="Output dir for logs/checkpoints")
    ap.add_argument("--backbone", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--enable_minibatch_ot", action="store_true")
    ap.add_argument("--bridge_noise_sigma", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    model = _get_model(
        backbone_signature=args.backbone,
        image_size=int(args.image_size),
        enable_minibatch_ot=bool(args.enable_minibatch_ot),
        bridge_noise_sigma=float(args.bridge_noise_sigma),
    )

    data_module = _get_data_module(
        args.train_shards,
        args.val_shards,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    fm = _get_filter_mappers(int(args.image_size))
    data_module.train_filters_mappers = fm
    data_module.eval_filters_mappers = fm

    training_config = TrainingConfig(
        learning_rate=float(args.lr),
        lr_scheduler_name=None,
        lr_scheduler_kwargs={},
        log_keys=["source_image", "target_image"],
        trainable_params=["denoiser.*"],
        optimizer_name="AdamW",
        optimizer_kwargs={"betas": (0.9, 0.999), "weight_decay": 0.01},
        log_samples_model_kwargs={"input_shape": (4, int(args.image_size) // 8, int(args.image_size) // 8), "num_steps": [1]},
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    csv = loggers.CSVLogger(save_dir=str(out_dir), name="csv")

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        every_n_train_steps=500,
        save_top_k=-1,
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=int(args.max_steps),
        accumulate_grad_batches=int(args.accum),
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=[csv],
        callbacks=[ckpt, lrmon],
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=str(out_dir),
    )

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "train_shards": args.train_shards,
        "val_shards": args.val_shards,
        "out_dir": str(out_dir),
        "backbone": args.backbone,
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "accum": int(args.accum),
        "num_workers": int(args.num_workers),
        "max_steps": int(args.max_steps),
        "lr": float(args.lr),
        "enable_minibatch_ot": bool(args.enable_minibatch_ot),
        "bridge_noise_sigma": float(args.bridge_noise_sigma),
        "seed": int(args.seed),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer.fit(pipeline, datamodule=data_module)


if __name__ == "__main__":
    # Avoid HF token warnings in logs.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
