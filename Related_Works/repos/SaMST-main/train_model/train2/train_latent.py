from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

project_root = os.path.abspath("../..")
sys.path.append(project_root)

from loss.vgg import Vgg16
from networks.latent_transfer_net import LatentTransformerNet
from train_model import utils


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SCHRODINGER_SRC = WORKSPACE_ROOT / "SchrodingerBridge" / "src"
if str(SCHRODINGER_SRC) not in sys.path:
    sys.path.append(str(SCHRODINGER_SRC))

from utils.inference import load_vae  # noqa: E402


def _save_named_model(transformer, device, save_dir, filename):
    transformer.eval().cpu()
    save_model_path = os.path.join(save_dir, filename)
    torch.save(transformer.state_dict(), save_model_path)
    print("\ntrained model saved at", save_model_path)
    transformer.to(device).train()


def _load_manifest(root: Path) -> dict[str, object] | None:
    manifest_path = root / ".latent_cache" / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class LatentRef:
    style: str
    index: int


class LatentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.by_style = self._load()
        self.styles = sorted(self.by_style.keys())
        self.refs = [
            LatentRef(style=style, index=index)
            for style in self.styles
            for index in range(len(self.by_style[style]))
        ]

    def _load(self) -> dict[str, list[torch.Tensor]]:
        manifest = _load_manifest(self.root)
        if manifest:
            styles = manifest.get("styles")
            if not isinstance(styles, dict):
                raise TypeError(f"Invalid manifest under {self.root}")
            loaded = {}
            for style, payload in sorted(styles.items()):
                if not isinstance(payload, dict):
                    continue
                packed_rel = payload.get("packed")
                if not packed_rel:
                    raise KeyError(f"Missing packed latent path for style={style} under {self.root}")
                packed_path = self.root / ".latent_cache" / str(packed_rel)
                obj = torch.load(packed_path, map_location="cpu", weights_only=False)
                if not isinstance(obj, dict) or not torch.is_tensor(obj.get("latents")):
                    raise TypeError(f"Unsupported packed latent payload: {packed_path}")
                latents = obj["latents"]
                loaded[style] = [latents[i] for i in range(latents.shape[0])]
            return loaded

        loaded = {}
        for style_dir in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            tensors = []
            for pt in sorted(style_dir.glob("*.pt")):
                payload = torch.load(pt, map_location="cpu", weights_only=False)
                if torch.is_tensor(payload):
                    tensors.append(payload.squeeze(0) if payload.ndim == 4 and payload.shape[0] == 1 else payload)
            if tensors:
                loaded[style_dir.name] = tensors
        return loaded

    def get(self, ref: LatentRef) -> torch.Tensor:
        return self.by_style[ref.style][ref.index].clone().float()


class LatentImageFolder(Dataset):
    def __init__(self, content_root: str | Path, style_names: list[str], max_per_style: int = 0):
        self.store = LatentStore(content_root)
        self.style_names = style_names
        self.refs = []
        for style in style_names:
            style_refs = [ref for ref in self.store.refs if ref.style == style]
            self.refs.extend(style_refs if max_per_style <= 0 else style_refs[:max_per_style])

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        return self.store.get(self.refs[idx]), 0


def check_paths(opt):
    try:
        if not os.path.exists(opt["save_model_dir"]):
            os.makedirs(opt["save_model_dir"])
        if opt["checkpoint_model_dir"] is not None and not os.path.exists(opt["checkpoint_model_dir"]):
            os.makedirs(opt["checkpoint_model_dir"])
    except OSError as exc:
        print(exc)
        sys.exit(1)


def _decode_latent_train(vae, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    dtype = torch.float16 if latent.device.type == "cuda" else torch.float32
    z = latent.to(dtype=dtype) / max(float(scaling_factor), 1e-8)
    decoded = vae.decode(z).sample
    decoded = (decoded + 1.0) / 2.0
    return torch.clamp(decoded, 0.0, 1.0).float()


def train(opt):
    device = torch.device("cuda" if opt["cuda"] else "cpu")

    np.random.seed(opt["seed"])
    torch.manual_seed(opt["seed"])

    style_names = [
        name
        for name in sorted(os.listdir(opt["style_image"]))
        if (Path(opt["style_image"]) / name).is_dir()
    ]
    style_num = len(style_names)
    print("total style number:", style_num)

    dataset = LatentImageFolder(opt["dataset"], style_names=[Path(name).stem for name in style_names], max_per_style=int(opt.get("max_train_per_style") or 0))
    train_loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)

    transformer = LatentTransformerNet(style_num=style_num, latent_channels=int(opt.get("latent_channels", 4)))
    print("# MODEL parameters:", sum(param.numel() for param in transformer.parameters()), "\n")
    begin_epoch = 0
    if opt["begin_checkpoint"] is not None:
        state_dict = torch.load(opt["begin_checkpoint"])
        transformer.load_state_dict(state_dict)
        begin_epoch = opt["begin_epoch"]
    transformer = transformer.to(device)

    optimizer = Adam(transformer.parameters(), opt["lr"])
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    if int(opt.get("loss_network_half") or 0):
        vgg = vgg.to(dtype=torch.float16)

    vae = load_vae(device=str(device), model_id=str(opt.get("vae_model", "ema")), cache_dir=opt.get("vae_cache_dir") or None, enable_xformers=False)
    vae.requires_grad_(False)
    vae.eval()

    style_batch = []
    style_root = Path(opt["style_latent_root"]).expanduser().resolve()
    for style_name in [Path(name).stem for name in style_names]:
        style_dir = style_root / style_name
        first_pt = sorted(style_dir.glob("*.pt"))[0]
        payload = torch.load(first_pt, map_location="cpu", weights_only=False)
        tensor = payload.squeeze(0) if torch.is_tensor(payload) and payload.ndim == 4 and payload.shape[0] == 1 else payload
        style_batch.append(tensor.float())
    style = torch.stack(style_batch).to(device)
    style_rgb = _decode_latent_train(vae, style, opt.get("latent_scaling_factor", 0.18215)) * 255.0
    style_rgb = style_rgb.to(dtype=next(vgg.parameters()).dtype)
    features_style = vgg(utils.normalize_batch(style_rgb.clone()))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    content_weight = float(opt["content_weight"])
    style_weight = float(opt["style_weight"])
    ae_weight = float(opt["ae_weight"])

    total_epochs = opt["epochs"]
    max_steps = int(opt.get("max_steps") or 0)
    global_step = 0
    step_model_template = opt.get("step_model_name_template") or "step_{step:06d}.model"

    for e in range(begin_epoch + 1, total_epochs + 1):
        transformer.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        agg_ae_loss = 0.0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            if n_batch < opt["batch_size"]:
                break

            count += n_batch
            optimizer.zero_grad()
            batch_style_id = [random.randint(1, style_num) for _ in range(count - n_batch, count)]
            for _ in range(n_batch):
                batch_style_id.append(0)

            x = x.to(device).repeat(2, 1, 1, 1)
            y_latent, embedding = transformer(x, style_id=batch_style_id)

            y = _decode_latent_train(vae, y_latent, opt.get("latent_scaling_factor", 0.18215)) * 255.0
            x_rgb = _decode_latent_train(vae, x, opt.get("latent_scaling_factor", 0.18215)) * 255.0
            target_dtype = next(vgg.parameters()).dtype
            y = utils.normalize_batch(y).to(dtype=target_dtype)
            x_rgb = utils.normalize_batch(x_rgb).to(dtype=target_dtype)

            y = torch.split(y, n_batch, dim=0)
            y1 = y[0]
            y2 = y[1]
            x_rgb = torch.split(x_rgb, n_batch, dim=0)
            x1 = x_rgb[0]
            x2 = x_rgb[1]

            features_y = vgg(y1.to(device))
            features_x = vgg(x1.to(device))

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.0
            style_ids = batch_style_id[0:n_batch]
            style_ids = [style_id - 1 for style_id in style_ids]
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[style_ids, :, :])
            style_loss *= style_weight

            ae_loss = ae_weight * mse_loss(y2.to(device), x2.to(device))
            total_loss = content_loss + style_loss + ae_loss
            total_loss.backward()
            optimizer.step()
            global_step += 1

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_ae_loss += ae_loss.item()

            if (batch_id + 1) % opt["log_interval"] == 0:
                mesg = (
                    f"{time.ctime()}\tEpoch {e}:\t[{count}/{len(dataset)}]\t"
                    f"content: {agg_content_loss / (batch_id + 1):.6f}\t"
                    f"style: {agg_style_loss / (batch_id + 1):.6f}\t"
                    f"ae: {agg_ae_loss / (batch_id + 1):.6f}\t"
                    f"total: {(agg_content_loss + agg_style_loss) / (batch_id + 1):.6f}"
                )
                print(mesg)

            if opt["checkpoint_model_dir"] is not None and (batch_id + 1) % opt["checkpoint_interval"] == 0:
                transformer.eval().cpu()
                ckpt_model_filename = f"ckpt_epoch_{e}_batch_id_{batch_id + 1}.pth"
                ckpt_model_path = os.path.join(opt["checkpoint_model_dir"], ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

            if max_steps > 0 and global_step >= max_steps:
                filename = step_model_template.format(step=global_step, epoch=e, batch_id=batch_id + 1)
                _save_named_model(transformer, device, opt["save_model_dir"], filename)
                print("Reached max_steps =", global_step)
                return

        if e % opt["save_interval"] == 0:
            _save_named_model(transformer, device, opt["save_model_dir"], f"epoch_{e}.model")

        if e % opt["step_size"] == 0:
            lr = opt["lr"] * (opt["weight_decay"] ** (e // opt["step_size"]))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("now learning rate: ", optimizer.state_dict()["param_groups"][0]["lr"])


def main():
    with open("train.yml", "r", encoding="utf-8") as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)
    random.seed(7)
    check_paths(opt)
    train(opt)


if __name__ == "__main__":
    main()
