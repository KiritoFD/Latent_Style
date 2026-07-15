from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STYLE_ROOT = WORKSPACE / "style_data" / "overfit50"
SB_SRC = ROOT / "src"
if str(SB_SRC) not in sys.path:
    sys.path.insert(0, str(SB_SRC))

from utils.modern_metrics import ClipEmbedder, VggGramEmbedder, compute_cmmd, list_style_images  # noqa: E402


STYLES = ["photo", "Hayao", "monet", "vangogh", "cezanne"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOCAL_CLIP_CANDIDATES = [
    WORKSPACE / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
    WORKSPACE / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32",
]


@dataclass(frozen=True)
class RunSpec:
    method: str
    run: str
    images_dir: Path
    source_root: Path = STYLE_ROOT
    summary_json: Path | None = None
    protocol_json: Path | None = None
    clip_style: float | None = None
    clip_content: float | None = None
    content_lpips: float | None = None
    transfer_clip_style: float | None = None


def parse_name(path: Path) -> tuple[str, str, str] | None:
    if "_to_" not in path.stem:
        return None
    prefix, target = path.stem.rsplit("_to_", 1)
    if "_" not in prefix:
        return None
    src_style, src_stem = prefix.split("_", 1)
    return src_style, src_stem, target


def source_path(source_root: Path, src_style: str, src_stem: str) -> Path | None:
    folder = source_root / src_style
    for ext in IMAGE_EXTS:
        candidate = folder / f"{src_stem}{ext}"
        if candidate.exists():
            return candidate
        prefixed = folder / f"{src_style}__{src_stem}{ext}"
        if prefixed.exists():
            return prefixed
    hits = list(folder.glob(f"{src_stem}.*"))
    if hits:
        return hits[0]
    hits = list(folder.glob(f"{src_style}__{src_stem}.*"))
    return hits[0] if hits else None


def collect_generated(images_dir: Path, source_root: Path) -> tuple[dict[str, list[Path]], list[tuple[Path, Path]]]:
    by_target: dict[str, list[Path]] = {}
    content_pairs: list[tuple[Path, Path]] = []
    for path in sorted(images_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_name(path)
        if parsed is None:
            continue
        src_style, src_stem, target = parsed
        by_target.setdefault(target, []).append(path)
        src = source_path(source_root, src_style, src_stem)
        if src is not None:
            content_pairs.append((src, path))
    return by_target, content_pairs


def resolve_clip_source() -> str:
    for candidate in LOCAL_CLIP_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "openai/clip-vit-base-patch32"


class InceptionFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        for p in self.parameters():
            p.requires_grad_(False)
        self.tf = T.Compose(
            [
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.net(batch)
        if isinstance(out, tuple):
            out = out[0]
        return out.float()

    @torch.inference_mode()
    def encode_paths(self, paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
        feats: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch.append(self.tf(img))
                img.close()
            tensor = torch.stack(batch).to(device)
            feats.append(self.forward(tensor).cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.empty((0, 2048), dtype=np.float32)


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    covmean = linalg.sqrtm(sigma_x.dot(sigma_y))
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma_x + np.eye(sigma_x.shape[0]) * eps).dot(sigma_y + np.eye(sigma_y.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(np.sum((mu_x - mu_y) ** 2) + np.trace(sigma_x) + np.trace(sigma_y) - 2.0 * np.trace(covmean))
    return max(0.0, value)


def polynomial_mmd_2(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    k_xx = (gamma * (x @ x.T) + coef0) ** degree
    k_yy = (gamma * (y @ y.T) + coef0) ** degree
    k_xy = (gamma * (x @ y.T) + coef0) ** degree
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = x.shape[0]
    n = y.shape[0]
    return float(k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2.0 * k_xy.mean())


def mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def load_basic_summary(spec: RunSpec) -> dict[str, Any]:
    if (
        spec.clip_style is not None
        or spec.clip_content is not None
        or spec.content_lpips is not None
        or spec.transfer_clip_style is not None
    ):
        return {
            "clip_style": spec.clip_style,
            "clip_content": spec.clip_content,
            "content_lpips": spec.content_lpips,
            "transfer_clip_style": spec.transfer_clip_style,
        }
    if spec.summary_json and spec.summary_json.exists():
        data = json.loads(spec.summary_json.read_text(encoding="utf-8"))
        overview = data.get("analysis", {}).get("all_pairs_overview", {})
        transfer = data.get("analysis", {}).get("style_transfer_ability", {})
        return {
            "clip_style": overview.get("clip_style"),
            "clip_content": overview.get("clip_content"),
            "content_lpips": overview.get("content_lpips"),
            "transfer_clip_style": transfer.get("clip_style"),
        }
    if spec.protocol_json and spec.protocol_json.exists():
        data = json.loads(spec.protocol_json.read_text(encoding="utf-8"))
        overall = data.get("overall", data)
        if "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                if isinstance(item, dict) and str(item.get("target", "")).upper() == "ALL":
                    overall = item
                    break
        return {
            "clip_style": overall.get("clip_style"),
            "clip_content": overall.get("clip_content"),
            "content_lpips": overall.get("content_lpips") or overall.get("lpips"),
            "transfer_clip_style": overall.get("transfer_clip_style") or overall.get("clip_style"),
        }
    return {}


def try_artfid(spec: RunSpec, content_pairs: list[tuple[Path, Path]], by_target: dict[str, list[Path]], batch_size: int, device: str) -> dict[str, float | None]:
    try:
        from utils.artfid_metric import (
            compute_artfid_content_distance_from_paths,
            compute_artfid_fid_from_paths,
            load_artfid_feature_extractor,
            load_artfid_lpips,
        )

        feature_model = load_artfid_feature_extractor(device=device)
        lpips_loss = load_artfid_lpips(device=device)
        art_fids: list[float | None] = []
        art_style_fids: list[float | None] = []
        for target, gen_paths in by_target.items():
            if not gen_paths:
                continue
            ref_paths = [str(p) for p in list_style_images(spec.source_root / target)]
            gen_paths_s = [str(p) for p in gen_paths]
            style_fid = compute_artfid_fid_from_paths(
                gen_paths_s,
                ref_paths,
                model=feature_model,
                batch_size=batch_size,
                device=device,
                ref_cache={},
                ref_cache_key=f"ref_{target}",
            )
            if style_fid is not None:
                art_style_fids.append(style_fid)
        src_paths = [str(src) for src, _ in content_pairs]
        gen_paths = [str(gen) for _, gen in content_pairs]
        content = compute_artfid_content_distance_from_paths(
            gen_paths,
            src_paths,
            loss_fn=lpips_loss,
            batch_size=batch_size,
            device=device,
        )
        style_mean = mean(art_style_fids)
        if style_mean is not None and content is not None:
            art_fids.append((1.0 + style_mean) * (1.0 + content))
        return {"artfid_fid": style_mean, "artfid_content_lpips": content, "artfid": mean(art_fids)}
    except Exception as exc:
        return {"artfid_fid": None, "artfid_content_lpips": None, "artfid": None, "artfid_error": str(exc)}


def evaluate(spec: RunSpec, *, device: str, batch_size: int, enable_artfid: bool) -> dict[str, Any]:
    by_target, content_pairs = collect_generated(spec.images_dir, spec.source_root)
    all_images = sum(len(v) for v in by_target.values())
    torch_device = torch.device(device)
    inception = InceptionFeat().to(torch_device)
    clip = ClipEmbedder(resolve_clip_source(), device=device)
    gram = VggGramEmbedder(device=device)

    ref_inception: dict[str, np.ndarray] = {}
    ref_clip: dict[str, torch.Tensor] = {}
    ref_gram: dict[str, dict[str, torch.Tensor]] = {}

    fid_vals: list[float | None] = []
    kid_vals: list[float | None] = []
    clip_fid_vals: list[float | None] = []
    cmmd_vals: list[float | None] = []
    gram_micro_vals: list[float | None] = []
    gram_macro_vals: list[float | None] = []

    for target in sorted(by_target):
        gen_paths = by_target[target]
        if not gen_paths:
            continue
        ref_paths = list_style_images(spec.source_root / target)
        if not ref_paths:
            continue
        if target not in ref_inception:
            ref_inception[target] = inception.encode_paths(ref_paths, torch_device, batch_size=batch_size)
            ref_clip[target] = clip.encode_paths(ref_paths, batch_size=batch_size)
            ref_gram[target] = gram.style_prototype(ref_paths, batch_size=batch_size)
        gen_inception = inception.encode_paths(gen_paths, torch_device, batch_size=batch_size)
        gen_clip_t = clip.encode_paths(gen_paths, batch_size=batch_size)
        gen_clip = gen_clip_t.numpy()
        fid_vals.append(frechet_distance(gen_inception, ref_inception[target]))
        kid_vals.append(polynomial_mmd_2(gen_inception, ref_inception[target]))
        clip_fid_vals.append(frechet_distance(gen_clip, ref_clip[target].numpy()))
        cmmd_vals.append(compute_cmmd(ref_clip[target], gen_clip_t, sigma=10.0))
        gm, gma = gram.gram_distances(gen_paths, ref_gram[target], batch_size=batch_size)
        gram_micro_vals.append(gm)
        gram_macro_vals.append(gma)

    basic = load_basic_summary(spec)
    art = try_artfid(spec, content_pairs, by_target, batch_size, device) if enable_artfid else {}
    clip_style = basic.get("clip_style")
    content_lpips = basic.get("content_lpips")
    ec_product = None
    if clip_style is not None and content_lpips is not None:
        ec_product = float(clip_style) * (1.0 - float(content_lpips))

    return {
        "method": spec.method,
        "run": spec.run,
        "images": all_images,
        "clip_style_up": clip_style,
        "clip_content_up": basic.get("clip_content"),
        "lpips_down": content_lpips,
        "ec_effectiveness": clip_style,
        "ec_coherence": (1.0 - float(content_lpips)) if content_lpips is not None else None,
        "ec_product_up": ec_product,
        "gram_micro_down": mean(gram_micro_vals),
        "gram_macro_down": mean(gram_macro_vals),
        "fid_down": mean(fid_vals),
        "kid_down": mean(kid_vals),
        "clip_fid_down": mean(clip_fid_vals),
        "clip_cmmd_down": mean(cmmd_vals),
        "artfid_down": art.get("artfid"),
        "artfid_fid_down": art.get("artfid_fid"),
        "artfid_content_lpips_down": art.get("artfid_content_lpips"),
        "artfid_error": art.get("artfid_error", ""),
        "images_dir": str(spec.images_dir),
    }


def default_specs() -> list[RunSpec]:
    ours = ROOT / "S-add__K-1_C-0_W-20_Col-0"
    rw = WORKSPACE / "Related_Works" / "run_511" / "complete_750"
    return [
        RunSpec("Ours", "epoch_0007", ours / "full_eval" / "epoch_0007" / "images", STYLE_ROOT, ours / "full_eval" / "epoch_0007" / "summary.json"),
        RunSpec("Ours", "epoch_0008", ours / "full_eval" / "epoch_0008" / "images", STYLE_ROOT, ours / "full_eval" / "epoch_0008" / "summary.json"),
        RunSpec("Ours", "residual_1p25", ours / "residual_scale_sweep_epoch7" / "residual_1p25" / "images", STYLE_ROOT, ours / "residual_scale_sweep_epoch7" / "residual_1p25" / "summary.json"),
        RunSpec("SaMST", "samst_strict", rw / "samst_strict" / "images", STYLE_ROOT, protocol_json=rw / "samst_strict" / "eval_protocol750_sbmatch.json"),
        RunSpec(
            "StyleID",
            "styleid_strict",
            rw / "styleid_strict" / "images",
            STYLE_ROOT,
            clip_style=0.7597,
            clip_content=0.5519,
            content_lpips=0.7497,
            transfer_clip_style=0.7597,
        ),
        RunSpec(
            "S2WAT",
            "s2wat_strict",
            rw / "s2wat_strict" / "images",
            STYLE_ROOT,
            clip_style=0.7139,
            clip_content=0.7465,
            content_lpips=0.5263,
            transfer_clip_style=0.7139,
        ),
        RunSpec(
            "AdaIN",
            "adain_v32k",
            rw / "adain_v32k" / "images",
            STYLE_ROOT,
            clip_style=0.7130,
            clip_content=0.6990,
            content_lpips=0.6298,
            transfer_clip_style=0.7130,
        ),
        RunSpec(
            "AdaIN",
            "adain_vgg19",
            rw / "adain_vgg19" / "images",
            STYLE_ROOT,
            clip_style=0.6930,
            clip_content=0.5991,
            content_lpips=0.6870,
            transfer_clip_style=0.6930,
        ),
        RunSpec(
            "AdaIN",
            "adain_bad",
            rw / "adain_bad" / "images",
            STYLE_ROOT,
            clip_style=0.6308,
            clip_content=0.5297,
            content_lpips=0.8490,
            transfer_clip_style=0.6308,
        ),
    ]


def load_specs_from_manifest(path: Path) -> list[RunSpec]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    specs: list[RunSpec] = []
    for row in rows:
        method = str(row.get("method", "")).strip()
        run = str(row.get("run", "")).strip()
        images_dir_raw = str(row.get("images_dir", "")).strip()
        if not method or not run or not images_dir_raw:
            continue
        specs.append(
            RunSpec(
                method=method,
                run=run,
                images_dir=Path(images_dir_raw),
                source_root=Path(row["source_root"]) if str(row.get("source_root", "")).strip() else STYLE_ROOT,
                summary_json=Path(row["summary_json"]) if str(row.get("summary_json", "")).strip() else None,
                protocol_json=Path(row["protocol_json"]) if str(row.get("protocol_json", "")).strip() else None,
                clip_style=float(row["clip_style"]) if str(row.get("clip_style", "")).strip() else None,
                clip_content=float(row["clip_content"]) if str(row.get("clip_content", "")).strip() else None,
                content_lpips=float(row["content_lpips"]) if str(row.get("content_lpips", "")).strip() else None,
                transfer_clip_style=float(row["transfer_clip_style"]) if str(row.get("transfer_clip_style", "")).strip() else None,
            )
        )
    return specs


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Selected Style Metrics",
        "",
        "| Method | Run | Images | CLIP-style up | LPIPS down | EC product up | Gram micro down | Gram macro down | FID down | KID down | CLIP-FID down | CLIP-CMMD down | ArtFID down |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['run']} | {row['images']} | {fmt(row['clip_style_up'])} | "
            f"{fmt(row['lpips_down'])} | {fmt(row['ec_product_up'])} | {fmt(row['gram_micro_down'])} | "
            f"{fmt(row['gram_macro_down'])} | {fmt(row['fid_down'])} | {fmt(row['kid_down'])} | "
            f"{fmt(row['clip_fid_down'])} | {fmt(row['clip_cmmd_down'])} | {fmt(row['artfid_down'])} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "selected_style_metrics.csv")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-artfid", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    specs = load_specs_from_manifest(args.manifest) if args.manifest is not None else default_specs()
    for spec in specs:
        if not spec.images_dir.exists():
            print(f"SKIP missing images: {spec.images_dir}")
            continue
        print(f"Evaluating {spec.method}/{spec.run}: {spec.images_dir}")
        rows.append(evaluate(spec, device=args.device, batch_size=args.batch_size, enable_artfid=bool(args.enable_artfid)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.output.with_suffix(".md"))
    print(args.output)
    print(args.output.with_suffix(".md"))
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
