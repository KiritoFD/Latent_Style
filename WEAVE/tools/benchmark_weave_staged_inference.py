from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import (
    LGTInference,
    decode_latent,
    encode_image,
    load_torchscript_vae_decoder,
    load_vae,
)


STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage WEAVE throughput benchmark.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--source-batch", type=int, default=2)
    parser.add_argument("--style-chunk", type=int, default=2)
    parser.add_argument("--decode-batch", type=int, default=16)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-cache-dir", default="")
    parser.add_argument("--vae-torchscript-decoder", default="")
    parser.add_argument("--compile-decoder", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-cache-dir", default="")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--warmup-decode", type=int, default=2)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--style-cache", default="")
    parser.add_argument("--latent-spool", default="")
    parser.add_argument("--decode-spool", default="")
    parser.add_argument("--skip-decode", action="store_true")
    return parser.parse_args()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_source_latents(path: Path) -> list[tuple[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    items = payload.get("items", payload)
    return sorted((str(key), value.to(dtype=torch.float16).contiguous()) for key, value in items.items())


def load_style_latents(test_dir: Path, *, vae, device: str) -> dict[int, torch.Tensor]:
    style_latents: dict[int, torch.Tensor] = {}
    for style_id, style_name in enumerate(STYLE_NAMES):
        paths = sorted(
            path for path in (test_dir / style_name).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not paths:
            raise FileNotFoundError(f"No reference image for {style_name}")
        from utils.run_evaluation import _load_eval_image_tensor

        image = _load_eval_image_tensor(paths[0]).unsqueeze(0).to(device)
        style_latents[style_id] = encode_image(vae, image, device).detach().cpu().to(torch.float16)
    return style_latents


def load_or_build_style_latents(args: argparse.Namespace, *, device: str) -> tuple[dict[int, torch.Tensor], float]:
    cache_path = Path(args.style_cache) if args.style_cache else None
    if cache_path is not None and cache_path.exists():
        t0 = time.perf_counter()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return {int(key): value.to(torch.float16).contiguous() for key, value in payload.items()}, time.perf_counter() - t0
    t0 = time.perf_counter()
    vae = load_vae(
        device=device,
        model_id=args.vae_model,
        cache_dir=args.vae_cache_dir or None,
        enable_xformers=False,
    )
    style_latents = load_style_latents(Path(args.test_dir), vae=vae, device=device)
    sync()
    elapsed = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(style_latents, cache_path)
    del vae
    torch.cuda.empty_cache()
    gc.collect()
    return style_latents, elapsed


def compile_decoder(vae, args: argparse.Namespace, device: str) -> None:
    if not args.compile_decoder:
        return
    from utils.inference import VAEDecodeWrapper, configure_torch_compile_cache

    configure_torch_compile_cache(args.compile_cache_dir)
    wrapper = VAEDecodeWrapper(vae).to(
        device=device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    ).eval()
    vae.compiled_decoder = torch.compile(
        wrapper,
        mode=args.compile_mode,
        fullgraph=bool(args.compile_fullgraph),
        dynamic=False,
    )


def fixed_batch_decode(
    vae,
    latents: torch.Tensor,
    *,
    batch_size: int,
    device: str,
    consume,
) -> tuple[int, float]:
    count = 0
    checksum = 0.0
    for start in range(0, latents.shape[0], batch_size):
        valid = min(batch_size, latents.shape[0] - start)
        batch = latents[start:start + valid]
        if valid < batch_size:
            batch = torch.cat([batch, batch[-1:].expand(batch_size - valid, -1, -1, -1)], dim=0)
        decoded = decode_latent(vae, batch.to(device, non_blocking=True), device=device)
        valid_decoded = decoded[:valid]
        checksum += float(valid_decoded.float().mean().item()) * valid
        consume(start, valid_decoded)
        count += valid
    return count, checksum / max(1, count)


def main() -> None:
    args = parse_args()
    device = "cuda"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    wall_start = time.perf_counter()
    if args.decode_spool:
        t0 = time.perf_counter()
        spool = torch.load(args.decode_spool, map_location="cpu", weights_only=False)
        latent_tensor = spool["latents"].to(torch.float16).contiguous()
        generated_meta = [(str(key), int(style_id)) for key, style_id in spool["meta"]]
        source_items = [(key, torch.empty(0)) for key in spool.get("sources", [])]
        timings["load_latent_spool"] = time.perf_counter() - t0
    else:
        source_items = load_source_latents(Path(args.source_cache))
        if args.max_sources > 0:
            source_items = source_items[:args.max_sources]
        if len(source_items) != 150:
            print(f"warning: expected 150 sources, got {len(source_items)}", flush=True)
        style_latents, timings["load_style_refs"] = load_or_build_style_latents(args, device=device)

        t0 = time.perf_counter()
        inference = LGTInference(args.checkpoint, device=device, num_steps=args.num_steps)
        sync()
        timings["load_bridge"] = time.perf_counter() - t0

        generated_latents: list[torch.Tensor] = []
        generated_meta: list[tuple[str, int]] = []
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for source_start in range(0, len(source_items), args.source_batch):
                source_group = source_items[source_start:source_start + args.source_batch]
                source_batch = torch.stack([item[1] for item in source_group]).to(device, non_blocking=True)
                for style_start in range(0, len(STYLE_NAMES), args.style_chunk):
                    style_ids = list(range(style_start, min(len(STYLE_NAMES), style_start + args.style_chunk)))
                    repeated = torch.cat([source_batch for _ in style_ids], dim=0)
                    target_ids = torch.tensor(
                        [style_id for style_id in style_ids for _ in source_group],
                        device=device,
                        dtype=torch.long,
                    )
                    style_batch = torch.cat(
                        [style_latents[style_id].expand(len(source_group), -1, -1, -1) for style_id in style_ids],
                        dim=0,
                    ).to(device, non_blocking=True)
                    generated = inference.generation_with_target_latent(
                        repeated,
                        target_ids,
                        num_steps=args.num_steps,
                        target_style_latent={"style_latent_tensor": style_batch},
                    )
                    generated_latents.append(generated.detach().to(device="cpu", dtype=torch.float16))
                    generated_meta.extend(
                        (source_key, style_id)
                        for style_id in style_ids
                        for source_key, _ in source_group
                    )
        sync()
        timings["bridge_generation"] = time.perf_counter() - t0
        latent_tensor = torch.cat(generated_latents, dim=0).contiguous()
        del generated_latents, inference
        torch.cuda.empty_cache()
        gc.collect()
        if args.latent_spool:
            spool_path = Path(args.latent_spool)
            spool_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"latents": latent_tensor, "meta": generated_meta, "sources": [key for key, _ in source_items]},
                spool_path,
            )

    if args.skip_decode:
        timings["wall_total"] = time.perf_counter() - wall_start
        result = {
            "device": torch.cuda.get_device_name(0),
            "source_count": len(source_items),
            "generated_count": int(latent_tensor.shape[0]),
            "num_steps": args.num_steps,
            "source_batch": args.source_batch,
            "style_chunk": args.style_chunk,
            "skip_decode": True,
            "timings_sec": timings,
        }
        (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return

    t0 = time.perf_counter()
    if args.vae_torchscript_decoder:
        vae = load_torchscript_vae_decoder(args.vae_torchscript_decoder, device=device)
    else:
        vae = load_vae(
            device=device,
            model_id=args.vae_model,
            cache_dir=args.vae_cache_dir or None,
            enable_xformers=False,
        )
        compile_decoder(vae, args, device)
    sync()
    timings["load_compile_vae"] = time.perf_counter() - t0

    warmup_batch = latent_tensor[:args.decode_batch]
    if warmup_batch.shape[0] < args.decode_batch:
        warmup_batch = torch.cat(
            [warmup_batch, warmup_batch[-1:].expand(args.decode_batch - warmup_batch.shape[0], -1, -1, -1)],
            dim=0,
        )
    with torch.inference_mode():
        for _ in range(max(0, args.warmup_decode)):
            decode_latent(vae, warmup_batch.to(device), device=device)
        sync()
        t0 = time.perf_counter()
        image_dir = output_dir / "images"
        save_pool = None
        if args.save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
            from torchvision.io import write_png
            save_pool = ThreadPoolExecutor(max_workers=max(1, int(args.save_workers)))

        def consume_decoded(start: int, images: torch.Tensor) -> None:
            if not args.save_images:
                return
            images_u8 = images.mul(255).round().clamp(0, 255).to(torch.uint8).cpu()
            for local_index, image in enumerate(images_u8):
                index = start + local_index
                source_key, style_id = generated_meta[index]
                source_stem = Path(source_key).stem
                output_path = image_dir / f"{index:04d}_{source_stem}_to_{STYLE_NAMES[style_id]}.png"
                if save_pool is None:
                    write_png(image, str(output_path))
                else:
                    save_pool.submit(write_png, image, str(output_path))

        decoded_count, decoded_checksum = fixed_batch_decode(
            vae,
            latent_tensor,
            batch_size=args.decode_batch,
            device=device,
            consume=consume_decoded,
        )
        sync()
        timings["vae_decode"] = time.perf_counter() - t0
        if save_pool is not None:
            t0 = time.perf_counter()
            save_pool.shutdown(wait=True)
            timings["image_save_join"] = time.perf_counter() - t0

    timings["wall_total"] = time.perf_counter() - wall_start
    result = {
        "device": torch.cuda.get_device_name(0),
        "source_count": len(source_items),
        "generated_count": int(latent_tensor.shape[0]),
        "decoded_count": int(decoded_count),
        "decoded_mean_checksum": float(decoded_checksum),
        "latent_shape": list(latent_tensor.shape[1:]),
        "num_steps": args.num_steps,
        "source_batch": args.source_batch,
        "style_chunk": args.style_chunk,
        "decode_batch": args.decode_batch,
        "compile_decoder": bool(args.compile_decoder),
        "vae_torchscript_decoder": str(args.vae_torchscript_decoder),
        "compile_mode": args.compile_mode,
        "compile_fullgraph": bool(args.compile_fullgraph),
        "save_images": bool(args.save_images),
        "save_workers": int(args.save_workers),
        "timings_sec": timings,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
