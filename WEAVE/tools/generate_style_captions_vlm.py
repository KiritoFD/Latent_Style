from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_style_captions_vlm")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _data_url(path: Path, *, max_edge: int = 1024, jpeg_quality: int = 85) -> str:
    with Image.open(path) as img:
        image = img.convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _call_api_with_retry(
    image_path: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_edge: int,
    jpeg_quality: int,
    max_retries: int = 5,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # 1. Prepare image data URL
    data_url = _data_url(image_path, max_edge=max_edge, jpeg_quality=jpeg_quality)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            # Use split timeout (connect, read) to prevent hangs
            response = requests.post(url, headers=headers, json=payload, timeout=(15, timeout))
            if response.status_code == 429:
                # Rate limit
                logger.warning(
                    "Rate limited (429) for %s. Retrying in %.2fs (attempt %d/%d)",
                    image_path.name, backoff, attempt, max_retries
                )
                time.sleep(backoff)
                backoff *= 2.0
                continue
            response.raise_for_status()
            data = response.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            
            if not content:
                raise ValueError("Received empty content from VLM API response.")
                
            # Short-circuit check: if blocked by safety / content moderation filters, don't keep retrying
            refusals = ["抱歉", "无法处理", "cannot process", "sorry", "inappropriate", "sensitive", "blocked"]
            if any(r in content.lower() for r in refusals):
                logger.warning(
                    "Image %s likely blocked by safety filters or refused by model. Content: %s",
                    image_path.name, content
                )
                return "style extraction blocked by safety filter"
                
            return content
        except requests.exceptions.Timeout:
            logger.warning(
                "Timeout for %s (attempt %d/%d).",
                image_path.name, attempt, max_retries
            )
            time.sleep(backoff)
            backoff *= 2.0
        except Exception as exc:
            logger.warning(
                "API call failed for %s (attempt %d/%d): %s",
                image_path.name, attempt, max_retries, exc
            )
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2.0
            
    raise RuntimeError(f"Failed to call VLM API after {max_retries} attempts.")


def process_image(
    image_path: Path,
    rel_path: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_edge: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    start_time = time.time()
    try:
        caption = _call_api_with_retry(
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
        elapsed = time.time() - start_time
        logger.info("Successfully labeled %s in %.2fs", rel_path, elapsed)
        return {"status": "success", "rel_path": rel_path, "caption": caption}
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("Failed to label %s in %.2fs: %s", rel_path, elapsed, exc)
        return {"status": "error", "rel_path": rel_path, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate style captions for training dataset images using Qwen VLM.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to classview train directory containing style subdirectories.")
    parser.add_argument("--output-json", type=Path, required=True, help="Final output path for compiled style_captions.json.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to intermediate JSONL file for resumability.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent threads to query Spark MaaS API.")
    parser.add_argument("--model", default=os.environ.get("XF_MAAS_MODEL_ID", "xopqwen36v35b"))
    parser.add_argument("--base-url", default=os.environ.get("XF_MAAS_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"))
    parser.add_argument("--api-key", default=os.environ.get("XF_MAAS_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to process (0 for no limit).")
    args = parser.parse_args()

    if not args.api_key.strip():
        raise ValueError("Missing api key. Set XF_MAAS_API_KEY environment variable or pass --api-key.")

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    system_prompt = (
        "You are an expert art critic and colorist. Your task is to extract ONLY the visual style of an image. "
        "You must respond with a clean, comma-separated list of keywords or short descriptive phrases. "
        "Do NOT output any conversational filler, greetings, or introductory sentences."
    )
    user_prompt = (
        "Analyze the image and describe its artistic style, medium, color palette, lighting, texture, and brushwork. "
        "CRITICAL REQUIREMENT: Do NOT mention any subjects, objects, characters, or scenes. "
        "Output ONLY the comma-separated list of style-related terms. Example output: 'oil painting, impressionism, thick impasto brushwork, warm golden hour lighting, pastel color palette'."
    )

    # 1. Collect all images and determine their relative paths (style/filename)
    all_images: list[tuple[Path, str]] = []
    for style_subdir in sorted(p for p in args.dataset_dir.iterdir() if p.is_dir()):
        for img_path in sorted(p for p in style_subdir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            rel_path = f"{style_subdir.name}/{img_path.name}"
            all_images.append((img_path, rel_path))

    logger.info("Found %d total training images to process across style subdirectories.", len(all_images))

    # 2. Load existing progress for resumability
    completed_captions: dict[str, str] = {}
    if args.output_jsonl.exists():
        try:
            with args.output_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("status") == "success":
                            completed_captions[obj["rel_path"]] = obj["caption"]
            logger.info("Loaded %d already completed style captions from %s", len(completed_captions), args.output_jsonl)
        except Exception as e:
            logger.warning("Error reading intermediate JSONL file %s: %s. Starting fresh.", args.output_jsonl, e)

    # Filter out already processed images
    pending_images = [(path, rel) for path, rel in all_images if rel not in completed_captions]
    if args.limit > 0:
        pending_images = pending_images[:args.limit]
    logger.info("Pending images to label: %d", len(pending_images))

    if not pending_images:
        logger.info("All images already processed. Writing final JSON.")
        args.output_json.write_text(json.dumps(completed_captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0

    # 3. Process pending images in parallel
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = args.output_jsonl.open("a", encoding="utf-8", buffering=1)

    failed_images: list[tuple[Path, str]] = []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_image = {
                executor.submit(
                    process_image,
                    img_path,
                    rel_path,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_edge=args.max_edge,
                    jpeg_quality=args.jpeg_quality,
                ): (img_path, rel_path)
                for img_path, rel_path in pending_images
            }

            for future in concurrent.futures.as_completed(future_to_image):
                img_path, rel_path = future_to_image[future]
                result = future.result()
                
                # Write intermediate result to JSONL
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                if result["status"] == "success":
                    completed_captions[rel_path] = result["caption"]
                else:
                    logger.error("Initial attempt failed for %s due to: %s", rel_path, result.get("error"))
                    failed_images.append((img_path, rel_path))

    except KeyboardInterrupt:
        logger.warning("Parallel execution interrupted by user. Saving current progress...")
    finally:
        jsonl_file.close()

    # 4. Sequential fallback retry pass for any failed images
    if failed_images:
        logger.info("Starting sequential fallback retry pass for %d failed images...", len(failed_images))
        jsonl_file = args.output_jsonl.open("a", encoding="utf-8", buffering=1)
        try:
            for img_path, rel_path in failed_images:
                logger.info("Retrying %s sequentially with downscaled resolution and longer timeout...", rel_path)
                time.sleep(2.0)  # Safe delay to reduce concurrent load/rate limits
                result = process_image(
                    img_path,
                    rel_path,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout + 30,  # extra timeout allowance
                    max_edge=512,  # downscale resolution to bypass payload/VLM constraints
                    jpeg_quality=75,  # downscale quality
                )
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["status"] == "success":
                    completed_captions[rel_path] = result["caption"]
                else:
                    logger.error("Sequential retry also failed for %s", rel_path)
        except KeyboardInterrupt:
            logger.warning("Fallback retry interrupted by user.")
        finally:
            jsonl_file.close()

    # 5. Compile final JSON output
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(completed_captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Successfully compiled final style captions mapping to %s. Total keys: %d", args.output_json, len(completed_captions))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
