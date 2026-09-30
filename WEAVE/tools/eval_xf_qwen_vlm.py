from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
from pathlib import Path

import requests
from PIL import Image


def _data_url(path: Path, *, max_edge: int = 1024, jpeg_quality: int = 85) -> str:
    with Image.open(path) as img:
        image = img.convert("RGB")
        if max(image.size) > max(1, int(max_edge)):
            image.thumbnail((int(max_edge), int(max_edge)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _message_content(prompt: str, image_paths: list[Path]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _data_url(path),
                },
            }
        )
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Call xf-yun OpenAI-compatible Qwen VLM with one or more local images.")
    parser.add_argument("--image", action="append", required=True, help="Local image path. Repeat for multiple images.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a careful visual evaluator for style-transfer research.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("XF_MAAS_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"))
    parser.add_argument("--model", default=os.environ.get("XF_MAAS_MODEL_ID", "xopqwen36v35b"))
    parser.add_argument("--api-key", default=os.environ.get("XF_MAAS_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    if not str(args.api_key).strip():
        raise ValueError("Missing api key. Set XF_MAAS_API_KEY or pass --api-key.")

    image_paths = [Path(p).resolve() for p in args.image]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    url = str(args.base_url).rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": str(args.model),
        "messages": [
            {"role": "system", "content": str(args.system)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(args.prompt)},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _data_url(path, max_edge=int(args.max_edge), jpeg_quality=int(args.jpeg_quality)),
                            },
                        }
                        for path in image_paths
                    ],
                ],
            },
        ],
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
    }
    response = requests.post(url, headers=headers, json=payload, timeout=int(args.timeout))
    response.raise_for_status()
    data = response.json()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
