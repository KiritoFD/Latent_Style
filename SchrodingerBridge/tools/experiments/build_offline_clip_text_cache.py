"""Build offline CLIP text token cache for 620 multimodal style transfer.

Reads style captions from a JSONL file, tokenizes them with a frozen CLIP text encoder,
and saves a .pt cache keyed by rel_path for use during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def load_captions(jsonl_path: str) -> dict[str, str]:
    captions: dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = entry.get("rel_path", "").strip()
            caption = entry.get("caption", "").strip()
            status = entry.get("status", "success")
            if rel_path and caption and status == "success":
                captions[rel_path] = caption
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline CLIP text token cache for 620 multimodal training.")
    parser.add_argument("--captions-jsonl", required=True, help="Path to train_style_captions.jsonl")
    parser.add_argument("--output", required=True, help="Output .pt cache path")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP text model name")
    parser.add_argument("--max-length", type=int, default=77, help="Max token length (CLIP default=77)")
    parser.add_argument("--hf-cache-dir", default="", help="HuggingFace cache directory")
    parser.add_argument("--device", default="cuda", help="Device for encoding")
    args = parser.parse_args()

    captions = load_captions(args.captions_jsonl)
    print(f"[clip_cache] Loaded {len(captions)} captions from {args.captions_jsonl}")

    model_kwargs = {}
    if args.hf_cache_dir:
        model_kwargs["cache_dir"] = args.hf_cache_dir
    model_kwargs["local_files_only"] = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **model_kwargs)
    text_encoder = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    text_encoder = text_encoder.text_model.to(device)
    text_encoder.eval()

    cache: dict[str, dict[str, torch.Tensor]] = {}
    batch_size = 64
    items = list(captions.items())

    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [caption for _, caption in batch]
            keys = [key for key, _ in batch]
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state.cpu()
            for j, key in enumerate(keys):
                cache[key] = {
                    "text_features": last_hidden_state[j],
                    "attention_mask": attention_mask[j].cpu(),
                    "caption": texts[j],
                }
            if (i // batch_size) % 10 == 0:
                print(f"[clip_cache] {i + len(batch)}/{len(items)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "model_name": args.model_name,
            "max_length": args.max_length,
            "feature_dim": last_hidden_state.shape[-1],
            "entries": cache,
        },
        output_path,
    )
    print(f"[clip_cache] Saved {len(cache)} entries to {output_path}")


if __name__ == "__main__":
    main()
