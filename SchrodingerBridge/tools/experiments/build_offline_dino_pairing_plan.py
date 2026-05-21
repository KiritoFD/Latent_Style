from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _normalize_patch_mean(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float().mean(dim=1), p=2, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline top-k pairing plan from cached DINO embeddings.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--cls-weight", type=float, default=0.7)
    parser.add_argument("--patch-weight", type=float, default=0.3)
    parser.add_argument("--styles", type=str, default="")
    args = parser.parse_args()

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows: list[dict[str, str]] = list(payload["rows"])
    cls_embeddings = F.normalize(payload["cls_embeddings"].float(), p=2, dim=-1)
    patch_embeddings = _normalize_patch_mean(payload["patch_embeddings"])
    styles = [x.strip() for x in str(args.styles).split(",") if x.strip()] or list(payload.get("styles", []))
    if not styles:
        raise ValueError("no styles found in cache or args")

    by_style: dict[str, list[int]] = {style: [] for style in styles}
    for idx, row in enumerate(rows):
        style = str(row["style"])
        if style in by_style:
            by_style[style].append(idx)

    out_pairs: dict[str, dict[str, dict[str, list[str]]]] = {}
    topk = max(1, int(args.topk))
    cls_weight = float(args.cls_weight)
    patch_weight = float(args.patch_weight)

    for src_idx, row in enumerate(rows):
        src_style = str(row["style"])
        src_stem = str(row["stem"])
        src_cls = cls_embeddings[src_idx : src_idx + 1]
        src_patch = patch_embeddings[src_idx : src_idx + 1]
        per_target: dict[str, list[str]] = {}

        for tgt_style in styles:
            candidate_indices = by_style.get(tgt_style, [])
            if not candidate_indices:
                continue
            if tgt_style == src_style:
                if src_stem in {rows[j]["stem"] for j in candidate_indices}:
                    per_target[tgt_style] = [src_stem]
                    continue

            tgt_idx = torch.as_tensor(candidate_indices, dtype=torch.long)
            tgt_cls = cls_embeddings.index_select(0, tgt_idx)
            tgt_patch = patch_embeddings.index_select(0, tgt_idx)
            cls_sim = torch.matmul(src_cls, tgt_cls.transpose(0, 1)).squeeze(0)
            patch_sim = torch.matmul(src_patch, tgt_patch.transpose(0, 1)).squeeze(0)
            score = cls_weight * cls_sim + patch_weight * patch_sim
            k = min(topk, int(score.numel()))
            best = torch.topk(score, k=k, dim=0, largest=True).indices.tolist()
            per_target[tgt_style] = [str(rows[candidate_indices[i]]["stem"]) for i in best]

        out_pairs.setdefault(src_style, {})[src_stem] = per_target

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(
            json.dumps(
                {
                    "cache": str(args.cache.resolve()),
                    "styles": styles,
                    "topk": topk,
                    "cls_weight": cls_weight,
                    "patch_weight": patch_weight,
                    "pairs": out_pairs,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        torch.save(
            {
                "cache": str(args.cache.resolve()),
                "styles": styles,
                "topk": topk,
                "cls_weight": cls_weight,
                "patch_weight": patch_weight,
                "pairs": out_pairs,
            },
            args.output,
        )
    print(f"[pairing-plan] wrote {args.output}")


if __name__ == "__main__":
    main()
