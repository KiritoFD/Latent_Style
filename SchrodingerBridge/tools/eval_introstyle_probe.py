from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from utils.introstyle_eval import (
    IntroStyleFeatureExtractor,
    introstyle_style_vector,
    mean_pool_scores,
    style_bank_paths,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_gen_path(images_dir: Path, gen_image: str) -> Path:
    name = Path(str(gen_image)).name
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(gen_image)
    if raw.exists():
        return raw
    raise FileNotFoundError(name)


def batched(items: list[Path], n: int) -> list[list[Path]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def encode_bank(
    extractor: IntroStyleFeatureExtractor,
    bank_root: Path,
    *,
    per_style_limit: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    banks = style_bank_paths(bank_root, per_style_limit=per_style_limit)
    out: dict[str, torch.Tensor] = {}
    for style, paths in banks.items():
        feats = extractor.encode_paths(paths, batch_size=batch_size)
        out[style] = introstyle_style_vector(feats)
    return out


def evaluate_one(
    *,
    extractor: IntroStyleFeatureExtractor,
    method: str,
    run: str,
    images_dir: Path,
    metrics_csv: Path,
    bank_vectors: dict[str, torch.Tensor],
    batch_size: int,
) -> dict:
    rows = load_rows(metrics_csv)
    all_paths: list[Path] = []
    metas: list[dict[str, str]] = []
    for row in rows:
        all_paths.append(resolve_gen_path(images_dir, row["gen_image"]))
        metas.append(row)

    score_rows: list[dict] = []
    for chunk_idx, chunk_paths in enumerate(batched(all_paths, batch_size)):
        chunk_metas = metas[chunk_idx * batch_size:(chunk_idx + 1) * batch_size]
        feats = extractor.encode_paths(chunk_paths, batch_size=len(chunk_paths))
        vecs = introstyle_style_vector(feats)
        scores = mean_pool_scores(vecs, bank_vectors, topk=8)
        style_names = sorted(bank_vectors.keys())
        for i, row in enumerate(chunk_metas):
            target = row["tgt_style"]
            src = row["src_style"]
            target_score = float(scores[target][i].item())
            source_score = float(scores[src][i].item())
            non_target_scores = [(name, float(scores[name][i].item())) for name in style_names if name != target]
            best_non_target_style, best_non_target_score = max(non_target_scores, key=lambda x: x[1])
            score_rows.append(
                {
                    "src_style": src,
                    "tgt_style": target,
                    "target_style_score": target_score,
                    "source_style_score": source_score,
                    "best_non_target_style": best_non_target_style,
                    "best_non_target_score": best_non_target_score,
                    "style_margin": target_score - best_non_target_score,
                }
            )

    transfer = [r for r in score_rows if r["src_style"] != r["tgt_style"]]
    identity = [r for r in score_rows if r["src_style"] == r["tgt_style"]]

    def mean(key: str, pool: list[dict]) -> float | None:
        if not pool:
            return None
        return float(sum(float(r[key]) for r in pool) / len(pool))

    return {
        "method": method,
        "run": run,
        "images": len(score_rows),
        "transfer_target_style_score": mean("target_style_score", transfer),
        "transfer_source_style_score": mean("source_style_score", transfer),
        "transfer_best_non_target_score": mean("best_non_target_score", transfer),
        "transfer_style_margin": mean("style_margin", transfer),
        "identity_target_style_score": mean("target_style_score", identity),
        "images_dir": str(images_dir),
        "metrics_csv": str(metrics_csv),
    }


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# IntroStyle Probe",
        "",
        "| Method | Run | Transfer target score | Transfer source score | Best non-target | Style margin | Identity target score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['run']} | {row['transfer_target_style_score']:.4f} | "
            f"{row['transfer_source_style_score']:.4f} | {row['transfer_best_non_target_score']:.4f} | "
            f"{row['transfer_style_margin']:.4f} | {row['identity_target_style_score']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--style-bank-root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bank_limit_per_style", type=int, default=64)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--up_ft_index", type=int, default=1)
    parser.add_argument("--ensemble_size", type=int, default=4)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    extractor = IntroStyleFeatureExtractor(
        model_id=str(args.model_id),
        device=str(args.device),
        t=int(args.t),
        up_ft_index=int(args.up_ft_index),
        ensemble_size=int(args.ensemble_size),
    )
    bank_vectors = encode_bank(
        extractor,
        args.style_bank_root,
        per_style_limit=int(args.bank_limit_per_style),
        batch_size=int(args.batch_size),
    )

    results: list[dict] = []
    for row in manifest_rows:
        method = str(row["method"]).strip()
        run = str(row["run"]).strip()
        images_dir = Path(str(row["images_dir"]).strip())
        metrics_csv = Path(str(row["metrics_csv"]).strip())
        if not images_dir.exists() or not metrics_csv.exists():
            print(f"SKIP {method}/{run}: missing images or metrics")
            continue
        print(f"Evaluating {method}/{run}")
        results.append(
            evaluate_one(
                extractor=extractor,
                method=method,
                run=run,
                images_dir=images_dir,
                metrics_csv=metrics_csv,
                bank_vectors=bank_vectors,
                batch_size=int(args.batch_size),
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "run",
                "images",
                "transfer_target_style_score",
                "transfer_source_style_score",
                "transfer_best_non_target_score",
                "transfer_style_margin",
                "identity_target_style_score",
                "images_dir",
                "metrics_csv",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, args.output_csv.with_suffix(".md"))
    print(args.output_csv)
    print(args.output_json)
    print(args.output_csv.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
