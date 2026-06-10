from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import torch

from utils.introstyle_eval import (
    IntroStyleFeatureExtractor,
    introstyle_style_vector,
    mean_pool_scores,
    resolve_introstyle_model_path,
    style_bank_paths,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK_CACHE_ROOT = WORKSPACE_ROOT / "eval_cache" / "introstyle_bank_vectors"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def normalize_host_path(raw: str) -> Path:
    text = str(raw).strip()
    if not text:
        return Path(text)

    candidate = Path(text)
    if candidate.exists():
        return candidate

    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        drive = text[5]
        remainder = text[7:].replace("/", "\\")
        return Path(f"{drive.upper()}:\\{remainder}") if remainder else Path(f"{drive.upper()}:\\")

    match = _WINDOWS_DRIVE_RE.match(text)
    if os.name != "nt" and match:
        drive = match.group(1).lower()
        remainder = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{remainder}") if remainder else Path(f"/mnt/{drive}")

    return candidate


def resolve_gen_path(images_dir: Path, gen_image: str) -> Path:
    name = Path(str(gen_image)).name
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(gen_image)
    if raw.exists():
        return raw
    want = _canonical_name(name)
    for candidate in images_dir.iterdir():
        if candidate.is_file() and _canonical_name(candidate.name) == want:
            return candidate
    raise FileNotFoundError(name)


def _repair_mojibake(text: str) -> str:
    raw = str(text)
    tried = {raw}
    current = raw
    for _ in range(2):
        try:
            fixed = current.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            break
        if not fixed or fixed in tried:
            break
        tried.add(fixed)
        current = fixed
    return current


def _canonical_name(text: str) -> str:
    repaired = _repair_mojibake(str(text))
    normalized = unicodedata.normalize("NFKD", repaired)
    ascii_only = normalized.encode("ascii", errors="ignore").decode("ascii")
    return ascii_only.replace("\\", "/").lower()


def image_cell(row: dict[str, str]) -> str:
    for key in ("gen_image", "image"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise KeyError("Expected one of gen_image/image in metrics row")


def batched(items: list[Path], n: int) -> list[list[Path]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def _bank_cache_meta(
    *,
    bank_root: Path,
    banks: dict[str, list[Path]],
    per_style_limit: int,
    model_id: str,
    device: str,
    t: int,
    up_ft_index: int,
    ensemble_size: int,
) -> dict[str, object]:
    fingerprint_items: list[str] = []
    for style, paths in sorted(banks.items()):
        for path in paths:
            try:
                stat = path.stat()
                stamp = f"{style}|{path.as_posix()}|{stat.st_size}|{int(stat.st_mtime)}"
            except OSError:
                stamp = f"{style}|{path.as_posix()}|missing"
            fingerprint_items.append(stamp)
    digest = hashlib.sha256("\n".join(fingerprint_items).encode("utf-8")).hexdigest()
    return {
        "bank_root": str(bank_root.resolve()),
        "per_style_limit": int(per_style_limit),
        "model_id": str(model_id),
        "device": str(device),
        "t": int(t),
        "up_ft_index": int(up_ft_index),
        "ensemble_size": int(ensemble_size),
        "bank_fingerprint_sha256": digest,
    }


def _default_bank_cache_path(
    *,
    bank_root: Path,
    per_style_limit: int,
    model_id: str,
    t: int,
    up_ft_index: int,
    ensemble_size: int,
) -> Path:
    payload = {
        "bank_root": str(bank_root.resolve()),
        "per_style_limit": int(per_style_limit),
        "model_id": str(model_id),
        "t": int(t),
        "up_ft_index": int(up_ft_index),
        "ensemble_size": int(ensemble_size),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return DEFAULT_BANK_CACHE_ROOT / f"introstyle_bank_{digest}.pt"


def encode_bank(
    extractor: IntroStyleFeatureExtractor,
    bank_root: Path,
    *,
    per_style_limit: int,
    batch_size: int,
    cache_path: Path | None,
    model_id: str,
    t: int,
    up_ft_index: int,
    ensemble_size: int,
) -> dict[str, torch.Tensor]:
    banks = style_bank_paths(bank_root, per_style_limit=per_style_limit)
    cache_meta = _bank_cache_meta(
        bank_root=bank_root,
        banks=banks,
        per_style_limit=per_style_limit,
        model_id=model_id,
        device=str(extractor.device),
        t=t,
        up_ft_index=up_ft_index,
        ensemble_size=ensemble_size,
    )
    if cache_path is not None and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, dict) and payload.get("meta") == cache_meta and isinstance(payload.get("vectors"), dict):
                print(f"[IntroStyle] bank cache hit: {cache_path}", flush=True)
                return {
                    str(style): tensor.to(device=extractor.device)
                    for style, tensor in payload["vectors"].items()
                }
        except Exception as exc:
            print(f"[IntroStyle] bank cache load failed: {cache_path} ({exc})", flush=True)
    out: dict[str, torch.Tensor] = {}
    for style, paths in banks.items():
        print(
            f"[IntroStyle] encoding bank style={style} images={len(paths)} batch_size={batch_size}",
            flush=True,
        )
        feats = extractor.encode_paths(paths, batch_size=batch_size)
        out[style] = introstyle_style_vector(feats)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "meta": cache_meta,
                    "vectors": {style: tensor.detach().cpu() for style, tensor in out.items()},
                },
                cache_path,
            )
            print(f"[IntroStyle] bank cache saved: {cache_path}", flush=True)
        except Exception as exc:
            print(f"[IntroStyle] bank cache save failed: {cache_path} ({exc})", flush=True)
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
        all_paths.append(resolve_gen_path(images_dir, image_cell(row)))
        metas.append(row)

    score_rows: list[dict] = []
    chunks = batched(all_paths, batch_size)
    total_chunks = len(chunks)
    for chunk_idx, chunk_paths in enumerate(chunks, start=1):
        chunk_metas = metas[(chunk_idx - 1) * batch_size:chunk_idx * batch_size]
        print(
            f"[IntroStyle] run={run} chunk={chunk_idx}/{total_chunks} images={len(chunk_paths)}",
            flush=True,
        )
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
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument("--modelscope-id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--modelscope-cache-dir", type=str, default="")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bank-batch-size", type=int, default=0)
    parser.add_argument("--bank_limit_per_style", type=int, default=64)
    parser.add_argument("--bank-cache-path", type=Path, default=None)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--up_ft_index", type=int, default=1)
    parser.add_argument("--ensemble_size", type=int, default=4)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    resolved_model_id = resolve_introstyle_model_path(
        model_id=str(args.model_id),
        modelscope_id=str(args.modelscope_id),
        modelscope_cache_dir=str(args.modelscope_cache_dir),
        allow_network=bool(args.allow_network),
    )
    extractor = IntroStyleFeatureExtractor(
        model_id=resolved_model_id,
        device=str(args.device),
        t=int(args.t),
        up_ft_index=int(args.up_ft_index),
        ensemble_size=int(args.ensemble_size),
    )
    bank_cache_path = Path(args.bank_cache_path).resolve() if args.bank_cache_path is not None else _default_bank_cache_path(
        bank_root=args.style_bank_root,
        per_style_limit=int(args.bank_limit_per_style),
        model_id=str(resolved_model_id),
        t=int(args.t),
        up_ft_index=int(args.up_ft_index),
        ensemble_size=int(args.ensemble_size),
    )
    bank_batch_size = int(args.bank_batch_size) if int(args.bank_batch_size) > 0 else int(args.batch_size)
    bank_vectors = encode_bank(
        extractor,
        args.style_bank_root,
        per_style_limit=int(args.bank_limit_per_style),
        batch_size=bank_batch_size,
        cache_path=bank_cache_path,
        model_id=str(resolved_model_id),
        t=int(args.t),
        up_ft_index=int(args.up_ft_index),
        ensemble_size=int(args.ensemble_size),
    )

    results: list[dict] = []
    for row in manifest_rows:
        method = str(row["method"]).strip()
        run = str(row["run"]).strip()
        images_dir = normalize_host_path(str(row["images_dir"]).strip())
        metrics_csv = normalize_host_path(str(row["metrics_csv"]).strip())
        if not images_dir.exists() or not metrics_csv.exists():
            print(f"SKIP {method}/{run}: missing images or metrics")
            continue
        print(f"Evaluating {method}/{run}", flush=True)
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
