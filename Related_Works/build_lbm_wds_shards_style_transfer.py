import argparse
import json
import random
from pathlib import Path

import webdataset as wds


def _list_images(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _write_shards(
    *,
    src_paths: list[Path],
    tgt_paths: list[Path],
    out_pattern: str,
    num_samples: int,
    seed: int,
):
    rng = random.Random(int(seed))
    if not src_paths:
        raise RuntimeError("No source images found")
    if not tgt_paths:
        raise RuntimeError("No target images found")

    src_order = list(src_paths)
    tgt_order = list(tgt_paths)
    rng.shuffle(src_order)
    rng.shuffle(tgt_order)

    # Unpaired: cycle through src; randomly sample tgt (with replacement).
    with wds.ShardWriter(out_pattern, maxcount=1000) as sink:
        for i in range(int(num_samples)):
            src_p = src_order[i % len(src_order)]
            tgt_p = tgt_order[rng.randrange(0, len(tgt_order))]
            key = f"{i:09d}"
            sink.write(
                {
                    "__key__": key,
                    "src.jpg": _read_bytes(src_p),
                    "tgt.jpg": _read_bytes(tgt_p),
                }
            )


def main():
    ap = argparse.ArgumentParser("Build WebDataset shards for unpaired style transfer (src.jpg + tgt.jpg)")
    ap.add_argument("--src_dir", required=True, help="Source distribution dir (images)")
    ap.add_argument("--tgt_dir", required=True, help="Target distribution dir (images)")
    ap.add_argument("--out_dir", required=True, help="Output dir for shards")
    ap.add_argument("--prefix", default="train", help="Shard prefix (train|val|...)")
    ap.add_argument("--num_samples", type=int, default=5000, help="How many samples to generate (unpaired)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    tgt_dir = Path(args.tgt_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_paths = _list_images(src_dir)
    tgt_paths = _list_images(tgt_dir)

    # WebDataset uses URL-like handlers; on Windows, absolute paths need an explicit file: scheme.
    out_pattern = "file:" + str(out_dir / f"{args.prefix}-%06d.tar")
    _write_shards(
        src_paths=src_paths,
        tgt_paths=tgt_paths,
        out_pattern=out_pattern,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )

    meta = {
        "src_dir": str(src_dir),
        "tgt_dir": str(tgt_dir),
        "out_dir": str(out_dir),
        "prefix": str(args.prefix),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "src_count": len(src_paths),
        "tgt_count": len(tgt_paths),
        "pattern": out_pattern,
    }
    (out_dir / f"{args.prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
