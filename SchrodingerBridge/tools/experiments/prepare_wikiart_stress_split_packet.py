from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _image_paths(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)


def _class_seed(base_seed: int, class_name: str) -> int:
    return int(base_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(class_name))


def _split_paths(paths: list[Path], *, train_count: int, test_count: int, seed: int) -> tuple[list[Path], list[Path]]:
    import random

    needed = int(train_count) + int(test_count)
    if len(paths) < needed:
        raise ValueError(f"Need {needed} images, found {len(paths)}")
    shuffled = list(paths)
    random.Random(int(seed)).shuffle(shuffled)
    test_paths = sorted(shuffled[:test_count], key=lambda p: p.name)
    train_paths = sorted(shuffled[test_count:test_count + train_count], key=lambda p: p.name)
    return train_paths, test_paths


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    chosen = "hardlink" if mode == "auto" else mode
    try:
        if chosen == "hardlink":
            os.link(src, dst)
            return "hardlink"
        if chosen == "symlink":
            os.symlink(src, dst)
            return "symlink"
        if chosen == "copy":
            shutil.copy2(src, dst)
            return "copy"
    except OSError:
        if mode != "auto":
            raise
    shutil.copy2(src, dst)
    return "copy"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(
    *,
    template: dict,
    config_path: Path,
    split_name: str,
    styles: list[str],
    remote_root_unix: str,
) -> None:
    cfg = json.loads(json.dumps(template))
    remote_split_root = f"{remote_root_unix.rstrip('/')}/{split_name}"
    cfg.setdefault("training", {})
    cfg.setdefault("data", {})
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("ablation", {})
    cfg["training"]["batch_size"] = 44
    cfg["training"]["test_image_dir"] = f"{remote_split_root}/classview/test"
    cfg["training"]["full_eval_cache_dir"] = "/mnt/i/Github/Latent_Style/eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "/mnt/i/Github/Latent_Style/eval_cache/hf"
    cfg["training"]["full_eval_batch_size"] = 4
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_defer_until_training_end"] = True
    cfg["training"]["full_eval_profile_timing"] = True
    cfg["data"]["data_root"] = f"{remote_split_root}/latents_ema/train"
    cfg["data"]["style_subdirs"] = list(styles)
    cfg["data"]["latent_cache_dir"] = f"{remote_split_root}/latents_ema/train/.latent_cache"
    cfg["data"]["pairing_cache_path"] = f"{remote_split_root}/latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
    cfg["checkpoint"]["save_dir"] = f"./exp/{split_name}_variant_f_b44_remote"
    cfg["ablation"]["name"] = f"{split_name}_variant_f_b44_remote"
    cfg["ablation"]["axis"] = "faraday_stress_splits"
    cfg["ablation"]["stage"] = split_name
    cfg["ablation"]["notes"] = "Faraday multi-split validation packet generated from selected_splits.json with fixed seed and no manual class cherry-picking."
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, object]:
    raw_root = Path(args.raw_root).resolve()
    selected = _load_json(Path(args.selected_json).resolve())
    template = _load_json(Path(args.template).resolve())
    output_root = Path(args.output_root).resolve()
    config_root = Path(args.config_root).resolve()

    packet_summary: dict[str, object] = {
        "schema": 1,
        "selected_json": str(Path(args.selected_json).resolve()),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "config_root": str(config_root),
        "packets": [],
    }

    for split in selected.get("selected_splits", []):
        split_name = str(split["name"]).strip()
        styles = list(split["styles"])
        split_slug = f"{split_name}_{_safe_slug('__'.join(styles))}"
        split_root = output_root / split_slug
        classview_root = split_root / "classview"
        packet = {
            "name": split_name,
            "slug": split_slug,
            "styles": styles,
            "score_mean_pairwise_clip_distance": split.get("mean_pairwise_clip_distance"),
            "score_min_pairwise_clip_distance": split.get("min_pairwise_clip_distance"),
            "local_root": str(split_root),
            "local_train_root": str(classview_root / "train"),
            "local_test_root": str(classview_root / "test"),
            "remote_unix_root": f"{args.remote_root_unix.rstrip('/')}/{split_slug}",
            "remote_windows_root": f"{args.remote_root_windows.rstrip('\\')}\\{split_slug}",
            "config_path": str((config_root / f"{split_slug}_variant_f_b44_remote.json").resolve()),
            "splits": {},
        }
        for style in styles:
            paths = _image_paths(raw_root / style)
            seed = _class_seed(int(args.seed), style)
            train_paths, test_paths = _split_paths(
                paths,
                train_count=int(args.train_per_class),
                test_count=int(args.test_per_class),
                seed=seed,
            )
            for split_kind, chosen in (("train", train_paths), ("test", test_paths)):
                out_dir = classview_root / split_kind / style
                records = []
                for src in chosen:
                    dst = out_dir / src.name
                    action = _link_or_copy(src, dst, mode=args.link_mode)
                    records.append({"source": str(src), "target": str(dst), "action": action})
                packet["splits"].setdefault(style, {})[split_kind] = {"count": len(chosen), "dir": str(out_dir), "records": records}
        config_path = config_root / f"{split_slug}_variant_f_b44_remote.json"
        _write_config(
            template=template,
            config_path=config_path,
            split_name=split_slug,
            styles=styles,
            remote_root_unix=args.remote_root_unix,
        )
        (split_root / "packet_manifest.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        packet_summary["packets"].append(packet)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "packet_summary.json"
    summary_path.write_text(json.dumps(packet_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return packet_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare classview packets and remote-ready configs for selected WikiArt stress splits.")
    parser.add_argument("--selected-json", required=True)
    parser.add_argument("--raw-root", default="F:/wikiart/wikiart")
    parser.add_argument("--output-root", default="F:/wikiart_faraday_splits")
    parser.add_argument("--config-root", default="G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/faraday_splits")
    parser.add_argument("--template", default="G:/GitHub/Latent_Style/SchrodingerBridge/configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json")
    parser.add_argument("--remote-root-unix", default="/mnt/i/wikiart_faraday_splits")
    parser.add_argument("--remote-root-windows", default="I:\\wikiart_faraday_splits")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
    return parser.parse_args()


def main() -> None:
    payload = prepare(parse_args())
    print(json.dumps({"output_root": payload["output_root"], "num_packets": len(payload["packets"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
