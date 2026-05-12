import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def run(cmd, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}. See log: {log_path}")


def _read_opt_file_kv(opt_path: Path) -> dict:
    """
    Parse CUT opt files like train_opt.txt (printed argparse table).
    Lines look like:
        netG: resnet_6blocks                 [default: resnet_9blocks]
    """
    if not opt_path.exists():
        return {}
    kv = {}
    for line in opt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        if line.strip().startswith("-"):
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if not key or not rest:
            continue
        # Drop "[default: ...]" and extra columns.
        if "[default:" in rest:
            rest = rest.split("[default:", 1)[0].strip()
        # Collapse multiple spaces.
        val = " ".join(rest.split())
        if val:
            kv[key] = val
    return kv


def parse_src_style_and_stem(testA_name: str):
    """
    testA filenames are created as: {style}__{orig_name}
    We want: src_style, src_stem (stem without extension).
    """
    p = Path(testA_name)
    if "__" not in p.name:
        return None
    src_style, rest = p.name.split("__", 1)
    stem = Path(rest).stem
    if not src_style or not stem:
        return None
    return src_style, stem


def main():
    ap = argparse.ArgumentParser("CUT inference 5x5 (serial)")
    ap.add_argument("--cut_repo", default="Related_Works/external/CUT")
    ap.add_argument("--python_exe", default="Related_Works/envs/cut/Scripts/python.exe")
    ap.add_argument("--datasets_root", default="Related_Works/runs/cut_5x5/datasets")
    ap.add_argument("--checkpoints_dir", default="Related_Works/runs/cut_5x5/checkpoints")
    ap.add_argument("--out_dir", default="Related_Works/runs/cut_5x5/infer_5x5")
    ap.add_argument("--results_dir", default="Related_Works/runs/cut_5x5/raw_results")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_test", type=int, default=1000000)
    ap.add_argument("--phase", default="val", help="CUT phase to run (e.g. test or val). Requires <phase>A/<phase>B dirs.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reuse_results", action="store_true", help="Skip test.py if raw results already exist")
    args = ap.parse_args()

    cut_repo = Path(args.cut_repo).resolve()
    py = str(Path(args.python_exe).resolve())
    datasets_root = Path(args.datasets_root).resolve()
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    img_out = out_dir / "images"
    logs_dir = out_dir / "logs"
    img_out.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for d in sorted(datasets_root.iterdir()):
        if d.is_dir() and d.name.startswith("to_"):
            targets.append(d.name[len("to_") :])
    if not targets:
        raise RuntimeError(f"No datasets found under: {datasets_root}")

    meta = {
        "method": "cut",
        "targets": targets,
        "datasets_root": str(datasets_root),
        "checkpoints_dir": str(ckpt_dir),
        "results_dir": str(results_dir),
        "size": args.size,
        "num_test": args.num_test,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    runs = []
    total_t0 = time.perf_counter()
    phase = str(args.phase).strip()
    if not phase:
        raise SystemExit("Empty --phase")

    # Serial: one target style per run.
    for tgt in targets:
        name = f"cut_to_{tgt}"
        dataroot = datasets_root / f"to_{tgt}"
        log_path = logs_dir / f"test_{name}.log"

        phase_a = dataroot / f"{phase}A"
        phase_b = dataroot / f"{phase}B"
        if not phase_a.is_dir() or not phase_b.is_dir():
            raise FileNotFoundError(
                f"Missing dataset split for phase='{phase}': {phase_a} or {phase_b}. "
                "Run build_cut_val_split.py (or create the folders) first."
            )

        raw_img_root = results_dir / name / f"{phase}_latest" / "images"
        fake_dir = raw_img_root / "fake_B"
        can_reuse = bool(args.reuse_results and fake_dir.exists() and any(fake_dir.iterdir()))

        # Match architecture/hparams to training to avoid shape mismatch.
        opt_path = ckpt_dir / name / "train_opt.txt"
        train_kv = _read_opt_file_kv(opt_path)
        netg = train_kv.get("netG", "")
        ngf = train_kv.get("ngf", "")
        ndf = train_kv.get("ndf", "")
        nce_layers = train_kv.get("nce_layers", "")
        nce_idt = train_kv.get("nce_idt", "")
        num_patches = train_kv.get("num_patches", "")

        if not can_reuse:
            run_t0 = time.perf_counter()
            cmd = [
                py,
                "test.py",
                "--dataroot",
                str(dataroot),
                "--name",
                name,
                "--CUT_mode",
                "CUT",
                "--phase",
                phase,
                "--num_test",
                str(int(args.num_test)),
                "--results_dir",
                str(results_dir),
                "--checkpoints_dir",
                str(ckpt_dir),
                "--load_size",
                str(int(args.size)),
                "--crop_size",
                str(int(args.size)),
                "--preprocess",
                "resize_and_crop",
            ]
            if netg:
                cmd += ["--netG", netg]
            if ngf:
                cmd += ["--ngf", ngf]
            if ndf:
                cmd += ["--ndf", ndf]
            if nce_layers:
                cmd += ["--nce_layers", nce_layers]
            if nce_idt:
                cmd += ["--nce_idt", nce_idt]
            if num_patches:
                cmd += ["--num_patches", num_patches]
            run(cmd, cwd=cut_repo, log_path=log_path)
            run_elapsed = time.perf_counter() - run_t0
        else:
            run_elapsed = 0.0

        # Collect fake_B images into unified {src}_{stem}_to_{tgt}.jpg names.
        if not fake_dir.exists():
            # Backward compatibility: some layouts flatten into images/*.png with *_fake_B suffix.
            fake_dir = raw_img_root
        if not fake_dir.exists():
            raise RuntimeError(f"Missing results dir: {raw_img_root}")

        fake_paths = sorted([p for p in fake_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not fake_paths:
            raise RuntimeError(f"No fake_B images found under: {fake_dir}")

        export_count = 0
        for fp in fake_paths:
            # fp.name example: photo__2013-11-08 16_45_24.jpg
            parsed = parse_src_style_and_stem(fp.name)
            if parsed is None:
                continue
            src_style, src_stem = parsed
            out_name = f"{src_style}_{src_stem}_to_{tgt}.jpg"
            out_path = img_out / out_name
            if out_path.exists() and not args.overwrite:
                continue
            im = Image.open(fp).convert("RGB")
            if im.size != (args.size, args.size):
                im = im.resize((args.size, args.size), Image.Resampling.LANCZOS)
            im.save(out_path, quality=95)
            export_count += 1

        runs.append(
            {
                "target": tgt,
                "name": name,
                "phase": phase,
                "reused_raw_results": bool(can_reuse),
                "raw_fake_count": int(len(fake_paths)),
                "exported_count": int(export_count),
                "test_wall_sec": float(run_elapsed),
                "avg_test_wall_sec_per_raw": float(run_elapsed / max(len(fake_paths), 1)),
                "avg_test_wall_sec_per_export": float(run_elapsed / max(export_count, 1)),
                "train_opt_path": str(opt_path),
            }
        )
        print(f"[CUT] done target={tgt} -> {img_out}")

    total_elapsed = time.perf_counter() - total_t0
    summary = {
        "method": "cut",
        "size": int(args.size),
        "num_test": int(args.num_test),
        "phase": phase,
        "targets": targets,
        "datasets_root": str(datasets_root),
        "checkpoints_dir": str(ckpt_dir),
        "results_dir": str(results_dir),
        "out_images_dir": str(img_out),
        "runs": runs,
        "total_wall_sec": float(total_elapsed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CUT] all done -> {img_out}")
    print(f"[CUT] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
