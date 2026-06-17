from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
RELATED_ROOT = WORKSPACE_ROOT / "Related_Works"
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
SB_ROOT = WORKSPACE_ROOT / "SchrodingerBridge"
SB_RUN_EVAL = SB_ROOT / "run_evaluation.py"
SB_SRC = SB_ROOT / "src"
RESULTS_ROOT = ROOT / "results"
TMP_ROOT = ROOT / "tmp"
ALL_STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEFAULT_PROTOCOL = "protocol_a_800"
DEFAULT_PROTOCOL_A_REFERENCE_IMAGES = (
    SB_ROOT
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)
TIMING_ROWS: list[dict[str, Any]] = []


@dataclass
class CloneSpec:
    name: str
    dest: Path
    url: str
    required_markers: tuple[str, ...]
    notes: str = ""


CLONE_SPECS: list[CloneSpec] = [
    CloneSpec(
        name="SaMST",
        dest=RELATED_ROOT / "SaMST-main",
        url="https://github.com/Chernobyllight/SaMST.git",
        required_markers=("README.md", "train_model"),
    ),
    CloneSpec(
        name="S2WAT",
        dest=RELATED_ROOT / "S2WAT-main",
        url="https://github.com/AlienZhang1996/S2WAT.git",
        required_markers=("README.md", "train.py", "test.py"),
    ),
    CloneSpec(
        name="StyTR-2",
        dest=RELATED_ROOT / "StyTR-2",
        url="https://github.com/diyiiyiii/StyTR-2.git",
        required_markers=("README.md", "test.py"),
    ),
    CloneSpec(
        name="AesPA-Net",
        dest=RELATED_ROOT / "AesPA-Net",
        url="https://github.com/Kibeom-Hong/AesPA-Net.git",
        required_markers=("README.md", "main.py"),
    ),
    CloneSpec(
        name="AesFA",
        dest=RELATED_ROOT / "AesFA",
        url="https://github.com/Sooyyoungg/AesFA.git",
        required_markers=("README.md", "test.py", "Config.py"),
    ),
    CloneSpec(
        name="ArtBank",
        dest=RELATED_ROOT / "ArtBank",
        url="https://github.com/Jamie-Cheung/ArtBank.git",
        required_markers=("README.md", "test.py"),
    ),
    CloneSpec(
        name="CycleGAN",
        dest=RELATED_ROOT / "pytorch-CycleGAN-and-pix2pix",
        url="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git",
        required_markers=("README.md", "train.py", "test.py"),
    ),
    CloneSpec(
        name="AdaIN",
        dest=RELATED_ROOT / "AdaIN-style-official",
        url="https://github.com/xunhuang1995/AdaIN-style.git",
        required_markers=("README.md", "test.py"),
        notes="May fail to clone on unstable network; script reports status explicitly.",
    ),
]


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(str(x) for x in cmd)
    print(f"[RUN] {printable}")
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def _run_timed(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    baseline: str,
    style: str,
    phase: str,
    protocol: str,
    output_root: Path,
) -> subprocess.CompletedProcess[str]:
    start = time.time()
    status = "ok"
    error = ""
    try:
        return _run(cmd, cwd=cwd)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        TIMING_ROWS.append(
            {
                "baseline": baseline,
                "style": style,
                "phase": phase,
                "protocol": protocol,
                "output_root": str(output_root),
                "elapsed_sec": round(time.time() - start, 3),
                "status": status,
                "error": error,
            }
        )


def _repo_ready(spec: CloneSpec) -> bool:
    if not spec.dest.exists():
        return False
    return all((spec.dest / marker).exists() for marker in spec.required_markers)


def ensure_repos(clone_missing: bool) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for spec in CLONE_SPECS:
        ready_before = _repo_ready(spec)
        cloned = False
        err = ""
        if (not ready_before) and clone_missing:
            try:
                if spec.dest.exists() and not any(spec.dest.iterdir()):
                    spec.dest.rmdir()
                if not spec.dest.exists():
                    _run(["git", "clone", "--depth", "1", spec.url, spec.dest], cwd=WORKSPACE_ROOT)
                    cloned = True
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        ready_after = _repo_ready(spec)
        status[spec.name] = {
            "path": str(spec.dest),
            "url": spec.url,
            "ready": ready_after,
            "cloned_now": cloned,
            "error": err,
            "notes": spec.notes,
        }
    return status


def _collect_content_subset(max_images_per_style: int = 0) -> list[Path]:
    out: list[Path] = []
    for style in ALL_STYLES:
        src = OVERFIT50 / style
        if not src.exists():
            continue
        files = sorted(src.glob("*.jpg"))
        if max_images_per_style > 0:
            files = files[:max_images_per_style]
        out.extend(files)
    return out


def _write_reference_manifest(reference_images_dir: Path, out_path: Path) -> Path:
    names = _load_reference_names(reference_images_dir)
    assert names is not None
    src_names = sorted({name.split("_to_", 1)[0] + ".jpg" for name in names})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(src_names) + "\n", encoding="utf-8")
    return out_path


def _prepare_mixed_content_dir(tag: str, max_images_per_style: int = 0) -> Path:
    out_dir = TMP_ROOT / tag / "content"
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in _collect_content_subset(max_images_per_style=max_images_per_style):
        content_style = img.parent.name
        dst = out_dir / f"{content_style}_{img.name}"
        if not dst.exists():
            shutil.copy2(img, dst)
    return out_dir


def _prepare_single_style_ref_dir(tag: str, target_style: str) -> Path:
    out_dir = TMP_ROOT / tag / f"style_{target_style}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = OVERFIT50 / target_style
    refs = sorted(src_dir.glob("*.jpg"))
    if not refs:
        raise FileNotFoundError(f"No style reference found for {target_style}: {src_dir}")
    dst = out_dir / refs[0].name
    if not dst.exists():
        shutil.copy2(refs[0], dst)
    return out_dir


def _copy_tree_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(src_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = dst_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            count += 1
    return count


def _baseline_result_dir(name: str, protocol: str | None = None, style: str | None = None) -> Path:
    root = RESULTS_ROOT / name
    if protocol:
        root = root / protocol
    if style is not None:
        root = root / style
    return root


def _load_reference_names(reference_images_dir: Path | None) -> set[str] | None:
    if reference_images_dir is None:
        return None
    if not reference_images_dir.exists():
        raise FileNotFoundError(f"Reference images dir not found: {reference_images_dir}")
    names = {
        p.name
        for p in reference_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if not names:
        raise RuntimeError(f"No reference images found in: {reference_images_dir}")
    return names


def _aggregate_baseline_images(
    result_root: Path,
    reference_images_dir: Path | None = None,
    allow_partial_reference: bool = False,
) -> Path:
    images_dir = result_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.iterdir():
        if old.is_file():
            old.unlink()
    reference_names = _load_reference_names(reference_images_dir)
    for child in sorted(result_root.iterdir()):
        if not child.is_dir() or child.name == "images":
            continue
        for img in child.rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if "_to_" not in img.stem:
                    continue
                if reference_names is not None and img.name not in reference_names:
                    continue
                dst = images_dir / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
    if reference_names is not None and not allow_partial_reference:
        found = {p.name for p in images_dir.iterdir() if p.is_file()}
        missing = sorted(reference_names - found)
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Aggregated {len(found)}/{len(reference_names)} reference images for {result_root}. "
                f"Missing examples: {preview}"
            )
    return images_dir


def _load_summary_metrics(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    return {
        "clip_style": all_pairs.get("clip_style"),
        "clip_content": all_pairs.get("clip_content"),
        "content_lpips": all_pairs.get("content_lpips"),
        "fid": all_pairs.get("fid"),
        "art_fid": all_pairs.get("art_fid"),
        "cmmd": all_pairs.get("cmmd"),
        "dino_structure": all_pairs.get("dino_structure"),
        "gram_micro": all_pairs.get("gram_micro"),
        "gram_macro": all_pairs.get("gram_macro"),
        "transfer_clip_style": transfer.get("clip_style"),
        "photo_to_art_clip_style": photo.get("clip_style"),
    }


def _append_modern_metrics_to_baseline_dir(eval_dir: Path) -> None:
    if str(SB_SRC) not in sys.path:
        sys.path.insert(0, str(SB_SRC))
    modern_metrics = importlib.import_module("utils.modern_metrics")
    cfg = modern_metrics.ModernMetricConfig(
        test_dir=OVERFIT50,
        device="cuda" if shutil.which("nvidia-smi") else "cpu",
        clip_model_name="openai/clip-vit-base-patch32",
        dino_model_name="facebook/dinov2-small",
        cmmd_sigma=10.0,
        batch_size=8,
    )
    modern_metrics.append_modern_metrics_to_summary(eval_dir, cfg)


def run_strong_eval(
    baseline: str,
    result_root: Path,
    enable_artfid: bool,
    artfid_photo_only: bool,
    reference_images_dir: Path | None,
    protocol: str,
    allow_partial_reference: bool,
) -> Path:
    images_dir = _aggregate_baseline_images(
        result_root,
        reference_images_dir=reference_images_dir,
        allow_partial_reference=allow_partial_reference,
    )
    if not any(images_dir.glob("*.jpg")) and not any(images_dir.glob("*.png")):
        raise RuntimeError(f"No aggregated images found for evaluation root: {images_dir}")
    cmd = [
        sys.executable,
        str(SB_RUN_EVAL),
        f"--output={result_root}",
        f"--test_dir={OVERFIT50}",
        f"--style_subdirs={','.join(ALL_STYLES)}",
        "--reuse_generated",
        "--force_regen",
    ]
    if enable_artfid:
        cmd.append("--eval_enable_art_fid")
        if artfid_photo_only:
            cmd.append("--eval_art_fid_photo_only")
    else:
        cmd.append("--no-eval_enable_art_fid")
    _run_timed(cmd, cwd=SB_ROOT, baseline=baseline, style="all", phase="eval", protocol=protocol, output_root=result_root)
    summary_path = result_root / "summary.json"
    if summary_path.exists():
        try:
            _append_modern_metrics_to_baseline_dir(result_root)
        except Exception as exc:
            print(f"[WARN] modern metrics append failed for {result_root.name}: {type(exc).__name__}: {exc}")
    return summary_path


def run_existing_wrapper(
    script_name: str,
    args: list[str],
    *,
    baseline: str,
    style: str,
    phase: str,
    protocol: str,
    output_root: Path,
) -> None:
    script = ROOT / "scripts" / script_name
    _run_timed(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        baseline=baseline,
        style=style,
        phase=phase,
        protocol=protocol,
        output_root=output_root,
    )


def run_cut(
    _: list[str],
    __: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
) -> None:
    args = [
        "--output_root",
        str(output_root),
        "--source_dir",
        str(WORKSPACE_ROOT / "Related_Works" / "runs" / "cut_5x5" / "infer_5x5" / "images"),
    ]
    run_existing_wrapper(
        "copy_cut_results.py",
        args,
        baseline="cut",
        style="all",
        phase="copy",
        protocol=protocol,
        output_root=output_root,
    )


def run_styleid(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
) -> None:
    max_images = "5" if smoke else "0"
    for style in styles:
        args = ["--style", style, "--max_images", max_images, "--output_root", str(output_root)]
        if reference_manifest is not None:
            args.extend(["--content_manifest", str(reference_manifest)])
        run_existing_wrapper("run_styleid.py", args, baseline="styleid", style=style, phase="infer", protocol=protocol, output_root=output_root)


def run_style_aligned(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
) -> None:
    max_images = "5" if smoke else "0"
    for style in styles:
        args = ["--style", style, "--max_images", max_images, "--output_root", str(output_root)]
        if reference_manifest is not None:
            args.extend(["--content_manifest", str(reference_manifest)])
        run_existing_wrapper("run_style_aligned.py", args, baseline="style_aligned", style=style, phase="infer", protocol=protocol, output_root=output_root)


def run_618_external(
    method: str,
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
    *,
    dry_run: bool = False,
    placeholder: bool = False,
) -> None:
    max_images_per_style = "1" if smoke else "30"
    for style in styles:
        args = [
            "--method",
            method,
            "--style",
            style,
            "--max_images",
            "0",
            "--max_images_per_style",
            max_images_per_style,
            "--output_root",
            str(output_root),
        ]
        if reference_manifest is not None:
            args.extend(["--content_manifest", str(reference_manifest)])
        if dry_run:
            args.append("--dry-run")
        if placeholder:
            args.append("--placeholder")
        run_existing_wrapper(
            "run_618_external.py",
            args,
            baseline=method,
            style=style,
            phase="infer",
            protocol=protocol,
            output_root=output_root,
        )


def run_samst(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
    mode: str = "infer",
    train_epochs: int = 100,
) -> None:
    run_mode = "smoke" if smoke else mode
    max_images = "5" if smoke else "0"
    for style in styles:
        phases = ["train", "infer"] if run_mode == "all" else [run_mode]
        for phase in phases:
            args = [
                "--style",
                style,
                "--mode",
                phase,
                "--epochs",
                str(train_epochs),
                "--max_images",
                max_images,
                "--output_root",
                str(output_root),
            ]
            if reference_manifest is not None:
                args.extend(["--content_manifest", str(reference_manifest)])
            run_existing_wrapper("run_samst.py", args, baseline="samst", style=style, phase=phase, protocol=protocol, output_root=output_root)


def run_s2wat(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
    mode: str = "infer",
    train_epochs: int = 2000,
) -> None:
    run_mode = "smoke" if smoke else mode
    max_images = "5" if smoke else "0"
    for style in styles:
        phases = ["train", "infer"] if run_mode == "all" else [run_mode]
        for phase in phases:
            args = [
                "--style",
                style,
                "--mode",
                phase,
                "--epochs",
                str(train_epochs),
                "--max_images",
                max_images,
                "--output_root",
                str(output_root),
            ]
            if reference_manifest is not None:
                args.extend(["--content_manifest", str(reference_manifest)])
            run_existing_wrapper("run_s2wat.py", args, baseline="s2wat", style=style, phase=phase, protocol=protocol, output_root=output_root)


def run_stytr2(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
) -> None:
    repo_dir = RELATED_ROOT / "StyTR-2"
    required = [
        repo_dir / "experiments" / "vgg_normalised.pth",
        repo_dir / "experiments" / "decoder_iter_160000.pth",
        repo_dir / "experiments" / "transformer_iter_160000.pth",
        repo_dir / "experiments" / "embedding_iter_160000.pth",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("StyTR-2 weights missing: " + "; ".join(missing))

    content_dir = _prepare_mixed_content_dir("stytr2", max_images_per_style=5 if smoke else 0)
    for target_style in styles:
        style_dir = _prepare_single_style_ref_dir("stytr2", target_style)
        raw_dir = TMP_ROOT / "stytr2" / "raw" / target_style
        out_dir = output_root / target_style
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_dir / "test.py"),
            "--content_dir",
            str(content_dir),
            "--style_dir",
            str(style_dir),
            "--output",
            str(raw_dir),
        ]
        _run(cmd, cwd=repo_dir)
        for img in sorted(raw_dir.glob("*_stylized_*.jpg")):
            stem = img.stem
            if "_stylized_" not in stem:
                continue
            src_name, style_name = stem.split("_stylized_", 1)
            target = target_style
            dst = out_dir / f"{src_name}_to_{target}.jpg"
            if not dst.exists():
                shutil.copy2(img, dst)


def run_aesfa(
    styles: list[str],
    smoke: bool,
    output_root: Path,
    protocol: str,
    reference_manifest: Path | None = None,
) -> None:
    repo_dir = RELATED_ROOT / "AesFA"
    ckpt = repo_dir / "ckpt" / "main" / "main.pth"
    vgg = repo_dir / "vgg_normalised.pth"
    if not ckpt.exists():
        raise RuntimeError(f"AesFA checkpoint missing: {ckpt}")
    if not vgg.exists():
        raise RuntimeError(f"AesFA VGG weights missing: {vgg}")

    content_dir = _prepare_mixed_content_dir("aesfa", max_images_per_style=5 if smoke else 0)
    original_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(repo_dir)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        config_mod = importlib.import_module("Config")
        test_mod = importlib.import_module("test")
        Config = config_mod.Config
        original_values = {
            "phase": Config.phase,
            "content_dir": Config.content_dir,
            "style_dir": Config.style_dir,
            "ckpt_dir": Config.ckpt_dir,
            "img_dir": Config.img_dir,
            "multi_to_multi": getattr(Config, "multi_to_multi", True),
            "test_content_size": getattr(Config, "test_content_size", 256),
            "test_style_size": getattr(Config, "test_style_size", 256),
            "data_num": Config.data_num,
            "vgg_model": Config.vgg_model,
        }
        for target_style in styles:
            out_dir = output_root / target_style
            out_dir.mkdir(parents=True, exist_ok=True)
            style_dir = _prepare_single_style_ref_dir("aesfa", target_style)
            Config.phase = "test"
            Config.content_dir = str(content_dir)
            Config.style_dir = str(style_dir)
            Config.ckpt_dir = str(repo_dir / "ckpt" / "main")
            Config.img_dir = str(TMP_ROOT / "aesfa" / "raw" / target_style)
            Config.multi_to_multi = True
            Config.test_content_size = 256
            Config.test_style_size = 256
            Config.data_num = 999999
            Config.vgg_model = str(vgg)
            test_mod.main()
            raw_dir = Path(Config.img_dir)
            for img in sorted(raw_dir.glob("*_stylized_*.jpg")):
                stem = img.stem
                src_name, _ = stem.split("_stylized_", 1)
                dst = out_dir / f"{src_name}_to_{target_style}.jpg"
                if not dst.exists():
                    shutil.copy2(img, dst)
        for key, value in original_values.items():
            setattr(Config, key, value)
    finally:
        os.chdir(original_cwd)
        sys.path = old_sys_path


def unsupported_adapter(name: str, reason: str) -> None:
    raise RuntimeError(f"{name} is cloned/registered but not fully automated yet: {reason}")


def write_summary(rows: list[dict[str, Any]], protocol: str) -> tuple[Path, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = protocol or "legacy"
    json_path = RESULTS_ROOT / f"unified_repro_eval_summary_{suffix}.json"
    csv_path = RESULTS_ROOT / f"unified_repro_eval_summary_{suffix}.csv"
    json_path.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "baseline",
        "protocol",
        "status",
        "styles",
        "result_root",
        "reference_images_dir",
        "images_dir",
        "summary_path",
        "elapsed_sec",
        "error",
        "clip_style",
        "clip_content",
        "content_lpips",
        "fid",
        "art_fid",
        "cmmd",
        "dino_structure",
        "gram_micro",
        "gram_macro",
        "transfer_clip_style",
        "photo_to_art_clip_style",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return json_path, csv_path


def write_timing_summary(protocol: str) -> tuple[Path, Path] | None:
    if not TIMING_ROWS:
        return None
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = protocol or "legacy"
    json_path = RESULTS_ROOT / f"runtime_summary_{suffix}.json"
    csv_path = RESULTS_ROOT / f"runtime_summary_{suffix}.csv"
    fieldnames = ["baseline", "style", "phase", "protocol", "output_root", "elapsed_sec", "status", "error"]
    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = [dict(row) for row in csv.DictReader(f)]
    merged_rows = existing_rows + TIMING_ROWS
    json_path.write_text(json.dumps({"runs": merged_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified baseline reproduction + evaluation entrypoint.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["cut", "styleid", "style_aligned", "samst", "s2wat", "stytr2", "aesfa"],
        help="Baselines to run. Registered clone-only baselines are: aespa, artbank, cyclegan, adain.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=ALL_STYLES,
        help="Target styles to reproduce.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run small-scale inference where supported.")
    parser.add_argument(
        "--external-dry-run",
        action="store_true",
        help="For 618 external methods, only write per-image job manifests instead of calling upstream repos.",
    )
    parser.add_argument(
        "--external-placeholder",
        action="store_true",
        help="For 618 external methods, create source-copy outputs for pipeline smoke tests only.",
    )
    parser.add_argument("--clone-missing", action="store_true", help="Clone missing official repos before execution.")
    parser.add_argument("--clone-only", action="store_true", help="Only ensure repos, do not run baselines.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip SchrodingerBridge evaluation.")
    parser.add_argument("--no-artfid", action="store_true", help="Disable ArtFID/FID during evaluation.")
    parser.add_argument("--artfid-photo-only", action="store_true", help="Only compute ArtFID/FID for photo->art directions.")
    parser.add_argument(
        "--train-mode",
        choices=["infer", "all", "train"],
        default="infer",
        help="Mode for trainable wrappers such as SaMST/S2WAT when --smoke is not set. Default reuses existing checkpoints for inference.",
    )
    parser.add_argument("--samst-epochs", type=int, default=100, help="SaMST training epochs when --train-mode includes training.")
    parser.add_argument("--s2wat-epochs", type=int, default=2000, help="S2WAT training epochs when --train-mode includes training.")
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
        help=f"Output protocol folder under each baseline. Use 'legacy' to write directly under results/<baseline>. Default: {DEFAULT_PROTOCOL}",
    )
    parser.add_argument(
        "--reference-images-dir",
        type=Path,
        default=None,
        help="Optional generated-image manifest directory. For protocol_a_800 this defaults to the selected Ours epoch_0001 images.",
    )
    parser.add_argument(
        "--allow-partial-reference",
        action="store_true",
        help="Allow generated images to be a subset of --reference-images-dir. Intended for smoke tests only.",
    )
    args = parser.parse_args()

    repo_status = ensure_repos(clone_missing=bool(args.clone_missing))
    print(json.dumps({"repo_status": repo_status}, indent=2, ensure_ascii=False))
    if args.clone_only:
        return 0

    adapters: dict[str, tuple[Any, str]] = {
        "cut": (run_cut, "fully automated via existing wrapper"),
        "styleid": (run_styleid, "fully automated via existing wrapper"),
        "style_aligned": (run_style_aligned, "fully automated via existing wrapper"),
        "stylegallery": (
            lambda styles, smoke, output_root, protocol, reference_manifest=None: run_618_external(
                "stylegallery",
                styles,
                smoke,
                output_root,
                protocol,
                reference_manifest,
                dry_run=bool(args.external_dry_run),
                placeholder=bool(args.external_placeholder),
            ),
            "618 external adapter; requires STYLEGALLERY_CMD or --external-dry-run",
        ),
        "ham": (
            lambda styles, smoke, output_root, protocol, reference_manifest=None: run_618_external(
                "ham",
                styles,
                smoke,
                output_root,
                protocol,
                reference_manifest,
                dry_run=bool(args.external_dry_run),
                placeholder=bool(args.external_placeholder),
            ),
            "618 external adapter; requires HAM_CMD or --external-dry-run",
        ),
        "scheduled_style_injection": (
            lambda styles, smoke, output_root, protocol, reference_manifest=None: run_618_external(
                "scheduled_style_injection",
                styles,
                smoke,
                output_root,
                protocol,
                reference_manifest,
                dry_run=bool(args.external_dry_run),
                placeholder=bool(args.external_placeholder),
            ),
            "618 external adapter; requires SCHEDULED_STYLE_INJECTION_CMD or --external-dry-run",
        ),
        "csgo": (
            lambda styles, smoke, output_root, protocol, reference_manifest=None: run_618_external(
                "csgo",
                styles,
                smoke,
                output_root,
                protocol,
                reference_manifest,
                dry_run=bool(args.external_dry_run),
                placeholder=bool(args.external_placeholder),
            ),
            "618 external adapter; requires CSGO_CMD/pretrained inference or --external-dry-run",
        ),
        "samst": (run_samst, "fully automated via existing wrapper"),
        "s2wat": (run_s2wat, "fully automated via existing wrapper"),
        "stytr2": (run_stytr2, "inference automated if official weights are provided"),
        "aesfa": (run_aesfa, "inference automated if official checkpoint and VGG weights are provided"),
        "aespa": (lambda *_: unsupported_adapter("AesPA-Net", "official repo has no ready-made paper-eval entry or weight downloader"), "clone-only"),
        "artbank": (lambda *_: unsupported_adapter("ArtBank", "official test path is hardcoded to local checkpoints/prompts"), "clone-only"),
        "cyclegan": (lambda *_: unsupported_adapter("CycleGAN", "dataset/train/test orchestration not yet specialized for this repo"), "clone-only"),
        "adain": (lambda *_: unsupported_adapter("AdaIN", "official repo clone is network-blocked in this session"), "clone-only"),
    }

    rows: list[dict[str, Any]] = []
    protocol = None if str(args.protocol).lower() == "legacy" else args.protocol
    reference_images_dir = args.reference_images_dir
    if reference_images_dir is None and protocol == DEFAULT_PROTOCOL and DEFAULT_PROTOCOL_A_REFERENCE_IMAGES.exists():
        reference_images_dir = DEFAULT_PROTOCOL_A_REFERENCE_IMAGES
    if reference_images_dir is not None:
        reference_images_dir = reference_images_dir.resolve()
    reference_manifest = None
    if reference_images_dir is not None:
        reference_manifest = _write_reference_manifest(reference_images_dir, TMP_ROOT / (protocol or "legacy") / "reference_content_manifest.txt")
    for baseline in args.baselines:
        key = baseline.lower()
        if key not in adapters:
            rows.append({"baseline": baseline, "status": "unknown", "styles": ",".join(args.styles), "error": "unregistered baseline"})
            continue
        fn, note = adapters[key]
        start = time.time()
        result_root = _baseline_result_dir(key, protocol=protocol)
        row: dict[str, Any] = {
            "baseline": key,
            "protocol": protocol or "legacy",
            "styles": ",".join(args.styles),
            "status": "started",
            "adapter_note": note,
            "error": "",
            "images_dir": str(result_root / "images"),
            "result_root": str(result_root),
            "reference_images_dir": str(reference_images_dir) if reference_images_dir else "",
        }
        try:
            if key in {"samst", "s2wat"}:
                epochs = args.samst_epochs if key == "samst" else args.s2wat_epochs
                fn(args.styles, args.smoke, result_root, protocol or "legacy", reference_manifest, args.train_mode, epochs)
            else:
                fn(args.styles, args.smoke, result_root, protocol or "legacy", reference_manifest)
            summary_path = None
            if not args.skip_eval:
                summary_path = run_strong_eval(
                    key,
                    result_root,
                    enable_artfid=not args.no_artfid,
                    artfid_photo_only=bool(args.artfid_photo_only),
                    reference_images_dir=reference_images_dir,
                    protocol=protocol or "legacy",
                    allow_partial_reference=bool(args.allow_partial_reference),
                )
                row["summary_path"] = str(summary_path)
                if summary_path.exists():
                    row.update(_load_summary_metrics(summary_path))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = round(time.time() - start, 2)
        rows.append(row)

    json_path, csv_path = write_summary(rows, protocol or "legacy")
    timing_paths = write_timing_summary(protocol or "legacy")
    print(f"[DONE] Summary JSON: {json_path}")
    print(f"[DONE] Summary CSV : {csv_path}")
    if timing_paths is not None:
        print(f"[DONE] Runtime JSON: {timing_paths[0]}")
        print(f"[DONE] Runtime CSV : {timing_paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
