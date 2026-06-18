from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent


def _find_workspace() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "Dataset" / "distinct5_512" / "test").is_dir() and (cwd / "Related_Works").is_dir():
        return cwd
    candidate = PIPELINE_ROOT.parent.parent
    if (candidate / "Dataset" / "distinct5_512" / "test").is_dir():
        return candidate
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "Dataset" / "distinct5_512" / "test").is_dir() and (parent / "Related_Works").is_dir():
            return parent
    raise FileNotFoundError("Cannot locate workspace containing Dataset/distinct5_512/test")


WORKSPACE = _find_workspace()
RELATED = WORKSPACE / "Related_Works"
RESULTS = RELATED / "baseline_pipeline" / "results"
DATA_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_VRAM_BUDGET_MB = 7900
VRAM_BUDGET_MB = DEFAULT_VRAM_BUDGET_MB


@dataclass(frozen=True)
class Case:
    src_style: str
    tgt_style: str
    content: Path
    style_ref: Path
    output: Path

    @property
    def output_name(self) -> str:
        return self.output.name


def _images(style: str) -> list[Path]:
    root = DATA_ROOT / style
    if not root.is_dir():
        raise FileNotFoundError(f"Missing WikiArt5 style dir: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _build_cases(output_root: Path, max_per_style: int, styles: list[str]) -> list[Case]:
    cases: list[Case] = []
    style_refs = {style: _images(style)[0] for style in styles}
    for src_style in styles:
        srcs = _images(src_style)
        if max_per_style > 0:
            srcs = srcs[:max_per_style]
        for content in srcs:
            for tgt_style in styles:
                out_name = f"{src_style}__{content.stem}__to__{tgt_style}.png"
                cases.append(
                    Case(
                        src_style=src_style,
                        tgt_style=tgt_style,
                        content=content,
                        style_ref=style_refs[tgt_style],
                        output=output_root / "images" / out_name,
                    )
                )
    return cases


def _run(cmd: list[str], cwd: Path, log_path: Path, timeout_sec: int = 900) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("[CMD] " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen([str(x) for x in cmd], cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while True:
            rc = proc.poll()
            if rc is not None:
                log.write(f"[RETURN] {rc}\n")
                return int(rc)
            used = _gpu_memory_used_mb()
            if used is not None and used >= VRAM_BUDGET_MB:
                proc.kill()
                log.write(f"[KILLED] GPU memory reached {used}MB over {VRAM_BUDGET_MB}MB budget\n")
                return 99
            if time.time() - started > timeout_sec:
                proc.kill()
                log.write(f"[TIMEOUT] exceeded {timeout_sec}s\n")
                return 98
            time.sleep(5)


def _gpu_memory_used_mb() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
        first = proc.stdout.strip().splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


def _guard_vram() -> tuple[bool, str]:
    used = _gpu_memory_used_mb()
    if used is None:
        return True, ""
    if used >= VRAM_BUDGET_MB:
        return False, f"GPU memory already {used}MB, refusing to start over {VRAM_BUDGET_MB}MB budget"
    return True, ""


def _copy_result(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Expected generated image not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _stylegallery(case: Case, work_dir: Path, log_path: Path) -> tuple[str, str]:
    ok, reason = _guard_vram()
    if not ok:
        return "blocked", reason
    repo = RELATED / "StyleGallery"
    if not repo.is_dir():
        return "blocked", f"missing repo: {repo}"
    out_dir = work_dir / case.output.stem
    out_img = out_dir / "result.png"
    sd15_path = RELATED / "StyleShot" / "runwayml" / "stable-diffusion-v1-5"
    model_name = sd15_path if (sd15_path / "model_index.json").is_file() else Path("runwayml/stable-diffusion-v1-5")
    cmd = [
        sys.executable,
        "demo.py",
        "--model_name",
        str(model_name),
        "--content_image",
        str(case.content),
        "--style_images",
        str(case.style_ref),
        "--output_folder",
        str(out_dir),
        "--noise_steps",
        "8",
        "--num_optimize_steps",
        "1",
        "--enable_gradient_checkpoint",
        "--c_ratio",
        "0.26",
    ]
    rc = _run(cmd, repo, log_path)
    if rc == 99:
        return "blocked", f"StyleGallery exceeded {VRAM_BUDGET_MB}MB GPU memory budget"
    if rc == 98:
        return "blocked", "StyleGallery timed out during one-case smoke run"
    if rc != 0:
        return "failed", f"StyleGallery demo.py returned {rc}"
    _copy_result(out_img, case.output)
    return "ok", ""


def _styleshot(case: Case, work_dir: Path, log_path: Path) -> tuple[str, str]:
    ok, reason = _guard_vram()
    if not ok:
        return "blocked", reason
    repo = RELATED / "StyleShot"
    if not repo.is_dir():
        return "blocked", f"missing repo: {repo}"
    out_img = work_dir / f"{case.output.stem}.png"
    prompt = f"a painting in {case.tgt_style.replace('_', ' ')} style"
    cmd = [
        sys.executable,
        "styleshot_image_driven_demo.py",
        "--style",
        str(case.style_ref),
        "--content",
        str(case.content),
        "--preprocessor",
        "Contour",
        "--prompt",
        prompt,
        "--output",
        str(out_img),
        "--steps",
        "8",
        "--size",
        "512",
        "--guidance_scale",
        "1.0",
    ]
    rc = _run(cmd, repo, log_path)
    if rc == 99:
        return "blocked", f"StyleShot exceeded {VRAM_BUDGET_MB}MB GPU memory budget"
    if rc == 98:
        return "blocked", "StyleShot timed out during one-case smoke run"
    if rc != 0:
        return "failed", f"StyleShot image demo returned {rc}"
    _copy_result(out_img, case.output)
    return "ok", ""


def _csgo(case: Case, work_dir: Path, log_path: Path) -> tuple[str, str]:
    ok, reason = _guard_vram()
    if not ok:
        return "blocked", reason
    repo = RELATED / "CSGO"
    if not repo.is_dir():
        return "blocked", f"missing repo: {repo}"
    out_img = work_dir / f"{case.output.stem}.png"
    prompt = f"a painting in {case.tgt_style.replace('_', ' ')} style"
    cmd = [
        sys.executable,
        "smoke_wikiart.py",
        "--content",
        str(case.content),
        "--style",
        str(case.style_ref),
        "--output",
        str(out_img),
        "--prompt",
        prompt,
        "--steps",
        "8",
        "--size",
        "512",
    ]
    rc = _run(cmd, repo, log_path, timeout_sec=1800)
    if rc == 99:
        return "blocked", f"CSGO exceeded {VRAM_BUDGET_MB}MB GPU memory budget"
    if rc == 98:
        return "blocked", "CSGO timed out during one-case smoke run"
    if rc != 0:
        return "failed", f"CSGO smoke_wikiart.py returned {rc}"
    _copy_result(out_img, case.output)
    return "ok", ""


def _scsa(case: Case, work_dir: Path, log_path: Path) -> tuple[str, str]:
    return (
        "blocked",
        (
            "Local HZAI-ZJNU/SCSA is a classification/detection Spatial and Channel Synergistic Attention repo, "
            "not a released style-transfer inference repo; README also says weights will be open-sourced later."
        ),
    )


RUNNERS = {
    "stylegallery": _stylegallery,
    "styleshot": _styleshot,
    "csgo": _csgo,
    "scsa": _scsa,
}


def main() -> int:
    global VRAM_BUDGET_MB
    parser = argparse.ArgumentParser(description="Run 618 WikiArt distinct5 baselines and place images beside SaMAM results.")
    parser.add_argument("--method", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--max-per-style", type=int, default=30, help="30 gives the full 5x5x30 protocol.")
    parser.add_argument("--styles", nargs="+", default=STYLES)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--vram-budget-mb", type=int, default=DEFAULT_VRAM_BUDGET_MB)
    args = parser.parse_args()

    method = args.method.lower()
    VRAM_BUDGET_MB = int(args.vram_budget_mb)
    output_root = args.output_root or (RESULTS / f"{method}_wikiart5_618")
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"{method}_run.log"

    cases = _build_cases(output_root, max_per_style=int(args.max_per_style), styles=list(args.styles))
    if args.limit_cases > 0:
        cases = cases[: int(args.limit_cases)]
    rows: list[dict[str, object]] = []
    started = time.time()
    runner = RUNNERS[method]
    for idx, case in enumerate(cases, start=1):
        row = {
            "idx": idx,
            "method": method,
            "src_style": case.src_style,
            "tgt_style": case.tgt_style,
            "content": str(case.content),
            "style_ref": str(case.style_ref),
            "output": str(case.output),
            "status": "started",
            "error": "",
        }
        if args.skip_existing and case.output.is_file():
            row["status"] = "skipped_existing"
        else:
            try:
                status, error = runner(case, work_dir, log_path)
                row["status"] = status
                row["error"] = error
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"[{method}] {idx}/{len(cases)} {case.output_name}: {row['status']}", flush=True)
        if row["status"] in {"blocked", "failed"}:
            break

    status_path = output_root / "repro_status.json"
    csv_path = output_root / "repro_status.csv"
    payload = {
        "method": method,
        "output_root": str(output_root),
        "images_dir": str(output_root / "images"),
        "requested_cases": len(cases),
        "completed_images": len([p for p in (output_root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS]),
        "elapsed_sec": round(time.time() - started, 3),
        "rows": rows,
    }
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["idx", "method", "src_style", "tgt_style", "content", "style_ref", "output", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if any(str(row["status"]) in {"failed", "blocked"} for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
