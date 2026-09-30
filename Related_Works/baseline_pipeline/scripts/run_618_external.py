from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"

ALL_STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

METHOD_DEFAULTS = {
    "stylegallery": {
        "display_name": "StyleGallery",
        "repo": WORKSPACE_ROOT / "Related_Works" / "repos" / "StyleGallery",
        "command_env": "STYLEGALLERY_CMD",
        "note": "Set STYLEGALLERY_CMD or pass --command-template after cloning the official repo.",
    },
    "ham": {
        "display_name": "HAM",
        "repo": WORKSPACE_ROOT / "Related_Works" / "repos" / "HAM",
        "command_env": "HAM_CMD",
        "note": "Set HAM_CMD or pass --command-template after cloning the official repo.",
    },
    "scheduled_style_injection": {
        "display_name": "Scheduled Style Injection",
        "repo": WORKSPACE_ROOT / "Related_Works" / "repos" / "ScheduledStyleInjection",
        "command_env": "SCHEDULED_STYLE_INJECTION_CMD",
        "note": "Set SCHEDULED_STYLE_INJECTION_CMD or pass --command-template after cloning the official repo.",
    },
    "csgo": {
        "display_name": "CSGO",
        "repo": WORKSPACE_ROOT / "Related_Works" / "repos" / "CSGO",
        "command_env": "CSGO_CMD",
        "note": "Prefer pretrained inference. Full IMAGStyle training is not a 4070/8GB first pass.",
    },
}


@dataclass(frozen=True)
class Job:
    src_style: str
    tgt_style: str
    src_path: Path
    style_ref: Path
    out_path: Path

    def template_vars(self) -> dict[str, str]:
        return {
            "workspace": str(WORKSPACE_ROOT),
            "pipeline_root": str(PIPELINE_ROOT),
            "content": str(self.src_path),
            "content_image": str(self.src_path),
            "style_ref": str(self.style_ref),
            "style_image": str(self.style_ref),
            "output": str(self.out_path),
            "output_image": str(self.out_path),
            "target_style": self.tgt_style,
            "tgt_style": self.tgt_style,
            "source_style": self.src_style,
            "src_style": self.src_style,
            "src_stem": self.src_path.stem,
            "src_name": self.src_path.name,
        }


def _read_manifest(content_manifest: Path | None) -> list[tuple[str, Path]] | None:
    if content_manifest is None:
        return None
    items: list[tuple[str, Path]] = []
    for raw in content_manifest.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name:
            continue
        if "_" not in name:
            raise ValueError(f"Cannot parse manifest entry without style prefix: {name}")
        src_style, img_name = name.split("_", 1)
        items.append((src_style, OVERFIT50 / src_style / img_name))
    return items


def _source_items(max_images: int, max_images_per_style: int, content_manifest: Path | None) -> list[tuple[str, Path]]:
    manifest_items = _read_manifest(content_manifest)
    if manifest_items is not None:
        items = manifest_items
    else:
        items = []
        for style in ALL_STYLES:
            style_dir = OVERFIT50 / style
            paths = sorted(p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if max_images_per_style > 0:
                paths = paths[:max_images_per_style]
            items.extend((style, p) for p in paths)
    if max_images > 0:
        items = items[:max_images]
    missing = [str(path) for _, path in items if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing content images: " + "; ".join(missing[:5]))
    return items


def _style_reference(target_style: str, ref_index: int) -> Path:
    style_dir = OVERFIT50 / target_style
    refs = sorted(p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not refs:
        raise FileNotFoundError(f"No style references found under {style_dir}")
    return refs[min(max(0, ref_index), len(refs) - 1)]


def _build_jobs(
    *,
    target_style: str,
    output_root: Path,
    max_images: int,
    max_images_per_style: int,
    content_manifest: Path | None,
    ref_index: int,
) -> list[Job]:
    style_ref = _style_reference(target_style, ref_index)
    jobs: list[Job] = []
    for src_style, src_path in _source_items(
        max_images=max_images,
        max_images_per_style=max_images_per_style,
        content_manifest=content_manifest,
    ):
        suffix = ".png"
        out_name = f"{src_style}_{src_path.stem}_to_{target_style}{suffix}"
        jobs.append(
            Job(
                src_style=src_style,
                tgt_style=target_style,
                src_path=src_path,
                style_ref=style_ref,
                out_path=output_root / target_style / out_name,
            )
        )
    return jobs


def _format_template(template: str, job: Job, repo: Path, work_dir: Path) -> list[str]:
    values = job.template_vars()
    values.update({"repo": str(repo), "work_dir": str(work_dir)})
    command = template.format(**values)
    if os.name == "nt":
        return command
    return shlex.split(command)


def _placeholder_image(job: Job) -> None:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(job.src_path) as im:
        im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS).save(job.out_path)


def _run_job(template: str, job: Job, repo: Path, work_dir: Path) -> tuple[str, str]:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _format_template(template, job, repo, work_dir)
    result = subprocess.run(cmd, cwd=str(repo if repo.is_dir() else work_dir), shell=isinstance(cmd, str))
    if result.returncode != 0:
        return "failed", f"command returned {result.returncode}"
    if not job.out_path.is_file():
        return "failed", f"command completed but did not create {job.out_path}"
    return "ok", ""


def run_external_method(
    *,
    method: str,
    target_style: str,
    max_images: int,
    max_images_per_style: int,
    output_root: Path,
    content_manifest: Path | None,
    repo: Path | None,
    command_template: str | None,
    dry_run: bool,
    placeholder: bool,
    ref_index: int,
) -> int:
    method_key = method.lower()
    defaults = METHOD_DEFAULTS.get(method_key, {})
    display_name = defaults.get("display_name", method)
    repo_path = (repo or defaults.get("repo") or (WORKSPACE_ROOT / "Related_Works" / "repos" / method)).resolve()
    env_name = defaults.get("command_env", f"{method_key.upper()}_CMD")
    template = command_template or os.environ.get(str(env_name), "")
    jobs = _build_jobs(
        target_style=target_style,
        output_root=output_root.resolve(),
        max_images=max_images,
        max_images_per_style=max_images_per_style,
        content_manifest=content_manifest,
        ref_index=ref_index,
    )
    started = time.time()
    rows: list[dict[str, str | int | float]] = []
    status = "ok"
    error = ""
    if not template and not dry_run and not placeholder:
        status = "blocked"
        error = f"No command template configured. {defaults.get('note', '')}".strip()
    elif not repo_path.exists() and not dry_run and not placeholder:
        status = "blocked"
        error = f"Repo path does not exist: {repo_path}"

    work_dir = output_root / "_work" / method_key / target_style
    work_dir.mkdir(parents=True, exist_ok=True)
    for idx, job in enumerate(jobs, start=1):
        row_status = status if status != "ok" else "pending"
        row_error = error
        if status == "ok":
            if dry_run:
                row_status = "dry_run"
            elif placeholder:
                _placeholder_image(job)
                row_status = "placeholder"
                row_error = "Copied source image as a pipeline smoke placeholder; do not report as method result."
            elif job.out_path.exists():
                row_status = "skipped_existing"
            else:
                row_status, row_error = _run_job(template, job, repo_path, work_dir)
        rows.append(
            {
                "idx": idx,
                "method": method_key,
                "display_name": display_name,
                "src_style": job.src_style,
                "tgt_style": job.tgt_style,
                "content": str(job.src_path),
                "style_ref": str(job.style_ref),
                "output": str(job.out_path),
                "status": row_status,
                "error": row_error,
            }
        )
    if any(row["status"] == "failed" for row in rows):
        status = "failed"
    if any(row["status"] == "blocked" for row in rows):
        status = "blocked"

    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method_key,
        "display_name": display_name,
        "status": status,
        "error": error,
        "repo": str(repo_path),
        "command_env": env_name,
        "command_template_configured": bool(template),
        "target_style": target_style,
        "job_count": len(rows),
        "elapsed_sec": round(time.time() - started, 3),
        "rows": rows,
    }
    status_path = output_root / f"{method_key}_{target_style}_618_external_status.json"
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{display_name}] {status}: {len(rows)} jobs -> {status_path}")
    if status in {"blocked", "failed"}:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 618 external style-transfer methods on WikiArt distinct5.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_DEFAULTS))
    parser.add_argument("--style", required=True, choices=ALL_STYLES)
    parser.add_argument("--max_images", type=int, default=0, help="Max source images before target-style expansion. 0 means all.")
    parser.add_argument(
        "--max_images_per_style",
        type=int,
        default=30,
        help="Max source images per source style when no content manifest is supplied. Default matches 5x5x30.",
    )
    parser.add_argument("--output_root", type=Path, default=PIPELINE_ROOT / "results" / "618_external")
    parser.add_argument("--content_manifest", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument(
        "--command-template",
        default=None,
        help=(
            "External per-image command. Placeholders include {content}, {style_ref}, {output}, "
            "{target_style}, {source_style}, {repo}, {work_dir}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only materialize the job/status manifest.")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Create source-copy placeholder outputs for pipeline smoke tests only.",
    )
    parser.add_argument("--style-ref-index", type=int, default=0, help="Reference image index within each target style.")
    args = parser.parse_args()
    manifest = args.content_manifest.resolve() if args.content_manifest else None
    repo = args.repo.resolve() if args.repo else None
    return run_external_method(
        method=args.method,
        target_style=args.style,
        max_images=args.max_images,
        max_images_per_style=args.max_images_per_style,
        output_root=args.output_root.resolve(),
        content_manifest=manifest,
        repo=repo,
        command_template=args.command_template,
        dry_run=bool(args.dry_run),
        placeholder=bool(args.placeholder),
        ref_index=int(args.style_ref_index),
    )


if __name__ == "__main__":
    raise SystemExit(main())
