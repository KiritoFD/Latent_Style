from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLASSES = ["Realism", "Impressionism", "Post_Impressionism", "Expressionism", "Symbolism"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STYLE_PROMPTS = {
    "Realism": "把图片转为Realism风格",
    "Impressionism": "把图片转为Impressionism风格",
    "Post_Impressionism": "把图片转为Post_Impressionism风格",
    "Expressionism": "把图片转为Expressionism风格",
    "Symbolism": "把图片转为Symbolism风格",
}


STYLE_PROMPTS = {
    "Realism": "\u628a\u56fe\u7247\u8f6c\u4e3aRealism\u98ce\u683c",
    "Impressionism": "\u628a\u56fe\u7247\u8f6c\u4e3aImpressionism\u98ce\u683c",
    "Post_Impressionism": "\u628a\u56fe\u7247\u8f6c\u4e3aPost_Impressionism\u98ce\u683c",
    "Expressionism": "\u628a\u56fe\u7247\u8f6c\u4e3aExpressionism\u98ce\u683c",
    "Symbolism": "\u628a\u56fe\u7247\u8f6c\u4e3aSymbolism\u98ce\u683c",
}


@dataclass(frozen=True)
class Job:
    src_style: str
    tgt_style: str
    src_path: Path
    out_path: Path

    @property
    def row_id(self) -> str:
        return f"{self.src_style}/{self.src_path.stem}->{self.tgt_style}"


def _paths_by_style(
    image_root: Path,
    classes: list[str],
    max_sources_per_style: int,
    exclude_source_substrings: list[str] | None = None,
) -> dict[str, list[Path]]:
    excludes = [item.lower() for item in (exclude_source_substrings or []) if item]
    paths_by_style: dict[str, list[Path]] = {}
    for style in classes:
        style_dir = image_root / style
        if not style_dir.is_dir():
            raise FileNotFoundError(f"Missing style image dir: {style_dir}")
        paths = sorted(
            [p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
        if excludes:
            paths = [p for p in paths if not any(token in p.name.lower() or token in p.stem.lower() for token in excludes)]
        if max_sources_per_style > 0:
            paths = paths[:max_sources_per_style]
        paths_by_style[style] = paths
    return paths_by_style


def _iter_source_images(
    image_root: Path,
    classes: list[str],
    max_sources_per_style: int,
    unit_order: bool = False,
    exclude_source_substrings: list[str] | None = None,
) -> list[tuple[str, Path]]:
    paths_by_style = _paths_by_style(image_root, classes, max_sources_per_style, exclude_source_substrings)
    items: list[tuple[str, Path]] = []
    if unit_order:
        max_count = max((len(paths) for paths in paths_by_style.values()), default=0)
        for idx in range(max_count):
            for style in classes:
                paths = paths_by_style[style]
                if idx < len(paths):
                    items.append((style, paths[idx]))
        return items
    for style in classes:
        paths = paths_by_style[style]
        for path in paths:
            items.append((style, path))
    return items


def _build_jobs(
    *,
    image_root: Path,
    output_root: Path,
    classes: list[str],
    max_sources_per_style: int,
    cross_only: bool,
    unit_order: bool,
    exclude_source_substrings: list[str] | None,
) -> list[Job]:
    jobs: list[Job] = []
    for src_style, src_path in _iter_source_images(
        image_root,
        classes,
        max_sources_per_style,
        unit_order,
        exclude_source_substrings,
    ):
        for tgt_style in classes:
            if cross_only and src_style == tgt_style:
                continue
            out_name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
            jobs.append(
                Job(
                    src_style=src_style,
                    tgt_style=tgt_style,
                    src_path=src_path,
                    out_path=output_root / tgt_style / out_name,
                )
            )
    return jobs


def _parse_size(size_text: str) -> tuple[int, int]:
    width_text, height_text = size_text.lower().split("x", 1)
    return int(width_text), int(height_text)


def _image_to_data_url(path: Path, input_resize: str = "") -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    if input_resize:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required for --input-resize.") from exc
        target = _parse_size(input_resize)
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != target:
                im = im.resize(target, Image.Resampling.LANCZOS)
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=95)
            raw = out.getvalue()
            mime = "image/jpeg"
    else:
        raw = path.read_bytes()
    payload = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _api_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


class RateLimiter:
    def __init__(self, rpm: float, launch_interval_sec: float = 0.0) -> None:
        if launch_interval_sec > 0.0:
            self.interval = float(launch_interval_sec)
        else:
            self.interval = 0.0 if rpm <= 0 else 60.0 / float(rpm)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> float:
        if self.interval <= 0.0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait > 0.0:
            time.sleep(wait)
        return wait


class ApiKeyRing:
    def __init__(self, keys: list[str]) -> None:
        clean = [k.strip() for k in keys if k.strip()]
        if not clean:
            raise RuntimeError("No Seedream API key configured.")
        self._keys = clean
        self._lock = threading.Lock()
        self._idx = 0

    @property
    def count(self) -> int:
        return len(self._keys)

    def next(self) -> tuple[int, str]:
        with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._keys)
        return idx, self._keys[idx]


def _request_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    transport: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], float, int, int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    if transport == "curl":
        start = time.time()
        cmd = [
            "curl.exe",
            "-sS",
            "-X",
            "POST",
            url,
            "--max-time",
            str(max(1, int(timeout))),
            "--data-binary",
            "@-",
            "-w",
            "\n__HTTP_STATUS__:%{http_code}",
        ]
        for key, value in headers.items():
            cmd[6:6] = ["-H", f"{key}: {value}"]
        proc = subprocess.run(
            cmd,
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(5, int(timeout) + 30),
        )
        elapsed = time.time() - start
        raw_all = proc.stdout.decode("utf-8", errors="replace")
        marker = "\n__HTTP_STATUS__:"
        if marker not in raw_all:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"curl failed after {elapsed:.3f}s rc={proc.returncode}: {err[:1200]}")
        raw, status_text = raw_all.rsplit(marker, 1)
        status = int(status_text.strip() or "0")
        if status >= 400 or proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            detail = raw.strip() or err.strip()
            raise RuntimeError(f"HTTP {status} after {elapsed:.3f}s: {detail[:1200]}")
        return json.loads(raw), elapsed, status, len(raw.encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            elapsed = time.time() - start
            raw = raw_bytes.decode("utf-8", errors="replace")
            return json.loads(raw), elapsed, int(getattr(resp, "status", 200)), len(raw_bytes)
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - start
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} after {elapsed:.3f}s: {err[:1200]}") from exc


def _get_json(
    *,
    url: str,
    api_key: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], float, int, int]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers, method="GET")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            elapsed = time.time() - start
            raw = raw_bytes.decode("utf-8", errors="replace")
            return json.loads(raw), elapsed, int(getattr(resp, "status", 200)), len(raw_bytes)
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - start
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} after {elapsed:.3f}s: {err[:1200]}") from exc


def _poll_async_task(
    *,
    base_url: str,
    task_id: str,
    api_key: str,
    task_type: str,
    timeout: float,
    poll_interval_sec: float,
    max_wait_sec: float,
) -> tuple[dict[str, Any], float, int, int]:
    url = _api_url(base_url, f"/v1/tasks/{task_id}")
    headers = {"X-ModelScope-Task-Type": task_type} if task_type else None
    deadline = time.time() + max(1.0, max_wait_sec)
    total_elapsed = 0.0
    while True:
        response, elapsed, status, response_bytes = _get_json(
            url=url,
            api_key=api_key,
            timeout=timeout,
            extra_headers=headers,
        )
        total_elapsed += elapsed
        task_status = str(response.get("task_status", ""))
        if task_status not in {"PENDING", "PROCESSING", "RUNNING"}:
            return response, total_elapsed, status, response_bytes
        if time.time() >= deadline:
            raise TimeoutError(f"Async task {task_id} still {task_status} after {max_wait_sec:.1f}s")
        time.sleep(max(1.0, poll_interval_sec))


def _download_url(url: str, timeout: float) -> tuple[bytes, float]:
    req = urllib.request.Request(url, headers={"User-Agent": "LANCET-seedream-baseline/1.0"})
    start = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), time.time() - start
        except Exception as exc:
            last_exc = exc
            if attempt >= 5:
                break
            time.sleep(float(attempt * 2))
    raise RuntimeError(f"download failed after retries: {last_exc}") from last_exc


def _extract_image_bytes(response: dict[str, Any], timeout: float) -> tuple[bytes, str, float]:
    output_images = response.get("output_images")
    if isinstance(output_images, list) and output_images:
        url = output_images[0]
        if isinstance(url, str) and url.startswith("http"):
            image_bytes, download_elapsed = _download_url(url, timeout)
            return image_bytes, url, download_elapsed

    data = response.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            b64 = first.get("b64_json") or first.get("b64")
            if isinstance(b64, str) and b64:
                return base64.b64decode(b64), "b64_json", 0.0
            url = first.get("url")
            if isinstance(url, str) and url:
                image_bytes, download_elapsed = _download_url(url, timeout)
                return image_bytes, url, download_elapsed
    # Some gateways wrap the OpenAI-like result.
    for key in ("result", "output", "image"):
        value = response.get(key)
        if isinstance(value, str) and value.startswith("http"):
            image_bytes, download_elapsed = _download_url(value, timeout)
            return image_bytes, value, download_elapsed
        if isinstance(value, str) and len(value) > 100:
            try:
                return base64.b64decode(value), key, 0.0
            except Exception:
                pass
    raise RuntimeError(f"No image found in response keys={list(response.keys())}")


def _resize_image_bytes(image_bytes: bytes, size_text: str, output_format: str) -> bytes:
    if not size_text:
        return image_bytes
    try:
        size = _parse_size(size_text)
    except Exception as exc:
        raise ValueError(f"--resize-output must look like WIDTHxHEIGHT, got {size_text!r}") from exc
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for --resize-output.") from exc
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        if im.size == size:
            return image_bytes
        im = im.resize(size, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        fmt = (output_format or "png").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        im.save(out, format=fmt)
        return out.getvalue()


def _payload_for_job(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    prompt = STYLE_PROMPTS.get(job.tgt_style, f"Transform the input into {job.tgt_style} painting style.")
    if args.prompt_suffix:
        prompt = f"{prompt} {args.prompt_suffix.strip()}"
    image_value = _image_to_data_url(job.src_path, str(args.input_resize))
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "size": args.size,
        "watermark": False,
        "sequential_image_generation": "disabled",
        "response_format": args.response_format,
    }
    if not args.omit_n:
        payload["n"] = 1
    if args.output_format:
        payload["output_format"] = args.output_format
    if args.optimize_prompt_mode:
        payload["optimize_prompt_options"] = {"mode": args.optimize_prompt_mode}
    if args.image_field in {"image", "image_input", "image_urls", "image_url"}:
        payload[args.image_field] = [image_value]
    else:
        raise ValueError(f"Unsupported image field: {args.image_field}")
    return payload


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "status",
        "src_style",
        "tgt_style",
        "src_path",
        "out_path",
        "prompt",
        "size",
        "elapsed_sec",
        "request_elapsed_sec",
        "download_elapsed_sec",
        "write_elapsed_sec",
        "rate_wait_sec",
        "attempts",
        "source",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    vals = sorted(float(v) for v in values)

    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "count": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "min": round(vals[0], 3),
        "p50": round(pct(0.5), 3),
        "p90": round(pct(0.9), 3),
        "p95": round(pct(0.95), 3),
        "max": round(vals[-1], 3),
    }


def _base_row(job: Job, status: str) -> dict[str, Any]:
    return {
        "row_id": job.row_id,
        "status": status,
        "src_style": job.src_style,
        "tgt_style": job.tgt_style,
        "src_path": str(job.src_path),
        "out_path": str(job.out_path),
        "prompt": STYLE_PROMPTS.get(job.tgt_style, f"Transform the input into {job.tgt_style} painting style."),
        "size": "",
        "elapsed_sec": 0.0,
        "request_elapsed_sec": 0.0,
        "download_elapsed_sec": 0.0,
        "write_elapsed_sec": 0.0,
        "rate_wait_sec": 0.0,
        "attempts": 0,
        "source": "",
        "error": "",
    }


def _run_one_job(
    *,
    args: argparse.Namespace,
    job: Job,
    job_index: int,
    num_jobs: int,
    url: str,
    limiter: RateLimiter,
    keyring: ApiKeyRing,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"[{job_index}/{num_jobs}] {job.row_id}")
    start = time.time()
    row = _base_row(job, "pending")
    attempts: list[dict[str, Any]] = []
    try:
        payload = _payload_for_job(args, job)
        row["prompt"] = str(payload.get("prompt", ""))
        row["size"] = str(payload.get("size", ""))
        if args.dry_run:
            row["status"] = "dry_run"
            row["source"] = "none"
            return row, attempts

        last_exc: Exception | None = None
        response: dict[str, Any] | None = None
        response_elapsed = 0.0
        response_status = 0
        response_bytes = 0
        rate_wait_total = 0.0
        used_attempts = 0
        for attempt in range(1, max(1, int(args.retries)) + 1):
            used_attempts = attempt
            rate_wait = limiter.acquire()
            api_key_index, api_key = keyring.next()
            rate_wait_total += rate_wait
            attempt_start = time.time()
            attempt_row = {
                "row_id": job.row_id,
                "attempt": attempt,
                "api_key_index": api_key_index,
                "status": "pending",
                "rate_wait_sec": round(rate_wait, 3),
                "request_elapsed_sec": 0.0,
                "http_status": 0,
                "response_bytes": 0,
                "error": "",
            }
            try:
                response, response_elapsed, response_status, response_bytes = _request_json(
                    url=url,
                    api_key=api_key,
                    payload=payload,
                    timeout=float(args.timeout),
                    transport=str(args.transport),
                    extra_headers={"X-ModelScope-Async-Mode": "true"} if args.async_mode else None,
                )
                if args.async_mode:
                    task_id = str(response.get("task_id", "")).strip()
                    if not task_id:
                        raise RuntimeError(f"Async response did not include task_id: {response}")
                    polled, poll_elapsed, poll_status, poll_bytes = _poll_async_task(
                        base_url=str(args.base_url),
                        task_id=task_id,
                        api_key=api_key,
                        task_type=str(args.async_task_type),
                        timeout=float(args.timeout),
                        poll_interval_sec=float(args.async_poll_interval_sec),
                        max_wait_sec=float(args.async_max_wait_sec),
                    )
                    response = polled
                    response_elapsed += poll_elapsed
                    response_status = poll_status
                    response_bytes += poll_bytes
                    if str(response.get("task_status", "")) != "SUCCEED":
                        raise RuntimeError(f"Async task {task_id} ended with response: {response}")
                    attempt_row["task_id"] = task_id
                attempt_row.update(
                    {
                        "status": "ok",
                        "request_elapsed_sec": round(response_elapsed, 3),
                        "http_status": response_status,
                        "response_bytes": response_bytes,
                    }
                )
                attempts.append(attempt_row)
                break
            except Exception as exc:
                last_exc = exc
                attempt_row.update(
                    {
                        "status": "failed",
                        "request_elapsed_sec": round(time.time() - attempt_start, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                attempts.append(attempt_row)
                print(
                    f"  attempt {attempt}/{int(args.retries)} failed for {job.row_id}: {attempt_row['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt >= int(args.retries):
                    raise
                if str(args.retry_backoff) == "constant":
                    sleep_sec = float(args.retry_sleep)
                else:
                    sleep_sec = float(args.retry_sleep) * attempt
                time.sleep(max(0.0, sleep_sec))

        if response is None:
            raise RuntimeError(f"No response: {last_exc}")

        image_bytes, source, download_elapsed = _extract_image_bytes(response, timeout=float(args.timeout))
        image_bytes = _resize_image_bytes(
            image_bytes,
            str(args.resize_output),
            str(args.output_format or "png"),
        )
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        write_start = time.time()
        job.out_path.write_bytes(image_bytes)
        write_elapsed = time.time() - write_start
        row.update(
            {
                "status": "ok",
                "request_elapsed_sec": round(response_elapsed, 3),
                "download_elapsed_sec": round(download_elapsed, 3),
                "write_elapsed_sec": round(write_elapsed, 3),
                "rate_wait_sec": round(rate_wait_total, 3),
                "attempts": used_attempts,
                "source": source,
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["attempts"] = len(attempts)
        print(f"  FAILED: {job.row_id}: {row['error']}", file=sys.stderr)
        if args.stop_on_error:
            raise
    finally:
        row["elapsed_sec"] = round(time.time() - start, 3)
    return row, attempts


def run(args: argparse.Namespace) -> int:
    api_keys = [
        item.strip()
        for item in os.environ.get(args.api_keys_env, "").replace(";", ",").split(",")
        if item.strip()
    ]
    if not api_keys:
        api_keys = [os.environ.get(args.api_key_env, "").strip()]
    if not any(api_keys) and not args.dry_run:
        raise RuntimeError(f"Set {args.api_key_env} before running non-dry-run generation.")

    classes = [item.strip() for item in args.classes.split(",") if item.strip()] if args.classes else CLASSES
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(
        image_root=image_root,
        output_root=output_root,
        classes=classes,
        max_sources_per_style=max(0, int(args.max_sources_per_style)),
        cross_only=bool(args.cross_only),
        unit_order=bool(args.unit_order),
        exclude_source_substrings=[item.strip() for item in str(args.exclude_source_substrings).split(",") if item.strip()],
    )
    if args.max_total > 0:
        jobs = jobs[: int(args.max_total)]

    url = _api_url(args.base_url, args.endpoint)
    jsonl_path = output_root / "seedream_manifest.jsonl"
    attempts_jsonl_path = output_root / "seedream_attempts.jsonl"
    csv_path = output_root / "seedream_manifest.csv"
    summary_path = output_root / "seedream_summary.json"
    rows: list[dict[str, Any]] = []
    limiter = RateLimiter(float(args.rpm), float(args.launch_interval_sec))
    keyring = ApiKeyRing(api_keys) if not args.dry_run else ApiKeyRing(["dry-run-key"])

    print(f"endpoint={url}")
    print(
        f"model={args.model} image_field={args.image_field} "
        f"response_format={args.response_format} transport={args.transport}"
    )
    print(
        f"jobs={len(jobs)} output_root={output_root} "
        f"concurrency={int(args.concurrency)} rpm={float(args.rpm):g} "
        f"launch_interval_sec={float(args.launch_interval_sec):g} api_keys={keyring.count}"
    )

    run_items: list[tuple[int, Job]] = []
    for idx, job in enumerate(jobs, start=1):
        if job.out_path.exists() and job.out_path.stat().st_size > 0 and not args.force:
            row = _base_row(job, "skipped_existing")
            row["source"] = "existing"
            row["size"] = str(args.size)
            if args.prompt_suffix:
                row["prompt"] = f"{row['prompt']} {args.prompt_suffix.strip()}"
            rows.append(row)
        else:
            run_items.append((idx, job))

    if rows:
        for row in list(rows):
            _append_jsonl(jsonl_path, row)
        _write_manifest_csv(csv_path, rows)

    max_workers = max(1, int(args.concurrency))
    if run_items:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    _run_one_job,
                    args=args,
                    job=job,
                    job_index=idx,
                    num_jobs=len(jobs),
                    url=url,
                    limiter=limiter,
                    keyring=keyring,
                ): (idx, job)
                for idx, job in run_items
            }
            for future in as_completed(future_map):
                row, attempts = future.result()
                rows.append(row)
                _append_jsonl(jsonl_path, row)
                for attempt_row in attempts:
                    _append_jsonl(attempts_jsonl_path, attempt_row)
                rows_sorted = sorted(rows, key=lambda r: str(r.get("row_id", "")))
                _write_manifest_csv(csv_path, rows_sorted)

    completed = sum(1 for r in rows if r.get("status") == "ok")
    skipped = sum(1 for r in rows if r.get("status") == "skipped_existing")
    failed = sum(1 for r in rows if r.get("status") == "failed")
    dry = sum(1 for r in rows if r.get("status") == "dry_run")
    request_times = [float(r.get("request_elapsed_sec", 0.0)) for r in rows if r.get("status") == "ok"]
    total_times = [float(r.get("elapsed_sec", 0.0)) for r in rows if r.get("status") == "ok"]
    download_times = [float(r.get("download_elapsed_sec", 0.0)) for r in rows if r.get("status") == "ok"]
    summary = {
        "model": args.model,
        "endpoint": url,
        "image_field": args.image_field,
        "response_format": args.response_format,
        "size": args.size,
        "image_root": str(image_root),
        "output_root": str(output_root),
        "classes": classes,
        "jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "dry": dry,
        "dry_run": bool(args.dry_run),
        "cross_only": bool(args.cross_only),
        "unit_order": bool(args.unit_order),
        "concurrency": max_workers,
        "rpm": float(args.rpm),
        "launch_interval_sec": float(args.launch_interval_sec),
        "api_key_count": keyring.count,
        "request_elapsed_sec": _timing_summary(request_times),
        "download_elapsed_sec": _timing_summary(download_times),
        "job_elapsed_sec": _timing_summary(total_times),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Seedream 4.5 external baseline on WikiArt512 test images.")
    parser.add_argument("--image-root", default="F:/wikiart_images_512_ema_test")
    parser.add_argument("--output-root", default="../results/seedream45_api/wikiart512_ema_test")
    parser.add_argument("--classes", default=",".join(CLASSES))
    parser.add_argument("--max-sources-per-style", type=int, default=0)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--cross-only", action="store_true")
    parser.add_argument("--unit-order", action="store_true", help="Interleave source images by per-style index so early outputs form complete 5x5 units.")
    parser.add_argument("--exclude-source-substrings", default="", help="Comma-separated substrings; matching source filenames are skipped during job enumeration.")
    parser.add_argument("--base-url", default="https://windhub.cc")
    parser.add_argument("--endpoint", default="/v1/images/generations")
    parser.add_argument("--model", default="doubao-seedream-4-5-251128")
    parser.add_argument("--image-field", default="image", choices=["image", "image_input", "image_urls", "image_url"])
    parser.add_argument("--response-format", default="b64_json", choices=["b64_json", "url"])
    parser.add_argument("--output-format", default="")
    parser.add_argument("--resize-output", default="")
    parser.add_argument("--omit-n", action="store_true")
    parser.add_argument("--optimize-prompt-mode", default="")
    parser.add_argument("--transport", default="urllib", choices=["urllib", "curl"])
    parser.add_argument("--async-mode", action="store_true", help="Submit ModelScope async image task and poll /v1/tasks/{task_id}.")
    parser.add_argument("--async-task-type", default="image_generation")
    parser.add_argument("--async-poll-interval-sec", type=float, default=20.0)
    parser.add_argument("--async-max-wait-sec", type=float, default=900.0)
    parser.add_argument("--input-resize", default="", help="Resize request input image to WIDTHxHEIGHT before base64 encoding.")
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--api-key-env", default="SEEDREAM_API_KEY")
    parser.add_argument("--api-keys-env", default="SEEDREAM_API_KEYS")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--retry-backoff", default="linear", choices=["linear", "constant"])
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rpm", type=float, default=5.0, help="Global request-per-minute limit; <=0 disables rate limiting.")
    parser.add_argument("--launch-interval-sec", type=float, default=0.0, help="Global interval between request launches. Overrides --rpm when >0.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
