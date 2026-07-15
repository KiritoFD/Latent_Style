from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import os
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


@dataclass
class RunSpec:
    method: str
    run: str
    images_dir: Path
    source_root: Path
    metrics_csv: Path


def _data_url(path: Path, *, max_edge: int = 1024, jpeg_quality: int = 85) -> str:
    import base64

    with Image.open(path) as img:
        image = img.convert("RGB")
        if max(image.size) > max(1, int(max_edge)):
            image.thumbnail((int(max_edge), int(max_edge)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    # Some locally generated manifests are written by Windows-side tools that
    # may leave a UTF-8 BOM on the first header cell. Use utf-8-sig so
    # DictReader sees the expected field names like "method".
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_ascii(text: object) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


def _load_manifest(path: Path, *, runs: set[str] | None = None) -> list[RunSpec]:
    rows = _read_rows(path)
    specs: list[RunSpec] = []
    for row in rows:
        run = str(row["run"]).strip()
        if runs and run not in runs:
            continue
        specs.append(
            RunSpec(
                method=str(row["method"]).strip(),
                run=run,
                images_dir=Path(str(row["images_dir"])).resolve(),
                source_root=Path(str(row["source_root"])).resolve(),
                metrics_csv=Path(str(row["metrics_csv"])).resolve(),
            )
        )
    return specs


def _source_index(source_root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for style_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
        for img in sorted([p for p in style_dir.iterdir() if p.is_file()]):
            stem = img.stem
            out[(style_dir.name, stem)] = img
            if "__" in stem:
                out[(style_dir.name, stem.split("__", 1)[1])] = img
    return out


def _gen_name(row: dict[str, str]) -> str:
    return Path(str(row.get("gen_image") or row.get("image") or "")).name


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


def _resolve_generated_image(images_dir: Path, row: dict[str, str]) -> Path | None:
    name = _gen_name(row)
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(row.get("gen_image") or row.get("image") or "")
    if raw.exists():
        return raw
    want = _canonical_name(name)
    for candidate in images_dir.iterdir():
        if candidate.is_file() and _canonical_name(candidate.name) == want:
            return candidate
    return None


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    src_style = _repair_mojibake(str(row.get("src_style", "")).strip())
    tgt_style = _repair_mojibake(str(row.get("tgt_style", "")).strip())
    src_token = _repair_mojibake(str(row.get("src_image") or row.get("src_stem") or "").strip())
    src_stem = Path(src_token).stem
    if "__" in src_stem:
        src_stem = src_stem.split("__", 1)[1]
    return src_style, tgt_style, src_stem


def _pick_target_ref(style_dir: Path, case_key: tuple[str, str, str]) -> Path:
    files = sorted([p for p in style_dir.iterdir() if p.is_file()])
    if not files:
        raise FileNotFoundError(style_dir)
    digest = hashlib.sha1(("|".join(case_key)).encode("utf-8", errors="ignore")).hexdigest()
    idx = int(digest[:8], 16) % len(files)
    return files[idx]


def _build_case_table(specs: list[RunSpec]) -> tuple[list[dict[str, Any]], list[str]]:
    if not specs:
        return [], []
    indexes = {spec.run: _source_index(spec.source_root) for spec in specs}
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in specs:
        rows = _read_rows(spec.metrics_csv)
        src_map = indexes[spec.run]
        for row in rows:
            key = _key(row)
            src = src_map.get((key[0], key[2]))
            gen = _resolve_generated_image(spec.images_dir, row)
            if src is None or not src.exists() or gen is None or not gen.exists():
                continue
            bucket = buckets.setdefault(
                key,
                {
                    "src_style": key[0],
                    "tgt_style": key[1],
                    "src_stem": key[2],
                    "source_image": src,
                    "candidates": {},
                },
            )
            bucket["candidates"][spec.run] = {"method": spec.method, "image": gen}
    run_order = [spec.run for spec in specs]
    cases = [bucket for bucket in buckets.values() if all(run in bucket["candidates"] for run in run_order)]
    return cases, run_order


def _round_robin_cases(cases: list[dict[str, Any]], *, key_field: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        key = str(case.get(key_field, ""))
        buckets.setdefault(key, []).append(case)
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda c: (str(c.get("src_style", "")), str(c.get("src_stem", "")), str(c.get("tgt_style", ""))))
    ordered_keys = sorted(buckets.keys())
    out: list[dict[str, Any]] = []
    while True:
        advanced = False
        for key in ordered_keys:
            if buckets[key]:
                out.append(buckets[key].pop(0))
                advanced = True
        if not advanced:
            break
    return out


def _hashed_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(case: dict[str, Any]) -> str:
        text = "|".join(
            [
                str(case.get("src_style", "")),
                str(case.get("src_stem", "")),
                str(case.get("tgt_style", "")),
            ]
        )
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    return sorted(cases, key=key)


def _fit_image(path: Path, *, side: int) -> Image.Image:
    with Image.open(path) as img:
        image = img.convert("RGB")
    image.thumbnail((side, side))
    canvas = Image.new("RGB", (side, side), (245, 245, 245))
    x = (side - image.width) // 2
    y = (side - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _compose_case_image(case: dict[str, Any], run_order: list[str], target_ref: Path, *, panel_side: int = 256) -> Image.Image:
    labels = [
        ("A Source", case["source_image"]),
        (f"B TargetRef {case['tgt_style']}", target_ref),
        *[
            (
                f"{chr(ord('C') + idx)} {case['candidates'][run]['method']}",
                case["candidates"][run]["image"],
            )
            for idx, run in enumerate(run_order)
        ],
    ]
    cols = len(labels)
    title_h = 28
    canvas = Image.new("RGB", (cols * panel_side, panel_side + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (label, path) in enumerate(labels):
        tile = _fit_image(Path(path), side=panel_side)
        x = idx * panel_side
        canvas.paste(tile, (x, title_h))
        draw.rectangle([x, 0, x + panel_side - 1, title_h - 1], fill=(245, 245, 245), outline=(220, 220, 220))
        draw.text((x + 6, 8), label, fill=(30, 30, 30), font=font)
    return canvas


def _composite_data_url(case: dict[str, Any], run_order: list[str], target_ref: Path, *, panel_side: int, jpeg_quality: int) -> str:
    import base64

    image = _compose_case_image(case, run_order, target_ref, panel_side=panel_side)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _messages_for_case(
    case: dict[str, Any],
    run_order: list[str],
    target_ref: Path,
    *,
    panel_side: int,
    jpeg_quality: int,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    tgt_style = case["tgt_style"]
    panel_labels = [chr(ord("C") + idx) for idx in range(len(run_order))]
    panel_desc = ", ".join(f"{label}={run}" for label, run in zip(panel_labels, run_order))
    if prompt_mode == "compact":
        prompt = (
            "Evaluate style-transfer candidates for research. "
            f"Target style: {tgt_style}. "
            "Panel A is source, panel B is target-style reference. "
            f"Candidate panels are {', '.join(panel_labels)}. "
            "Judge style specificity, structure preservation, and artifact control. "
            "Return JSON only with keys best_overall, best_style_specificity, best_structure, "
            "best_artifact_control, confidence_1_to_5, and scores. "
            f"Use panel letters only for winners and score keys: {', '.join(panel_labels)}. "
            "Each score entry must contain integer 1-5 values for style_specificity, "
            "structure_preservation, artifact_control, plus a short note."
        )
    else:
        prompt = (
            "You are evaluating style-transfer outputs for research. "
            "A single comparison panel is attached. "
            f"Target style is {tgt_style}. "
            "Panel A is the source content image. "
            "Panel B is a target-style reference image. "
            f"Candidate panels are {', '.join(panel_labels)} for the same source-target task. "
            f"Methods are ordered as: {panel_desc}. "
            "Judge target-style specificity, structure preservation, and artifact control. "
            "Return strict JSON with keys: best_overall, best_style_specificity, best_structure, "
            "best_artifact_control, confidence_1_to_5, and scores. "
            f"Use panel letters only for winners and score keys: {', '.join(panel_labels)}. "
            "Each value must contain integers 1-5 for style_specificity, "
            "structure_preservation, artifact_control, plus a short note."
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt,
        },
        {"type": "image_url", "image_url": {"url": _composite_data_url(case, run_order, target_ref, panel_side=panel_side, jpeg_quality=jpeg_quality)}},
    ]
    if prompt_mode != "compact":
        content.append(
            {
                "type": "text",
                "text": "Respond with JSON only. In scores, use only panel letters as keys.",
            }
        )
    return content


def _remap_panel_letter_predictions(parsed: dict[str, Any], run_order: list[str]) -> dict[str, Any]:
    panel_to_run = {chr(ord("C") + idx): run for idx, run in enumerate(run_order)}

    def _map_value(value: Any) -> Any:
        text = str(value).strip()
        return panel_to_run.get(text, value)

    out = dict(parsed)
    for key in ["best_overall", "best_style_specificity", "best_structure", "best_artifact_control"]:
        out[key] = _map_value(parsed.get(key))

    scores = parsed.get("scores") or {}
    remapped_scores: dict[str, Any] = {}
    for key, value in scores.items():
        remapped_scores[str(panel_to_run.get(str(key).strip(), key))] = value
    out["scores"] = remapped_scores
    return out


def _call_api(*, base_url: str, api_key: str, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, timeout: int) -> dict[str, Any]:
    url = str(base_url).rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful visual evaluator for style-transfer research."},
            {"role": "user", "content": messages},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run xf-yun Qwen VLM pairwise review on Distinct5 method outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["LBM-Knee_e13", "LBM-PS-v2_e13", "Seedream_repaired750"])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("XF_MAAS_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"))
    parser.add_argument("--model", default=os.environ.get("XF_MAAS_MODEL_ID", "xopqwen36v35b"))
    parser.add_argument("--api-key", default=os.environ.get("XF_MAAS_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--panel-side", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-seconds", type=float, default=3.0)
    parser.add_argument("--error-jsonl", type=Path, default=None)
    parser.add_argument("--case-order", choices=["native", "round_robin_target", "round_robin_source", "hashed"], default="native")
    parser.add_argument("--prompt-mode", choices=["detailed", "compact"], default="compact")
    args = parser.parse_args()

    if not str(args.api_key).strip():
        raise ValueError("Missing api key. Set XF_MAAS_API_KEY or pass --api-key.")

    specs = _load_manifest(args.manifest, runs=set(args.runs))
    cases, run_order = _build_case_table(specs)
    if str(args.case_order) == "round_robin_target":
        cases = _round_robin_cases(cases, key_field="tgt_style")
    elif str(args.case_order) == "round_robin_source":
        cases = _round_robin_cases(cases, key_field="src_style")
    elif str(args.case_order) == "hashed":
        cases = _hashed_cases(cases)
    if args.limit > 0:
        cases = cases[: int(args.limit)]

    done: set[str] = set()
    if args.resume and args.output_jsonl.exists():
        for line in args.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(str(obj.get("case_id", "")))
            except Exception:
                continue
    failed: set[str] = set()
    if args.resume and args.error_jsonl is not None and args.error_jsonl.exists():
        for line in args.error_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                failed.add(str(obj.get("case_id", "")))
            except Exception:
                continue

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        case_id = f"{case['src_style']}__{case['src_stem']}__to__{case['tgt_style']}"
        if case_id in done:
            continue
        target_style_dir = specs[0].source_root / case["tgt_style"]
        target_ref = _pick_target_ref(target_style_dir, (case["src_style"], case["tgt_style"], case["src_stem"]))
        raw = None
        parsed = None
        last_error: str | None = None
        attempt_side = int(args.panel_side)
        attempt_quality = int(args.jpeg_quality)
        for attempt in range(1, max(1, int(args.max_retries)) + 1):
            try:
                messages = _messages_for_case(
                    case,
                    run_order,
                    target_ref,
                    panel_side=attempt_side,
                    jpeg_quality=attempt_quality,
                    prompt_mode=str(args.prompt_mode),
                )
                raw = _call_api(
                    base_url=str(args.base_url),
                    api_key=str(args.api_key),
                    model=str(args.model),
                    messages=messages,
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    timeout=int(args.timeout),
                )
                content = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                parsed = _remap_panel_letter_predictions(_extract_json(content), run_order)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{idx}/{len(cases)}] {_safe_ascii(case_id)} attempt {attempt} failed: {_safe_ascii(last_error)} "
                    f"(panel_side={attempt_side}, jpeg_quality={attempt_quality})",
                    flush=True,
                )
                attempt_side = max(128, int(round(attempt_side * 0.85)))
                attempt_quality = max(55, int(round(attempt_quality * 0.9)))
                if attempt < max(1, int(args.max_retries)):
                    time.sleep(max(0.0, float(args.retry_seconds)))
        if parsed is None or raw is None:
            error_record = {
                "case_id": case_id,
                "src_style": case["src_style"],
                "tgt_style": case["tgt_style"],
                "src_stem": case["src_stem"],
                "target_ref": str(target_ref),
                "last_error": last_error,
                "final_panel_side": attempt_side,
                "final_jpeg_quality": attempt_quality,
            }
            if args.error_jsonl is not None:
                args.error_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.error_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            continue
        record = {
            "case_id": case_id,
            "src_style": case["src_style"],
            "tgt_style": case["tgt_style"],
            "src_stem": case["src_stem"],
            "target_ref": str(target_ref),
            "raw_response": raw,
            "parsed": parsed,
        }
        with args.output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        row = {
            "case_id": case_id,
            "src_style": case["src_style"],
            "tgt_style": case["tgt_style"],
            "best_overall": parsed.get("best_overall"),
            "best_style_specificity": parsed.get("best_style_specificity"),
            "best_structure": parsed.get("best_structure"),
            "best_artifact_control": parsed.get("best_artifact_control"),
            "confidence_1_to_5": parsed.get("confidence_1_to_5"),
        }
        scores = parsed.get("scores") or {}
        for run in run_order:
            block = scores.get(run) or {}
            row[f"{run}_style_specificity"] = block.get("style_specificity")
            row[f"{run}_structure_preservation"] = block.get("structure_preservation")
            row[f"{run}_artifact_control"] = block.get("artifact_control")
            row[f"{run}_note"] = block.get("note")
        csv_rows.append(row)
        print(f"[{idx}/{len(cases)}] {_safe_ascii(case_id)} -> {_safe_ascii(parsed.get('best_overall'))}", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(float(args.sleep_seconds))

    if args.output_jsonl.exists():
        merged_rows: list[dict[str, Any]] = []
        for line in args.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            parsed = obj.get("parsed") or {}
            scores = parsed.get("scores") or {}
            row = {
                "case_id": obj.get("case_id"),
                "src_style": obj.get("src_style"),
                "tgt_style": obj.get("tgt_style"),
                "best_overall": parsed.get("best_overall"),
                "best_style_specificity": parsed.get("best_style_specificity"),
                "best_structure": parsed.get("best_structure"),
                "best_artifact_control": parsed.get("best_artifact_control"),
                "confidence_1_to_5": parsed.get("confidence_1_to_5"),
            }
            for run in run_order:
                block = scores.get(run) or {}
                row[f"{run}_style_specificity"] = block.get("style_specificity")
                row[f"{run}_structure_preservation"] = block.get("structure_preservation")
                row[f"{run}_artifact_control"] = block.get("artifact_control")
                row[f"{run}_note"] = block.get("note")
            merged_rows.append(row)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
            writer.writeheader()
            writer.writerows(merged_rows)
    print(args.output_jsonl)
    print(args.output_csv)
    if args.error_jsonl is not None:
        print(args.error_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
