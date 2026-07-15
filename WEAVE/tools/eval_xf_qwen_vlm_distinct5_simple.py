from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_ascii(text: object) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


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


def _gen_name(row: dict[str, str]) -> str:
    return Path(str(row.get("gen_image") or row.get("image") or "")).name


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


def _source_index(source_root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for style_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
        for img in sorted([p for p in style_dir.iterdir() if p.is_file()]):
            stem = img.stem
            out[(style_dir.name, stem)] = img
            if "__" in stem:
                out[(style_dir.name, stem.split("__", 1)[1])] = img
    return out


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


def _pick_target_ref(style_dir: Path, case_key: tuple[str, str, str]) -> Path:
    files = sorted([p for p in style_dir.iterdir() if p.is_file()])
    if not files:
        raise FileNotFoundError(style_dir)
    digest = hashlib.sha1(("|".join(case_key)).encode("utf-8", errors="ignore")).hexdigest()
    idx = int(digest[:8], 16) % len(files)
    return files[idx]


def _build_case_table(specs: list[RunSpec]) -> tuple[list[dict[str, Any]], list[str]]:
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


def _fit_image(path: Path, *, side: int) -> Image.Image:
    with Image.open(path) as img:
        image = img.convert("RGB")
    image.thumbnail((side, side))
    canvas = Image.new("RGB", (side, side), (245, 245, 245))
    x = (side - image.width) // 2
    y = (side - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _compose_case_image(case: dict[str, Any], run_order: list[str], target_ref: Path, *, panel_side: int = 128) -> bytes:
    labels = [
        ("A Source", case["source_image"]),
        ("B Target", target_ref),
        *[
            (f"{chr(ord('C') + idx)} {case['candidates'][run]['method']}", case["candidates"][run]["image"])
            for idx, run in enumerate(run_order)
        ],
    ]
    title_h = 28
    canvas = Image.new("RGB", (len(labels) * panel_side, panel_side + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (label, path) in enumerate(labels):
        x = idx * panel_side
        canvas.paste(_fit_image(Path(path), side=panel_side), (x, title_h))
        draw.rectangle([x, 0, x + panel_side - 1, title_h - 1], fill=(245, 245, 245), outline=(220, 220, 220))
        draw.text((x + 6, 8), label, fill=(30, 30, 30), font=font)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=55, optimize=True)
    return buf.getvalue()


def _call_api(
    *,
    api_key: str,
    model: str,
    target_style: str,
    candidate_panels: list[str],
    image_bytes: bytes,
    timeout: int,
) -> dict[str, Any]:
    import base64

    panel_letters = ", ".join(candidate_panels)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful visual evaluator for style-transfer research."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Target style is {target_style}. "
                            "Panel A is the source image. Panel B is a target-style reference. "
                            f"Candidate outputs are panels {panel_letters}. "
                            f"Compare only {panel_letters} on style specificity, structure preservation, and artifact control. "
                            "Return JSON only with keys best_overall, best_style_specificity, best_structure, "
                            "best_artifact_control, confidence_1_to_5, and scores. "
                            f"Use only panel letters {panel_letters} as winner values and score keys. "
                            "Each score entry must contain integer 1-5 values for style_specificity, "
                            "structure_preservation, artifact_control, plus a short note."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")},
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 384,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = str(os.environ.get("XF_MAAS_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2")).rstrip("/") + "/chat/completions"
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    repaired = text.replace("\r\n", "\n").replace("\r", "\n")
    repaired = repaired.replace(",\n}", "\n}").replace(",\n]", "\n]")
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Some VLM responses are almost-Python dicts or contain minor JSON syntax drift.
    # literal_eval is only attempted after trimming to the outermost braces above.
    try:
        obj = ast.literal_eval(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise json.JSONDecodeError("Unable to parse model JSON after fallback repair", repaired, 0)


def _remap_panel_predictions(parsed: dict[str, Any], run_order: list[str]) -> dict[str, Any]:
    panel_to_run = {chr(ord("C") + idx): run for idx, run in enumerate(run_order)}
    out = dict(parsed)
    for key in ["best_overall", "best_style_specificity", "best_structure", "best_artifact_control"]:
        value = str(parsed.get(key, "")).strip()
        out[key] = panel_to_run.get(value, value)
    raw_scores = parsed.get("scores") or {}
    scores: dict[str, Any] = {}
    for panel_key, value in raw_scores.items():
        scores[panel_to_run.get(str(panel_key).strip(), str(panel_key).strip())] = value
    out["scores"] = scores
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simplified xf-yun Qwen VLM review on Distinct5 method outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--error-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=os.environ.get("XF_MAAS_MODEL_ID", "xopqwen36v35b"))
    parser.add_argument("--api-key", default=os.environ.get("XF_MAAS_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    args = parser.parse_args()

    if not str(args.api_key).strip():
        raise ValueError("Missing api key.")

    specs = _load_manifest(args.manifest, runs=set(args.runs))
    cases, run_order = _build_case_table(specs)
    cases = _round_robin_cases(cases, key_field="tgt_style")
    if args.limit > 0:
        cases = cases[: int(args.limit)]

    done: set[str] = set()
    if args.resume and args.output_jsonl.exists():
        for line in args.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("case_id", "")))
            except Exception:
                pass

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.output_jsonl.exists():
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
            rows.append(row)

    for idx, case in enumerate(cases, start=1):
        case_id = f"{case['src_style']}__{case['src_stem']}__to__{case['tgt_style']}"
        if case_id in done:
            continue
        target_style_dir = specs[0].source_root / case["tgt_style"]
        target_ref = _pick_target_ref(target_style_dir, (case["src_style"], case["tgt_style"], case["src_stem"]))
        try:
            candidate_panels = [chr(ord("C") + panel_idx) for panel_idx in range(len(run_order))]
            image_bytes = _compose_case_image(case, run_order, target_ref, panel_side=128)
            raw = _call_api(
                api_key=str(args.api_key),
                model=str(args.model),
                target_style=str(case["tgt_style"]),
                candidate_panels=candidate_panels,
                image_bytes=image_bytes,
                timeout=int(args.timeout),
            )
            content = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            parsed = _remap_panel_predictions(_extract_json(content), run_order)
        except Exception as exc:
            error_record = {
                "case_id": case_id,
                "src_style": case["src_style"],
                "tgt_style": case["tgt_style"],
                "src_stem": case["src_stem"],
                "last_error": f"{type(exc).__name__}: {exc}",
                "raw_content": content if "content" in locals() else None,
                "raw_response": raw if "raw" in locals() else None,
                "finish_reason": ((((raw or {}).get("choices") or [{}])[0].get("finish_reason")) if "raw" in locals() and raw else None),
            }
            with args.error_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(cases)}] {_safe_ascii(case_id)} failed: {_safe_ascii(error_record['last_error'])}", flush=True)
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
        rows.append(row)
        _write_csv(args.output_csv, rows)
        print(f"[{idx}/{len(cases)}] {_safe_ascii(case_id)} -> {_safe_ascii(parsed.get('best_overall'))}", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(float(args.sleep_seconds))

    print(args.output_jsonl)
    print(args.output_csv)
    print(args.error_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
