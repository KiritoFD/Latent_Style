from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
GENERIC_VLM = SB_ROOT / "tools" / "eval_xf_qwen_vlm.py"


def _run(cmd: list[str]) -> int:
    print("[run_xf_vlm_failure_diagnosis] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _extract_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = raw.replace("\r\n", "\n").replace("\r", "\n")
    repaired = repaired.replace(",\n}", "\n}").replace(",\n]", "\n]")
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(repaired)
    except Exception as exc:
        raise json.JSONDecodeError("Unable to parse model JSON", repaired, 0) from exc
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Parsed object is not a dict", repaired, 0)
    return obj


def _build_prompt(*, target_style: str, target_ref_count: int, has_baseline: bool) -> str:
    baseline_text = (
        "Panel D is the baseline output. Compare panel C against panel D and state whether the failure is intrinsic to our method or shared by both outputs. "
        if has_baseline
        else "There is no baseline output in this case. "
    )
    ref_label = "Panel B" if target_ref_count == 1 else f"Panels B1-B{target_ref_count}"
    return (
        "You are diagnosing why a style-transfer result is not good enough for research iteration. "
        "Be concrete and visual, not polite. "
        "Panel A is the source image. "
        f"{ref_label} are target-style reference images from the desired style domain: {target_style}. "
        "Panel C is our generated output. "
        f"{baseline_text}"
        "Judge whether panel C fails mainly because of: "
        "1) weak style injection, "
        "2) structure drift, "
        "3) brightness/contrast or color-stat mismatch, "
        "4) texture oversmoothing / plastic look, "
        "5) local artifacts or broken regions. "
        "Return JSON only with keys: "
        "verdict, primary_failure_mode, secondary_failure_modes, "
        "scores, evidence, comparison_to_baseline, recommendations, concise_summary. "
        "scores must be a dict with integer 1-5 values for style_strength, structure_preservation, "
        "photometric_alignment, texture_richness, artifact_control. "
        "evidence must be a list of short bullet-like strings grounded in visible image regions. "
        "recommendations must be a list of short actionable suggestions for model research, not post-editing. "
        "primary_failure_mode must be one of: weak_style_injection, structure_drift, photometric_mismatch, oversmoothing, local_artifacts, no_major_failure. "
        "If a baseline is present, comparison_to_baseline must say whether panel C is better, worse, or mixed versus panel D and why."
    )


def _write_markdown(path: Path, *, parsed: dict[str, Any], raw_text: str, args: argparse.Namespace) -> None:
    lines = [
        "# XF VLM Failure Diagnosis",
        "",
        f"- source: `{Path(args.source).resolve()}`",
        f"- generated: `{Path(args.generated).resolve()}`",
        f"- target_style: `{args.target_style}`",
        f"- target_refs: `{len(args.target_ref)}`",
        f"- baseline: `{Path(args.baseline).resolve()}`" if args.baseline else "- baseline: `none`",
        f"- model: `{args.model}`",
        "",
        "## Parsed",
        "",
        "```json",
        json.dumps(parsed, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Raw",
        "",
        "```text",
        raw_text.strip(),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an xf-yun Qwen VLM failure diagnosis for one style-transfer case.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--target-style", required=True)
    parser.add_argument("--target-ref", action="append", required=True, help="Repeat for one or more target-style reference images.")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--system", default="You are a careful visual evaluator for style-transfer research.")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="xopqwen36v35b")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    image_paths = [Path(args.source).resolve(), *[Path(p).resolve() for p in args.target_ref], Path(args.generated).resolve()]
    if str(args.baseline).strip():
        image_paths.append(Path(args.baseline).resolve())
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = (WORKSPACE / output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_json = output_prefix.with_suffix(".raw.json")
    parsed_json = output_prefix.with_suffix(".parsed.json")
    summary_md = output_prefix.with_suffix(".md")

    prompt = _build_prompt(
        target_style=str(args.target_style),
        target_ref_count=len(list(args.target_ref)),
        has_baseline=bool(str(args.baseline).strip()),
    )
    cmd = [
        sys.executable,
        str(GENERIC_VLM),
        "--image",
        str(Path(args.source).resolve()),
    ]
    for ref in args.target_ref:
        cmd.extend(["--image", str(Path(ref).resolve())])
    cmd.extend(["--image", str(Path(args.generated).resolve())])
    if str(args.baseline).strip():
        cmd.extend(["--image", str(Path(args.baseline).resolve())])
    cmd.extend(
        [
            "--prompt",
            prompt,
            "--system",
            str(args.system),
            "--output-json",
            str(raw_json),
            "--model",
            str(args.model),
            "--temperature",
            str(float(args.temperature)),
            "--max-tokens",
            str(int(args.max_tokens)),
            "--timeout",
            str(int(args.timeout)),
            "--max-edge",
            str(int(args.max_edge)),
            "--jpeg-quality",
            str(int(args.jpeg_quality)),
        ]
    )
    if str(args.base_url).strip():
        cmd.extend(["--base-url", str(args.base_url)])
    if str(args.api_key).strip():
        cmd.extend(["--api-key", str(args.api_key)])
    rc = _run(cmd)
    if rc != 0:
        return rc

    raw = json.loads(raw_json.read_text(encoding="utf-8"))
    raw_text = str((((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
    parsed = _extract_json(raw_text)
    parsed_json.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary_md, parsed=parsed, raw_text=raw_text, args=args)

    print(raw_json)
    print(parsed_json)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
