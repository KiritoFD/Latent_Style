from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_EXCLUDE_PREFIXES = [
    "ablation",
    "checkpoint.save_dir",
    "training.resume_checkpoint",
    "training.resume_optimizer",
    "training.resume_training_state",
    "training.resume_prefer_local_checkpoint",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _path_is_excluded(path: str, prefixes: list[str]) -> bool:
    if not path:
        return False
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


def _values_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    return a == b


def _collect_overrides(
    baseline: Any,
    variant: Any,
    *,
    prefix: str,
    exclude_prefixes: list[str],
    out: dict[str, Any],
) -> None:
    if prefix and _path_is_excluded(prefix, exclude_prefixes):
        return
    if isinstance(baseline, dict) and isinstance(variant, dict):
        for key in sorted(set(baseline.keys()) | set(variant.keys())):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in variant:
                continue
            if key not in baseline:
                if not _path_is_excluded(child_prefix, exclude_prefixes):
                    out[child_prefix] = copy.deepcopy(variant[key])
                continue
            _collect_overrides(
                baseline[key],
                variant[key],
                prefix=child_prefix,
                exclude_prefixes=exclude_prefixes,
                out=out,
            )
        return
    if isinstance(baseline, list) and isinstance(variant, list):
        if baseline != variant and prefix and not _path_is_excluded(prefix, exclude_prefixes):
            out[prefix] = copy.deepcopy(variant)
        return
    if not _values_equal(baseline, variant):
        if prefix and not _path_is_excluded(prefix, exclude_prefixes):
            out[prefix] = copy.deepcopy(variant)


def _variant_name(path: Path, mode: str) -> str:
    if mode == "stem":
        return path.stem
    if mode == "filename":
        return path.name
    return path.parent.name or path.stem


def _expand_variant_configs(paths: list[Path], variant_dir: Path | None, glob_pattern: str) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            out.append(path)
            seen.add(resolved)
    if variant_dir is not None:
        for path in sorted(variant_dir.glob(glob_pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved not in seen:
                out.append(path)
                seen.add(resolved)
    return out


def _run_probe(
    *,
    script_name: str,
    baseline_config: Path,
    variant_spec: Path,
    output_dir: Path,
    device: str,
    checkpoint: Path | None,
    seed: int,
    input_seed: int,
    batch_size: int,
    style_id: int,
    latent_size: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(HERE / script_name),
        "--config",
        str(baseline_config),
        "--variant-spec",
        str(variant_spec),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--seed",
        str(seed),
        "--input-seed",
        str(input_seed),
        "--batch-size",
        str(batch_size),
        "--latent-size",
        str(latent_size),
    ]
    if script_name == "probe_training_variant_effect.py":
        cmd.extend(
            [
                "--target-style-id",
                str(style_id),
                "--source-style-id",
                "0",
            ]
        )
    else:
        cmd.extend(
            [
                "--style-id",
                str(style_id),
            ]
        )
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    stdout_tail = str(proc.stdout or "").splitlines()[-80:]
    stderr_tail = str(proc.stderr or "").splitlines()[-80:]
    result = {
        "script": script_name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "output_dir": str(output_dir),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with rc={proc.returncode}\n"
            + "\n".join(stdout_tail + stderr_tail)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable variant-spec directly from real config.json files, then optionally "
            "run the existing config-effect / training-effect probes on that exact family."
        )
    )
    parser.add_argument("--baseline-config", type=Path, required=True, help="Baseline config JSON.")
    parser.add_argument(
        "--variant-config",
        type=Path,
        action="append",
        default=[],
        help="Variant config JSON. Repeat for multiple files.",
    )
    parser.add_argument("--variant-dir", type=Path, default=None, help="Optional directory to scan for configs.")
    parser.add_argument(
        "--glob",
        type=str,
        default="*/config.json",
        help="Glob used under --variant-dir. Default: */config.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated specs and probe outputs.")
    parser.add_argument(
        "--name-mode",
        choices=("parent", "stem", "filename"),
        default="parent",
        help="How to derive variant names from config paths.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Dot-path prefix to ignore when diffing. Can be repeated.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint passed to both probes.")
    parser.add_argument("--device", type=str, default="cpu", help="Probe device.")
    parser.add_argument("--seed", type=int, default=0, help="Probe model init seed.")
    parser.add_argument("--input-seed", type=int, default=123, help="Probe input seed.")
    parser.add_argument("--batch-size", type=int, default=2, help="Probe batch size.")
    parser.add_argument("--style-id", type=int, default=1, help="Probe style id.")
    parser.add_argument("--latent-size", type=int, default=32, help="Probe latent size.")
    parser.add_argument("--skip-config-effect", action="store_true", help="Skip probe_config_effectiveness.py.")
    parser.add_argument("--skip-training-effect", action="store_true", help="Skip probe_training_variant_effect.py.")
    parser.add_argument("--allow-empty-overrides", action="store_true", help="Keep variants even if their diff is empty.")
    args = parser.parse_args()

    baseline_config = args.baseline_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_prefixes = list(dict.fromkeys(DEFAULT_EXCLUDE_PREFIXES + [str(x) for x in args.exclude_prefix]))
    baseline_payload = _load_json(baseline_config)
    variant_configs = _expand_variant_configs(list(args.variant_config), args.variant_dir, args.glob)
    if not variant_configs:
        raise ValueError("No variant configs found. Provide --variant-config or --variant-dir.")

    variants: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for config_path in variant_configs:
        payload = _load_json(config_path)
        name = _variant_name(config_path, args.name_mode)
        if name in seen_names:
            raise ValueError(f"Duplicate variant name derived from config path: {name}")
        seen_names.add(name)
        overrides: dict[str, Any] = {}
        _collect_overrides(
            baseline_payload,
            payload,
            prefix="",
            exclude_prefixes=exclude_prefixes,
            out=overrides,
        )
        if not overrides and not args.allow_empty_overrides:
            continue
        variants.append({"name": name, "overrides": overrides})
        manifest.append(
            {
                "name": name,
                "config_path": str(config_path.resolve()),
                "override_count": len(overrides),
                "overrides": overrides,
            }
        )

    if not variants:
        raise ValueError("No non-empty variant diffs were found after exclusions.")

    variant_spec_path = output_dir / "variant_spec.json"
    manifest_path = output_dir / "variant_manifest.json"
    summary_path = output_dir / "summary.json"
    variant_spec_path.write_text(json.dumps({"variants": variants}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "baseline_config": str(baseline_config),
        "variant_config_count": len(variant_configs),
        "variant_spec_count": len(variants),
        "exclude_prefixes": exclude_prefixes,
        "variant_spec_path": str(variant_spec_path),
        "variant_manifest_path": str(manifest_path),
        "probe_runs": [],
    }

    if not args.skip_config_effect:
        summary["probe_runs"].append(
            _run_probe(
                script_name="probe_config_effectiveness.py",
                baseline_config=baseline_config,
                variant_spec=variant_spec_path,
                output_dir=output_dir / "config_effect_probe",
                device=args.device,
                checkpoint=args.checkpoint,
                seed=args.seed,
                input_seed=args.input_seed,
                batch_size=args.batch_size,
                style_id=args.style_id,
                latent_size=args.latent_size,
            )
        )
    if not args.skip_training_effect:
        summary["probe_runs"].append(
            _run_probe(
                script_name="probe_training_variant_effect.py",
                baseline_config=baseline_config,
                variant_spec=variant_spec_path,
                output_dir=output_dir / "training_effect_probe",
                device=args.device,
                checkpoint=args.checkpoint,
                seed=args.seed,
                input_seed=args.input_seed,
                batch_size=args.batch_size,
                style_id=args.style_id,
                latent_size=args.latent_size,
            )
        )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(variant_spec_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
