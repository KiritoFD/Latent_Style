"""Extract core metrics from a summary.json file."""
import json
import sys
from pathlib import Path


def extract(summary_path: str) -> None:
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    name = Path(summary_path).parts[-4] if len(Path(summary_path).parts) >= 4 else Path(summary_path).stem
    print(f"=== {name} ===")
    print(f"top keys: {list(data.keys())}")
    # Core metrics may live under "analysis.all_pairs_overview" or nested
    analysis = data.get("analysis", {})
    if isinstance(analysis, dict):
        for sub_key, sub_val in analysis.items():
            if isinstance(sub_val, dict):
                # Look for the overview that holds scalar metrics
                for k, v in sub_val.items():
                    if isinstance(v, (int, float)):
                        print(f"  analysis.{sub_key}.{k}: {v:.4f}" if isinstance(v, float) else f"  analysis.{sub_key}.{k}: {v}")
                    elif isinstance(v, str) and len(v) < 80:
                        print(f"  analysis.{sub_key}.{k}: {v}")
            elif isinstance(sub_val, (int, float)):
                print(f"  analysis.{sub_key}: {sub_val:.4f}" if isinstance(sub_val, float) else f"  analysis.{sub_key}: {sub_val}")
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict) and metrics:
        print(f"  [metrics] keys: {list(metrics.keys())[:30]}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    # Common alternative top-level locations
    for key in ("clip_style", "clip_s", "lpips", "musiq", "dino_style", "dino_content",
                "clip_s_score", "lpips_score", "musiq_score", "dino_style_score", "dino_content_score"):
        if key in data:
            print(f"  [top] {key}: {data[key]}")
    # Settings (postprocess config)
    settings = data.get("settings", {})
    if isinstance(settings, dict) and settings:
        pp_keys = {k: v for k, v in settings.items()
                   if any(t in k.lower() for t in ("postprocess", "affine", "strength", "denoise", "spectral"))}
        if pp_keys:
            print(f"  [settings postprocess]: {pp_keys}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: _extract_summary.py <summary.json> [more...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        try:
            extract(p)
        except Exception as e:
            print(f"ERROR for {p}: {e}")
