"""Extract key metrics from a summary.json file."""
import json
import sys
from pathlib import Path


def extract(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    a = d.get("analysis", {})
    s = a.get("style_transfer_ability", {})
    p = a.get("all_pairs_overview", {})
    t = d.get("timings_sec", {})
    return {
        "transfer_clip_style": s.get("clip_style"),
        "transfer_content_lpips": s.get("content_lpips"),
        "allpairs_clip_style": p.get("clip_style"),
        "allpairs_content_lpips": p.get("content_lpips"),
        "generated_count": d.get("generated_count"),
        "wall_total": t.get("wall_total"),
        "lancet_generation": t.get("lancet_generation"),
        "lpips": t.get("lpips"),
        "clip": t.get("clip"),
    }


if __name__ == "__main__":
    summary_path = Path(sys.argv[1])
    print(json.dumps(extract(summary_path), indent=2))
