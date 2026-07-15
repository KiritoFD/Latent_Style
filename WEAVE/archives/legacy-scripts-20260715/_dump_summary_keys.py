"""Dump summary.json top-level keys and aggregate metrics."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def dump_structure(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=== {path} ===")
    print("TOP KEYS:", list(data.keys()))
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k} keys: {list(v.keys())[:15]}")
            if "aggregate" in v:
                agg = v["aggregate"]
                print(f"    aggregate keys: {list(agg.keys())}")
    # Look for clip/lpips anywhere in top 2 levels
    for k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)):
                    kl = (k + "." + k2).lower()
                    if "clip" in kl or "lpips" in kl or "dino" in kl:
                        print(f"  METRIC {k}.{k2} = {v2}")
        elif isinstance(v, (int, float)):
            kl = k.lower()
            if "clip" in kl or "lpips" in kl or "dino" in kl:
                print(f"  METRIC {k} = {v}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    for p in paths:
        try:
            dump_structure(p)
        except Exception as e:
            print(f"ERROR {p}: {e}")
