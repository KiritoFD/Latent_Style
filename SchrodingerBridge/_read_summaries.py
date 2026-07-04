"""Read summary.json and print key metrics + config."""
from __future__ import annotations
import json
from pathlib import Path


def read_summary(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"NOT FOUND: {path}")
        return
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Print top-level keys
    print(f"Top-level keys: {list(data.keys())}")

    # Analysis section
    analysis = data.get("analysis", {})
    aps = analysis.get("all_pairs_overview", {})
    print(f"all_pairs_overview: clip_style={aps.get('clip_style')}, content_lpips={aps.get('content_lpips')}")

    # Config section (if exists)
    config = data.get("config", {})
    if config:
        model = config.get("model", {})
        training = config.get("training", {})
        print(f"model.endpoint_lock_ll={model.get('endpoint_lock_ll')}")
        print(f"model.endpoint_adain_scale={model.get('endpoint_adain_scale')}")
        print(f"training.full_eval_num_steps={training.get('full_eval_num_steps')}")
    else:
        print("No config section in summary.json")

    # Timings
    timings = data.get("timings_sec", {})
    print(f"timings: {dict(timings)}")


# Check the main summary
print("=== 630_local_t11_long30ep/full_eval/epoch_0001/summary.json ===")
read_summary("I:/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_long30ep/full_eval/epoch_0001/summary.json")

# Also check if there are other lock_ll experiment dirs
print("\n=== Searching for lock_ll experiment dirs ===")
exp_root = Path("I:/Github/Latent_Style/SchrodingerBridge/exp")
for d in sorted(exp_root.iterdir()):
    if d.is_dir() and "lock" in d.name.lower():
        print(f"Found: {d.name}")
        # Look for summary.json
        for s in d.rglob("summary.json"):
            print(f"  summary: {s}")
            read_summary(str(s))

# Also check cspb results (from the sweep we ran earlier)
print("\n=== CSPB results ===")
cspb_root = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/cspb")
if cspb_root.exists():
    for d in sorted(cspb_root.iterdir()):
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                read_summary(str(summary_path))
                print()
