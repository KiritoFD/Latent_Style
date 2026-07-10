"""Check eval progress. Run on remote."""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for name in ["t11_repro_fulleval", "t11e2_fulleval"]:
    for ext in [".out", ".err"]:
        path = rf"C:\Users\Administrator\logs\{name}{ext}"
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            continue
        print(f"\n=== {name}{ext} ({len(lines)} lines) ===")
        # Show last 20 lines
        for line in lines[-20:]:
            print(line[:200])
        # Show key lines
        key_lines = [l for l in lines if any(k in l.lower() for k in
                     ["generating batch", "phase", "clip_style", "content_lpips",
                      "summary", "error", "traceback", "done", "complete", "transfer"])]
        if key_lines and len(key_lines) > 20:
            print(f"\n--- key lines (first 5 + last 10) ---")
            for l in key_lines[:5]:
                print(l[:200])
            print("...")
            for l in key_lines[-10:]:
                print(l[:200])
