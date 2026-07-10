"""Extract D7b complete metrics from remote summary.json and dino results, compare with baseline."""
import json
import urllib.request
import subprocess
import sys

REMOTE = "administrator@100.115.18.62"
PORT = "2222"

# Read remote files via ssh + python
remote_script = r"""
import json
summary_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\d7b_deep_backbone_lr5e5\full_eval\epoch_0015\summary.json"
dino_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\d7b_deep_backbone_lr5e5.json"

with open(summary_path, "r") as f:
    summary = json.load(f)
with open(dino_path, "r") as f:
    dino = json.load(f)

# Extract key metrics
metrics = {}
metrics["clip_s"] = summary.get("clip_style_mean", summary.get("clip_s", None))
metrics["1_LPIPS"] = summary.get("1_minus_lpips_mean", summary.get("lpips_mean", None))
# Check if 1-LPIPS needs conversion
lpips_raw = summary.get("lpips_mean", None)
if lpips_raw is not None and lpips_raw < 0.5:
    # It's raw LPIPS, convert to 1-LPIPS
    metrics["1_LPIPS"] = 1.0 - lpips_raw

metrics["dino_sty"] = dino.get("dino_sty", dino.get("style_mean", None))
metrics["dino_con"] = dino.get("dino_con", dino.get("content_mean", None))
metrics["dino_str"] = dino.get("dino_str", dino.get("style_rendition", None))

# Also dump all keys for inspection
print("SUMMARY_KEYS=" + str(list(summary.keys())))
print("DINO_KEYS=" + str(list(dino.keys())))
print("METRICS=" + json.dumps(metrics, indent=2))
print("RAW_SUMMARY=" + json.dumps(summary, indent=2)[:2000])
print("RAW_DINO=" + json.dumps(dino, indent=2)[:2000])
"""

# Write remote script
with open(r"g:\GitHub\Latent_Style\SchrodingerBridge\scripts\_remote_extract_d7b.py", "w") as f:
    f.write(remote_script)

# scp and run
import os
scp_cmd = ["scp", "-P", PORT, "-o", "LogLevel=ERROR",
           r"g:\GitHub\Latent_Style\SchrodingerBridge\scripts\_remote_extract_d7b.py",
           f"{REMOTE}:C:/Users/Administrator/_remote_extract_d7b.py"]
subprocess.run(scp_cmd, check=True)

ssh_cmd = ["ssh", "-p", PORT, "-o", "LogLevel=ERROR", REMOTE,
           "python C:\\Users\\Administrator\\_remote_extract_d7b.py"]
result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)
