"""Extract all D9 metrics for all 3 configs."""
import json
import subprocess
import sys

REMOTE = "administrator@100.115.18.62"
PORT = "2222"

remote_script = r"""
import json

configs = ["d9a_swd16_15ep", "d9b_swd20_ep05_15ep", "d9c_swd24_15ep"]
results = {}

for exp in configs:
    summary_path = rf"I:\Github\Latent_Style\SchrodingerBridge\exp\{exp}\full_eval\epoch_0015\summary.json"
    dino_path = rf"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\{exp}.json"

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        with open(dino_path, "r") as f:
            dino = json.load(f)

        overview = summary.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = overview.get("clip_style", None)
        lpips_raw = overview.get("content_lpips", None)
        one_minus_lpips = 1.0 - lpips_raw if lpips_raw is not None else None

        dino_sty = dino.get("dino_style", None)
        dino_con = dino.get("dino_content", None)
        dino_str = dino.get("dino_structure", None)

        results[exp] = {
            "clip_s": clip_s,
            "1_LPIPS": one_minus_lpips,
            "dino_sty": dino_sty,
            "dino_con": dino_con,
            "dino_str": dino_str,
        }
    except Exception as e:
        results[exp] = {"error": str(e)}

print("RESULTS=" + json.dumps(results, indent=2))
"""

with open(r"g:\GitHub\Latent_Style\SchrodingerBridge\scripts\_remote_extract_all_d9.py", "w") as f:
    f.write(remote_script)

scp_cmd = ["scp", "-P", PORT, "-o", "LogLevel=ERROR",
           r"g:\GitHub\Latent_Style\SchrodingerBridge\scripts\_remote_extract_all_d9.py",
           f"{REMOTE}:C:/Users/Administrator/_remote_extract_all_d9.py"]
subprocess.run(scp_cmd, check=True)

ssh_cmd = ["ssh", "-p", PORT, "-o", "LogLevel=ERROR", REMOTE,
           "python C:\\Users\\Administrator\\_remote_extract_all_d9.py"]
result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)
