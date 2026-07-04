"""Probe final_works directory: list entries with mtime and size only."""
import os, sys, subprocess

def run(cmd, timeout=25):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERR:{e}"

path = "/mnt/i/Github/Latent_Style/final_works"
entries = run(f'ls -1 "{path}" 2>/dev/null')
if entries in ("TIMEOUT", "") or entries.startswith("ERR:"):
    print(f"ERROR: {entries}")
    sys.exit(0)

for entry in entries.split("\n"):
    if not entry:
        continue
    full = f"{path}/{entry}"
    # Check if it's a file or directory
    type_check = run(f'if [ -d "{full}" ]; then echo dir; elif [ -f "{full}" ]; then echo file; else echo other; fi')
    if type_check == "dir":
        mtime = run(f'stat -c "%y" "{full}"').split(".")[0]
        size = run(f'du -sh "{full}" 2>/dev/null', timeout=120)
        if size == "TIMEOUT" or not size:
            size = "?"
        print(f"{entry}\t{full}\t{mtime}\t{size}\tDIR")
    elif type_check == "file":
        mtime = run(f'stat -c "%y" "{full}"').split(".")[0]
        size = run(f'ls -lh "{full}" 2>/dev/null')
        # parse size from ls -lh output
        size_str = "?"
        if size and size != "TIMEOUT":
            parts = size.split()
            if len(parts) >= 5:
                size_str = parts[4]
        print(f"{entry}\t{full}\t{mtime}\t{size_str}\tFILE")
    else:
        print(f"{entry}\t{full}\t\t\tOTHER")
