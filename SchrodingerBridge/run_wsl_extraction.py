import subprocess
import os
import time

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"
local_script = r"g:\GitHub\Latent_Style\SchrodingerBridge\extract_metrics_remote.py"
local_result = r"g:\GitHub\Latent_Style\SchrodingerBridge\metrics.txt"

print("=" * 70)
print("FC-SB Phase 2 v4 Metrics Extraction (WSL Mode)")
print("=" * 70)

def run(cmd, desc, timeout=30):
    print(f"\n[STEP] {desc}")
    print(f"[CMD] {cmd[:150]}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout
        )
        if result.stdout.strip():
            print(f"[OUT] {result.stdout.strip()[:1500]}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"[ERR] {result.stderr.strip()[:500]}")
        ok = result.returncode == 0
        print(f"[{'OK' if ok else 'FAIL'}]")
        return ok, result.stdout
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return False, ""

# Step 1: Upload script to remote Windows machine
ok, _ = run(
    f'scp -P {ssh_port} "{local_script}" {ssh_host}:C:\\Users\\Administrator\\extract_metrics_remote.py',
    "Upload script to Windows host"
)
if not ok:
    exit(1)

# Step 2: Execute via WSL (try different WSL approaches)
wsl_commands = [
    # Approach 1: Direct wsl command
    f'ssh -p {ssh_port} {ssh_host} wsl -e python3 /mnt/c/Users/Administrator/extract_metrics_remote.py',
    
    # Approach 2: WSL with bash -c
    f'ssh -p {ssh_port} {ssh_host} wsl bash -c "python3 /mnt/c/Users/Administrator/extract_metrics_remote.py"',
    
    # Approach 3: Enter WSL first
    f'ssh -p {ssh_port} {ssh_host} wsl -- python3 /mnt/c/Users/Administrator/extract_metrics_remote.py',
]

output = None
for i, cmd in enumerate(wsl_commands, 1):
    print(f"\n--- Attempt {i}/{len(wsl_commands)} ---")
    ok, out = run(cmd, f"Execute via WSL (approach {i})", timeout=60)
    if ok and out and len(out) > 10:
        output = out
        break

# Step 3: Save results
if output:
    with open(local_result, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\n[SAVED] Results saved to {local_result}")
else:
    print("\n[WARN] WSL execution failed or no output")

# Step 4: Display results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if os.path.exists(local_result) and os.path.getsize(local_result) > 0:
    with open(local_result, 'r', encoding='utf-8') as f:
        content = f.read()
    print("\n" + content)
    print("\n[SUCCESS]")
else:
    print("\n[ERROR] No results obtained!")
    # Try alternative: check if we can read the file directly from Windows
    print("\n[ALT] Trying to read from Windows path directly...")
    ok2, out2 = run(
        f'ssh -p {ssh_port} {ssh_host} type C:\\Users\\Administrator\\metrics.txt 2>nul || echo FILE_NOT_FOUND',
        "Read from Windows path"
    )
    if ok2 and "FILE_NOT_FOUND" not in out2:
        with open(local_result, 'w', encoding='utf-8') as f:
            f.write(out2)
        print("\n" + out2)

print(f"\n[FILE] {local_result}")
print("=" * 70)
