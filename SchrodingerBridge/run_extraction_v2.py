import subprocess
import os
import tempfile

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"
remote_project_path = "/home/xy/Latent_Style/SchrodingerBridge"
local_script_path = r"g:\GitHub\Latent_Style\SchrodingerBridge\extract_metrics_remote.py"
local_result_path = r"g:\GitHub\Latent_Style\SchrodingerBridge\metrics.txt"

def run_cmd(cmd, description):
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"[CMD]  {cmd[:100]}..." if len(cmd) > 100 else f"[CMD]  {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result.stdout.strip():
        print(f"[OUT]\n{result.stdout.strip()}")
    if result.stderr.strip() and result.returncode != 0:
        print(f"[ERR]\n{result.stderr.strip()[:500]}")
    status = "✓ SUCCESS" if result.returncode == 0 else f"✗ FAILED (code {result.returncode})"
    print(f"[{status}]")
    return result.returncode == 0

print("=" * 70)
print("FC-SB Phase 2 v4 - Metrics Extraction Tool")
print("=" * 70)

# Step 1: Upload script
if not run_cmd(
    f'scp -P {ssh_port} "{local_script_path}" {ssh_host}:~/extract_metrics_remote.py',
    "Upload extraction script to remote server"
):
    exit(1)

# Step 2: Execute script on remote (using a simpler approach)
# First, let's check if the directory exists and create a wrapper script
wrapper_script = """#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge || exit 1
python3 extract_metrics_remote.py > ~/metrics.txt 2>&1
echo "EXIT_CODE=$?"
"""

# Write wrapper to temp file and upload it
with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8') as f:
    f.write(wrapper_script)
    wrapper_local = f.name

if not run_cmd(
    f'scp -P {ssh_port} {wrapper_local} {ssh_host}:~/run_metrics.sh',
    "Upload wrapper script"
):
    os.unlink(wrapper_local)
    exit(1)

os.unlink(wrapper_local)

# Step 3: Make executable and run
if not run_cmd(
    f'ssh -p {ssh_port} {ssh_host} chmod +x ~/run_metrics.sh && bash ~/run_metrics.sh',
    "Execute metrics extraction on remote server"
):
    print("\n[WARNING] Execution may have issues, attempting download...")

# Step 4: Download results
if not run_cmd(
    f'scp -P {ssh_port} {ssh_host}:~/metrics.txt "{local_result_path}"',
    "Download results to local machine"
):
    # Try alternative: cat the file via SSH
    print("\n[ALT] Trying to retrieve output via SSH...")
    success = run_cmd(
        f'ssh -p {ssh_port} {ssh_host} cat ~/metrics.txt 2>/dev/null || echo "FILE_NOT_FOUND"',
        "Retrieve output directly"
    )
    if success:
        # Save the output manually
        result = subprocess.run(
            f'ssh -p {ssh_port} {ssh_host} cat ~/metrics.txt 2>/dev/null',
            shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        with open(local_result_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        print(f"\n[SAVED] Output saved to {local_result_path}")

# Step 5: Display results
print("\n" + "=" * 70)
print("EXTRACTION RESULTS")
print("=" * 70)

if os.path.exists(local_result_path):
    with open(local_result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("\n" + content)
else:
    print("\n[ERROR] No results file found!")

print("\n" + "=" * 70)
print(f"Complete! Results saved to: {local_result_path}")
print("=" * 70)
