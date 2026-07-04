import subprocess
import os

# Configuration
ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"
remote_project_path = "/home/xy/Latent_Style/SchrodingerBridge"
local_script_path = r"g:\GitHub\Latent_Style\SchrodingerBridge\extract_metrics_remote.py"
local_result_path = r"g:\GitHub\Latent_Style\SchrodingerBridge\metrics.txt"

def run_command(cmd, description):
    print(f"\n[INFO] {description}")
    print(f"[CMD]  {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"[OUT]  {result.stdout}")
    if result.stderr:
        print(f"[ERR]  {result.stderr}")
    if result.returncode != 0:
        print(f"[FAIL] Exit code: {result.returncode}")
        return False
    return True

print("=" * 70)
print("FC-SB Phase 2 v4 Metrics Extraction")
print("=" * 70)

# Step 1: Upload Python script to remote server
success = run_command(
    f'scp -P {ssh_port} "{local_script_path}" {ssh_host}:~/extract_metrics_remote.py',
    "Uploading extraction script to remote server..."
)

if not success:
    print("\n[ERROR] Failed to upload script. Exiting.")
    exit(1)

# Step 2: Execute the script on remote server
success = run_command(
    f'ssh -p {ssh_port} {ssh_host} "cd {remote_project_path} && python3 extract_metrics_remote.py > ~/metrics.txt 2>&1"',
    "Executing metrics extraction on remote server..."
)

if not success:
    print("\n[WARNING] Script execution may have failed. Trying to download results anyway...")

# Step 3: Download results
success = run_command(
    f'scp -P {ssh_port} {ssh_host}:~/metrics.txt "{local_result_path}"',
    "Downloading results from remote server..."
)

if not success:
    print("\n[ERROR] Failed to download results. Exiting.")
    exit(1)

# Step 4: Read and display results
print("\n" + "=" * 70)
print("EXTRACTION RESULTS")
print("=" * 70)

if os.path.exists(local_result_path):
    with open(local_result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(content)
else:
    print("[ERROR] Result file not found locally.")

print("\n" + "=" * 70)
print("DONE - Results saved to:", local_result_path)
print("=" * 70)
