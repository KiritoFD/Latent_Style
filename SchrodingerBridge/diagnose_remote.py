import subprocess
import os

# Configuration
ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"
remote_project_path = "/home/xy/Latent_Style/SchrodingerBridge"

def run_ssh_command(command):
    """Run a command on the remote server via SSH"""
    full_cmd = f'ssh -p {ssh_port} {ssh_host} "{command}"'
    print(f"\n[SSH] {command}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result.stdout.strip():
        print(f"[OUT] {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"[ERR] {result.stderr.strip()}")
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

print("=" * 70)
print("Remote Server Diagnostics")
print("=" * 70)

# Test 1: Check if remote project directory exists
print("\n--- Test 1: Check remote directory ---")
success, out, err = run_ssh_command(f"ls -la {remote_project_path}/")
if not success:
    print(f"[FAIL] Directory does not exist or cannot access")

# Test 2: Check if exp/p3_remote_10h exists
print("\n--- Test 2: Check experiment directories ---")
success, out, err = run_ssh_command(f"ls -la {remote_project_path}/exp/p3_remote_10h/ 2>&1 | head -20")
if not success:
    print(f"[FAIL] Experiment directory issue")

# Test 3: Look for summary.json files
print("\n--- Test 3: Search for summary.json files ---")
success, out, err = run_ssh_command(f"find {remote_project_path}/exp/p3_remote_10h -name 'summary.json' 2>/dev/null | head -20")

# Test 4: Check Python availability
print("\n--- Test 4: Check Python ---")
success, out, err = run_ssh_command("which python3 && python3 --version")

print("\n" + "=" * 70)
print("Diagnostics Complete")
print("=" * 70)
