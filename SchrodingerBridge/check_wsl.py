import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Checking for WSL on Remote Windows Server")
print("=" * 70)

def run_remote(cmd, desc):
    print(f"\n[TEST] {desc}")
    full_cmd = f'ssh -p {ssh_port} {ssh_host} {cmd}'
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    out = result.stdout.strip()
    err = result.stderr.strip()
    if out:
        print(f"OUT: {out[:500]}")
    if err and result.returncode != 0:
        print(f"ERR: {err[:300]}")
    return result.returncode == 0, out, err

# Test WSL availability
print("\n--- WSL Checks ---")
ok1, out1, _ = run_remote("wsl --list --verbose", "List WSL distributions")
ok2, out2, _ = run_remote("wsl -e uname -a", "Test WSL execution")
ok3, out3, _ = run_remote("wsl -e python3 --version", "Check Python in WSL")

# Test direct Windows paths
print("\n--- Windows Path Checks ---")
ok4, out4, _ = run_remote("dir C:\\Users\\Administrator\\ 2>&1 | head -20", "Check Windows home directory")
ok5, out5, _ = run_remote("if exist C:\\Users\\Administrator\\Latent_Style (echo EXISTS) else (echo NOT_FOUND)", "Check Latent_Style on Windows")

# Try to find the project
print("\n--- Project Location Search ---")
ok6, out6, _ = run_remote("where /R C:\\Users\\Administrator SchrodingerBridge 2>nul | findstr /i summary.json", "Search for project files")
ok7, out7, _ = run_remote("dir /S /B C:\\Users\\Administrator\\*summary.json 2>nul | head -10", "Search for summary.json files")

print("\n" + "=" * 70)
if ok2 and "Linux" in out2:
    print("[RESULT] WSL is available!")
    print(f"[INFO] {out2}")
elif ok1:
    print("[RESULT] WSL installed but may need configuration")
    print(f"[INFO] {out1}")
else:
    print("[RESULT] No WSL detected or not accessible via SSH")
print("=" * 70)
