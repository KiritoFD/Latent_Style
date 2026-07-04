import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("WSL Environment Exploration")
print("=" * 70)

def wsl(command, desc, timeout=30):
    print(f"\n[CHECK] {desc}")
    cmd = f'ssh -p {ssh_port} {ssh_host} wsl {command}'
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
        out = result.stdout.strip()
        if out:
            print(f"[OUT]\n{out[:1500]}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"[ERR] {result.stderr.strip()[:300]}")
        return out
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return ""

# Basic WSL environment
wsl("whoami", "WSL user")
wsl("pwd", "WSL current directory")
wsl("ls -la /home/", "Home directories")
wsl("ls -la ~/", "User home contents")

# Search for Latent_Style anywhere
print("\n--- Searching for project ---")
wsl("find / -maxdepth 4 -type d -name 'SchrodingerBridge' 2>/dev/null | head -10", "Find SchrodingerBridge dir (depth 4)")
wsl("find / -maxdepth 5 -type d -name 'Latent_Style' 2>/dev/null | head -10", "Find Latent_Style dir (depth 5)")

# Check common locations
locations = [
    "/mnt/c/Users/Administrator/Latent_Style/SchrodingerBridge",
    "/mnt/c/Users/Administrator",
    "/mnt/d/",
    "/mnt/e/",
]
for loc in locations:
    wsl(f"test -d '{loc}' && ls '{loc}' 2>/dev/null | head -20 || echo 'NOT_FOUND'", f"Check {loc}")

# Check if there are any experiment results anywhere
print("\n--- Searching for any JSON result files ---")
wsl("find /mnt/c/Users/Administrator -name 'summary.json' 2>/dev/null | head -20", "summary.json on C drive")
wsl("find /mnt/d -name 'summary.json' 2>/dev/null | head -20", "summary.json on D drive")
wsl("find /mnt/e -name 'summary.json' 2>/dev/null | head -20", "summary.json on E drive")

# Look for fc_sb directories
print("\n--- Searching for fc_sb experiment dirs ---")
wsl("find /mnt/c/Users/Administrator -type d -name 'fc_sb_*' 2>/dev/null | head -20", "fc_sb dirs on C drive")

print("\n" + "=" * 70)
