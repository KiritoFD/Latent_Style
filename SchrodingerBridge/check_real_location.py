import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Checking Actual Project Location: /mnt/i/Github/")
print("=" * 70)

def wsl(command, desc):
    print(f"\n[CHECK] {desc}")
    cmd = f'ssh -p {ssh_port} {ssh_host} wsl {command}'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=30
    )
    out = result.stdout.strip()
    if out:
        print(f"[OUT]\n{out[:3000]}")
    return out

# Check if the actual project exists at /mnt/i/
wsl("test -d '/mnt/i/Github/Latent_Style/SchrodingerBridge' && echo 'EXISTS' || echo 'NOT_FOUND'", "Check project at /mnt/i/")
wsl("ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/ 2>/dev/null | head -30", "List p3_remote_10h at real location")

# Search for summary.json in the real project location
wsl("find '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h' -name 'summary.json' 2>/dev/null | head -20", "Find summary.json files")

# Check each experiment directory
exps = ["fc_sb_kernel7", "fc_sb_floor0", "fc_sb_curriculum", "fc_sb_fiber_ep", "fc_sb_wavelet"]
for exp in exps:
    path = f"/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/{exp}"
    wsl(f"test -d '{path}' && ls -laR '{path}' 2>/dev/null | head -40 || echo 'DIR_NOT_FOUND'", f"Check {exp} at real location")

# Also search for any full_eval directories
print("\n--- Searching for any evaluation results ---")
wsl("find '/mnt/i/Github/Latent_Style/SchrodingerBridge' -type d -name 'full_eval' 2>/dev/null | head -20", "full_eval dirs")
wsl("find '/mnt/i/Github/Latent_Style/SchrodingerBridge' -name 'summary.json' 2>/dev/null | head -30", "All summary.json files")

# Check if there are any checkpoint or output files
print("\n--- Looking for any experiment outputs ---")
for exp in exps:
    base = f"/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/{exp}"
    wsl(f"find '{base}' -type f 2>/dev/null | head -20 || echo 'NO_FILES'", f"Files in {exp}")

print("\n" + "=" * 70)
