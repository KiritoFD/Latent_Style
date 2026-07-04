import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Checking Experiment Directory Contents")
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
        print(f"[OUT]\n{out[:2000]}")
    return out

exps = ["fc_sb_kernel7", "fc_sb_floor0", "fc_sb_curriculum", "fc_sb_fiber_ep", "fc_sb_wavelet"]

# Check each experiment directory in detail
for exp in exps:
    print(f"\n{'='*60}")
    print(f"Experiment: {exp}")
    print('='*60)
    
    base_path = f"/home/xy/exp/p3_remote_10h/{exp}"
    
    # List all contents recursively (limited depth)
    wsl(f"ls -laR '{base_path}' 2>/dev/null | head -40", f"Full listing of {exp}")
    
    # Specifically look for full_eval or summary files
    wsl(f"find '{base_path}' -type f \( -name '*.json' -o -name '*.txt' -o -name '*.log' \) 2>/dev/null | head -20", f"Find result files in {exp}")

# Also check if there's a different location where results might be stored
print("\n\n--- Alternative locations check ---")
wsl("find /home/xy/exp/p3_remote_10h -name 'full_eval' -type d 2>/dev/null", "Find full_eval dirs")
wsl("find /home/xy -path '*/fc_sb_*/full_eval/summary.json' 2>/dev/null", "Find summary.json in any fc_sb/full_eval")

# Check if results might be in the main project under a different structure
wsl("find /home/xy/Latent_Style/SchrodingerBridge -type d -name 'fc_sb_*' 2>/dev/null | head -20", "fc_sb in main project")
wsl("ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/ 2>/dev/null || echo 'NO_EXP_DIR'", "Main project exp dir")

# Maybe experiments haven't finished yet? Check for any output files at all
print("\n--- Any output files? ---")
wsl("find /home/xy/exp/p3_remote_10h -type f 2>/dev/null | head -30", "All files in p3_remote_10h")

print("\n" + "=" * 70)
