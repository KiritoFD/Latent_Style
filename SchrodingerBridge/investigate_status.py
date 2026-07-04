import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Investigating Experiment Status")
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

# Check the phase2 master log for experiment status
wsl("cat /home/xy/exp/p3_remote_10h/phase2_master.log 2>/dev/null | tail -100", "Phase 2 master log (last 100 lines)")

# Check run scripts to understand what they do
wsl("head -50 /home/xy/exp/p3_remote_10h/run_phase2.sh 2>/dev/null", "run_phase2.sh content")

# Search more broadly - maybe results are in a completely different location
print("\n--- Broad search for any experiment outputs ---")
wsl("find /home/xy -name 'summary.json' 2>/dev/null | head -30", "All summary.json files")
wsl("find /home/xy -type d -name 'full_eval' 2>/dev/null | head -20", "All full_eval directories")

# Check if there are any recent modified directories that might contain results
wsl("find /home/xy -maxdepth 5 -type f -mtime -7 -name '*.json' 2>/dev/null | head -30", "Recently modified JSON files")

# Look in the main project directory structure
wsl("ls -la /home/xy/Latent_Style/SchrodingerBridge/ 2>/dev/null | head -30", "Main project root")
wsl("find /home/xy/Latent_Style/SchrodingerBridge -maxdepth 3 -type d -name '*eval*' 2>/dev/null | head -20", "Eval-related dirs in project")

# Maybe check if there's an output or results directory somewhere
wsl("find /home/xy -maxdepth 4 -type d \( -name 'output' -o -name 'results' -o -name 'outputs' \) 2>/dev/null | head -20", "Output/results directories")

# Check home directory for any result files
wsl("ls -lht /home/xy/*.json 2>/dev/null | head -20 || echo 'NO_JSON_IN_HOME'", "JSON files in home")

print("\n" + "=" * 70)
