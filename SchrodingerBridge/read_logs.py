import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("=" * 70)
print("Reading Experiment Logs")
print("=" * 70)

def wsl(command, desc):
    print(f"\n[READ] {desc}")
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
        print(f"[CONTENT]\n{out[:5000]}")
    else:
        print("[EMPTY or NOT FOUND]")
    return out

# Read the master log file
wsl("cat /home/xy/exp/p3_remote_10h/phase2_master.log", "Full phase2_master.log")

# Also check other log files in the directory
wsl("ls -lht /home/xy/exp/p3_remote_10h/*.log 2>/dev/null", "List all log files")
wsl("cat /home/xy/exp/p3_remote_10h/run_phase2.sh 2>/dev/null | head -100", "run_phase2.sh script")

# Check if there are any text files with results info
wsl("cat /home/xy/final_all.txt 2>/dev/null", "final_all.txt from home")
wsl("cat /home/xy/final_results.txt 2>/dev/null", "final_results.txt from home")

# Check diag files for any error information
wsl("cat /home/xy/diag_out.txt 2>/dev/null", "diag_out.txt")
wsl("cat /home/xy/f3_diag.txt 2>/dev/null | tail -50", "f3_diag.txt (last 50 lines)")

print("\n" + "=" * 70)
