"""Kill all chain_D3_D4_D5.py processes, then report remaining python processes."""
import subprocess
import os
import time

# Kill all python processes running chain_D3_D4_D5.py
result = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True
)
print("=== Current python processes ===")
print(result.stdout)

# Parse and kill chain_D3_D4_D5 processes
lines = result.stdout.strip().split("\n")
killed = []
for line in lines:
    if "chain_D3_D4_D5" in line.lower():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            pid = parts[-1].strip()
            if pid.isdigit():
                print(f"Killing chain_D3_D4_D5 PID={pid}: {line.strip()}")
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                killed.append(pid)

# Also kill the bat process
result2 = subprocess.run(
    ["wmic", "process", "where", "name='cmd.exe'", "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True
)
for line in result2.stdout.strip().split("\n"):
    if "_run_chain_D3_D4_D5" in line.lower() or "chain_D3_D4_D5" in line.lower():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            pid = parts[-1].strip()
            if pid.isdigit():
                print(f"Killing bat/cmd PID={pid}: {line.strip()}")
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                killed.append(pid)

print(f"\n=== Killed {len(killed)} processes: {killed} ===")

# Wait and report remaining
time.sleep(2)
result3 = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True
)
print("\n=== Remaining python processes ===")
print(result3.stdout)
