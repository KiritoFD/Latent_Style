import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

# Test different command formats
print("Testing SSH command formats...\n")

# Format 1: Simple command
cmd1 = f'ssh -p {ssh_port} {ssh_host} pwd'
print(f"Format 1: {cmd1}")
r1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
print(f"  Result: {r1.stdout.strip()} (exit: {r1.returncode})")

# Format 2: Command with quotes
cmd2 = f'ssh -p {ssh_port} {ssh_host} "pwd"'
print(f"\nFormat 2: {cmd2}")
r2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
print(f"  Result: {r2.stdout.strip()} (exit: {r2.returncode})")
if r2.stderr:
    print(f"  Error: {r2.stderr.strip()}")

# Format 3: Complex command
cmd3 = f'ssh -p {ssh_port} {ssh_host} "ls /home/xy/"'
print(f"\nFormat 3: {cmd3}")
r3 = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
print(f"  Result: {r3.stdout.strip()[:200]} (exit: {r3.returncode})")
if r3.stderr:
    print(f"  Error: {r3.stderr.strip()[:200]}")

# Format 4: Using bash -c
cmd4 = f'ssh -p {ssh_port} {ssh_host} bash -c "pwd"'
print(f"\nFormat 4: {cmd4}")
r4 = subprocess.run(cmd4, shell=True, capture_output=True, text=True)
print(f"  Result: {r4.stdout.strip()} (exit: {r4.returncode})")
if r4.stderr:
    print(f"  Error: {r4.stderr.strip()}")
