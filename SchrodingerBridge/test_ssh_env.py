import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

print("Testing SSH connection...\n")

# Test basic connectivity
cmd = f'ssh -p {ssh_port} {ssh_host} bash -c "hostname && whoami && pwd"'
print(f"Command: {cmd}")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(f"Exit code: {result.returncode}")
print(f"Output:\n{result.stdout}")
if result.stderr:
    print(f"Error:\n{result.stderr}")

# Check if we're in WSL or remote
print("\n--- Checking environment ---")
cmd2 = f'ssh -p {ssh_port} {ssh_host} bash -c "uname -a && cat /etc/os-release 2>/dev/null | head -3"'
result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(result2.stdout)
