import subprocess

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"

# Simple test
cmd = f'ssh -p {ssh_port} {ssh_host} echo "Hello from remote server"'
print(f"Running: {cmd}")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(f"Return code: {result.returncode}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
