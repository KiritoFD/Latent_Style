"""Launch latent256 training as a detached process via Start-Process."""
import sys
import subprocess
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Use PowerShell Start-Process to create a detached process
ps_cmd = (
    'Start-Process -FilePath python '
    '-ArgumentList "run.py --config configs\\630_latent_256.json" '
    '-RedirectStandardOutput "logs\\latent256_train.log" '
    '-RedirectStandardError "logs\\latent256_train_err.log" '
    '-WorkingDirectory "C:\\Users\\Administrator" '
    '-WindowStyle Hidden -PassThru'
)

print("Launching latent256 training via Start-Process...")
result = subprocess.run(
    ['powershell', '-Command', ps_cmd],
    capture_output=True, text=True, timeout=30
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"returncode: {result.returncode}")

# Wait a bit and check
print("\nWaiting 15 seconds...")
time.sleep(15)

# Check if python process started
print("\n=== Python processes ===")
result = subprocess.run(
    ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)

# Check GPU
print("=== GPU memory ===")
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)
