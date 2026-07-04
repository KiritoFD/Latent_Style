"""Launch pixel256 training as a detached process via Start-Process."""
import sys
import subprocess
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ps_cmd = (
    'Start-Process -FilePath python '
    '-ArgumentList "run.py --config configs\\630_pixel_256.json" '
    '-RedirectStandardOutput "logs\\pixel256_train.log" '
    '-RedirectStandardError "logs\\pixel256_train_err.log" '
    '-WorkingDirectory "C:\\Users\\Administrator" '
    '-WindowStyle Hidden -PassThru'
)

print("Launching pixel256 training via Start-Process...")
result = subprocess.run(
    ['powershell', '-Command', ps_cmd],
    capture_output=True, text=True, timeout=30
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"returncode: {result.returncode}")

print("\nWaiting 20 seconds for training to start...")
time.sleep(20)

print("\n=== Python processes ===")
result = subprocess.run(
    ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)

print("=== GPU memory ===")
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)

print("=== pixel256 log (last 10 lines) ===")
try:
    with open(r'C:\Users\Administrator\logs\pixel256_train.log', 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in lines[-10:]:
        print(line, end='')
except Exception as e:
    print(f"Error: {e}")
