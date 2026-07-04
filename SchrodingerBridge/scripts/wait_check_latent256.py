"""Wait and check latent256 training status."""
import sys
import time
import subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Wait for training to start
print("Waiting 30 seconds for training to start...")
time.sleep(30)

# Check python processes
print("\n=== Python processes ===")
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# Check GPU
print("=== GPU memory ===")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# Check latent256 log
log_path = r'C:\Users\Administrator\logs\latent256_train.log'
print(f"\n=== latent256 log (last 20 lines) ===")
try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in lines[-20:]:
        print(line, end='')
except Exception as e:
    print(f"Error: {e}")
