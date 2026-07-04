"""Check GPU and python processes on remote."""
import sys
import subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check nvidia-smi
print("=== nvidia-smi ===")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
except Exception as e:
    print(f"Error: {e}")

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

# Check scheduled task status
print("\n=== Scheduled task status ===")
try:
    result = subprocess.run(
        ['schtasks', '/query', '/tn', 'latent256_train', '/v', '/fo', 'list'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
except Exception as e:
    print(f"Error: {e}")
