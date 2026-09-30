"""Test if detached python process can write to log."""
import sys
import subprocess
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PYTHON = r'C:\Program Files\Python312\python.exe'
LOG = r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\test_detached.log'

DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x08000000

# Simple test: python -c "print('hello')"
cmd = [PYTHON, '-c', 'import sys; print("hello from detached", flush=True); import time; time.sleep(5); print("done")']

print(f"Launching test detached process...")
with open(LOG, 'w') as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, creationflags=DETACHED_FLAGS, close_fds=True)
print(f"PID: {proc.pid}")

# Wait 3 seconds and check log
time.sleep(3)
with open(LOG, 'r') as f:
    content = f.read()
print(f"Log content after 3s: {content!r}")

print("==TEST_DETACHED_DONE==")
