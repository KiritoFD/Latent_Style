"""Check remote status: running python processes and recent log tails."""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SSH = ["ssh", "-p", "2222", "-o", "LogLevel=ERROR", "administrator@100.115.18.62"]

# Use a here-string style PowerShell command via a temp .ps1 on remote is overkill;
# instead run a compact one-liner with single quotes inside double quotes.
ps_cmd = (
    "Get-Process python -ErrorAction SilentlyContinue | "
    "Select-Object Id,StartTime | Format-Table -AutoSize; "
    "Write-Output '=== t11_repro log tail ==='; "
    "Get-Content C:\\Users\\Administrator\\logs\\t11_repro_train_eval.out -Tail 8 -ErrorAction SilentlyContinue; "
    "Write-Output '=== t11e2 log tail ==='; "
    "Get-Content C:\\Users\\Administrator\\logs\\t11e2_train_eval.out -Tail 8 -ErrorAction SilentlyContinue"
)

# Wrap in powershell -Command with & { } block to avoid escaping issues
full = ["powershell", "-NoProfile", "-Command", "& { " + ps_cmd + " }"]
result = subprocess.run(SSH + full, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
print("exit:", result.returncode)
