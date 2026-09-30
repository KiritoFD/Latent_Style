"""Check running python processes on remote."""
import subprocess
result = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True
)
print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
