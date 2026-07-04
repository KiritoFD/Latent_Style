"""Check remote training status via log files."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def tail_file(path, n=15):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return lines[-n:] if len(lines) > n else lines
    except FileNotFoundError:
        return [f"[NOT FOUND: {path}]\n"]
    except Exception as e:
        return [f"[ERROR: {e}]\n"]

print("=== pixel256_train.log (last 15 lines) ===")
for line in tail_file(r'C:\Users\Administrator\logs\pixel256_train.log', 15):
    print(line, end='')

print("\n=== latent256_train.log (last 15 lines) ===")
for line in tail_file(r'C:\Users\Administrator\logs\latent256_train.log', 15):
    print(line, end='')
