"""Check D3/D4/D5 eval progress by parsing logs and output dirs."""
import os
import re
import sys
import io
from pathlib import Path

# Force UTF-8 output to avoid GBK encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_DIR = Path(r"C:\Users\Administrator\logs")
EXP_ROOT = Path(r"I:\Github\Latent_Style\WEAVE\exp\rebuttal")

def count_summary_updated(log_path):
    """Count 'Summary updated' occurrences in a log file."""
    if not log_path.exists():
        return 0, []
    epochs = []
    pattern = re.compile(r"epoch_(\d+)[/\\]summary\.json")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Summary updated" in line:
                m = pattern.search(line)
                if m:
                    epochs.append(int(m.group(1)))
    return len(epochs), sorted(set(epochs))

def check_oracle_json(out_dir):
    """Check if oracle_regret.json exists in output dir."""
    p = out_dir / "oracle_regret.json"
    return p.exists()

def main():
    print("=" * 70)
    print("D3/D4/D5 PROGRESS CHECK")
    print("=" * 70)

    # D3 eval log
    d3_log = LOG_DIR / "rebuttal_D3_eval.log"
    n, epochs = count_summary_updated(d3_log)
    d3_out = EXP_ROOT / "expA_D3_seed42"
    d3_done = check_oracle_json(d3_out)
    print(f"\n[D3] rebuttal_D3_eval.log:")
    print(f"  Summary updated count: {n}")
    print(f"  Epochs evaluated: {epochs}")
    print(f"  Output dir: {d3_out}")
    print(f"  oracle_regret.json exists: {d3_done}")

    # D4 eval log (may not exist yet)
    d4_log = LOG_DIR / "rebuttal_D4_eval.log"
    if d4_log.exists():
        n, epochs = count_summary_updated(d4_log)
        d4_out = EXP_ROOT / "expA_D4_seed42"
        d4_done = check_oracle_json(d4_out)
        print(f"\n[D4] rebuttal_D4_eval.log:")
        print(f"  Summary updated count: {n}")
        print(f"  Epochs evaluated: {epochs}")
        print(f"  oracle_regret.json exists: {d4_done}")
    else:
        print(f"\n[D4] eval log not yet created (D4 not started)")

    # D4 training log
    d4_train = LOG_DIR / "rebuttal_D4_train.log"
    if d4_train.exists():
        size = d4_train.stat().st_size
        print(f"  D4 train log size: {size} bytes")
        # Get last 5 lines
        with open(d4_train, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"  Last 5 lines of D4 train log:")
        for line in lines[-5:]:
            try:
                print(f"    {line.rstrip()}")
            except Exception:
                print(f"    [unprintable line, len={len(line)}]")

    # D5 eval log
    d5_log = LOG_DIR / "rebuttal_D5_eval.log"
    if d5_log.exists():
        n, epochs = count_summary_updated(d5_log)
        d5_out = EXP_ROOT / "expA_D5_seed42"
        d5_done = check_oracle_json(d5_out)
        print(f"\n[D5] rebuttal_D5_eval.log:")
        print(f"  Summary updated count: {n}")
        print(f"  Epochs evaluated: {epochs}")
        print(f"  oracle_regret.json exists: {d5_done}")
    else:
        print(f"\n[D5] eval log not yet created (D5 not started)")

    # D5 training log
    d5_train = LOG_DIR / "rebuttal_D5_train.log"
    if d5_train.exists():
        size = d5_train.stat().st_size
        print(f"  D5 train log size: {size} bytes")
        with open(d5_train, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"  Last 5 lines of D5 train log:")
        for line in lines[-5:]:
            try:
                print(f"    {line.rstrip()}")
            except Exception:
                print(f"    [unprintable line, len={len(line)}]")

    # Chain runner log
    chain_log = LOG_DIR / "chain_D3_D4_D5.log"
    if chain_log.exists():
        with open(chain_log, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"\n[Chain] chain_D3_D4_D5.log (last 15 lines):")
        for line in lines[-15:]:
            print(f"  {line.rstrip()}")

    # Check if python processes are still running
    import subprocess
    print(f"\n[Processes] python.exe processes:")
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n")[:20]:
            if line.strip():
                print(f"  {line.strip()[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
