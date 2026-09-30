"""Check T11 evo training/eval logs on remote."""
import subprocess
import sys

REMOTE = "administrator@100.115.18.62"
PORT = "2222"

def remote_tail(path, n=40):
    cmd = f'ssh -p {PORT} -o LogLevel=ERROR {REMOTE} "powershell -Command Get-Content -Path {path} -Tail {n}"'
    # Escape issue: use Get-Content with -Path
    r = subprocess.run(
        ["ssh", "-p", PORT, "-o", "LogLevel=ERROR", REMOTE,
         "powershell", "-Command",
         f"Get-Content -Path '{path}' -Tail {n}"],
        capture_output=True, text=True, timeout=30,
    )
    return r.stdout, r.stderr

def remote_run(py_cmd):
    """Run a python one-liner on remote."""
    r = subprocess.run(
        ["ssh", "-p", PORT, "-o", "LogLevel=ERROR", REMOTE,
         "python", "-c", py_cmd],
        capture_output=True, text=True, timeout=60,
    )
    return r.stdout, r.stderr

if __name__ == "__main__":
    # Check t11_repro log tail
    print("=" * 60)
    print("=== t11_repro_15ep_train_eval.out (tail 50) ===")
    print("=" * 60)
    out, err = remote_tail(r"C:\Users\Administrator\logs\t11_repro_15ep_train_eval.out", 50)
    print(out)
    if err:
        print("STDERR:", err[:500])

    print("\n" + "=" * 60)
    print("=== t11e1_ll05_15ep_train_eval.out (tail 30) ===")
    print("=" * 60)
    out, err = remote_tail(r"C:\Users\Administrator\logs\t11e1_ll05_15ep_train_eval.out", 30)
    print(out)
    if err:
        print("STDERR:", err[:500])

    print("\n" + "=" * 60)
    print("=== t11e2_extrap05_15ep_train_eval.out (tail 30) ===")
    print("=" * 60)
    out, err = remote_tail(r"C:\Users\Administrator\logs\t11e2_extrap05_15ep_train_eval.out", 30)
    print(out)
    if err:
        print("STDERR:", err[:500])

    # Check exp dirs for checkpoints
    print("\n" + "=" * 60)
    print("=== Exp directories ===")
    print("=" * 60)
    py = (
        "import os; "
        "base=r'I:\Github\Latent_Style\SchrodingerBridge\exp'; "
        "[print(d, sorted(os.listdir(os.path.join(base,d)))[:5]) "
        "for d in sorted(os.listdir(base)) "
        "if 't11' in d.lower() and os.path.isdir(os.path.join(base,d))]"
    )
    out, err = remote_run(py)
    print(out)
    if err:
        print("STDERR:", err[:500])

    # Check for summary.json in eval output
    print("\n" + "=" * 60)
    print("=== Looking for summary.json ===")
    print("=" * 60)
    py = (
        "import os, json; "
        "base=r'I:\Github\Latent_Style\SchrodingerBridge\exp'; "
        "[print(d, f) for d in sorted(os.listdir(base)) "
        "if 't11' in d.lower() and os.path.isdir(os.path.join(base,d)) "
        "for root,dirs,files in os.walk(os.path.join(base,d)) "
        "for f in files if f=='summary.json']"
    )
    out, err = remote_run(py)
    print(out)
    if err:
        print("STDERR:", err[:500])
