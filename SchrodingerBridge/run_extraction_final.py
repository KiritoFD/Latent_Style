import subprocess
import os

ssh_host = "administrator@100.115.18.62"
ssh_port = "2222"
remote_project = "/home/xy/Latent_Style/SchrodingerBridge"
local_script = r"g:\GitHub\Latent_Style\SchrodingerBridge\extract_metrics_remote.py"
local_result = r"g:\GitHub\Latent_Style\SchrodingerBridge\metrics.txt"

print("=" * 70)
print("FC-SB Phase 2 v4 Metrics Extraction")
print("=" * 70)

def run(cmd, desc):
    print(f"\n[STEP] {desc}")
    print(f"[CMD] {cmd[:120]}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=60
    )
    if result.stdout.strip():
        print(f"[OUT] {result.stdout.strip()[:1000]}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"[ERR] {result.stderr.strip()[:500]}")
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}]")
    return ok, result.stdout

# Step 1: Upload script
print("\n" + "-" * 70)
ok, _ = run(
    f'scp -P {ssh_port} "{local_script}" {ssh_host}:~/extract_metrics_remote.py',
    "Upload Python script to remote server"
)
if not ok:
    print("[FATAL] Cannot upload script")
    exit(1)

# Step 2: Execute Python directly (avoiding shell interpretation issues)
ok, out = run(
    f'ssh -p {ssh_port} {ssh_host} python3 /home/xy/extract_metrics_remote.py',
    "Execute metrics extraction on remote server"
)

# If direct execution fails, try with cd
if not ok:
    print("\n[RETRY] Trying with cd to project directory...")
    ok, out = run(
        f'ssh -p {ssh_port} {ssh_host} "cd {remote_project} && python3 ../extract_metrics_remote.py"',
        "Retry: Execute from project directory"
    )

# Step 3: Get output (either from file or stdout)
if ok and out:
    # Save stdout as result
    with open(local_result, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f"\n[SAVED] Output saved from stdout to {local_result}")
else:
    # Try to read from remote file if it was created
    print("\n[ALT] Attempting to download results file...")
    ok_dl, out_dl = run(
        f'scp -P {ssh_port} {ssh_host}:~/metrics.txt "{local_result}""',
        "Download metrics.txt from remote"
    )
    
    if not ok_dl:
        # Last resort: cat the file via SSH
        print("\n[LAST RESORT] Reading output via SSH...")
        ok_cat, out_cat = run(
            f'ssh -p {ssh_port} {ssh_host} cat ~/metrics.txt 2>/dev/null || echo "NO_OUTPUT_FILE"',
            "Cat remote output file"
        )
        if ok_cat and "NO_OUTPUT_FILE" not in out_cat:
            with open(local_result, 'w', encoding='utf-8') as f:
                f.write(out_cat)
            print(f"[SAVED] Retrieved via SSH cat")

# Step 4: Display final results
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

if os.path.exists(local_result) and os.path.getsize(local_result) > 0:
    with open(local_result, 'r', encoding='utf-8') as f:
        content = f.read()
    print("\n" + content)
    print("\n[SUCCESS] Extraction complete!")
else:
    print("\n[ERROR] No results obtained!")
    if os.path.exists(local_result):
        os.remove(local_result)

print(f"\n[FILE] {local_result}")
print("=" * 70)
