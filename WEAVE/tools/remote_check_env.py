#!/usr/bin/env python3
"""Remote environment baseline check script for SchrodingerBridge reproduction."""
import os, sys, subprocess, json
sys.stdout.reconfigure(line_buffering=True)

SEP = "=" * 70

def run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    sys.stdout.flush()

# ─── 1. Python version and key packages ────────────────────────────────
section("1. Python Version & Key Packages")
py_ver = run("python --version")
print(f"Python: {py_ver}")
packages = ["torch", "diffusers", "transformers", "lpips", "accelerate", "mamba_ssm"]
for pkg in packages:
    ver = run(f'python -c "import {pkg}; print({pkg}.__version__)"')
    if "[ERROR]" in ver or "No module" in ver or "ModuleNotFoundError" in ver:
        print(f"  {pkg}: NOT INSTALLED")
    else:
        print(f"  {pkg}: {ver}")

# ─── 2. GPU info ───────────────────────────────────────────────────────
section("2. GPU Information")
gpu_info = run('python -c "import torch; print(torch.cuda.device_count()); [print(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_mem/1024**3) for i in range(torch.cuda.device_count())]"')
print(gpu_info)

# ─── 3. distinct5_512 dataset integrity ────────────────────────────────
section("3. distinct5_512 Dataset Integrity")
data_candidates = [
    r"I:\data\distinct5_512",
    r"I:\datasets\distinct5_512",
    r"D:\data\distinct5_512",
    r"I:\GitHub\Latent_Style\data\distinct5_512",
    r"I:\GitHub\Latent_Style\SchrodingerBridge\data\distinct5_512",
]
dataset_root = None
for c in data_candidates:
    if os.path.isdir(c):
        dataset_root = c
        break

if dataset_root is None:
    # Quick targeted search under I:\GitHub only (not full I:\)
    find_result = run(r'dir /s /b "I:\GitHub\distinct5_512" 2>nul', timeout=10)
    if find_result and "[ERROR]" not in find_result:
        for line in find_result.split("\n"):
            if line.strip() and os.path.isdir(line.strip()):
                dataset_root = line.strip()
                break

if dataset_root:
    print(f"Dataset root: {dataset_root}")
    for split, expected_styles, expected_per_style in [("test", 5, 30), ("train", 5, 1000)]:
        split_path = os.path.join(dataset_root, split)
        if os.path.isdir(split_path):
            style_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            n_styles = len(style_dirs)
            counts = {}
            for sd in sorted(style_dirs):
                sd_path = os.path.join(split_path, sd)
                files = [f for f in os.listdir(sd_path) if os.path.isfile(os.path.join(sd_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
                counts[sd] = len(files)
            ok = n_styles == expected_styles and all(v == expected_per_style for v in counts.values())
            status = "OK" if ok else "MISMATCH"
            print(f"  {split}: {status} | {n_styles} styles x {counts} (expected {expected_styles}x{expected_per_style})")
        else:
            print(f"  {split}: MISSING ({split_path})")
else:
    print("  Dataset NOT FOUND at any expected location")

# ─── 4. HuggingFace model cache ────────────────────────────────────────
section("4. HuggingFace Model Cache")
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hub_cache = os.path.join(hf_home, "hub")
extra_cache_dirs = [
    os.path.expanduser("~/.cache/huggingface"),
    r"C:\Users\Administrator\.cache\huggingface",
    r"I:\cache\huggingface",
    r"I:\huggingface_cache",
]
all_cache_dirs = [hub_cache] + extra_cache_dirs
seen = set()
sd15_found = False
sdturbo_found = False
for cache_dir in all_cache_dirs:
    if cache_dir in seen or not os.path.isdir(cache_dir):
        continue
    seen.add(cache_dir)
    try:
        entries = os.listdir(cache_dir)
    except:
        continue
    model_dirs = [e for e in entries if e.startswith("models--")]
    if model_dirs:
        print(f"  Cache dir: {cache_dir} ({len(model_dirs)} models)")
        for md in sorted(model_dirs):
            md_lower = md.lower()
            if "stable" in md_lower or "sd-turbo" in md_lower or "runwayml" in md_lower:
                print(f"    -> {md}")
            if ("v1-5" in md_lower or "stable-diffusion-v1-5" in md_lower or "runwayml" in md_lower) and "stable" in md_lower:
                sd15_found = True
            if "turbo" in md_lower:
                sdturbo_found = True
if not sd15_found:
    print("  SD 1.5: NOT FOUND in HF cache")
if not sdturbo_found:
    print("  SD-Turbo: NOT FOUND in HF cache")

# ─── 5. AdaIN weights ─────────────────────────────────────────────────
section("5. AdaIN Weights (decoder.pth)")
adain_search_paths = [
    r"I:\GitHub\Latent_Style\Related_Works",
    r"I:\GitHub\Latent_Style\SchrodingerBridge",
]
adain_found = False
for sp in adain_search_paths:
    if not os.path.isdir(sp):
        continue
    result = run(f'dir /s /b "{sp}\\decoder.pth" 2>nul', timeout=10)
    if result and "[ERROR]" not in result and result.strip():
        for line in result.strip().split("\n"):
            if line.strip():
                print(f"  FOUND: {line.strip()}")
                adain_found = True
if not adain_found:
    print("  decoder.pth: NOT FOUND in expected dirs")

# ─── 6. SaMAM - mamba_ssm in WSL ──────────────────────────────────────
section("6. SaMAM - mamba_ssm in WSL")
mamba_result = run(r'wsl bash -c "python3 -c \"import mamba_ssm; print(mamba_ssm.__version__)\""', timeout=15)
if "[ERROR]" in mamba_result or "No module" in mamba_result or "ModuleNotFoundError" in mamba_result:
    print(f"  mamba_ssm in WSL: NOT AVAILABLE ({mamba_result[:120]})")
else:
    print(f"  mamba_ssm in WSL: {mamba_result}")

# ─── 7. StyleID code ──────────────────────────────────────────────────
section("7. StyleID Code")
styleid_path = r"I:\GitHub\Latent_Style\Related_Works\repos\StyleID"
if os.path.isdir(styleid_path):
    entries = os.listdir(styleid_path)
    key_files = [f for f in entries if f.endswith(('.py', '.md', '.txt', '.yaml', '.yml'))]
    print(f"  EXISTS: {styleid_path}")
    print(f"  Key files ({len(key_files)}): {', '.join(sorted(key_files)[:12])}")
else:
    print(f"  NOT FOUND: {styleid_path}")

# ─── 8. CUT code ──────────────────────────────────────────────────────
section("8. CUT Code")
cut_path = r"I:\GitHub\Latent_Style\Related_Works\repos\external\CUT"
if os.path.isdir(cut_path):
    entries = os.listdir(cut_path)
    key_files = [f for f in entries if f.endswith(('.py', '.md', '.txt', '.yaml', '.yml'))]
    print(f"  EXISTS: {cut_path}")
    print(f"  Key files ({len(key_files)}): {', '.join(sorted(key_files)[:12])}")
else:
    print(f"  NOT FOUND: {cut_path}")

# ─── 9. S2WAT code + pre-trained VGG weights ──────────────────────────
section("9. S2WAT Code & VGG Weights")
s2wat_candidates = [
    r"I:\GitHub\Latent_Style\Related_Works\repos\S2WAT",
    r"I:\GitHub\Latent_Style\Related_Works\repos\external\S2WAT",
    r"I:\GitHub\Latent_Style\Related_Works\repos\s2wat",
]
s2wat_found = False
for sp in s2wat_candidates:
    if os.path.isdir(sp):
        print(f"  Code EXISTS: {sp}")
        entries = os.listdir(sp)
        key_files = [f for f in entries if f.endswith(('.py', '.md'))]
        print(f"  Key files ({len(key_files)}): {', '.join(sorted(key_files)[:12])}")
        s2wat_found = True
        # Check for VGG weights
        vgg_result = run(f'dir /s /b "{sp}\\*vgg*" 2>nul', timeout=10)
        if vgg_result and "[ERROR]" not in vgg_result and vgg_result.strip():
            print(f"  VGG weights found:")
            for line in vgg_result.strip().split("\n")[:5]:
                if line.strip():
                    print(f"    {line.strip()}")
        else:
            print(f"  VGG weights: NOT FOUND under {sp}")
        break
if not s2wat_found:
    print("  S2WAT code: NOT FOUND at any expected location")

# Check VGG weights under Related_Works broadly (not whole I:\)
vgg_rw = run(r'dir /s /b "I:\GitHub\Latent_Style\Related_Works\*vgg*.pth" 2>nul', timeout=10)
if vgg_rw and "[ERROR]" not in vgg_rw and vgg_rw.strip():
    lines = [l.strip() for l in vgg_rw.strip().split("\n") if l.strip()]
    if lines:
        print(f"  VGG .pth in Related_Works ({len(lines)} total):")
        for l in lines[:5]:
            print(f"    {l}")

# ─── 10. Disk space on I: ─────────────────────────────────────────────
section("10. Disk Space on I:\\")
disk_info = run(r'powershell -command "Get-PSDrive I | Select-Object Used,Free | ConvertTo-Json"', timeout=10)
if disk_info and "[ERROR]" not in disk_info:
    try:
        info = json.loads(disk_info)
        used_gb = info.get("Used", 0) / (1024**3)
        free_gb = info.get("Free", 0) / (1024**3)
        total_gb = used_gb + free_gb
        print(f"  I:\\ Total: {total_gb:.1f} GB | Used: {used_gb:.1f} GB | Free: {free_gb:.1f} GB")
    except:
        print(f"  Raw: {disk_info[:200]}")
else:
    dir_info = run(r"dir I:\ 2>nul", timeout=5)
    print(f"  Raw dir: {dir_info[:200] if dir_info else 'N/A'}")

section("DONE")
print("Check complete.")
