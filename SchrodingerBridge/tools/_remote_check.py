"""远程环境检查脚本 - 在远程 Windows Python 上执行."""
import sys, os, json

# 检查 Python 和 torch
print("=== Python 环境 ===")
print(f"Python: {sys.version}")
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError as e:
    print(f"torch import error: {e}")

# 检查 SchrodingerBridge 代码
sb_dir = r"I:\Github\Latent_Style\SchrodingerBridge"
print(f"\n=== SchrodingerBridge 代码 ({sb_dir}) ===")
print(f"exists: {os.path.exists(sb_dir)}")

# 检查 src 目录
src_dir = os.path.join(sb_dir, "src")
if os.path.exists(src_dir):
    print(f"src files: {len(os.listdir(src_dir))}")
    # 检查关键文件
    for f in ["spectral_losses620.py", "config_schema.py", "trainer.py", "run.py"]:
        path = os.path.join(src_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {f}: {size} bytes")
        else:
            print(f"  {f}: NOT FOUND")

# 检查 630 配置
configs_dir = os.path.join(sb_dir, "configs")
if os.path.exists(configs_dir):
    cfgs_630 = [f for f in os.listdir(configs_dir) if "630" in f]
    print(f"\n630 configs: {len(cfgs_630)}")
    for f in sorted(cfgs_630)[:10]:
        print(f"  {f}")

# 检查 docs/630
docs_630 = os.path.join(sb_dir, "docs", "630")
print(f"\ndocs/630 exists: {os.path.exists(docs_630)}")
if os.path.exists(docs_630):
    state_dir = os.path.join(docs_630, "state")
    if os.path.exists(state_dir):
        progress_path = os.path.join(state_dir, "progress.json")
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            print(f"  progress.json iteration: {prog.get('iteration')}")
            print(f"  progress.json keys: {list(prog.keys())[:10]}")

# 检查 exp 目录中的 630 系列
exp_dir = os.path.join(sb_dir, "exp")
if os.path.exists(exp_dir):
    exps_630 = [d for d in os.listdir(exp_dir) if "630" in d]
    print(f"\n630 experiments: {len(exps_630)}")
    for d in sorted(exps_630)[:10]:
        print(f"  {d}")
    # 也检查 SaMam 相关
    exps_samam = [d for d in os.listdir(exp_dir) if "samam" in d.lower()]
    print(f"samam experiments: {len(exps_samam)}")
    for d in exps_samam[:10]:
        print(f"  {d}")

# 检查 git 状态
print(f"\n=== Git 状态 ===")
import subprocess
try:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=sb_dir)
    print(f"HEAD: {result.stdout.strip()[:20]}")
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd=sb_dir)
    changed = [l for l in result.stdout.strip().split("\n") if l.strip()]
    print(f"changed files: {len(changed)}")
    result = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True, cwd=sb_dir)
    print(f"recent commits:\n{result.stdout.strip()}")
except Exception as e:
    print(f"git error: {e}")

# 搜索 SaMam 评估结果
print(f"\n=== SaMam 评估结果搜索 ===")
for search_dir in [
    os.path.join(sb_dir, "exp"),
    os.path.join(sb_dir, "baseline_pipeline"),
    r"I:\Github\Latent_Style\exp",
]:
    if not os.path.exists(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f == "curve_summary.json" and "samam" in root.lower():
                print(f"  FOUND: {os.path.join(root, f)}")
            elif f == "summary.json" and "samam" in root.lower():
                print(f"  FOUND: {os.path.join(root, f)}")

# 检查数据集
print(f"\n=== 数据集 ===")
dataset_dirs = [
    r"I:\Github\Latent_Style\Dataset",
    r"I:\wikiart_distinct5_samam_512_classview",
    r"I:\Github\Latent_Style\SchrodingerBridge\I??wikiart_distinct5_samam_512_latents_ema",
]
for d in dataset_dirs:
    print(f"  {d}: exists={os.path.exists(d)}")
