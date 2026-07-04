"""验证远程代码同步结果."""
import sys, os
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")

# 检查 spectral_losses620 是否有 v3 修改
try:
    from spectral_losses620 import SpectralODEObjective620
    print("OK: spectral_losses620 imported")
    # 检查 endpoint_style_enabled 属性
    import inspect
    src = inspect.getsource(SpectralODEObjective620.__init__)
    if "endpoint_style_enabled" in src:
        print("OK: v3 endpoint style loss 已同步")
    else:
        print("WARNING: v3 修改未找到")
except Exception as e:
    print(f"FAIL: {e}")

# 检查 config_schema 是否有 v3 配置项
try:
    from config_schema import ExperimentConfig
    import inspect
    src = inspect.getsource(ExperimentConfig)
    if "spectral_w_endpoint_style_lh" in src:
        print("OK: config_schema v3 配置项已同步")
    else:
        print("WARNING: config_schema v3 配置项未找到")
except Exception as e:
    print(f"FAIL: {e}")

# 检查关键文件大小
sb_dir = r"I:\Github\Latent_Style\SchrodingerBridge"
for f in ["src/spectral_losses620.py", "src/config_schema.py", "src/trainer.py", "src/spectral_bridge620.py", "src/style_encoder620.py", "src/blocks620.py"]:
    path = os.path.join(sb_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {f}: {size} bytes")
    else:
        print(f"  {f}: NOT FOUND")

# 检查 630 配置
configs_dir = os.path.join(sb_dir, "configs")
cfgs_630 = [f for f in os.listdir(configs_dir) if "630" in f]
print(f"\n630 configs: {len(cfgs_630)}")
for f in sorted(cfgs_630):
    print(f"  {f}")

# 检查 docs/630
docs_630 = os.path.join(sb_dir, "docs", "630")
if os.path.exists(docs_630):
    state_dir = os.path.join(docs_630, "state")
    if os.path.exists(state_dir):
        print(f"\ndocs/630/state files: {os.listdir(state_dir)}")

# 检查数据集路径
print("\n=== 数据集路径检查 ===")
dataset_paths = [
    r"I:\wikiart_distinct5_samam_512_classview",
    r"I:\wikiart_distinct5_samam_512_classview\train",
    r"I:\wikiart_distinct5_samam_512_classview\test",
]
for p in dataset_paths:
    print(f"  {p}: exists={os.path.exists(p)}")
    if os.path.exists(p) and os.path.isdir(p):
        items = os.listdir(p)
        print(f"    items: {items[:5]}...")

# 检查 latent cache
latent_paths = [
    r"I:\Github\Latent_Style\SchrodingerBridge\cache",
    r"I:\Github\Latent_Style\latents",
]
for p in latent_paths:
    if os.path.exists(p):
        print(f"  {p}: exists, items={os.listdir(p)[:5]}")
    else:
        print(f"  {p}: NOT FOUND")
