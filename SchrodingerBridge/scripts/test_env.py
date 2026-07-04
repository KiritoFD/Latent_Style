import sys
print("python:", sys.executable)
print("version:", sys.version)
mods = ["torch", "transformers", "diffusers", "pyiqa", "lpips", "numpy", "PIL"]
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print(f"OK {m} {v}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
try:
    import torch
    print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info(0)
        print(f"vram free={free/1e9:.2f}GB total={total/1e9:.2f}GB")
except Exception as e:
    print("cuda check failed:", e)
