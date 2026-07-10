import torch
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
