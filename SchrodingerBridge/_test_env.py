import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("torch import error:", e)
