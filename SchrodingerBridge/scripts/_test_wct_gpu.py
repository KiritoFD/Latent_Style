"""Quick test: verify GPU WCT works and is fast. Run on remote."""
import os
import sys
import time
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")

from spectral_bridge620 import _wct_match_fiber, _wct_match_fiber_keep_mean

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# Create test tensors: (B, C, H, W) — typical fiber shape
# C=4 (latent_channels), H=W=32 (after DWT on 64x64 latent)
B, C, H, W = 2, 4, 32, 32
content = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
style = torch.randn(1, C, H, W, device=device, dtype=torch.float32)

# Warmup
_ = _wct_match_fiber(content, style)
if device == "cuda":
    torch.cuda.synchronize()

# Time GPU WCT
t0 = time.time()
for _ in range(100):
    result = _wct_match_fiber(content, style)
if device == "cuda":
    torch.cuda.synchronize()
t_gpu = (time.time() - t0) / 100 * 1000  # ms per call
print(f"\n_wct_match_fiber (GPU): {t_gpu:.2f} ms/call")
print(f"  input shape: {content.shape}, output shape: {result.shape}")
print(f"  output mean: {result.mean().item():.4f}, std: {result.std().item():.4f}")

# Test keep_mean variant
_ = _wct_match_fiber_keep_mean(content, style)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    result2 = _wct_match_fiber_keep_mean(content, style)
if device == "cuda":
    torch.cuda.synchronize()
t_gpu2 = (time.time() - t0) / 100 * 1000
print(f"\n_wct_match_fiber_keep_mean (GPU): {t_gpu2:.2f} ms/call")

# Compare with CPU
content_cpu = content.cpu()
style_cpu = style.cpu()
t0 = time.time()
for _ in range(10):
    result_cpu = _wct_match_fiber(content_cpu, style_cpu)
t_cpu = (time.time() - t0) / 10 * 1000
print(f"\n_wct_match_fiber (CPU): {t_cpu:.2f} ms/call")
print(f"  Speedup: {t_cpu/t_gpu:.1f}x")

# Verify outputs match (GPU vs CPU)
diff = (result.cpu() - result_cpu).abs().max().item()
print(f"\nMax diff GPU vs CPU: {diff:.6e}")

# Test with BFloat16 (common training dtype)
if device == "cuda":
    content_bf = content.to(torch.bfloat16)
    style_bf = style.to(torch.bfloat16)
    try:
        result_bf = _wct_match_fiber(content_bf, style_bf)
        print(f"\nBFloat16 input: OK, output dtype={result_bf.dtype}")
    except Exception as e:
        print(f"\nBFloat16 input: FAIL - {e}")

print("\n=== WCT GPU test complete ===")
