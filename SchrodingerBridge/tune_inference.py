"""
Tune Ours inference: find optimal batch size + pipeline config
Starts small, monitors VRAM, stops before OOM.
"""
import torch, time, sys, gc
sys.path.insert(0, r'G:\GitHub\Latent_Style\SchrodingerBridge\src')
from utils.inference import LGTInference, load_vae, decode_latent, encode_image

device = 'cuda'
torch.set_float32_matmul_precision('high')
ckpt = r'G:\GitHub\Latent_Style\SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0\epoch_0007.pt'

print("Loading models...")
lgt = LGTInference(ckpt, device=device, num_steps=1, step_size=1.0)
vae = load_vae(device)

from PIL import Image
from torchvision import transforms as T
tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])
dummy = tfm(Image.new('RGB', (512,512))).unsqueeze(0)

# Reserve ~1GB for safety
SAFE_MB = 1024
TOTAL_MB = 8192
MAX_SAFE = TOTAL_MB - SAFE_MB

print(f"\n{'batch':>5s} {'model_ms':>8s} {'vae_ms':>7s} {'e2e_ms':>7s} {'pipe_ms':>7s} {'speedup':>7s} {'VRAM_MB':>7s} {'status':>10s}")
print("-" * 65)

best = {"bs": 1, "pipe_ms": 999, "e2e_ms": 999}

for bs in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    batch = dummy.expand(bs, -1, -1, -1).to(device)
    
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lat_src = encode_image(vae, batch, device)
            lat_x0 = lgt.inversion(lat_src)
            sty = torch.full((bs,), 1, dtype=torch.long, device=device)
            _ = lgt.generation(lat_x0, sty)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f'{bs:5d} {"OOM":>8s}')
            break
        raise
    
    N = max(1, 80 // max(1, bs))
    peak_init = torch.cuda.max_memory_allocated() / 1024**2
    if peak_init > MAX_SAFE:
        print(f'{bs:5d} {"VRAM_HIGH":>8s}  peak={peak_init:.0f}MB > {MAX_SAFE}MB')
        break
    
    # ---- Model only ----
    with torch.autocast('cuda', dtype=torch.bfloat16):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N):
            lgt.generation(lat_x0, sty)
        torch.cuda.synchronize()
        t_m = (time.perf_counter() - t0) / N / bs
    
    # ---- VAE decode ----
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lat_gen = lgt.generation(lat_x0, sty)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N):
        decode_latent(vae, lat_gen, device)
    torch.cuda.synchronize()
    t_v = (time.perf_counter() - t0) / N / bs
    
    # ---- Pipelined: model inference on stream1, VAE decode on stream2 ----
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lat_gen_init = lgt.generation(lat_x0, sty)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N):
        with torch.cuda.stream(s1):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lat_gen2 = lgt.generation(lat_x0, sty)
        with torch.cuda.stream(s2):
            img2 = decode_latent(vae, lat_gen2, device)
    torch.cuda.synchronize()
    t_p = (time.perf_counter() - t0) / N / bs
    
    peak = torch.cuda.max_memory_allocated() / 1024**2
    e2e_ms = (t_m + t_v) * 1000
    pipe_ms = t_p * 1000
    sp = e2e_ms / pipe_ms if pipe_ms > 0 else 1
    
    if pipe_ms < best["pipe_ms"]:
        best = {"bs": bs, "pipe_ms": pipe_ms, "e2e_ms": e2e_ms}
    
    status = "OK"
    if peak > 5000: status = "WARM"
    if peak > 6500: status = "HOT"
    print(f'{bs:5d} {t_m*1000:8.2f} {t_v*1000:7.2f} {e2e_ms:7.2f} {pipe_ms:7.2f} {sp:7.2f}x {peak:7.0f} {status:>10s}')

# Report best
e2e_750 = best["e2e_ms"] * 750 / 1000
pipe_750 = best["pipe_ms"] * 750 / 1000
print(f"\n=== BEST: batch={best['bs']} | E2E={best['e2e_ms']:.1f}ms/img | pipe={best['pipe_ms']:.1f}ms/img ===")
print(f"750 images: E2E={e2e_750:.0f}s | pipelined={pipe_750:.0f}s")
print(f"Speedup vs current (batch=20, 85.4s): {85.4/max(pipe_750,0.1):.1f}x")

del lgt, vae; torch.cuda.empty_cache()
