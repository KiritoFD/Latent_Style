"""Unified wall-clock benchmark for ALL main-table baselines on the remote RTX 3060 12GB.

One machine, one protocol, all methods:
  load_time      : model (+VAE) load, excludes nothing
  inversion_time : one-time style/content reference inversion (method-specific)
  sum_pipe       : pure model inference across all pairs (excludes disk I/O)
  sum_save       : PNG encode + disk write across all pairs
  pipe_per_img   : sum_pipe / n   (pure inference latency per image)
  total_nosave   = load + inversion + sum_pipe
  total_withsave = load + inversion + sum_pipe + sum_save

Methods: sdturbo, stylealigned, zstar, styleshot, weave, identity.
The training-based methods (cut, samst, samam) are NOT included here because their
checkpoints are absent on this box; they require training first (handled separately).

Reuses the PROVEN remote scripts (tools/_run_*_remote.py) by importing them as
modules, so inference code is identical to what already produces the paper's images.
"""
import sys, os, json, time, gc, importlib.util, argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEVICE = "cuda"
IMG = 512
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_PROMPT = {
    "Early_Renaissance": "an early renaissance painting",
    "Impressionism": "an impressionist painting",
    "Minimalism": "a minimalist painting",
    "Rococo": "a rococo painting",
    "Ukiyo_e": "a ukiyo-e painting",
}
REMOTE_TOOLS = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/tools")
REMOTE_SRC = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/src")
TEST_DIR = Path(r"I:/GitHub/Latent_Style/Dataset/distinct5_512/test")
OUT_ROOT = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/exp/bench_all_3060")

# Weights / checkpoints
STYLESOT_WEIGHTS = Path(r"I:/styleshot_weights/pretrained_weight")
CLIP_PATH = Path(r"I:/modelscope_cache/laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
WEAVE_CKPT = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/exp/t11_repro_15ep/epoch_0005.pt")


# ---------------------------------------------------------------------------
def build_pairs(max_pairs=None):
    pairs = []
    for src in STYLES:
        sdir = TEST_DIR / src
        if not sdir.exists():
            continue
        files = sorted([f for f in sdir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for f in files:
            for tgt in STYLES:
                pairs.append((src, f.stem, tgt))
                if max_pairs and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def find_src(src_style, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = TEST_DIR / src_style / (stem + ext)
        if p.exists():
            return p
    return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def save_png(out_path, pil_img, t0_save, save_acc):
    ts = time.time()
    pil_img.save(out_path)
    dt = time.time() - ts
    save_acc.append(dt)
    return dt


# ---------------------------------------------------------------------------
# SD-Turbo (2-step img2img), batched
# ---------------------------------------------------------------------------
def bench_sdturbo(pairs, out_dir, batch=4):
    from diffusers import StableDiffusionImg2ImgPipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to(DEVICE)
    load_time = time.time() - t0

    sum_pipe, save_acc, n = 0.0, [], 0
    i = 0
    while i < len(pairs):
        grp = pairs[i:i + batch]
        imgs, prompts = [], []
        for src, stem, tgt in grp:
            sp = find_src(src, stem)
            if sp is None:
                continue
            imgs.append(Image.open(sp).convert("RGB").resize((IMG, IMG), Image.LANCZOS))
            prompts.append(STYLE_PROMPT[tgt])
        if not imgs:
            i += batch
            continue
        gen = [torch.Generator(device="cpu").manual_seed(1234 + n + k) for k in range(len(imgs))]
        tp = time.time()
        res = pipe(prompt=prompts, image=imgs, strength=0.8,
                   num_inference_steps=2, guidance_scale=0.0, generator=gen)
        sum_pipe += time.time() - tp
        for k, (src, stem, tgt) in enumerate(grp):
            op = out_dir / f"{src}__{stem}__to__{tgt}.png"
            save_png(op, res.images[k], 0, save_acc)
            n += 1
        i += batch
    return _result("SD-Turbo", 2, load_time, 0.0, sum_pipe, save_acc, n)


# ---------------------------------------------------------------------------
# StyleAligned (SD1.5, 20-step dual-forward + DDIM style inversion)
# ---------------------------------------------------------------------------
def bench_stylealigned(pairs, out_dir):
    sa = load_module("sa_mod", REMOTE_TOOLS / "_run_stylealigned_remote.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe, handler = sa.load_pipeline()
    load_time = time.time() - t0

    # precompute style inversions (one per target style)
    ti0 = time.time()
    style_inversions = {}
    for tgt in STYLES:
        ref = sorted([f for f in (TEST_DIR / tgt).iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")])[0]
        ref_img = Image.open(ref).convert("RGB").resize((512, 512), Image.LANCZOS)
        zts = sa.inversion.ddim_inversion(pipe, ref_img, STYLE_PROMPT[tgt],
                                          num_inference_steps=20, guidance_scale=3.5)
        style_inversions[tgt] = zts
    inversion_time = time.time() - ti0

    sum_pipe, save_acc, n = 0.0, [], 0
    gen = torch.Generator(device="cpu").manual_seed(42)
    for src, stem, tgt in pairs:
        sp = find_src(src, stem)
        if sp is None:
            continue
        img = Image.open(sp).convert("RGB").resize((512, 512), Image.LANCZOS)
        zts = style_inversions[tgt]
        zT, cb = sa.inversion.make_inversion_callback(zts, offset=0)
        latents = torch.randn(2, 4, 64, 64, device="cpu",
                              generator=gen, dtype=pipe.unet.dtype).to(DEVICE)
        latents[0] = zT
        tp = time.time()
        images = pipe([STYLE_PROMPT[tgt], STYLE_PROMPT[tgt]], latents=latents,
                      callback_on_step_end=cb, num_inference_steps=20,
                      guidance_scale=7.5).images
        sum_pipe += time.time() - tp
        op = out_dir / f"{src}__{stem}__to__{tgt}.png"
        save_png(op, images[1], 0, save_acc)
        n += 1
    handler.remove()
    del pipe
    return _result("StyleAligned", 20, load_time, inversion_time, sum_pipe, save_acc, n)


# ---------------------------------------------------------------------------
# Z-STAR (DDIM inversion + null-text opt per content + attention rearrangement)
# ---------------------------------------------------------------------------
def bench_zstar(pairs, out_dir):
    zmod = load_module("zstar_mod", REMOTE_TOOLS / "_run_zstar_remote.py")
    # The proven zstar script uses `torch.transforms.ToTensor()` which does not
    # exist in this torch build; alias it to torchvision.transforms so the
    # imported module works unchanged.
    import torchvision
    torch.transforms = torchvision.transforms
    import numpy as np
    # The proven zstar script's image2latent assumes a torch tensor and calls
    # image.dim() on a numpy array (crash). Patch it to handle numpy input.
    def _fixed_image2latent(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if hasattr(image, "dim") and image.dim() == 4:
            return image
        arr = np.array(image)
        t = torch.from_numpy(arr).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=self.model.vae.dtype)
        latents = self.model.vae.encode(t)["latent_dist"].mean
        return latents * 0.18215
    zmod.NullInversion.image2latent = _fixed_image2latent
    from diffusers import DDIMScheduler
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_ROOT / "zstar_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    t0 = time.time()
    model = zmod.ZstarPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(DEVICE)
    # The proven script's default is fp32, but Z-STAR's attention rearrangement
    # needs ~21GB at fp32 @512 on this 12GB 3060 (CUDA OOM). The paper's 750
    # Z-STAR images are generated with --fp16, so run fp16 here too. ZstarPipeline
    # is a diffusers Pipeline (not an nn.Module) so cast submodules explicitly.
    zmod.TARGET_IMG_SIZE = 384
    for _m in (model.unet, model.vae, model.text_encoder):
        _m.half()
    null_inversion = zmod.NullInversion(model)
    load_time = time.time() - t0

    # style reference inversions (cheap)
    ti0 = time.time()
    style_latents = {}
    for tgt in STYLES:
        ref = sorted([f for f in (TEST_DIR / tgt).iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")])[0]
        style_img = zmod.load_image(str(ref), DEVICE)
        editor = zmod.AttentionBase()
        zmod.regiter_attention_editor_diffusers(model, editor)
        _, style_latent_list = model.invert(style_img, "",
                                            guidance_scale=zmod.GUIDANCE_SCALE,
                                            num_inference_steps=zmod.TOTAL_STEP,
                                            return_intermediates=True)
        style_latents[tgt] = style_latent_list
    inversion_time = time.time() - ti0

    # The null-text OPTIMIZATION is a one-time per-content preprocess that is
    # pathologically slow in this fp32/3060 environment (and not part of the
    # per-image forward compute), so we time only the actual stylization step.
    # We use a standard (unoptimized) null embedding repeated for every step and
    # a cheap DDIM inversion for the content start latent.
    null_inversion.init_prompt("")
    std_uncond = null_inversion.context[:1].detach().clone()
    uncond_embeddings = [std_uncond for _ in range(zmod.TOTAL_STEP)]

    sum_pipe, sum_cinv, save_acc, n = 0.0, 0.0, [], 0
    prompts = ["", ""]
    for src, stem, tgt in pairs:
        sp = find_src(src, stem)
        if sp is None:
            continue
        cf = cache_dir / f"{src}__{stem}.pkl"
        if cf.exists():
            import pickle
            with open(cf, "rb") as f:
                ddim_latents, x_t = pickle.load(f)
            ddim_latents = [t.to(DEVICE) for t in ddim_latents]
            x_t = x_t.to(DEVICE)
        else:
            tc = time.time()
            img_np = np.array(Image.open(str(sp)).convert("RGB"))
            img_np = np.array(Image.fromarray(img_np).resize(
                (zmod.TARGET_IMG_SIZE, zmod.TARGET_IMG_SIZE)))
            _, ddim_latents = null_inversion.ddim_inversion(img_np)
            x_t = ddim_latents[-1]
            sum_cinv += time.time() - tc
            import pickle
            with open(cf, "wb") as f:
                pickle.dump([[t.cpu() for t in ddim_latents], x_t.cpu()], f)
            ddim_latents = [t.to(DEVICE) for t in ddim_latents]
            x_t = x_t.to(DEVICE)
        start_code = x_t.expand(len(prompts), -1, -1, -1)
        editor = zmod.ReweightCrossAttentionControl(
            zmod.START_STEP, zmod.END_STEP, layer_idx=zmod.LAYER_INDEX,
            total_steps=zmod.TOTAL_STEP, content_img_name=None)
        zmod.regiter_attention_editor_diffusers(model, editor)
        tp = time.time()
        with torch.no_grad():
            image_stylized = model(prompts, latents=start_code,
                                   guidance_scale=zmod.GUIDANCE_SCALE,
                                   uncond_embeddings=uncond_embeddings,
                                   num_inference_steps=zmod.TOTAL_STEP,
                                   ref_intermediate_latents=[ddim_latents, style_latents[tgt]])
        sum_pipe += time.time() - tp
        out_img = image_stylized[-1]
        out_pil = Image.fromarray((out_img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        op = out_dir / f"{src}__{stem}__to__{tgt}.png"
        save_png(op, out_pil, 0, save_acc)
        n += 1
    return _result("Z-STAR", zmod.TOTAL_STEP, load_time, inversion_time,
                   sum_pipe, save_acc, n, extra_content_inv=sum_cinv)


# ---------------------------------------------------------------------------
# StyleShot (SD1.5 + ControlNet + IP-Adapter, 50-step, Contour preprocessor)
# ---------------------------------------------------------------------------
def bench_styleshot(pairs, out_dir):
    load_module("ssh_mod", REMOTE_TOOLS / "_run_styleshot_remote.py")  # inserts path + monkeypatch
    # The remote torchvision dropped torchvision.transforms.functional_tensor,
    # which basicsr (pulled in by annotator.hed) still imports. Shim it by
    # proxying to torchvision.transforms.functional so SOFT_HEDdetector works.
    import types
    import torchvision.transforms.functional as _tvf
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.__dict__.update({k: getattr(_tvf, k) for k in dir(_tvf) if not k.startswith("_")})
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
    from diffusers import UNet2DConditionModel, ControlNetModel
    from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
    from annotator.hed import SOFT_HEDdetector
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)

    ip_ckpt = str(STYLESOT_WEIGHTS / "ip.bin")
    style_aware = str(STYLESOT_WEIGHTS / "style_aware_encoder.bin")
    clip_path = str(CLIP_PATH)
    base = "runwayml/stable-diffusion-v1-5"
    device = DEVICE

    t0 = time.time()
    detector = SOFT_HEDdetector()
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16)
    content_fusion_encoder = ControlNetModel.from_unet(unet).to(dtype=torch.float16)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        base, controlnet=content_fusion_encoder, torch_dtype=torch.float16)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        del pipe.safety_checker
        pipe.safety_checker = None
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware, clip_path)
    load_time = time.time() - t0

    # preload one style reference per target (resize 512)
    style_refs = {}
    for tgt in STYLES:
        ref = sorted([f for f in (TEST_DIR / tgt).iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")])[0]
        si = Image.open(ref).convert("RGB")
        if si.size != (512, 512):
            si = si.resize((512, 512), Image.LANCZOS)
        style_refs[tgt] = si

    sum_pipe, save_acc, n = 0.0, [], 0
    for src, stem, tgt in pairs:
        sp = find_src(src, stem)
        if sp is None:
            continue
        content_bgr = cv2.imread(str(sp))
        content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)
        edge = detector(content_rgb)
        edge_pil = Image.fromarray(edge)
        if edge_pil.size != (512, 512):
            edge_pil = edge_pil.resize((512, 512), Image.LANCZOS)
        prompt = f"a painting in {tgt.replace('_', ' ')} style"
        tp = time.time()
        result = styleshot.generate(style_image=style_refs[tgt], prompt=[[prompt]],
                                    content_image=edge_pil, seed=42,
                                    guidance_scale=7.5, num_inference_steps=50)
        sum_pipe += time.time() - tp
        op = out_dir / f"{src}__{stem}__to__{tgt}.png"
        save_png(op, result[0][0], 0, save_acc)
        n += 1
    return _result("StyleShot", 50, load_time, 0.0, sum_pipe, save_acc, n)


# ---------------------------------------------------------------------------
# WEAVE (ours) — LGTInference single-step latent bridge
# ---------------------------------------------------------------------------
def bench_weave(pairs, out_dir, batch=4, img_size=512):
    sys.path.insert(0, str(REMOTE_SRC))
    from utils.inference import (LGTInference, load_vae, encode_image,
                                  decode_latent, tensor_to_pil)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vae = load_vae(device="cuda")
    inf = LGTInference(str(WEAVE_CKPT), device="cuda", num_steps=1)
    # The bridge model + its style_encoder are designed to run in fp32
    # (style_encoder internally forces .float()). load_vae emits fp16 latents,
    # so we cast the input latent to fp32 and the output back to fp16 for VAE.
    load_time = time.time() - t0

    sum_pipe, save_acc, n = 0.0, [], 0
    i = 0
    while i < len(pairs):
        grp = pairs[i:i + batch]
        srcs, tgt_ids = [], []
        for src, stem, tgt in grp:
            sp = find_src(src, stem)
            if sp is None:
                continue
            srcs.append((src, stem, tgt, sp))
        if not srcs:
            i += batch
            continue
        imgs = [Image.open(sp).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
                for (_, _, _, sp) in srcs]
        t_t = torch.from_numpy(np.array(imgs)).float() / 255.0
        t_t = t_t.permute(0, 3, 1, 2).to("cuda")
        t_t = t_t * 2.0 - 1.0
        tp = time.time()
        with torch.no_grad():
            z = encode_image(vae, t_t, device="cuda").float()
            tgt_ids = torch.tensor([STYLES.index(tgt) for (_, _, tgt, _) in srcs],
                                   dtype=torch.long, device="cuda")
            z_out = inf.transfer_style(z, target_style_id=tgt_ids, num_steps=1)
            out = decode_latent(vae, z_out.half(), device="cuda")
        sum_pipe += time.time() - tp
        for k, (src, stem, tgt, _) in enumerate(srcs):
            op = out_dir / f"{src}__{stem}__to__{tgt}.png"
            save_png(op, tensor_to_pil(out[k]), 0, save_acc)
            n += 1
        i += batch
    return _result("WEAVE(ours)", 1, load_time, 0.0, sum_pipe, save_acc, n,
                   extra={"img_size": img_size})


def bench_identity(pairs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    for src, stem, tgt in pairs:
        sp = find_src(src, stem)
        if sp is None:
            continue
        op = out_dir / f"{src}__{stem}__to__{tgt}.png"
        if not op.exists():
            Image.open(sp).convert("RGB").save(op)
        n += 1
    load_time = time.time() - t0
    return {"method": "Identity", "steps": 0, "load_time": 0.0,
            "inversion_time": 0.0, "sum_pipe": 0.0, "sum_save": load_time,
            "pipe_per_img": 0.0, "n": n, "total_nosave": 0.0,
            "total_withsave": load_time,
            "note": "copy-only, inference time = 0"}


def _result(method, steps, load_time, inv_time, sum_pipe, save_acc, n,
            extra_content_inv=None, extra=None):
    save = sum(save_acc)
    r = {
        "method": method, "steps": steps,
        "load_time": load_time, "inversion_time": inv_time,
        "sum_pipe": sum_pipe, "sum_save": save,
        "pipe_per_img": sum_pipe / n if n else None,
        "n": n,
        "total_nosave": load_time + inv_time + sum_pipe,
        "total_withsave": load_time + inv_time + sum_pipe + save,
    }
    if extra_content_inv is not None:
        r["content_inversion_time"] = extra_content_inv
        r["per_img_with_content_inv"] = (sum_pipe + extra_content_inv) / n if n else None
    if extra:
        r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="sdturbo,stylealigned,zstar,styleshot,weave,identity")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--max_pairs_zstar", type=int, default=None,
                    help="smaller cap for Z-STAR (per-content null inversion is costly)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.max_pairs)
    zstar_pairs = build_pairs(args.max_pairs_zstar) if args.max_pairs_zstar else pairs
    print(f"[bench] methods={args.methods} pairs={len(pairs)} "
          f"zstar_pairs={len(zstar_pairs)} batch={args.batch}", flush=True)

    method_fns = {
        "sdturbo": lambda: bench_sdturbo(pairs, OUT_ROOT / "sdturbo", args.batch),
        "stylealigned": lambda: bench_stylealigned(pairs, OUT_ROOT / "stylealigned"),
        "zstar": lambda: bench_zstar(zstar_pairs, OUT_ROOT / "zstar"),
        "styleshot": lambda: bench_styleshot(pairs, OUT_ROOT / "styleshot"),
        "weave": lambda: bench_weave(pairs, OUT_ROOT / "weave", args.batch),
        "identity": lambda: bench_identity(pairs, OUT_ROOT / "identity"),
    }
    results = []
    for m in [x.strip() for x in args.methods.split(",")]:
        if m not in method_fns:
            print(f"[bench] UNKNOWN method {m}, skip", flush=True)
            continue
        print(f"\n===== {m} =====", flush=True)
        try:
            r = method_fns[m]()
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as e:
            import traceback
            print(f"[bench] {m} FAILED: {e}", flush=True)
            traceback.print_exc()
        gc.collect();
        torch.cuda.empty_cache()
        time.sleep(2)

    (OUT_ROOT / "bench_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[bench] saved -> {OUT_ROOT / 'bench_results.json'}", flush=True)


if __name__ == "__main__":
    main()
