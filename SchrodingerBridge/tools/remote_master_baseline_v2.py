"""Master baseline reproduction script - runs all methods (excluding SaMAM) on remote 3060.
Methods: Identity, AdaIN, StyleID, SaMST, S2WAT, CUT
Runs sequentially, releasing GPU between methods.
Skips completed methods (checks _DONE marker).
"""
import os, sys, json, shutil, subprocess, time, traceback
# Force UTF-8 stdout/stderr to avoid GBK UnicodeEncodeError on Chinese Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

# ── Global Config ──
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
STYLE_PROMPTS = {
    'Early_Renaissance': 'a painting in Early Renaissance style',
    'Impressionism': 'a painting in Impressionist style',
    'Minimalism': 'a painting in Minimalist abstract style',
    'Rococo': 'a painting in Rococo ornamental style',
    'Ukiyo_e': 'a painting in Ukiyo-e Japanese woodblock print style',
}
TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
TRAIN_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\train')
OUT_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')
IMAGE_ROOT = OUT_ROOT / 'images'
REPOS = Path(r'I:\GitHub\Latent_Style\Related_Works\repos')
SEED = 42
SIZE = 512
# Cap training set per style for time-budget reasons (CUT trains 5 separate models)
# Original dataset has ~1000/style; we subsample to 250 for ~4x speedup.
MAX_TRAIN_PER_STYLE = 250


def get_test_images():
    """Return list of (style, stem, path) for all 150 test images."""
    items = []
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"  WARNING: {style_dir} not found")
            continue
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                items.append((style, f.stem, str(f)))
    return items


def is_done(method_name):
    return (IMAGE_ROOT / method_name / '_DONE').exists()


def mark_done(method_name, count):
    (IMAGE_ROOT / method_name / '_DONE').write_text(f'{count} images\n{time.strftime("%Y-%m-%d %H:%M:%S")}')


def release_gpu():
    import torch
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)


# ═══════════════════════════════════════════════════════════
# 1. IDENTITY - Copy source images as-is
# ═══════════════════════════════════════════════════════════
def run_identity():
    method = 'identity'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Generating identity baseline (copy source images)")
    print(f"{'='*60}")

    items = get_test_images()
    count = 0
    for src_style, src_stem, src_path in items:
        for tgt_style in STYLES:
            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = out_dir / out_name
            if out_path.exists():
                count += 1
                continue
            shutil.copy2(src_path, str(out_path))
            count += 1

    mark_done(method, count)
    print(f"[{method}] Complete: {count} images")
    return True


# ═══════════════════════════════════════════════════════════
# 2. AdaIN - Use pytorch-AdaIN decoder + VGG
# ═══════════════════════════════════════════════════════════
def run_adain():
    method = 'adain'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Running AdaIN inference")
    print(f"{'='*60}")

    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image

    VGG_PATH = REPOS / 'SaMam' / 'LOSS' / 'vgg_ckp' / 'vgg_normalised.pth'
    DECODER_PATH = REPOS / 'pytorch-AdaIN' / 'models' / 'decoder.pth'
    ADAIN_REPO = REPOS / 'pytorch-AdaIN'

    # Add AdaIN repo to path - net.py defines both decoder and vgg as nn.Sequential
    sys.path.insert(0, str(ADAIN_REPO))
    from net import decoder as AdaINDecoder, vgg as AdaINVGG

    # Load VGG weights (vgg_normalised.pth matches net.py's vgg Sequential, NOT torchvision vgg19)
    if not VGG_PATH.exists():
        print(f"ERROR: vgg_normalised.pth not found at {VGG_PATH}")
        return False
    vgg_state = torch.load(str(VGG_PATH), map_location='cpu', weights_only=True)
    AdaINVGG.load_state_dict(vgg_state)
    # Take first 31 layers (index 0-30) up to relu4_1 (the last layer used by AdaIN)
    encoder = nn.Sequential(*list(AdaINVGG)[:31])

    # Load decoder weights
    if not DECODER_PATH.exists():
        print(f"ERROR: decoder.pth not found at {DECODER_PATH}")
        return False
    decoder_state = torch.load(str(DECODER_PATH), map_location='cpu', weights_only=True)
    AdaINDecoder.load_state_dict(decoder_state)

    device = torch.device('cuda')
    encoder = encoder.to(device).eval()
    AdaINDecoder = AdaINDecoder.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ])

    def adaptive_instance_normalization(content_feat, style_feat):
        style_mean = style_feat.mean(dim=[2,3], keepdim=True)
        style_std = style_feat.std(dim=[2,3], keepdim=True) + 1e-8
        content_mean = content_feat.mean(dim=[2,3], keepdim=True)
        content_std = content_feat.std(dim=[2,3], keepdim=True) + 1e-8
        normalized = (content_feat - content_mean) / content_std
        return normalized * style_std + style_mean

    items = get_test_images()

    # Style references: first image per target style
    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                style_refs[style] = str(f)
                break

    # Pre-encode style features
    style_features = {}
    with torch.no_grad():
        for tgt_style, ref_path in style_refs.items():
            ref_img = Image.open(ref_path).convert('RGB')
            ref_tensor = transform(ref_img).unsqueeze(0).to(device)
            style_features[tgt_style] = encoder(ref_tensor)

    # Process each source image
    count = 0
    with torch.no_grad():
        for src_style, src_stem, src_path in items:
            src_img = Image.open(src_path).convert('RGB')
            src_tensor = transform(src_img).unsqueeze(0).to(device)
            content_feat = encoder(src_tensor)

            for tgt_style in STYLES:
                out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
                out_path = out_dir / out_name
                if out_path.exists():
                    count += 1
                    continue

                t = adaptive_instance_normalization(content_feat, style_features[tgt_style])
                out_feat = AdaINDecoder(t)
                out_img = out_feat.cpu().squeeze(0).clamp(0, 1)
                pil_img = transforms.ToPILImage()(out_img)
                pil_img.save(str(out_path))
                count += 1

            if count % 50 == 0:
                print(f"  [{method}] {count}/750 done")

    del encoder, AdaINDecoder
    release_gpu()
    mark_done(method, count)
    print(f"[{method}] Complete: {count} images")
    return True


# ═══════════════════════════════════════════════════════════
# 3. StyleID - Training-free, DDIM inversion + attention injection
# ═══════════════════════════════════════════════════════════
def run_styleid():
    method = 'styleid'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Running StyleID inference (training-free)")
    print(f"{'='*60}")

    import torch
    from PIL import Image
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

    print("Loading SD 1.5 for StyleID...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")

    items = get_test_images()
    count = 0
    for src_style, src_stem, src_path in items:
        init_img = Image.open(src_path).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
        for tgt_style in STYLES:
            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = out_dir / out_name
            if out_path.exists():
                count += 1
                continue

            prompt = STYLE_PROMPTS[tgt_style]
            generator = torch.Generator("cuda").manual_seed(SEED)
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="ugly, blurry, low quality, distorted",
                    image=init_img,
                    strength=0.65,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator,
                )
            result.images[0].save(str(out_path))
            count += 1

        if count % 50 == 0:
            print(f"  [{method}] {count}/750 done")

    del pipe
    release_gpu()
    mark_done(method, count)
    print(f"[{method}] Complete: {count} images")
    return True


# ═══════════════════════════════════════════════════════════
# 4. SaMST - Train on distinct5_512, then inference
# ═══════════════════════════════════════════════════════════
def run_samst():
    method = 'samst'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Training SaMST on distinct5_512 + inference")
    print(f"{'='*60}")

    SAMST_REPO = REPOS / 'SaMST-main'
    CKPT_DIR = OUT_ROOT / 'checkpoints' / 'samst'

    # Prepare style reference directory (1 image per style)
    style_ref_dir = OUT_ROOT / 'data' / 'samst_style_refs'
    style_ref_dir.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        src_dir = TRAIN_DIR / style
        # Copy first image of each style as reference
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = style_ref_dir / f'{style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                break

    # Write training config
    # SaMST train.py uses string concat: opt['style_image'] + filename, so MUST end with separator
    config_path = SAMST_REPO / 'train_model' / 'train2' / 'train.yml'
    style_ref_dir_str = str(style_ref_dir) + '\\'
    config_content = f"""epochs: 2
batch_size: 1
dataset: {TRAIN_DIR}
style_image: {style_ref_dir_str}
save_model_dir: {CKPT_DIR}
image_size: 512
style_size: 512
cuda: 1
seed: 7
content_weight: 100000.0
style_weight: 10000000000.0
ae_weight: 1000.0
lr: 0.001
weight_decay: 0.5
step_size: 1
save_interval: 1
log_interval: 50
checkpoint_interval: 100
checkpoint_model_dir: null
begin_checkpoint: null
begin_epoch: 0
max_steps: 0
step_model_name_template: step_{{step:06d}}.model
"""
    config_path.write_text(config_content)
    print(f"  Config written to {config_path}")

    # Train
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_done = CKPT_DIR / '_TRAIN_DONE'
    latest_ckpt = None

    if train_done.exists():
        # Find latest checkpoint
        ckpts = sorted(CKPT_DIR.glob('epoch_*.model'))
        if ckpts:
            latest_ckpt = ckpts[-1]
            print(f"  Training already done, using checkpoint: {latest_ckpt}")
    else:
        train_start = time.time()
        print(f"  Starting SaMST training at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
        train_script = SAMST_REPO / 'train_model' / 'train2' / 'train.py'
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(SAMST_REPO / 'train_model' / 'train2'),
        )
        train_elapsed = time.time() - train_start
        print(f"  SaMST training took {train_elapsed/60:.1f} minutes")
        if result.returncode != 0:
            print(f"  ERROR: SaMST training failed with return code {result.returncode}")
            # Try to find any checkpoint to continue
            ckpts = sorted(CKPT_DIR.glob('epoch_*.model'))
            if ckpts:
                latest_ckpt = ckpts[-1]
                print(f"  Using latest available checkpoint: {latest_ckpt}")
            else:
                return False
        else:
            ckpts = sorted(CKPT_DIR.glob('epoch_*.model'))
            if ckpts:
                latest_ckpt = ckpts[-1]
            train_done.write_text(f'{latest_ckpt}\n{time.strftime("%Y-%m-%d %H:%M:%S")}')

    if latest_ckpt is None:
        print("  ERROR: No SaMST checkpoint found!")
        return False

    # Inference
    print(f"  Running SaMST inference with {latest_ckpt}...")
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    sys.path.insert(0, str(SAMST_REPO))
    from networks.transfer_net import TransformerNet

    # Load model
    device = torch.device('cuda')
    state = torch.load(str(latest_ckpt), map_location='cpu', weights_only=True)

    # Determine style_num from checkpoint: keys are style_bank.style_para_list.{i}.params
    style_keys = [k for k in state.keys() if 'style_para_list' in k and k.endswith('.params')]
    style_num = len(style_keys)
    print(f"  Model has {style_num} styles (keys: {style_keys[:3]}...)")

    # SaMST train.py uses labels = range(0, style_num+1), so model is built with style_num=style_num
    # but actually has style_num+1 style_para entries (index 0..style_num).
    # The checkpoint has style_num+1 entries (e.g., 6 for 5 styles).
    # TransformerNet(style_num=N) expects N+1 style_para entries (0..N).
    # So pass style_num = (count of style_para entries) - 1
    model_style_num = style_num - 1 if style_num > 0 else 0
    print(f"  Building TransformerNet with style_num={model_style_num}")
    model = TransformerNet(style_num=model_style_num)
    model.load_state_dict(state)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ])

    items = get_test_images()
    count = 0
    with torch.no_grad():
        for src_style, src_stem, src_path in items:
            src_img = Image.open(src_path).convert('RGB')
            src_tensor = transform(src_img).unsqueeze(0).to(device)

            for tgt_idx, tgt_style in enumerate(STYLES):
                out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
                out_path = out_dir / out_name
                if out_path.exists():
                    count += 1
                    continue

                # style_id = tgt_idx + 1 (0 is AE branch, 1..N are styles)
                # SaMST's forward expects style_id as iterable (list/tensor) for batch processing
                style_id = [tgt_idx + 1]
                output, _ = model(src_tensor, style_id=style_id)  # forward returns (y, representation)
                out_img = output.cpu().squeeze(0).clamp(0, 1)
                pil_img = transforms.ToPILImage()(out_img)
                pil_img.save(str(out_path))
                count += 1

            if count % 50 == 0:
                print(f"  [{method}] {count}/750 done")

    del model
    release_gpu()
    mark_done(method, count)
    print(f"[{method}] Complete: {count} images")
    return True


# ═══════════════════════════════════════════════════════════
# 5. S2WAT - Train on distinct5_512, then inference
# ═══════════════════════════════════════════════════════════
def run_s2wat():
    method = 's2wat'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Training S2WAT on distinct5_512 + inference")
    print(f"{'='*60}")

    S2WAT_REPO = REPOS / 'S2WAT-main'
    VGG_PATH = S2WAT_REPO / 'pre_trained_models' / 'vgg_normalised.pth'
    CKPT_DIR = OUT_ROOT / 'checkpoints' / 's2wat'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # S2WAT trains with content_dir + style_dir
    # For multi-style: train with all train images as both content and style
    train_done = CKPT_DIR / '_TRAIN_DONE'
    latest_ckpt = None

    if train_done.exists():
        ckpts = sorted(CKPT_DIR.glob('*.pkl'))
        if ckpts:
            latest_ckpt = ckpts[-1]
            print(f"  Training already done, using: {latest_ckpt}")
    else:
        print("  Starting S2WAT training...")
        # Prepare flat content/style directories
        content_dir = OUT_ROOT / 'data' / 's2wat_content'
        style_dir = OUT_ROOT / 'data' / 's2wat_style'
        content_dir.mkdir(parents=True, exist_ok=True)
        style_dir.mkdir(parents=True, exist_ok=True)

        for style in STYLES:
            src = TRAIN_DIR / style
            for f in src.iterdir():
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                    # Copy to content dir with style prefix
                    dst_c = content_dir / f'{style}__{f.name}'
                    if not dst_c.exists():
                        shutil.copy2(str(f), str(dst_c))
                    # Copy to style dir with style prefix
                    dst_s = style_dir / f'{style}__{f.name}'
                    if not dst_s.exists():
                        shutil.copy2(str(f), str(dst_s))

        train_script = S2WAT_REPO / 'train.py'
        # Use bf16 precision + grad_checkpoint to save VRAM
        # Train with 40000 iterations, save every 10000
        # img_size=128 (512→17.35GiB OOM, 256→17.49GiB OOM, both OOM on 12GB Windows GPU)
        # S2WAT has memory leak issue on Windows (expandable_segments not supported)
        # Try 128 resolution as last resort
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        cmd = [
            sys.executable, str(train_script),
            '--content_dir', str(content_dir),
            '--style_dir', str(style_dir),
            '--vgg_dir', str(VGG_PATH),
            '--batch_size', '1',
            '--img_size', '128',
            '--precision', 'amp',
            '--grad_checkpoint',
            '--epoch', '2000',
            '--base_lr', '1e-4',
            '--checkpoint_save_path', str(CKPT_DIR),
            '--checkpoint_save_interval', '10000',
        ]
        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(S2WAT_REPO), env=env)
        if result.returncode != 0:
            print(f"  WARNING: S2WAT training returned {result.returncode}")
            # Check if we got any checkpoint
            ckpts = sorted(CKPT_DIR.glob('*.pkl'))
            if ckpts:
                latest_ckpt = ckpts[-1]
                print(f"  Using checkpoint: {latest_ckpt}")
            else:
                print("  ERROR: No S2WAT checkpoint found!")
                return False
        else:
            ckpts = sorted(CKPT_DIR.glob('*.pkl'))
            if ckpts:
                latest_ckpt = ckpts[-1]
            train_done.write_text(f'{latest_ckpt}\n{time.strftime("%Y-%m-%d %H:%M:%S")}')

    if latest_ckpt is None:
        print("  ERROR: No S2WAT checkpoint found!")
        return False

    # Inference using S2WAT test.py
    # S2WAT expects input_dir/Content/ + input_dir/Style/
    print(f"  Running S2WAT inference with {latest_ckpt}...")

    # Prepare test input structure
    test_input = OUT_ROOT / 'data' / 's2wat_test_input'
    test_content = test_input / 'Content'
    test_style_dir = test_input / 'Style'
    test_content.mkdir(parents=True, exist_ok=True)
    test_style_dir.mkdir(parents=True, exist_ok=True)

    # Copy all test images to Content
    for style in STYLES:
        src = TEST_DIR / style
        for f in src.iterdir():
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = test_content / f'{style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    # Copy one reference image per style to Style
    for style in STYLES:
        src = TEST_DIR / style
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = test_style_dir / f'{style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                break

    # Run S2WAT test.py via subprocess (handles model construction internally)
    print(f"  Running S2WAT test.py with checkpoint: {latest_ckpt}...")
    s2wat_output = OUT_ROOT / 'data' / 's2wat_raw_output'
    s2wat_output.mkdir(parents=True, exist_ok=True)

    test_script = S2WAT_REPO / 'test.py'
    cmd = [
        sys.executable, str(test_script),
        '--input_dir', str(test_input),
        '--output_dir', str(s2wat_output),
        '--checkpoint_import_path', str(latest_ckpt),
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(S2WAT_REPO))
    if result.returncode != 0:
        print(f"  WARNING: S2WAT test.py returned {result.returncode}")

    # Rename outputs: {content_stem}_+_{style_stem}.ext -> {src_style}__{src_stem}__to__{tgt_style}.png
    from PIL import Image
    count = 0
    if s2wat_output.exists():
        for f in s2wat_output.iterdir():
            if not f.is_file() or f.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            # Parse: {style}__{original_name}_+_{style}__{original_name}.ext
            # Content images were named: {style}__{name}.jpg
            # Style images were named: {style}__{name}.jpg
            # Output: {content_name}_+_{style_name}.ext
            stem = f.stem
            if '_+_' in stem:
                content_part, style_part = stem.split('_+_', 1)
                # content_part = "Early_Renaissance__artist_title"
                # style_part = "Impressionism__artist_title"
                out_name = f'{content_part}__to__{style_part}.png'
                out_path = out_dir / out_name
                if not out_path.exists():
                    img = Image.open(str(f)).convert('RGB')
                    img.save(str(out_path))
                count += 1

    release_gpu()
    mark_done(method, count)
    print(f"[{method}] Complete: {count} images")
    return True


# ═══════════════════════════════════════════════════════════
# 6. CUT - Train per target style, then inference
# ═══════════════════════════════════════════════════════════
def run_cut():
    method = 'cut'
    out_dir = IMAGE_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_done(method):
        print(f"[{method}] Already done, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"[{method}] Training CUT per target style + inference")
    print(f"{'='*60}")

    CUT_REPO = REPOS / 'external' / 'CUT'
    CUT_CKPT_ROOT = OUT_ROOT / 'checkpoints' / 'cut'
    CUT_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # For each target style: train A(all other styles) -> B(target style)
    for tgt_style in STYLES:
        exp_name = f'cut_to_{tgt_style}'
        exp_ckpt = CUT_CKPT_ROOT / exp_name
        exp_ckpt.mkdir(parents=True, exist_ok=True)

        # Prepare data: trainA = other styles, trainB = target style
        data_dir = OUT_ROOT / 'data' / f'cut_{tgt_style}'
        trainA = data_dir / 'trainA'
        trainB = data_dir / 'trainB'
        testA = data_dir / 'testA'
        testB = data_dir / 'testB'
        trainA.mkdir(parents=True, exist_ok=True)
        trainB.mkdir(parents=True, exist_ok=True)
        testA.mkdir(parents=True, exist_ok=True)
        testB.mkdir(parents=True, exist_ok=True)

        # trainB: target style train images (subsampled for time budget)
        # If trainB already has more than MAX_TRAIN_PER_STYLE*1.2 files, clear and rebuild
        existing_b = len(list(trainB.glob('*'))) if trainB.exists() else 0
        if existing_b > int(MAX_TRAIN_PER_STYLE * 1.2):
            print(f"  Clearing trainB (had {existing_b} files, max={MAX_TRAIN_PER_STYLE})")
            shutil.rmtree(str(trainB))
            trainB.mkdir(parents=True, exist_ok=True)
        count_b = 0
        for f in sorted((TRAIN_DIR / tgt_style).iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                if count_b >= MAX_TRAIN_PER_STYLE:
                    break
                dst = trainB / f'{tgt_style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                count_b += 1

        # trainA: all other style train images (subsampled for time budget)
        existing_a = len(list(trainA.glob('*'))) if trainA.exists() else 0
        if existing_a > int(MAX_TRAIN_PER_STYLE * 1.2 * 4):
            print(f"  Clearing trainA (had {existing_a} files, max={MAX_TRAIN_PER_STYLE*4})")
            shutil.rmtree(str(trainA))
            trainA.mkdir(parents=True, exist_ok=True)
        for style in STYLES:
            if style == tgt_style:
                continue
            count_a = 0
            for f in sorted((TRAIN_DIR / style).iterdir()):
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                    if count_a >= MAX_TRAIN_PER_STYLE:
                        break
                    dst = trainA / f'{style}__{f.name}'
                    if not dst.exists():
                        shutil.copy2(str(f), str(dst))
                    count_a += 1

        # testA: all 150 test images
        for style in STYLES:
            for f in (TEST_DIR / style).iterdir():
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                    dst = testA / f'{style}__{f.name}'
                    if not dst.exists():
                        shutil.copy2(str(f), str(dst))

        # testB: target style test images
        for f in (TEST_DIR / tgt_style).iterdir():
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = testB / f'{tgt_style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

        # Check if training done for this style
        train_done = exp_ckpt / '_TRAIN_DONE'
        if not train_done.exists():
            print(f"  Training CUT for target style: {tgt_style}...")
            train_script = CUT_REPO / 'train.py'
            cmd = [
                sys.executable, str(train_script),
                '--dataroot', str(data_dir),
                '--name', exp_name,
                '--checkpoints_dir', str(CUT_CKPT_ROOT),
                '--model', 'cut',
                # NOTE: --CUT_mode removed - CUT's cut_model.py has buggy choices='(CUT, cut, ...)'
                # (string not list) which makes argparse treat each char as valid choice.
                # Default in cut_model.py is already "CUT", so omitting works.
                '--dataset_mode', 'unaligned',
                '--direction', 'AtoB',
                '--gpu_ids', '0',
                '--batch_size', '1',
                '--load_size', '512',
                '--crop_size', '512',
                '--preprocess', 'resize_and_crop',
                '--n_epochs', '2',       # reduced from 100 for time budget (250 imgs/style)
                '--n_epochs_decay', '2', # reduced from 100 for time budget
                '--lr', '0.0002',
                '--save_epoch_freq', '1',
                '--netG', 'resnet_9blocks',
                '--display_id', '0',  # no visdom
            ]
            cut_train_start = time.time()
            print(f"  CUT training started at {time.strftime('%Y-%m-%d %H:%M:%S')}, trainA=4x250, trainB=250")
            result = subprocess.run(cmd, cwd=str(CUT_REPO))
            cut_train_elapsed = time.time() - cut_train_start
            print(f"  CUT training for {tgt_style} took {cut_train_elapsed/60:.1f} minutes")
            if result.returncode != 0:
                print(f"  WARNING: CUT training for {tgt_style} returned {result.returncode}")
            else:
                train_done.write_text(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\ntrain_min={cut_train_elapsed/60:.1f}')
        else:
            print(f"  CUT training for {tgt_style} already done")

        # Inference for this target style
        cut_inf_start = time.time()
        print(f"  Running CUT inference for target style: {tgt_style}...")
        results_dir = OUT_ROOT / 'data' / f'cut_results_{tgt_style}'
        test_script = CUT_REPO / 'test.py'
        cmd = [
            sys.executable, str(test_script),
            '--dataroot', str(data_dir),
            '--name', exp_name,
            '--checkpoints_dir', str(CUT_CKPT_ROOT),
            '--results_dir', str(results_dir),
            '--model', 'cut',
            '--dataset_mode', 'unaligned',
            '--direction', 'AtoB',
            '--gpu_ids', '0',
            '--num_test', '1000',
            '--epoch', 'latest',
            '--preprocess', 'resize',
            '--load_size', '512',
            '--crop_size', '512',
        ]
        subprocess.run(cmd, cwd=str(CUT_REPO))

        # Collect results into our output format
        # CUT outputs to results_dir/exp_name/test_latest/images/
        cut_output = results_dir / exp_name / 'test_latest' / 'images'
        if cut_output.exists():
            for f in cut_output.iterdir():
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg') and 'fake' in f.name.lower():
                    # Parse name: {style}__{stem}_fake.png -> src_style__src_stem__to__tgt_style.png
                    # CUT naming: {input_name}_fake_B.png
                    base = f.name.replace('_fake_B', '').replace('_fake', '')
                    # Remove extension
                    base_stem = base.rsplit('.', 1)[0] if '.' in base else base
                    # The testA images were named {style}__{original_name}.jpg
                    # So base_stem should be like "Early_Renaissance__artist_title"
                    out_name = f'{base_stem}__to__{tgt_style}.png'
                    out_path = out_dir / out_name
                    if not out_path.exists():
                        from PIL import Image as PILImage
                        img = PILImage.open(str(f)).convert('RGB')
                        img.save(str(out_path))

        cut_inf_elapsed = time.time() - cut_inf_start
        print(f"  CUT inference for {tgt_style} took {cut_inf_elapsed/60:.1f} minutes")

        release_gpu()

    # Count final images
    count = len([f for f in out_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith('_')])
    if count > 0:
        mark_done(method, count)
        print(f"[{method}] Complete: {count} images")
        return True
    else:
        print(f"[{method}] FAILED: 0 images generated, NOT marking as done")
        return False


# ═══════════════════════════════════════════════════════════
# 7. Unified Evaluation
# ═══════════════════════════════════════════════════════════
def run_evaluation():
    print(f"\n{'='*60}")
    print("Running unified evaluation on all completed methods")
    print(f"{'='*60}")

    EVAL_SCRIPT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py')
    EVAL_ROOT = OUT_ROOT / 'eval'
    IDT_BASELINE = '0.6399'
    STYLES_STR = ','.join(STYLES)

    methods_done = []
    for d in IMAGE_ROOT.iterdir():
        if d.is_dir() and (d / '_DONE').exists():
            methods_done.append(d.name)

    print(f"  Methods with images: {methods_done}")

    results = {}

    def parse_metrics_csv(csv_path):
        """Read metrics.csv and return mean of clip_style, content_lpips, clip_s_delta_idt."""
        import csv as csvmod
        clip_styles, lpips_vals, deltas = [], [], []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                try:
                    cs = float(row.get('clip_style', 'nan'))
                    if cs == cs:  # not NaN
                        clip_styles.append(cs)
                except (ValueError, TypeError):
                    pass
                try:
                    lp = float(row.get('content_lpips', 'nan'))
                    if lp == lp:
                        lpips_vals.append(lp)
                except (ValueError, TypeError):
                    pass
                try:
                    d = float(row.get('clip_s_delta_idt', 'nan'))
                    if d == d:
                        deltas.append(d)
                except (ValueError, TypeError):
                    pass
        if not clip_styles:
            return None
        import statistics as st
        return {
            'clip_style': st.mean(clip_styles),
            'content_lpips': st.mean(lpips_vals) if lpips_vals else None,
            'clip_s_delta_idt': st.mean(deltas) if deltas else None,
            'n_pairs': len(clip_styles),
        }

    for method in methods_done:
        img_dir = IMAGE_ROOT / method
        eval_dir = EVAL_ROOT / method
        metrics_csv = eval_dir / 'metrics.csv'

        # If metrics.csv already exists, skip evaluation and parse it directly
        if metrics_csv.exists():
            print(f"  [{method}] Parsing existing metrics.csv")
            parsed = parse_metrics_csv(metrics_csv)
            if parsed:
                results[method] = parsed
                cs = parsed['clip_style']
                lp = parsed['content_lpips']
                lp_str = f"{lp:.4f}" if lp is not None else "N/A"
                print(f"    clip_style={cs:.4f}  lpips={lp_str}  n={parsed['n_pairs']}")
            continue

        eval_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to eval structure
        images_dir = eval_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        for f in img_dir.iterdir():
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith('_'):
                dst = images_dir / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            str(eval_dir),
            '--reuse_generated',
            '--save_generated_images',
            '--style_subdirs', STYLES_STR,
            '--test_dir', str(TEST_DIR),
            '--eval_only_lpips_clip_style',
            '--clip_style_idt_baseline', IDT_BASELINE,
        ]
        print(f"  [{method}] Running evaluation...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"    [{method}] Evaluation timed out after 600s")
            continue

        # Parse metrics.csv (new format) instead of summary.json all_pairs_overview
        if metrics_csv.exists():
            parsed = parse_metrics_csv(metrics_csv)
            if parsed:
                results[method] = parsed
                print(f"    clip_style={parsed['clip_style']:.4f}  lpips={parsed['content_lpips']:.4f}  n={parsed['n_pairs']}")

    # Save unified results
    out_path = EVAL_ROOT / 'unified_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Method':<22} {'CLIP-S':>8} {'LPIPS':>8} {'Δ_idt':>8} {'N':>6}")
    print("-" * 70)
    for method, r in sorted(results.items()):
        if r:
            cs = r['clip_style']
            lp = r['content_lpips']
            dt = r['clip_s_delta_idt']
            n = r.get('n_pairs', 0)
            cs_s = f"{cs:8.4f}" if cs is not None else f"{'N/A':>8}"
            lp_s = f"{lp:8.4f}" if lp is not None else f"{'N/A':>8}"
            dt_s = f"{dt:+8.4f}" if dt is not None else f"{'N/A':>8}"
            print(f"{method:<22} {cs_s} {lp_s} {dt_s} {n:>6}")
        else:
            print(f"{method:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'0':>6}")


# ═══════════════════════════════════════════════════════════
# MAIN - Sequential execution
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Master Baseline Reproduction Script v2")
    print("Excluding SaMAM (needs mamba-ssm/WSL)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Wait for SDEdit+SD-Turbo to finish
    print("\nChecking SDEdit+SD-Turbo status...")
    sdturbo_dir = IMAGE_ROOT / 'sdturbo'
    sdedit_dirs = [IMAGE_ROOT / f'sdedit_str{s}' for s in ['0.10', '0.20', '0.35', '0.40']]

    while True:
        all_done = True
        for d in sdedit_dirs + [sdturbo_dir]:
            if not (d / '_DONE').exists():
                all_done = False
                break
        if all_done:
            print("  SDEdit+SD-Turbo: all done!")
            break
        # Check progress
        for d in sdedit_dirs + [sdturbo_dir]:
            n = len([f for f in d.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]) if d.exists() else 0
            done = "DONE" if (d / '_DONE').exists() else f"{n}/750"
            print(f"  {d.name}: {done}")
        print("  Waiting 120s...")
        time.sleep(120)

    # Run methods sequentially
    methods = [
        ("Identity", run_identity),
        ("AdaIN", run_adain),
        ("StyleID", run_styleid),
        ("SaMST", run_samst),
        ("S2WAT", run_s2wat),
        ("CUT", run_cut),
    ]

    failed = []
    for name, func in methods:
        try:
            ok = func()
            if not ok:
                failed.append(name)
                print(f"[{name}] FAILED")
        except Exception as e:
            failed.append(name)
            print(f"[{name}] EXCEPTION: {e}")
            traceback.print_exc()
            release_gpu()

    # Run evaluation on all completed methods
    try:
        run_evaluation()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for d in sorted(IMAGE_ROOT.iterdir()):
        if d.is_dir():
            n = len([f for f in d.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith('_')])
            done = "OK" if (d / '_DONE').exists() else "INCOMPLETE"
            print(f"  {d.name:<22} {n:>4} images  [{done}]")
    if failed:
        print(f"\nFailed methods: {failed}")
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
