"""
图像级风格后处理：将模型输出对齐到目标风格的色调分布。
支持多种方法：直方图匹配、颜色迁移、手动参数调节。
"""
import os, sys, numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE, "_style_postproc_results")
os.makedirs(RESULT_DIR, exist_ok=True)

CONTENT_IMG = os.path.join(BASE, "teaser_content_photo_vangogh.jpg")       # 原始内容图
STYLE_REF   = r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\72_fewshot\data\5p1_shot01\test\Post_Impressionism\vincent-van-gogh_road-with-cypresses-1890.jpg"  # 梵高参考
INPUT_IMG   = os.path.join(BASE, "_3keys_sweep_results", "09_k1w10_k2lock03.png")  # 模型输出（最佳结果）

# ============================================================
# Utility
# ============================================================
def load_img(path, size=(512,512)):
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)

def to_np(img):
    return np.array(img).astype(np.float32) / 255.0

def from_np(arr):
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def save(img, name):
    path = os.path.join(RESULT_DIR, f"{name}.png")
    img.save(path)
    print(f"[SAVE] {path}")
    return path

# ============================================================
# Method 1: Histogram Matching (per channel)
# ============================================================
def histogram_match(source, reference):
    """Match each channel's histogram of source to reference."""
    matched = np.zeros_like(source)
    for c in range(3):
        src_flat = source[:,:,c].ravel()
        ref_flat = reference[:,:,c].ravel()
        # Get sorted indices
        src_sorted = np.argsort(src_flat)
        ref_sorted = np.argsort(ref_flat)
        # Map source values to reference quantiles
        mapped = np.empty_like(src_sorted, dtype=np.float32)
        mapped[src_sorted] = ref_flat[ref_sorted]
        matched[:,:,c] = mapped.reshape(source.shape[:2])
    return matched

# ============================================================
# Method 2: Color Transfer (mean/std in Lab space)
# ============================================================
def color_transfer_lab(source, reference):
    """Reinhard color transfer in Lab space."""
    from scipy import stats
    
    def _to_lab(img):
        lab = img.astype(np.float32)
        # Simple approximation: treat RGB as linear and convert roughly to Lab
        # Use the fact that for style transfer, mean/std transfer works even in luma+chroma
        # Here we use a simpler approach: YCbCr-like decomposition
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        luma = 0.299*r + 0.587*g + 0.114*b
        cb = 0.5*(b - luma) / (1 - 0.114) + 0.5
        cr = 0.5*(r - luma) / (1 - 0.299) + 0.5
        return np.stack([luma, cb, cr], axis=-1), img
    
    def _from_lab(lab_data, orig):
        luma, cb, cr = lab_data[:,:,0], lab_data[:,:,1], lab_data[:,:,2]
        r = cr * (1 - 0.299) * 2 + luma - 0.5*(1-0.299)*2*cb
        b = cb * (1 - 0.114) * 2 + luma - 0.5*(1-0.114)*2*cr
        g = (luma - 0.299*r - 0.114*b) / 0.587
        return np.clip(np.stack([r,g,b], axis=-1), 0, 1)
    
    src_lab, _ = _to_lab(source)
    ref_lab, _ = _to_lab(reference)
    
    result = np.zeros_like(src_lab)
    for c in range(3):
        s_mean, s_std = src_lab[:,:,c].mean(), src_lab[:,:,c].std()
        r_mean, r_std = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std()
        if s_std > 1e-6:
            result[:,:,c] = (src_lab[:,:,c] - s_mean) / s_std * r_std + r_mean
        else:
            result[:,:,c] = src_lab[:,:,c]
    
    return _from_lab(result, source)

# ============================================================
# Method 3: Manual parameter adjustment
# ============================================================
def adjust_params(img, brightness=1.0, contrast=1.0, saturation=1.0, 
                  sharpness=1.0, color_temp=0.0):
    """Apply manual image adjustments."""
    result = img
    if brightness != 1.0:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)
    if sharpness != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(sharpness)
    if color_temp != 0.0:
        arr = np.array(result).astype(np.float32)
        # Warm: shift red up, blue down; Cool: opposite
        factor = color_temp / 100.0
        arr[:,:,0] = np.clip(arr[:,:,0] + factor * 20, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] - factor * 20, 0, 255)
        result = Image.fromarray(arr.astype(np.uint8))
    return result

# ============================================================
# Method 4: Adaptive local contrast (CLAHE-like)
# ============================================================
def adaptive_local_contrast(img, clip_limit=2.0, grid_size=8):
    """Apply CLAHE-like local contrast enhancement."""
    arr = np.array(img).astype(np.uint8)
    h, w = arr.shape[:2]
    result = np.zeros_like(arr, dtype=np.float32)
    
    cell_h, cell_w = h // grid_size, w // grid_size
    
    for c in range(3):
        channel = arr[:,:,c].astype(np.float32)
        # Local mean subtraction with Gaussian weighting
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(channel, size=max(h//grid_size//2, 10))
        # Normalize local contrast
        local_std = uniform_filter((channel - local_mean)**2, size=max(h//grid_size//2, 10)) ** 0.5
        local_std = np.maximum(local_std, 1e-6)
        
        enhanced = (channel - local_mean) / local_std * clip_limit + local_mean
        result[:,:,c] = np.clip(enhanced, 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))

# ============================================================
# Method 5: Style-aware color mapping (match dominant colors)
# ============================================================
def dominant_color_match(source, reference, n_colors=5):
    """Extract dominant colors from reference and remap source palette."""
    from sklearn.cluster import KMeans
    
    src_reshaped = source.reshape(-1, 3)
    ref_reshaped = reference.reshape(-1, 3)
    
    # Subsample for speed
    n_samples = min(5000, len(ref_reshaped))
    np.random.seed(42)
    ref_idx = np.random.choice(len(ref_reshaped), n_samples, replace=False)
    
    try:
        kmeans_ref = KMeans(n_clusters=n_colors, random_state=42, n_init=1)
        kmeans_ref.fit(ref_reshaped[ref_idx])
        ref_centers = kmeans_ref.cluster_centers_
        
        kmeans_src = KMeans(n_clusters=n_colors, random_state=42, n_init=1)
        kmeans_src.fit(src_reshaped[::10])  # subsample source
        src_centers = kmeans_src.cluster_centers_
        
        # Map each source pixel to nearest ref center via source cluster
        src_labels = kmeans_src.predict(src_reshaped)
        # Match centers by sorting luminance
        src_order = np.argsort(src_centers.mean(axis=1))
        ref_order = np.argsort(ref_centers.mean(axis=1))
        
        new_pixels = src_reshaped.copy()
        for i in range(n_colors):
            mask = src_labels == i
            new_pixels[mask] = ref_centers[ref_order[np.where(src_order == i)[0][0]]]
        
        return new_pixels.reshape(source.shape)
    except ImportError:
        print("[WARN] sklearn not available, skipping dominant color match")
        return source

# ============================================================
# Method 6: Film-like tone curve (S-curve for dramatic effect)
# ============================================================
def apply_tone_curve(img, shadow=25, highlight=235, black=20, white=240):
    """Apply S-curve tone mapping for more dramatic look."""
    arr = np.array(img).astype(np.float32)
    
    # Normalize to 0-1
    arr_norm = arr / 255.0
    
    # S-curve: lift shadows, compress highlights
    curve = lambda x: np.where(
        x < 0.5,
        0.5 * (2*x) ** (shadow/50.0),
        1 - 0.5 * (2*(1-x)) ** (highlight/250.0)
    )
    
    result = curve(arr_norm)
    
    # Black point / white point clipping
    result = (result - black/255.0) / ((white - black)/255.0)
    result = np.clip(result, 0, 1)
    
    return Image.fromarray((result * 255).astype(np.uint8))

# ============================================================
# Run all methods
# ============================================================
def main():
    print("="*60)
    print("Style Post-processing Experiments")
    print("="*60)
    
    content = load_img(CONTENT_IMG)
    style_ref = load_img(STYLE_REF)
    model_out = load_img(INPUT_IMG)
    
    content_np = to_np(content)
    style_np = to_np(style_ref)
    model_np = to_np(model_out)
    
    results = {}
    
    # ── Originals ──
    results['00_original_content'] = content
    results['01_original_style_ref'] = style_ref
    results['02_model_output'] = model_out
    
    # ── Histogram Matching ──
    print("[PROC] Histogram matching...")
    hist_matched = histogram_match(model_np, style_np)
    results['03_hist_match_full'] = from_np(hist_matched)
    
    # Luminosity-preserving hist match (match chroma only)
    model_lum = 0.299*model_np[:,:,0] + 0.587*model_np[:,:,1] + 0.114*model_np[:,:,2]
    style_lum = 0.299*style_np[:,:,0] + 0.587*style_np[:,:,1] + 0.114*style_np[:,:,2]
    hist_rgb = histogram_match(model_np, style_np)
    # Replace luminance with original
    for c in range(3):
        hist_rgb[:,:,c] = hist_rgb[:,:,c] * (style_lum + 1e-6) / (hist_rgb[:,:,0]*0.299 + hist_rgb[:,:,1]*0.587 + hist_rgb[:,:,2]*0.114 + 1e-6)
    results['04_hist_match_luma_preserve'] = from_np(hist_rgb)
    
    # ── Color Transfer (Lab) ──
    print("[PROC] Color transfer (Lab space)...")
    ct_result = color_transfer_lab(model_np, style_np)
    results['05_color_transfer'] = from_np(ct_result)
    
    # ── Manual adjustments grid ──
    print("[PROC] Manual parameter sweep...")
    param_combos = [
        ("06_bright120",     dict(brightness=1.2)),
        ("07_contr130",      dict(contrast=1.3)),
        ("08_sat140",        dict(saturation=1.4)),
        ("09_bright115_contr125", dict(brightness=1.15, contrast=1.25)),
        ("10_bright110_contr120_sat120", dict(brightness=1.1, contrast=1.2, saturation=1.2)),
        ("11_contr140_sat130_sharp13", dict(contrast=1.4, saturation=1.3, sharpness=1.3)),
        ("12_bright108_contr135_sat125", dict(brightness=1.08, contrast=1.35, saturation=1.25)),
        ("13_warm30_contr125_sat115", dict(color_temp=30, contrast=1.25, saturation=1.15)),
        ("14_cool20_contr130_sat125", dict(color_temp=-20, contrast=1.3, saturation=1.25)),
        ("15_dramatic_Scurve", None),  # special case
        ("16_film_look", None),         # special case
    ]
    
    for name, params in param_combos:
        if params is None:
            continue
        results[name] = adjust_params(model_out, **params)
        save(results[name], name)
    
    # Special: S-curve
    results['15_dramatic_Scurve'] = apply_tone_curve(model_out, shadow=30, highlight=230, black=15, white=245)
    save(results['15_dramatic_Scurve'], '15_dramatic_Scurve')
    
    # Film look: slight fade + warm + contrast
    film = adjust_params(model_out, brightness=1.05, contrast=1.15, saturation=0.95, color_temp=15)
    film_arr = np.array(film).astype(np.float32)
    # Add slight fade (lift blacks)
    film_arr = film_arr * 0.92 + 20
    results['16_film_look'] = Image.fromarray(np.clip(film_arr, 0, 255).astype(np.uint8))
    save(results['16_film_look'], '16_film_look')
    
    # ── Combined: Hist match + enhancement ──
    print("[PROC] Combined methods...")
    hist_base = from_np(hist_matched)
    results['17_hist_match_plus_contr130'] = adjust_params(hist_base, contrast=1.3, saturation=1.15, sharpness=1.1)
    save(results['17_hist_match_plus_contr130'], '17_hist_match_plus_contr130')
    
    ct_base = from_np(ct_result)
    results['18_ct_plus_contr125_sat120'] = adjust_params(ct_base, contrast=1.25, saturation=1.2, sharpness=1.1)
    save(results['18_ct_plus_contr125_sat120'], '18_ct_plus_contr125_sat120')
    
    # Van Gogh specific: boost yellows/blues, high contrast, painterly feel
    vg_style = adjust_params(model_out, contrast=1.45, saturation=1.35, sharpness=1.2, color_temp=10)
    # Boost yellow-blue channels
    vg_arr = np.array(vg_style).astype(np.float32)
    # Enhance yellow (R+G) and blue
    vg_arr[:,:,0] = np.clip(vg_arr[:,:,0] * 1.05, 0, 255)   # Red
    vg_arr[:,:,1] = np.clip(vg_arr[:,:,1] * 1.08, 0, 255)   # Green → yellow boost
    vg_arr[:,:,2] = np.clip(vg_arr[:,:,2] * 1.02, 0, 255)   # Blue (slight)
    results['19_vangogh_style'] = Image.fromarray(np.clip(vg_arr, 0, 255).astype(np.uint8))
    save(results['19_vangogh_style'], '19_vangogh_style')
    
    # Best guess: combine all insights
    best = adjust_params(from_np(hist_matched), brightness=1.08, contrast=1.35, saturation=1.25, sharpness=1.15)
    best_arr = np.array(best).astype(np.float32)
    best_arr[:,:,1] = np.clip(best_arr[:,:,1] * 1.04, 0, 255)  # slight yellow push
    results['20_best_combined'] = Image.fromarray(np.clip(best_arr, 0, 255).astype(np.uint8))
    
    # Fix: uint8 type
    results['20_best_combined'] = Image.fromarray(np.clip(best_arr, 0, 255).astype(np.uint8))
    save(results['20_best_combined'], '20_best_combined')
    
    # ── Generate comparison grid ──
    print("[PROC] Generating comparison grid...")
    
    keys = sorted(results.keys())
    n = len(keys)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, key in enumerate(keys):
        ax = axes[i]
        ax.imshow(results[key])
        ax.set_title(key, fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Style Post-processing Comparison\n(Content→Model Output→Various Enhancements)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(RESULT_DIR, "_postproc_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[DONE] Grid saved: {grid_path}")
    print(f"[DONE] All {n} variants saved to: {RESULT_DIR}")

if __name__ == "__main__":
    main()
