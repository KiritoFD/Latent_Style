from PIL import Image
import numpy as np
import colorsys

in_path = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio.png"
out_path = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027_v4\framework_sfm_main.png"

im = Image.open(in_path).convert("RGBA")
arr = np.array(im, dtype=np.float32)

h, w, _ = arr.shape
bg_thresh = 30  # pixels darker than this are considered black background

for y in range(h):
    for x in range(w):
        r, g, b, a = arr[y, x]
        if a < 10:
            arr[y, x] = [255, 255, 255, 0]
            continue
        # near-black background -> white
        if max(r, g, b) < bg_thresh:
            arr[y, x] = [255, 255, 255, a]
            continue
        # keep hue, invert value for light-theme readability
        # convert to HSV, invert V, keep H and S
        rd, gd, bd = r / 255.0, g / 255.0, b / 255.0
        hue, sat, val = colorsys.rgb_to_hsv(rd, gd, bd)
        # map value: invert around 1.0, but keep some contrast for mid tones
        new_val = 1.0 - val
        # ensure foreground is visible (not too light, not too dark)
        new_val = np.clip(new_val, 0.15, 0.9)
        # slightly boost saturation for readability
        new_sat = min(sat * 1.15, 1.0)
        nr, ng, nb = colorsys.hsv_to_rgb(hue, new_sat, new_val)
        arr[y, x] = [nr * 255, ng * 255, nb * 255, a]

# clip and convert to uint8
out = np.clip(arr, 0, 255).astype(np.uint8)
out_im = Image.fromarray(out, "RGBA")
# composite on white background to remove alpha
white = Image.new("RGBA", out_im.size, (255, 255, 255, 255))
final = Image.alpha_composite(white, out_im).convert("RGB")
final.save(out_path, "PNG")
print(f"saved {out_path} ({final.size})")
