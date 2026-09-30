"""Check image counts and re-run DINO for seed42_r5 if needed."""
import json
import os
import subprocess
import sys

base = r"I:\Github\Latent_Style\SchrodingerBridge"
dino_dir = os.path.join(base, "exp", "seed3", "_dino")

# Check all image directories
for tag in ["seed42_d5", "seed42_p2a", "seed42_r5", "seed123_d5", "seed123_p2a", "seed123_r5", "seed2024_d5"]:
    img_dir = os.path.join(base, "exp", "seed3", f"{tag}_eval", "full_eval", "epoch_0005", "images")
    if os.path.isdir(img_dir):
        n = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
    else:
        n = -1
    dino_path = os.path.join(dino_dir, f"{tag}.json")
    dino_n = ""
    if os.path.isfile(dino_path):
        with open(dino_path) as f:
            d = json.load(f)
        dino_n = f" dino_n={d.get('n_images')}"
    print(f"{tag}: images={n}{dino_n}")

# Check if seed42_r5 DINO needs re-running
r5_dino = os.path.join(dino_dir, "seed42_r5.json")
r5_imgs = os.path.join(base, "exp", "seed3", "seed42_r5_eval", "full_eval", "epoch_0005", "images")
r5_n = len([f for f in os.listdir(r5_imgs) if f.endswith(".png")]) if os.path.isdir(r5_imgs) else 0

if os.path.isfile(r5_dino):
    with open(r5_dino) as f:
        d = json.load(f)
    if d.get("n_images", 0) != r5_n:
        print(f"\nMISMATCH: seed42_r5 has {r5_n} images but DINO json says {d.get('n_images')}")
        print("Deleting old DINO json and re-running...")
        os.remove(r5_dino)
        # Re-run DINO
        cmd = [
            "python", "_compute_dino.py",
            "--images_dir", r5_imgs,
            "--test_dir", r"I:\datasets\wikiarts20_512_test",
            "--dataset", "wikiart",
            "--output", r5_dino,
            "--hf_cache", r"C:\Users\Administrator\.cache\huggingface\hub",
            "--max_refs", "30",
            "--style_subdirs", "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism",
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=base, capture_output=True, text=True)
        print("STDOUT:", proc.stdout[-500:] if proc.stdout else "")
        print("STDERR:", proc.stderr[-500:] if proc.stderr else "")
        print("Exit code:", proc.returncode)
    else:
        print(f"\nOK: seed42_r5 DINO matches ({r5_n} images)")
