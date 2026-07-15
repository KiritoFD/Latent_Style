"""Check image counts and re-run DINO for seed2024_p2a if needed."""
import json
import os
import subprocess

base = r"I:\Github\Latent_Style\SchrodingerBridge"
dino_dir = os.path.join(base, "exp", "seed3", "_dino")

# Check all image directories
for tag in ["seed42_d5", "seed42_p2a", "seed42_r5", "seed123_d5", "seed123_p2a", "seed123_r5", "seed2024_d5", "seed2024_p2a", "seed2024_r5"]:
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

# Fix seed2024_p2a DINO
p2a_dino = os.path.join(dino_dir, "seed2024_p2a.json")
p2a_imgs = os.path.join(base, "exp", "seed3", "seed2024_p2a_eval", "full_eval", "epoch_0005", "images")
p2a_n = len([f for f in os.listdir(p2a_imgs) if f.endswith(".png")]) if os.path.isdir(p2a_imgs) else 0

if os.path.isfile(p2a_dino):
    with open(p2a_dino) as f:
        d = json.load(f)
    if d.get("n_images", 0) != p2a_n:
        print(f"\nMISMATCH: seed2024_p2a has {p2a_n} images but DINO json says {d.get('n_images')}")
        print("Deleting old DINO json and re-running...")
        os.remove(p2a_dino)
        cmd = [
            "python", "_compute_dino.py",
            "--images_dir", p2a_imgs,
            "--test_dir", r"I:\datasets\legacy256_overfit50\test",
            "--dataset", "p2a",
            "--output", p2a_dino,
            "--hf_cache", r"C:\Users\Administrator\.cache\huggingface\hub",
            "--max_refs", "30",
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=base, capture_output=True, text=True)
        print("STDOUT:", proc.stdout[-500:] if proc.stdout else "")
        print("STDERR:", proc.stderr[-500:] if proc.stderr else "")
        print("Exit code:", proc.returncode)
    else:
        print(f"\nOK: seed2024_p2a DINO matches ({p2a_n} images)")
