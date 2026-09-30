import shutil, pathlib, os

SRC = pathlib.Path("/mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/c")
DST = pathlib.Path("/mnt/i/Github/Latent_Style/exp_samam/curve_pngs")

STEPS = ["000250", "000500", "001000", "002000", "003000", "004000",
         "005000", "006000", "007000", "008000", "010000", "012000",
         "015000", "017500", "020000"]

for step in STEPS:
    step_dir = SRC / f"step_{step}" / "images"
    dst_dir = DST / f"step_{step}" / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying step_{step}...")
    count = 0
    for src_file in step_dir.glob("*.png"):
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        count += 1
    print(f"  Done: {count} files")

print("All done.")