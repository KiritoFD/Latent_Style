"""Check if swd_replace_with_mse code exists on remote."""
f = open(r"I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py", encoding="utf-8").read()
lines = f.split("\n")
for i, l in enumerate(lines):
    if "swd_replace" in l:
        print(f"{i+1}: {l}")
print("---")
# Also check config_schema
f2 = open(r"I:\Github\Latent_Style\SchrodingerBridge\src\config_schema.py", encoding="utf-8").read()
for i, l in enumerate(f2.split("\n")):
    if "swd_replace" in l:
        print(f"config_schema:{i+1}: {l}")
