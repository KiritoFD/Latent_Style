"""Check the exact loss calculation code on remote."""
f = open(r"I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py", encoding="utf-8").read()
lines = f.split("\n")
# Print lines 675-720
for i in range(674, min(720, len(lines))):
    print(f"{i+1}: {lines[i]}")
