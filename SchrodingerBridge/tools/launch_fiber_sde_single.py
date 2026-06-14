import subprocess
import sys

# Launch single Fiber-SDE eval on topogate e2
cmd = [
    "ssh", "-p", "2222", "-o", "LogLevel=ERROR",
    "administrator@100.115.18.62",
    "wsl", "-d", "Ubuntu-26.04",
    "python", "/mnt/i/Github/Latent_Style/SchrodingerBridge/src/run.py",
    "--config", "/mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_fiber_sde_fiber_sigma0p02.json",
    "--resume", "/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0002.pt"
]

print("Launching Fiber-SDE sigma=0.02 on topogate e2...")
print("Theory: noise × TopoGate breaks ODE mean collapse")
print("Expected: style 0.69-0.72, LPIPS 0.32-0.35")
print()

result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)
