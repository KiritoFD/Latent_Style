import subprocess
import os
import sys

# Change to the working directory
os.chdir('/home/xy/Latent_Style/SchrodingerBridge')

print(f"Working directory: {os.getcwd()}")
print(f"Starting evaluation at {subprocess.check_output(['date']).decode().strip()}")

ckpt = "/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
config = "/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json"
styles = "Hayao,cezanne,monet,photo,vangogh"

print(f"Checkpoint: {ckpt}")
print(f"Config: {config}")
print(f"Exists: {os.path.exists(ckpt)}, {os.path.exists(config)}")

# Run evaluation
cmd = [
    'python', 'run.py',
    '--config', config,
    '--eval_only',
    '--checkpoint_path', ckpt,
    '--style_subdirs', styles
]

print(f"Running: {' '.join(cmd)}")

with open('/mnt/c/Users/Administrator/eval_direct_log.txt', 'w') as f:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line.strip())
        f.write(line)
        f.flush()

print(f"Evaluation complete at {subprocess.check_output(['date']).decode().strip()}")
