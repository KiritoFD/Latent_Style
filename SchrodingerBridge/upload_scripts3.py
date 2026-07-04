import subprocess, base64

files = [
    'generate_focused_ablation.py',
    'base_focused.json',
    'run_focused_ablation.sh',
    'start_focused_remote.sh',
    'collect_results.py'
]

for f in files:
    local_path = f'tools/massive_ablation/{f}'
    remote_path = f'/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/massive_ablation/{f}'
    with open(local_path, 'rb') as fp:
        file_bytes = fp.read()
    
    print(f'Uploading {f}...')
    b64_bytes = base64.b64encode(file_bytes)
    
    cmd = ['ssh', '-p', '2222', '-o', 'StrictHostKeyChecking=no', 'administrator@100.115.18.62', f'wsl bash -c "base64 -d > {remote_path}"']
    p = subprocess.run(cmd, input=b64_bytes, check=True)

print('Done!')
