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
        b64 = base64.b64encode(fp.read()).decode('utf-8')
    cmd = f"wsl bash -c \"echo '{b64}' | base64 -d > {remote_path}\""
    print(f'Uploading {f}...')
    subprocess.run(['ssh', '-p', '2222', '-o', 'StrictHostKeyChecking=no', 'administrator@100.115.18.62', cmd], check=True)
print('Done!')
