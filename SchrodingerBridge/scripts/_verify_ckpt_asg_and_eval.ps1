$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Verify checkpoint contains ASG weights
Write-Output "=== Verifying ASG weights in checkpoint ==="
python -c @"
import torch
ckpt = torch.load(r'I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
asg_keys = [k for k in sd.keys() if 'asg' in k.lower()]
print(f'ASG keys in checkpoint: {len(asg_keys)}')
for k in asg_keys[:10]:
    print(f'  {k}: shape={tuple(sd[k].shape)}, zero={bool((sd[k].abs().sum()==0))}')
"@

# Run evaluation
Write-Output "=== Running evaluation ==="
$evalOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep"
if (Test-Path $evalOut) {
    Remove-Item $evalOut -Recurse -Force -ErrorAction SilentlyContinue
}

python run_evaluation.py `
    --checkpoint "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt" `
    --output $evalOut `
    --batch_size 2 `
    --ref_feature_batch_size 2 `
    --vae_decode_batch_size 16 `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\eval_asg_retrain.log"

Write-Output "EXIT_CODE=$LASTEXITCODE"
