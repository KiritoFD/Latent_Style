# Round 6 evaluation script: fetch checkpoint from remote, run full_eval + DINO locally
param(
    [Parameter(Mandatory=$true)]
    [string]$ExpName,
    [Parameter(Mandatory=$true)]
    [int]$Epoch,
    [string]$RemoteExpDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break",
    [string]$LocalExpDir = "g:/GitHub/Latent_Style/SchrodingerBridge/exp/dino_s_break",
    [string]$RemoteHost = "administrator@100.115.18.62",
    [int]$RemotePort = 2222,
    [string]$ConfigName = ""  # optional: override config name
)

$ErrorActionPreference = "Stop"

# Derive paths
$ckptDir = "$LocalExpDir\${ExpName}"
$ckptFile = "$ckptDir\epoch_$($Epoch.ToString('0000')).pt"
$configFile = if ($ConfigName) { "g:/GitHub/Latent_Style/SchrodingerBridge/configs/exp_${ConfigName}.json" } else { "g:/GitHub/Latent_Style/SchrodingerBridge/configs/exp_${ExpName}.json" }

Write-Host "=========================================="
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] EVAL $ExpName epoch=$Epoch"
Write-Host "=========================================="

# Step 1: Fetch checkpoint from remote
if (-not (Test-Path $ckptFile)) {
    Write-Host "Fetching checkpoint from remote..."
    if (-not (Test-Path $ckptDir)) {
        New-Item -ItemType Directory -Path $ckptDir -Force | Out-Null
    }
    & scp.exe -P $RemotePort -o LogLevel=ERROR "${RemoteHost}:$RemoteExpDir/${ExpName}/epoch_$($Epoch.ToString('0000')).pt" "$ckptFile"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to fetch checkpoint"
        exit 1
    }
} else {
    Write-Host "Checkpoint already exists: $ckptFile"
}

# Step 2: Run full_eval locally
$evalOutDir = "$ckptDir\full_eval\epoch_$($Epoch.ToString('0000'))"
if (Test-Path "$evalOutDir\summary.json") {
    Write-Host "full_eval already done: $evalOutDir\summary.json"
} else {
    Write-Host "Running full_eval..."
    Set-Location "g:/GitHub/Latent_Style/SchrodingerBridge/src"
    & python utils/run_evaluation.py --checkpoint $ckptFile --output $evalOutDir --config $configFile --batch_size 2 --generation_batch_size 2 --ref_feature_batch_size 2 --max_src_samples 30 --max_ref_compare 24 --max_ref_cache 80 --num_steps 12 --step_size 1.0 --eval_only_lpips_clip_style
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: full_eval failed"
        exit 1
    }
}

# Step 3: Extract CLIP-S and LPIPS
Write-Host "Extracting CLIP-S and LPIPS..."
& python -c "import json; s=json.load(open(r'$evalOutDir\summary.json')); o=s['analysis']['all_pairs_overview']; print('CLIP-S=',o['clip_style']); print('LPIPS=',o['content_lpips'])"

# Step 4: Run DINO eval
$dinoOut = "$LocalExpDir\_dino\${ExpName}_d5.json"
if (Test-Path $dinoOut) {
    Write-Host "DINO eval already done: $dinoOut"
} else {
    Write-Host "Running DINO eval..."
    $dinoScript = "g:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_dino_style.py"
    if (Test-Path $dinoScript) {
        & python $dinoScript --images_dir "$evalOutDir\images" --output $dinoOut
    } else {
        Write-Host "DINO script not found, searching..."
        Get-ChildItem -Path "g:/GitHub/Latent_Style/SchrodingerBridge" -Filter "eval_dino*" -Recurse | Select-Object -First 1 | ForEach-Object {
            Write-Host "Found: $($_.FullName)"
            & python $_.FullName --images_dir "$evalOutDir\images" --output $dinoOut
        }
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: DINO eval failed"
        exit 1
    }
}

# Step 5: Extract DINO results
if (Test-Path $dinoOut) {
    Write-Host "Extracting DINO results..."
    & python -c "import json; d=json.load(open(r'$dinoOut')); print('DINO-C=',d.get('dino_content','N/A')); print('DINO-S=',d.get('dino_style','N/A'))"
}

Write-Host "=========================================="
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] EVAL COMPLETE $ExpName"
Write-Host "=========================================="
