# Eval Distinct5-512 MUSIQ for all baselines + Launch SaMam W20 v2
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$REPO\src;$USER_SITE;$REPO\scripts"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"

$logFile = "$REPO\logs\eval_d5_musiq.log"
"=== D5-512 MUSIQ START $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Encoding utf8

# Distinct5-512 baselines from baseline_v2/images
$targets = @(
    @{ name = "adain_d5";    dir = "$REPO\exp\baseline_v2\images\adain" },
    @{ name = "wct_d5";      dir = "$REPO\exp\baseline_v2\images\wct_v32k" },
    @{ name = "sdturbo_d5";  dir = "$REPO\exp\baseline_v2\images\sdturbo" },
    @{ name = "styleid_d5";  dir = "$REPO\exp\baseline_v2\images\styleid" },
    @{ name = "cut_d5";      dir = "$REPO\exp\baseline_v2\images\cut" },
    @{ name = "samst_d5";    dir = "$REPO\exp\baseline_v2\images\samst" },
    @{ name = "samam_d5";    dir = "$REPO\exp\baseline_v2\images\samam" },
    @{ name = "identity_d5"; dir = "$REPO\exp\baseline_v2\images\identity" }
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.dir)) {
        "[$($t.name)] SKIP: dir not found" | Out-File $logFile -Append -Encoding utf8
        continue
    }
    $out = "$REPO\exp\_eval_$($t.name)_musiq.json"
    "[$($t.name)] START $(Get-Date -Format 'HH:mm:ss')" | Out-File $logFile -Append -Encoding utf8
    & $PYTHON -u "$REPO\scripts\_eval_unified.py" `
        --image-dir $t.dir `
        --dataset distinct5 `
        --output $out `
        --max-images 750 `
        --skip-clip --skip-lpips 2>&1 | Out-File $logFile -Append -Encoding utf8
    if (Test-Path $out) {
        $r = Get-Content $out -Raw | ConvertFrom-Json
        $msg = "  MUSIQ=$($r.musiq)"
        Write-Host $msg
        $msg | Out-File $logFile -Append -Encoding utf8
    }
    "[$($t.name)] DONE" | Out-File $logFile -Append -Encoding utf8
}

"=== D5-512 MUSIQ END $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8

# Phase 2: Launch SaMam W20 v2 in background (after D5 eval done, VRAM free)
Write-Host "=== Launching SaMam W20 v2 ==="
"=== SaMam W20 v2 LAUNCH $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8

# Use schtasks for persistence
schtasks /Delete /TN "samam_w20_v2" /F 2>$null | Out-Null
schtasks /Create /TN "samam_w20_v2" `
    /TR "`"$PYTHON`" -u `"$REPO\scripts\_gen_samam_wiki20_v2.py`"" `
    /SC ONCE /ST 23:59 /SD (Get-Date -Format "yyyy/MM/dd") `
    /RU SYSTEM /RL HIGHEST /F
schtasks /Run /TN "samam_w20_v2"

Write-Host "=== SaMam W20 v2 launched ==="
"=== SaMam W20 v2 LAUNCHED ===" | Out-File $logFile -Append -Encoding utf8
