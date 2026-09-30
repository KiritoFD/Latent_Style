# Remote batch runner for 630 tuning experiments
# Runs 5 experiments sequentially with error handling
# Each experiment: train from scratch (5 epochs, patience=2) + eval each epoch
$ErrorActionPreference = "Continue"

$PROJECT_ROOT = "I:\Github\Latent_Style\SchrodingerBridge"
Set-Location $PROJECT_ROOT

$LOG_DIR = "$PROJECT_ROOT\exp\630_remote_logs"
if (!(Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

$EXPERIMENTS = @(
    @{name="a1_dwt_strong_style";       config="630_remote_a1_dwt_strong_style.json"},
    @{name="a2_cosine_heun_dwt_balanced";config="630_remote_a2_cosine_heun_dwt_balanced.json"},
    @{name="p1_spectral_rebalance";     config="630_remote_p1_spectral_rebalance.json"},
    @{name="t1_lowfreq_style";          config="630_remote_t1_lowfreq_style.json"},
    @{name="p2_swd_flow_balance";       config="630_remote_p2_swd_flow_balance.json"}
)

$TOTAL_START = Get-Date
$RESULTS = @()

Write-Host "=========================================="
Write-Host "630 Remote Tuning Batch Runner"
Write-Host "Start: $TOTAL_START"
Write-Host "Experiments: $($EXPERIMENTS.Count)"
Write-Host "Target: transfer_clip>0.6914 AND transfer_lpips<0.3387"
Write-Host "=========================================="

foreach ($exp in $EXPERIMENTS) {
    $name = $exp.name
    $config = $exp.config
    $configPath = "$PROJECT_ROOT\configs\$config"
    $logFile = "$LOG_DIR\${name}.log"
    $expStart = Get-Date

    Write-Host ""
    Write-Host "[$($RESULTS.Count + 1)/$($EXPERIMENTS.Count)] Running: $name"
    Write-Host "  Config: $configPath"
    Write-Host "  Log: $logFile"
    Write-Host "  Start: $expStart"

    if (!(Test-Path $configPath)) {
        Write-Host "  ERROR: Config not found, skipping"
        $RESULTS += @{name=$name; status="CONFIG_NOT_FOUND"; duration_sec=0}
        continue
    }

    # Run experiment with error handling
    try {
        # Training + eval (full_eval_each_epoch=true in config)
        $proc = Start-Process -FilePath "python" `
            -ArgumentList "src/run.py --config configs/$config" `
            -WorkingDirectory $PROJECT_ROOT `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $logFile `
            -RedirectStandardError "$logFile.err"

        # Wait for completion with timeout (4 hours = 14400 sec)
        $timeout = 14400
        $waited = 0
        while (!$proc.HasExited -and $waited -lt $timeout) {
            Start-Sleep -Seconds 30
            $waited += 30
            # Print GPU status every 5 minutes
            if ($waited % 300 -eq 0) {
                $gpu = nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>$null
                Write-Host "  [$($waited/60)min] GPU: $gpu  PID=$($proc.Id)"
            }
        }

        if (!$proc.HasExited) {
            Write-Host "  TIMEOUT: Killing process"
            $proc | Stop-Process -Force
            $RESULTS += @{name=$name; status="TIMEOUT"; duration_sec=$waited}
            continue
        }

        $exitCode = $proc.ExitCode
        $expEnd = Get-Date
        $duration = ($expEnd - $expStart).TotalSeconds

        if ($exitCode -eq 0) {
            Write-Host "  SUCCESS: exit=$exitCode  duration=$([math]::Round($duration/60,1))min"
            $RESULTS += @{name=$name; status="SUCCESS"; duration_sec=[math]::Round($duration,0); exitCode=$exitCode}
        } else {
            Write-Host "  FAILED: exit=$exitCode  duration=$([math]::Round($duration/60,1))min"
            $RESULTS += @{name=$name; status="FAILED"; duration_sec=[math]::Round($duration,0); exitCode=$exitCode}
        }
    } catch {
        $expEnd = Get-Date
        $duration = ($expEnd - $expStart).TotalSeconds
        Write-Host "  EXCEPTION: $_"
        $RESULTS += @{name=$name; status="EXCEPTION"; duration_sec=[math]::Round($duration,0); error=$_.ToString()}
    }

    # Cool down between experiments (let GPU memory release)
    Write-Host "  Cooling down 15s..."
    Start-Sleep -Seconds 15
}

$TOTAL_END = Get-Date
$TOTAL_DURATION = ($TOTAL_END - $TOTAL_START).TotalSeconds

Write-Host ""
Write-Host "=========================================="
Write-Host "Batch Complete: $TOTAL_END"
Write-Host "Total Duration: $([math]::Round($TOTAL_DURATION/3600,2))h"
Write-Host "=========================================="
foreach ($r in $RESULTS) {
    $dur = if ($r.duration_sec) { "$([math]::Round($r.duration_sec/60,1))min" } else { "N/A" }
    Write-Host "  $($r.name): $($r.status)  ($dur)"
}
Write-Host "==========================================")

# Save results summary
$summary = @{
    start = $TOTAL_START.ToString()
    end = $TOTAL_END.ToString()
    total_duration_sec = [math]::Round($TOTAL_DURATION, 0)
    results = $RESULTS
}
$summary | ConvertTo-Json -Depth 3 | Out-File "$LOG_DIR\batch_summary.json"
Write-Host "Summary saved: $LOG_DIR\batch_summary.json"
