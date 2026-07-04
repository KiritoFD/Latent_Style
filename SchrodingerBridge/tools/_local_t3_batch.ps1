# Local batch runner for T3 experiments — with NaN retry logic
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"

$EXPERIMENTS = @(
    @{name="t3a_adain_ll_005"; config="630_local_t3_adain_ll_t3a.json"; save_dir="630_local_t3_adain_ll_t3a"},
    @{name="t3b_adain_ll_010"; config="630_local_t3_adain_ll_t3b.json"; save_dir="630_local_t3_adain_ll_t3b"},
    @{name="t3c_adain_ll_015"; config="630_local_t3_adain_ll_t3c.json"; save_dir="630_local_t3_adain_ll_t3c"}
)

$MAX_RETRIES = 3

function Test-Epoch1NaN {
    param([string]$saveDir)
    $csvFiles = Get-ChildItem "exp\$saveDir\logs\training_*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if (-not $csvFiles) { return $true }
    $csv = Get-Content $csvFiles[0].FullName
    if ($csv.Count -lt 2) { return $true }
    $loss = $csv[1].Split(',')[1]
    return ($loss -eq "nan" -or $loss -eq "inf" -or $loss -eq "")
}

function Invoke-CudaCleanup {
    Write-Host "  [cleanup] Releasing CUDA cache..."
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('cache cleared')" 2>$null | Out-Null
    Start-Sleep -Seconds 2
}

$TOTAL_START = Get-Date
Write-Host "=== T3 Batch Start: $TOTAL_START ==="

foreach ($exp in $EXPERIMENTS) {
    $name = $exp.name
    $config = $exp.config
    $saveDir = $exp.save_dir

    for ($attempt = 1; $attempt -le $MAX_RETRIES; $attempt++) {
        $expStart = Get-Date
        Write-Host "`n[$name] Attempt $attempt/$MAX_RETRIES Start: $expStart"

        # Clean previous failed attempt
        if ($attempt -gt 1) {
            Remove-Item -Recurse -Force "exp\$saveDir" -ErrorAction SilentlyContinue
            Invoke-CudaCleanup
        } else {
            # Even on first attempt, clear CUDA cache
            Invoke-CudaCleanup
        }

        try {
            $proc = Start-Process -FilePath "python" `
                -ArgumentList "src/run.py --config configs/$config" `
                -NoNewWindow -PassThru
            $proc.WaitForExit()
            $exitCode = $proc.ExitCode
            $dur = [math]::Round((Get-Date - $expStart).TotalSeconds / 60, 1)

            if ($exitCode -eq 0) {
                # Check for NaN in epoch 1
                $isNan = Test-Epoch1NaN -saveDir $saveDir
                if ($isNan) {
                    Write-Host "  [$name] Exit=0 but epoch1=NaN, will retry"
                    if ($attempt -lt $MAX_RETRIES) {
                        Write-Host "  [$name] NaN detected, retrying..."
                        continue
                    } else {
                        Write-Host "  [$name] NaN after $MAX_RETRIES attempts, GIVING UP"
                    }
                } else {
                    Write-Host "  [$name] SUCCESS in ${dur}min"
                    break
                }
            } else {
                Write-Host "  [$name] FAILED exit=$exitCode in ${dur}min"
                if ($attempt -lt $MAX_RETRIES) {
                    Write-Host "  [$name] Retrying..."
                    continue
                } else {
                    Write-Host "  [$name] FAILED after $MAX_RETRIES attempts, GIVING UP"
                }
            }
        } catch {
            Write-Host "  [$name] EXCEPTION: $_"
            if ($attempt -lt $MAX_RETRIES) { continue }
        }
        Start-Sleep -Seconds 3
    }
}

$TOTAL_END = Get-Date
Write-Host "`n=== T3 Batch Complete: $TOTAL_END ==="
Write-Host "Total: $([math]::Round((Get-Date - $TOTAL_START).TotalMinutes, 1))min"
