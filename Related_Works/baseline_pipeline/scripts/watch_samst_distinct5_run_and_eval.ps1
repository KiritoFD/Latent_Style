param(
    [string]$RunRoot = "G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b1_e5_20260603",
    [int]$Epoch = 5,
    [int]$PollSeconds = 30,
    [string]$ReferenceRoot = "G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602"
)

$ErrorActionPreference = "Stop"

$runRootPath = (Resolve-Path $RunRoot).Path
$refRootPath = (Resolve-Path $ReferenceRoot).Path
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = (Resolve-Path (Join-Path $scriptRoot "..\..\..")).Path
$evalBundle = Join-Path $scriptRoot "run_samst_distinct5_eval_bundle.py"
$compareScript = Join-Path $scriptRoot "compare_samst_distinct5_epochs.py"
$runLog = Join-Path $runRootPath "run.log"
$watchLog = Join-Path $runRootPath "watch_eval.log"
$epochTag = ("epoch_{0:d4}" -f $Epoch)
$evalRoot = Join-Path $runRootPath "eval_bundle"
$summaryPath = Join-Path $evalRoot ("eval_epoch{0}\{1}\summary.json" -f $Epoch, $epochTag)
$artfidPath = Join-Path $evalRoot ("eval_epoch{0}\{1}\aggregate_targetwise_artfid.json" -f $Epoch, $epochTag)
$compareOut = Join-Path $evalRoot "compare_e5_vs_e15"
$refSummary = Join-Path $refRootPath "eval_epoch15\epoch_0015\summary.json"
$refArtfid = Join-Path $refRootPath "eval_epoch15\epoch_0015\aggregate_targetwise_artfid.json"
$trackedTrainYml = Join-Path $workspaceRoot "Related_Works\repos\SaMST-main\train_model\train2\train.yml"

function Write-WatchLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchLog -Value "[$stamp] $Message"
}

function Restore-TrackedTrainYmlIfNeeded {
    if (-not (Test-Path $trackedTrainYml -PathType Leaf)) {
        Write-WatchLog "Tracked train.yml not found; skip restore."
        return
    }
    $current = Get-Content -Path $trackedTrainYml -Raw
    $markers = @(
        "F:\wikiart_distinct5_samam_512_classview_real",
        "content_train_subset",
        "samst_distinct5_512_real_b1_e5_20260603",
        "samst_distinct5_512_real_b2_e5_20260603"
    )
    $needsRestore = $false
    foreach ($marker in $markers) {
        if ($current -like "*$marker*") {
            $needsRestore = $true
            break
        }
    }
    if (-not $needsRestore) {
        Write-WatchLog "Tracked train.yml does not match rerun markers; skip restore."
        return
    }
    $repoRelative = "Related_Works/repos/SaMST-main/train_model/train2/train.yml"
    $restored = git -C $workspaceRoot show ("HEAD:" + $repoRelative)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($restored)) {
        throw "Failed to restore tracked train.yml from git HEAD."
    }
    Set-Content -Path $trackedTrainYml -Value $restored -Encoding UTF8
    Write-WatchLog "Restored tracked train.yml from git HEAD."
}

Write-WatchLog "Watcher started. run_root=$runRootPath epoch=$Epoch poll=$PollSeconds"

while ($true) {
    if (Test-Path $summaryPath -PathType Leaf) {
        Write-WatchLog "Eval summary already exists at $summaryPath; watcher exiting."
        exit 0
    }

    if (-not (Test-Path $runLog -PathType Leaf)) {
        Write-WatchLog "Run log not found yet: $runLog"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $content = Get-Content -Path $runLog -Raw
    if ($content -match "finished=") {
        Write-WatchLog "Detected training completion in run.log; launching eval bundle."
        break
    }

    Write-WatchLog "Training still running; sleeping $PollSeconds sec."
    Start-Sleep -Seconds $PollSeconds
}

try {
    Restore-TrackedTrainYmlIfNeeded

    & py -3 $evalBundle `
        --run-root $runRootPath `
        --epochs $Epoch `
        1>> $watchLog 2>&1
    Write-WatchLog "Eval bundle completed."

    if (-not (Test-Path $summaryPath -PathType Leaf)) {
        throw "Expected summary missing after eval bundle: $summaryPath"
    }

    $compareArgs = @(
        "-3",
        $compareScript,
        "--label-a", ("e{0}" -f $Epoch),
        "--summary-a", $summaryPath,
        "--artfid-a", $artfidPath,
        "--label-b", "e15",
        "--summary-b", $refSummary,
        "--artfid-b", $refArtfid,
        "--output-dir", $compareOut
    )
    & py @compareArgs 1>> $watchLog 2>&1
    Write-WatchLog "Comparison against retained e15 completed."
}
catch {
    Write-WatchLog ("Watcher failed: " + $_.Exception.Message)
    throw
}
