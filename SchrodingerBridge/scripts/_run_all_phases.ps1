# Orchestrator: Run all 3 baseline-fill phases sequentially via schtasks
#
# Phase 1: WikiArt-20 StyleID/CUT/SaMST (12000 images per method)
# Phase 2: 256 baselines (SD-Turbo/StyleID/CUT) + seedream LPIPS
# Phase 3: MUSIQ re-run for 512 Distinct5 + WikiArt-20
#
# This script registers a one-time scheduled task (SYSTEM account) that runs
# all 3 phases sequentially, surviving SSH disconnects.
#
# Usage (on remote):
#   powershell -ExecutionPolicy Bypass -File C:\Users\Administrator\_run_all_phases.ps1
#
# Or to just register the schtask and start it:
#   powershell -ExecutionPolicy Bypass -File C:\Users\Administrator\_run_all_phases.ps1 -RegisterOnly
#
# Or to run phases inline (no schtasks, blocks current session):
#   powershell -ExecutionPolicy Bypass -File C:\Users\Administrator\_run_all_phases.ps1 -Inline

param(
    [switch]$RegisterOnly,
    [switch]$Inline,
    [string]$TaskName = "baseline_fill_phases"
)

$ErrorActionPreference = "Continue"

# ── Paths ──
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$LOG_DIR = "$REPO\logs"
$LOG = "$LOG_DIR\run_all_phases.log"
$RESULTS_JSON = "$REPO\exp\_baseline_fill_results.json"

# Phase scripts (in execution order)
$PHASE_SCRIPTS = @(
    @{ name = "phase1"; file = "$REPO\scripts\_phase1_wiki20_scm.ps1"; desc = "WikiArt-20 StyleID/CUT/SaMST" },
    @{ name = "phase2"; file = "$REPO\scripts\_phase2_256_baselines.ps1"; desc = "256 baselines + seedream LPIPS" },
    @{ name = "phase3"; file = "$REPO\scripts\_phase3_musiq_rerun.ps1"; desc = "MUSIQ re-run for 512+wiki20" }
)

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\exp" | Out-Null

# ── Runner: execute a single phase script with try/catch ──
function Invoke-Phase($phase) {
    $name = $phase.name
    $file = $phase.file
    $desc = $phase.desc

    "" | Tee-Object -FilePath $LOG -Append
    "==========================================================" | Tee-Object -FilePath $LOG -Append
    "=== Phase ${name}: $desc ===" | Tee-Object -FilePath $LOG -Append
    "=== Started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
    "=== Script: $file" | Tee-Object -FilePath $LOG -Append
    "==========================================================" | Tee-Object -FilePath $LOG -Append

    if (-not (Test-Path $file)) {
        "  [ERROR] Phase script not found: $file" | Tee-Object -FilePath $LOG -Append
        return @{ phase = $name; status = "script not found"; exit_code = -1 }
    }

    try {
        $phaseLog = "$LOG_DIR\phase_${name}_wrapper.log"
        $proc = Start-Process -FilePath "powershell.exe" `
            -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoProfile", "-File", $file) `
            -NoNewWindow -PassThru -WorkingDirectory $REPO `
            -RedirectStandardOutput $phaseLog -RedirectStandardError "$phaseLog.err"
        $proc.WaitForExit()
        $ec = $proc.ExitCode

        "  [$name] Finished at $(Get-Date) exit_code=$ec" | Tee-Object -FilePath $LOG -Append
        if (Test-Path $phaseLog) {
            "  [$name] Last 10 lines of output:" | Tee-Object -FilePath $LOG -Append
            Get-Content $phaseLog -Tail 10 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
        return @{ phase = $name; status = "completed"; exit_code = $ec }
    } catch {
        "  [$name] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        return @{ phase = $name; status = "exception"; error = "$_" }
    }
}

# ── Mode: Inline (run phases in current session) ──
if ($Inline) {
    "=== Run All Phases (inline) started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
    "  mode: inline (blocks current session)" | Tee-Object -FilePath $LOG -Append
    "  phases: $($PHASE_SCRIPTS.Count)" | Tee-Object -FilePath $LOG -Append

    $phaseResults = @()
    foreach ($phase in $PHASE_SCRIPTS) {
        $r = Invoke-Phase $phase
        $phaseResults += $r
        # Continue to next phase regardless of exit code (try/catch per phase)
    }

    # ── Final summary ──
    "" | Tee-Object -FilePath $LOG -Append
    "==========================================================" | Tee-Object -FilePath $LOG -Append
    "=== All phases completed at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
    "==========================================================" | Tee-Object -FilePath $LOG -Append
    foreach ($r in $phaseResults) {
        "  Phase $($r.phase): status=$($r.status) exit=$($r.exit_code)" | Tee-Object -FilePath $LOG -Append
    }
    "  Results JSON: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append

    if (Test-Path $RESULTS_JSON) {
        "  --- Results summary ---" | Tee-Object -FilePath $LOG -Append
        try {
            $results = Get-Content $RESULTS_JSON -Raw | ConvertFrom-Json -AsHashtable
            $results | ConvertTo-Json -Depth 5 | Tee-Object -FilePath $LOG -Append
        } catch {
            "  (failed to parse results JSON)" | Tee-Object -FilePath $LOG -Append
        }
    }
    exit 0
}

# ── Mode: Register schtask (default) ──
# Creates a one-time scheduled task under SYSTEM that runs all 3 phases
$wrapperScript = @"
`$ErrorActionPreference = "Continue"
`$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
`$LOG_DIR = "`$REPO\logs"
`$LOG = "`$LOG_DIR\run_all_phases.log"
New-Item -ItemType Directory -Force -Path `$LOG_DIR | Out-Null

"=== Run All Phases (schtask) started at `$(Get-Date) ===" | Tee-Object -FilePath `$LOG -Append

# Run each phase sequentially
`$phaseScripts = @(
    "`$REPO\scripts\_phase1_wiki20_scm.ps1",
    "`$REPO\scripts\_phase2_256_baselines.ps1",
    "`$REPO\scripts\_phase3_musiq_rerun.ps1"
)

foreach (`$ps in `$phaseScripts) {
    "  Running: `$ps at `$(Get-Date)" | Tee-Object -FilePath `$LOG -Append
    if (Test-Path `$ps) {
        try {
            & powershell.exe -ExecutionPolicy Bypass -NoProfile -File `$ps 2>&1 | Tee-Object -FilePath `$LOG -Append
            "  Finished: `$ps exit=`$LASTEXITCODE at `$(Get-Date)" | Tee-Object -FilePath `$LOG -Append
        } catch {
            "  EXCEPTION in `$ps : `$_ at `$(Get-Date)" | Tee-Object -FilePath `$LOG -Append
        }
    } else {
        "  [ERROR] Script not found: `$ps" | Tee-Object -FilePath `$LOG -Append
    }
}

"=== All phases completed at `$(Get-Date) ===" | Tee-Object -FilePath `$LOG -Append
"  Results: `$REPO\exp\_baseline_fill_results.json" | Tee-Object -FilePath `$LOG -Append
"@

$wrapperPath = "$env:TEMP\_run_all_phases_wrapper.ps1"
$wrapperScript | Out-File -FilePath $wrapperPath -Encoding utf8 -Force

# Register the scheduled task
"=== Registering scheduled task: $TaskName ===" | Tee-Object -FilePath $LOG -Append
"  Wrapper script: $wrapperPath" | Tee-Object -FilePath $LOG -Append

# Remove existing task if any
try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    "  Removed existing task (if any)" | Tee-Object -FilePath $LOG -Append
} catch {
    # Task doesn't exist, continue
}

# Create one-time task running as SYSTEM
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$wrapperPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(10)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 48)

try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Force | Out-Null
    "  Task registered successfully" | Tee-Object -FilePath $LOG -Append

    if (-not $RegisterOnly) {
        "  Starting task..." | Tee-Object -FilePath $LOG -Append
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 3
        $task = Get-ScheduledTask -TaskName $TaskName
        $info = $task | Get-ScheduledTaskInfo
        "  Task State: $($task.State)" | Tee-Object -FilePath $LOG -Append
        "  Last Run Time: $($info.LastRunTime)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  -RegisterOnly: task registered but not started." | Tee-Object -FilePath $LOG -Append
        "  To start: Start-ScheduledTask -TaskName '$TaskName'" | Tee-Object -FilePath $LOG -Append
    }
} catch {
    "  [ERROR] Failed to register task: $_" | Tee-Object -FilePath $LOG -Append
    "  Falling back to inline execution..." | Tee-Object -FilePath $LOG -Append
    & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $wrapperPath
}

# Verify phase scripts exist
"" | Tee-Object -FilePath $LOG -Append
"=== Phase script verification ===" | Tee-Object -FilePath $LOG -Append
foreach ($phase in $PHASE_SCRIPTS) {
    if (Test-Path $phase.file) {
        $size = (Get-Item $phase.file).Length
        "  OK: $($phase.file) ($size bytes)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  MISSING: $($phase.file)" | Tee-Object -FilePath $LOG -Append
    }
}

# Verify Python helper scripts exist
$helpers = @(
    "$REPO\scripts\_gen_diffusion_baseline.py",
    "$REPO\scripts\_gen_samst_wiki20.py",
    "$REPO\scripts\_compute_musiq_batch.py"
)
"" | Tee-Object -FilePath $LOG -Append
"=== Python helper verification ===" | Tee-Object -FilePath $LOG -Append
foreach ($h in $helpers) {
    if (Test-Path $h) {
        $size = (Get-Item $h).Length
        "  OK: $h ($size bytes)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  MISSING: $h" | Tee-Object -FilePath $LOG -Append
    }
}

"=== Orchestrator setup completed at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  Monitor: Get-Content '$LOG' -Wait" | Tee-Object -FilePath $LOG -Append
"  Task status: schtasks /Query /TN $TaskName /V /FO LIST" | Tee-Object -FilePath $LOG -Append
"  Stop task: schtasks /End /TN $TaskName" | Tee-Object -FilePath $LOG -Append
"  Results: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append
