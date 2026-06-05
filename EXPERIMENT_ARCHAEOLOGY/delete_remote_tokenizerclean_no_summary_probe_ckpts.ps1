param(
    [Parameter(Mandatory = $true)]
    [string]$PolicyCsv,

    [string]$Root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$expRoot = Join-Path $Root 'exp'
if (!(Test-Path $expRoot)) {
    throw "Expected exp root not found: $expRoot"
}

$policy = Import-Csv -LiteralPath $PolicyCsv |
    Where-Object { $_.policy_action -eq 'delete_probe_calibration_checkpoint_only' }

$ledger = New-Object System.Collections.Generic.List[object]
$resolvedExpRoot = (Resolve-Path -LiteralPath $expRoot).Path

foreach ($row in $policy) {
    $dirPath = Join-Path $expRoot $row.exp_dir
    if (!(Test-Path $dirPath)) {
        $ledger.Add([pscustomobject]@{
            timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            action = 'missing_dir'
            exp_dir = $row.exp_dir
            path = $dirPath
            size_mb = ''
            reason = $row.reason
        })
        continue
    }

    $resolvedDir = (Resolve-Path -LiteralPath $dirPath).Path
    if (!$resolvedDir.StartsWith($resolvedExpRoot + '\')) {
        throw "Refusing to inspect outside exp root: $resolvedDir"
    }

    $files = @(Get-ChildItem -LiteralPath $resolvedDir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in '.pt', '.ckpt', '.pth' })

    foreach ($file in $files) {
        $resolvedFile = (Resolve-Path -LiteralPath $file.FullName).Path
        if (!$resolvedFile.StartsWith($resolvedDir + '\')) {
            throw "Refusing to delete outside selected exp dir: $resolvedFile"
        }

        $sizeMb = $file.Length / 1MB
        Remove-Item -LiteralPath $resolvedFile -Force
        $ledger.Add([pscustomobject]@{
            timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            action = 'deleted_checkpoint'
            exp_dir = $row.exp_dir
            path = $resolvedFile
            size_mb = ('{0:N3}' -f $sizeMb)
            reason = $row.reason
        })
    }
}

$ledger | ConvertTo-Csv -NoTypeInformation
