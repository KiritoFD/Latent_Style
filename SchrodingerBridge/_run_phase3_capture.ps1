Set-Location I:\Github\Latent_Style\SchrodingerBridge
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3_stderr.log"
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3_stdout.log"

# Use Start-Process to avoid fortrl error and capture stderr/stdout
$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy Bypass -File I:\Github\Latent_Style\SchrodingerBridge\_p4_run_phase3_ablations.ps1" `
    -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge" `
    -RedirectStandardError $errLog `
    -RedirectStandardOutput $outLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

Write-Output "PHASE3_EXIT_CODE: $($proc.ExitCode)"
