Set-Location I:\Github\Latent_Style\SchrodingerBridge
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3b_stderr.log"
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3b_stdout.log"

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy Bypass -File I:\Github\Latent_Style\SchrodingerBridge\_p4_run_phase3b_ablations.ps1" `
    -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge" `
    -RedirectStandardError $errLog `
    -RedirectStandardOutput $outLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

Write-Output "PHASE3B_EXIT_CODE: $($proc.ExitCode)"
