$logPath = 'I:\Github\Latent_Style\SchrodingerBridge\mnt\i\Github\Latent_Style\SchrodingerBridge\exp\20250618_lite_ot_vertical\h0_vertical_fm\resume.out'
if (Test-Path $logPath) {
    Get-Item $logPath | Select-Object Length, LastWriteTime
    Get-Content $logPath -Tail 80
} else {
    Write-Output "resume.out missing"
}
