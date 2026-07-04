$sshCmd = "C:\Windows\System32\OpenSSH\ssh.exe"

# Kill existing tmux session
Write-Host "=== Killing phase2 session ===" -ForegroundColor Cyan
$remoteCmd = """wsl -- bash -c 'tmux kill-session -t phase2 2>&1 || echo no_session'"""
$p = New-Object System.Diagnostics.Process
$p.StartInfo.FileName = $sshCmd
$p.StartInfo.Arguments = "-p 2222 administrator@100.115.18.62 $remoteCmd"
$p.StartInfo.UseShellExecute = $false
$p.StartInfo.RedirectStandardOutput = $true
$p.Start() | Out-Null
$p.WaitForExit()

Start-Sleep -Seconds 2

# Restart phase2
Write-Host "`n=== Final Restart of Phase 2 ===" -ForegroundColor Yellow
$remoteCmd2 = """wsl -- python3 /mnt/c/Users/Administrator/start_phase2.py"""
$p2 = New-Object System.Diagnostics.Process
$p2.StartInfo.FileName = $sshCmd
$p2.StartInfo.Arguments = "-p 2222 administrator@100.115.18.62 $remoteCmd2"
$p2.StartInfo.UseShellExecute = $false
$p2.StartInfo.RedirectStandardOutput = $true
$p2.StartInfo.RedirectStandardError = $true
$p2.Start() | Out-Null
$o2 = $p2.StandardOutput.ReadToEnd()
$e2 = $p2.StandardError.ReadToEnd()
$p2.WaitForExit()

Write-Host $o2
if ($e2) { Write-Host "STDERR: $e2" -ForegroundColor Red }

# Wait and verify training is running
Start-Sleep -Seconds 25
Write-Host "`n=== Verification Check (after 25s) ===" -ForegroundColor Green
$remoteCmd3 = """wsl -- tail -30 /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/fc_sb_kernel7/focused.log"""
$p3 = New-Object System.Diagnostics.Process
$p3.StartInfo.FileName = $sshCmd
$p3.StartInfo.Arguments = "-p 2222 administrator@100.115.18.62 $remoteCmd3"
$p3.StartInfo.UseShellExecute = $false
$p3.StartInfo.RedirectStandardOutput = $true
$p3.StartInfo.RedirectStandardError = $true
$p3.Start() | Out-Null
$o3 = $p3.StandardOutput.ReadToEnd()
$e3 = $p3.StandardError.ReadToEnd()
$p3.WaitForExit()
Write-Host $o3
