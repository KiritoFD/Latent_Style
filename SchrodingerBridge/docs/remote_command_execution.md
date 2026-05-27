# Remote Command Execution Notes

Last verified: 2026-05-27.

This document is the source of truth for running SchrodingerBridge commands on
the current remote Windows GPU box.

## Canonical SSH Target

Use this exact SSH endpoint:

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

For one-shot commands:

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command `"Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'; python --version`""
```

Verified facts:

```text
remote repo: I:\Github\Latent_Style\SchrodingerBridge
python: Python 3.12.10
gpu: NVIDIA Graphics Device, 12288 MiB
```

## Use Encoded PowerShell For Multi-Line Commands

Inline quoting through local PowerShell -> ssh -> remote PowerShell is fragile.
For anything longer than one simple command, encode the remote script locally:

```powershell
$script = @'
Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'
python --version
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
'@
$enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($script))
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -EncodedCommand $enc"
```

PowerShell `-EncodedCommand` expects UTF-16LE, which is what
`[System.Text.Encoding]::Unicode` emits.

## Copy Files To Remote

Prefer `scp` with forward slashes on the remote destination:

```powershell
scp -P 2222 -o LogLevel=ERROR `
  SchrodingerBridge\tools\experiments\run_vae_backend_256_probe.py `
  administrator@100.115.18.62:"I:/Github/Latent_Style/SchrodingerBridge/tools/experiments/run_vae_backend_256_probe.py"
```

For several files, copy them to `_codex_tmp/` first, then move them on the
remote with an encoded PowerShell command.

## Long-Running Jobs

Do not use `Start-Process` for training launched through SSH. Use Windows Task
Scheduler (`schtasks`) so the job survives SSH disconnects.

Create a remote `.bat` wrapper first. Example tokenizer job:

```bat
@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
if not exist exp\vae_backend_256_probe mkdir exp\vae_backend_256_probe
echo [%date% %time%] start neutral tokenizer spiral > exp\vae_backend_256_probe\tokenizer_neutral_spiral_status.txt
python -u tools\experiments\run_vae_backend_256_probe.py --variants ema_style_vocab_neutral_w34,ema_style_vocab_neutral_w36_stylepush --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents > exp\vae_backend_256_probe\tokenizer_neutral_spiral_stdout.log 2> exp\vae_backend_256_probe\tokenizer_neutral_spiral_stderr.log
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] exit %EXITCODE% >> exp\vae_backend_256_probe\tokenizer_neutral_spiral_status.txt
exit /b %EXITCODE%
```

Upload it:

```powershell
scp -P 2222 -o LogLevel=ERROR `
  SchrodingerBridge\_codex_tmp\run_style_vocab_neutral_spiral_remote.bat `
  administrator@100.115.18.62:"I:/Github/Latent_Style/SchrodingerBridge/_codex_tmp/run_style_vocab_neutral_spiral_remote.bat"
```

Create and start the scheduled task:

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /delete /tn LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL /f"

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /create /tn LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL /tr I:\Github\Latent_Style\SchrodingerBridge\_codex_tmp\run_style_vocab_neutral_spiral_remote.bat /sc once /st 00:00 /f && schtasks /run /tn LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL"
```

## Status Checks

Query the scheduled task:

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /query /tn LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL /fo LIST /v"
```

Check GPU and active SchrodingerBridge Python processes:

```powershell
$script = @'
Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'SchrodingerBridge|run_vae_backend_256_probe|src\\run.py' } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
'@
$enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($script))
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -EncodedCommand $enc"
```

Tail logs:

```powershell
$script = @'
Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'
Get-Content 'exp\vae_backend_256_probe\tokenizer_neutral_spiral_status.txt' -Tail 20 -ErrorAction SilentlyContinue
Get-Content 'exp\vae_backend_256_probe\tokenizer_neutral_spiral_stdout.log' -Tail 80 -ErrorAction SilentlyContinue
Get-Content 'exp\vae_backend_256_probe\tokenizer_neutral_spiral_stderr.log' -Tail 80 -ErrorAction SilentlyContinue
'@
$enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($script))
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -EncodedCommand $enc"
```

If waiting is needed, sleep locally for two minutes and re-check:

```powershell
Start-Sleep -Seconds 120
```

## Stop A Job

```powershell
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /end /tn LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL"
```

If a Python process survives, inspect before killing it:

```powershell
$script = @'
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'SchrodingerBridge|run_vae_backend_256_probe|src\\run.py' } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
'@
$enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($script))
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -EncodedCommand $enc"
```

## Current Tokenizer Mainline

The clean tokenizer spiral must not manually boost Hayao through sampling or
loss weights. Hayao is a diagnostic slice only.

Current main variants:

```text
ema_style_vocab_neutral_w34
ema_style_vocab_neutral_w36_stylepush
```

Expected output files:

```text
I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_probe\tokenizer_neutral_spiral_status.txt
I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_probe\tokenizer_neutral_spiral_stdout.log
I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_probe\tokenizer_neutral_spiral_stderr.log
I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_probe\vae_backend_256_results.csv
```
