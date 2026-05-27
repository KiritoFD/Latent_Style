@echo off
schtasks /Create /TN LANCET_BodyDualFull /TR "cmd.exe /c I:\Github\Latent_Style\SchrodingerBridge\run_ema_bodydual_full_remote.cmd" /SC ONCE /ST 23:59 /F
schtasks /Run /TN LANCET_BodyDualFull
