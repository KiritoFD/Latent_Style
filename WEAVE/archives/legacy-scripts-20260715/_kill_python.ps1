$ErrorActionPreference = "SilentlyContinue"
Stop-Process -Id 23652 -Force
Stop-Process -Name python -Force
Write-Output "killed"
