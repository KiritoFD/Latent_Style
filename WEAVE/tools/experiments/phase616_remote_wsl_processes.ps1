Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'wsl.exe' } |
    Select-Object ProcessId, CommandLine
