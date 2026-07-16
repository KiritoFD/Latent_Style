Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" |
    Select-Object ProcessId, Name, CommandLine |
    Format-List
