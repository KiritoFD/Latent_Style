param(
    [int]$ProcessId
)

Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -eq $ProcessId } |
    Select-Object ProcessId, Name, CommandLine
