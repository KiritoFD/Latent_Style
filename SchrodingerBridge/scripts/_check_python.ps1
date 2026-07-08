gwmi Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId,CommandLine | Format-List
