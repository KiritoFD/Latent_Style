# Run the metrics extraction script on remote via Python
$script = @'
cd C:\Users\Administrator
& "C:\Program Files\Python312\python.exe" _extract_wikiarts15_metrics.py
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
