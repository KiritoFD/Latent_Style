# Create wikiarts-15 256 test set on remote
$script = @'
cd C:\Users\Administrator
& "C:\Program Files\Python312\python.exe" create_wikiarts15_256_test.py
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
