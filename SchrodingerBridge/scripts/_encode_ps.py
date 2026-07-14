"""Generate base64 EncodedCommand for PowerShell via SSH to avoid quoting hell."""
import base64
import sys

cmd = sys.stdin.read()
encoded = base64.b64encode(cmd.encode("utf-16-le")).decode()
print(encoded)
