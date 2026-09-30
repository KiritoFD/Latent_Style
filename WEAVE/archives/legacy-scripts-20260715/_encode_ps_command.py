import base64
import sys

ps = sys.argv[1]
encoded = base64.b64encode(ps.encode("utf-16-le")).decode()
print(encoded)
