#!/bin/bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
python3 -c "
import base64
with open('/tmp/code_updates.b64', 'r') as f:
    data = base64.b64decode(f.read())
with open('/tmp/code_updates.zip', 'wb') as f:
    f.write(data)
print('Decoded zip')

import zipfile
z = zipfile.ZipFile('/tmp/code_updates.zip')
z.extractall()
z.close()
print('Extracted code updates')
"