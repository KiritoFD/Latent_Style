#!/bin/bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
python3 -c "
import zipfile
z = zipfile.ZipFile('/tmp/code_updates.zip')
z.extractall()
z.close()
print('Extracted code updates')
"