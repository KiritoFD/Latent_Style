import zipfile
import os
import base64

# Decode base64
with open(r'C:\Users\Administrator\code_updates.b64', 'r') as f:
    data = base64.b64decode(f.read())
with open(r'C:\Users\Administrator\code_updates.zip', 'wb') as f:
    f.write(data)
print('Decoded zip')

# Extract to target location
os.chdir(r'G:\GitHub\Latent_Style\SchrodingerBridge')
z = zipfile.ZipFile(r'C:\Users\Administrator\code_updates.zip')
for name in z.namelist():
    target = os.path.join('.', name)
    if name.endswith('/'):
        os.makedirs(target, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'wb') as f:
            f.write(z.read(name))
z.close()
print('Extracted code updates')
