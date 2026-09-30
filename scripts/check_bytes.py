with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_620_spatial_bridge.sh', 'rb') as f:
    c = f.read()
print('Has CR:', b'\r' in c)
print('First 20 bytes:', repr(c[:20]))
