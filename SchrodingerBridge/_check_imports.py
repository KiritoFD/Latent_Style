import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
try:
    from spectral_bridge620 import SpectralODEBridge620
    print("OK: spectral_bridge620 imported")
except Exception as e:
    print(f"FAIL: {e}")
try:
    from config_schema import ModelConfig, BridgeConfig
    print("OK: config_schema imported")
except Exception as e:
    print(f"FAIL: {e}")
