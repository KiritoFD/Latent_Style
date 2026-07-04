import sys
sys.path.insert(0, r'G:\GitHub\Latent_Style\SchrodingerBridge')

print("=== Testing Imports ===")
try:
    from src.config_schema import *
    print("CONFIG_SCHEMA OK")
except Exception as e:
    print(f"CONFIG_SCHEMA FAILED: {e}")

try:
    from src.model620 import *
    print("MODEL OK")
except Exception as e:
    print(f"MODEL FAILED: {e}")

try:
    from src.losses620 import *
    print("LOSSES OK")
except Exception as e:
    print(f"LOSSES FAILED: {e}")

try:
    import run
    print("RUN OK")
except Exception as e:
    print(f"RUN FAILED: {e}")

print("=== Import Test Complete ===")