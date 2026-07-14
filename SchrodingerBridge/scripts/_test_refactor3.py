import sys
sys.path.insert(0, 'src')
from spectral_bridge620 import SpectralODEBridge620, SpectralVelocityHead, _wct_match_fiber
from spectral_losses620 import SpectralODEObjective620
from blocks620 import SpatialBridgeBlock620
from spectral620 import dwt2_lowpass, dwt2_haar, dwt2_haar_lowpass
import spectral_bridge620 as sb

# Verify _wct_match_fiber_keep_mean is gone
assert not hasattr(sb, '_wct_match_fiber_keep_mean'), '_wct_match_fiber_keep_mean should be removed'
# Verify _wct_match_fiber is still there (used by per_subband mode)
assert hasattr(sb, '_wct_match_fiber'), '_wct_match_fiber should still exist'

print('OK: all imports pass after refactor batch 3')
print('OK: _wct_match_fiber_keep_mean removed')
print('OK: _wct_match_fiber retained')
