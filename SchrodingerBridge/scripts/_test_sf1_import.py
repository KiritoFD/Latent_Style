"""Quick import test for 712 SF1 subband time schedule."""
import sys
sys.path.insert(0, 'src')
from spectral620 import subband_gamma, subband_gamma_tensor
print('gamma(0.25, early_peak)=', subband_gamma(0.25, 'early_peak'))
print('gamma(0.75, early_peak)=', subband_gamma(0.75, 'early_peak'))
print('gamma(0.75, late_burst)=', subband_gamma(0.75, 'late_burst'))
print('gamma(0.25, late_burst)=', subband_gamma(0.25, 'late_burst'))
from spectral_losses620 import SpectralODEObjective620
print('losses import OK')
from spectral_bridge620 import SpectralODEBridge620
print('bridge import OK')
print('ALL TESTS PASSED')
