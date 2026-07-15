import sys
sys.path.insert(0, 'src')
from spectral_losses620 import SpectralODEObjective620
print('OK: spectral_losses620 imports')
# Verify contrastive SWD is gone
import spectral_losses620 as sl
assert not hasattr(sl, '_style_contrastive_swd'), '_style_contrastive_swd should be removed'
print('OK: contrastive SWD removed')
