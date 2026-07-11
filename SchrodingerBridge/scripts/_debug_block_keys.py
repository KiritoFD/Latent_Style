import sys
sys.path.insert(0, 'src')
from blocks620 import SpatialBridgeBlock620
b = SpatialBridgeBlock620(dim=64, num_heads=4)
asg_keys = [k for k in b.state_dict().keys() if 'asg' in k]
print(f'Direct block asg keys: {len(asg_keys)}')
print(f'asg keys: {asg_keys}')
print(f'All keys: {list(b.state_dict().keys())}')
