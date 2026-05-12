import os
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

out_dir = r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/pre_trained_models"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "vgg_normalised.pth")

# custom vgg in S2WAT
custom = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
)

m = vgg19(weights=VGG19_Weights.DEFAULT)
feat = m.features
src_conv_idx = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
dst_conv_idx = [2,5,9,12,16,19,22,25,29,32,35,38,42,45,48,51]

# first 1x1 conv as identity
with torch.no_grad():
    custom[0].weight.zero_()
    custom[0].bias.zero_()
    for c in range(3):
        custom[0].weight[c,c,0,0] = 1.0

for s, d in zip(src_conv_idx, dst_conv_idx):
    custom[d].weight.data.copy_(feat[s].weight.data)
    custom[d].bias.data.copy_(feat[s].bias.data)

torch.save(custom.state_dict(), out_path)
print(out_path)
print('saved_ok', os.path.getsize(out_path))
