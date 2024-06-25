from torchtsmixer import tsmixer, TSMixer

import torch



m = TSMixer(10, 5, 2, output_channels=4)
x = torch.randn(3, 10, 2)
y = m(x)
print(y.shape)
