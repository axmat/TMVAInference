import torch
from torch import nn
import numpy as np

conv = nn.Conv2d(2, 4, (3,3), groups=2)
x =torch.from_numpy(np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3), dtype=torch.float32)
y = conv.forward(x)

print(y)
