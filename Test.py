import torch
import torch.nn as nn

if __name__ == '__main__':
    input = torch.randn(16,7,6,16)

    flatten = nn.Flatten(start_dim=-2)

    print(  flatten(input).shape)