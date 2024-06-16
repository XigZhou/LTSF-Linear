import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from mlp_mixer_pytorch import MLPMixer

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        print("self.token_mix(x).shape----",self.token_mix(x).shape)
        x = x + self.token_mix(x)
        print("self.channel_mix(x).shape---",self.channel_mix(x).shape)
        x = x + self.channel_mix(x)

        return x

# pip install mlp-mixer-pytorch
class MLPMixer_other(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        # model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
        #                  dim=512, depth=8, token_dim=256, channel_dim=2048)
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        self.to1=nn.Conv2d(in_channels, dim, patch_size, patch_size)
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
        # stride: _size_2_t = 1,
        self.to2=Rearrange('b c h w -> b (h w) c')

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        print("x.shape>>>>", x.shape)
        # x = self.to_patch_embedding(x)
        x = self.to1(x)
        print("to1.shape>>>>", x.shape)
        x = self.to2(x)
        print("to2.shape>>>>", x.shape)
        print("self.to_patch_embedding(x)>>>>",x.shape)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)





if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    x = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)

    result = x(img)

    print(x.forward(img).shape)

    # test for layernorm
    y = torch.ones(1,196,768)
    layernorm = nn.LayerNorm(512)
    # print(layernorm(y).shape)

    #一维卷积 + 1*1卷积成理解
    z = torch.ones(1, 196, 512)
    conv1 = nn.Conv1d(in_channels=196, out_channels=196*4, kernel_size=1)
    # input = torch.randn(32, 35, 256)
    # input = input.permute(0, 2, 1)
    output = conv1(z)
    print("conv1 shape = ",output.shape)

    #全局平局池化层
    data = [[[1., 1.], [3., 4.]],[[1., 2.], [3., 4.]]]
    gl= torch.tensor(data)

    global_pooling = Reduce('b n c -> b c', 'mean')
    m = global_pooling(gl)
    print(m.shape)
    print(gl)
    print(m)
    print(global_pooling(z).shape)
    print(torch.ones(1,2))
    model = MLPMixer(image_size=224, channels=3, patch_size=16, dim=512, depth=1, num_classes=1000, expansion_factor=4,
             expansion_factor_token=0.5, dropout=0.)
    # model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
    #                  dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

    # model = MixerBlock( patch_size=16, num_classes=1000,
    #                  dim=512,token_dim=256, channel_dim=2048)
