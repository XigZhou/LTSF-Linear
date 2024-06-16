# import torch
# import numpy as np
# from torch import nn
# from einops.layers.torch import Rearrange, Reduce
# from mlp_mixer_pytorch import MLPMixer
#
# from functools import partial
#
#
#
# class PreNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         return self.fn(self.norm(x)) + x
#
# def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
#     inner_dim = int(dim * expansion_factor)
#     x = nn.Sequential(
#         dense(dim, inner_dim),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         dense(inner_dim, dim),
#         nn.Dropout(dropout)
#     )
#     # print(x)
#     return x
#
#
# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """
#
#     def __init__(self, *, look_back_windows, channels, number_of_values, dim, depth, predict_len, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
#         super(Model, self).__init__()
#         self.look_back_windows = look_back_windows
#         self.channels = channels
#         self.number_of_values = number_of_values
#         self.dim = dim
#         self.depth = depth
#         self.predict_len = predict_len
#         self.expansion_factor = expansion_factor
#         self.expansion_factor_token = expansion_factor_token
#         self.dropout = dropout
#
#     def forward(self, x):
#
#         num_patches = self.look_back_windows//self.number_of_values
#         chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
#
#         result= nn.Sequential(
#             Rearrange('b c s ss -> b c (s ss)'),
#
#             # nn.Linear((number_of_values ** 2) * channels, dim),#共享权重
#             nn.Linear((self.number_of_values ** 2) , self.dim),  # 共享权重，时间序列我没有channel
#             *[nn.Sequential(
#                 PreNormResidual(self.dim, FeedForward(num_patches, self.expansion_factor, self.dropout, chan_first)),##  return self.fn(self.norm(x)) + x
#                 PreNormResidual(self.dim, FeedForward(self.dim, self.expansion_factor_token, self.dropout, chan_last))
#
#             ) for _ in range(self.depth)],
#             nn.LayerNorm(self.dim),
#             # Reduce('b n c -> b c', 'mean'),
#             nn.Linear(self.dim, self.predict_len)
#         )
#         # print(result)
#         return result(x)
#
# #
# # if __name__ == "__main__":
# #     img = torch.ones([16, 13, 7, 7])
# #     x=Rearrange('b c s ss -> b c (s ss)')
# #     print(x(img).shape)
# #     print((7 ** 2) * 3)
# #     z=nn.Linear((7 ** 2), 128)
# #     print(z(x(img)).shape)
# #
# #
# #     model = TimeMLPMixer(look_back_windows=91, channels=3, number_of_values=7, dim=128, depth=1, predict_len=7*7, expansion_factor=4,
# #              expansion_factor_token=0.5, dropout=0.)
# #
# #
# #     parameters = filter(lambda p: p.requires_grad, model.parameters())
# #     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# #     print('Trainable Parameters: %.3fM' % parameters)
# #
# #     out_img = model(img)
# #
# #     result = Rearrange('b c (s ss) -> b c s ss',s=7,ss=7)
# #     print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
# #     print("Shape of result :", result(out_img).shape)
# #     # model = MixerBlock( patch_size=16, num_classes=1000,
# #     #                  dim=512,token_dim=256, channel_dim=2048)
from torch import nn, reshape
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # print("row14,x.shape=",x.shape)
        # print("rself.norm(x)=", self.norm(x).shape)
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    x = nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )
    # print(x)
    return x

def MLPMixer(*, look_back_windows, channels, number_of_values, dim, depth, predict_len, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):

    num_patches  = look_back_windows // number_of_values
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear


    result= nn.Sequential(
        # Rearrange('b (c r) l ) -> b c (r l)', r=number_of_values,c=num_patches),
        # x=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),

        nn.Linear((number_of_values ** 2) , dim),#共享权重
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),##  return self.fn(self.norm(x)) + x
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))

        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        # Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, number_of_values ** 2)
    )



    return result