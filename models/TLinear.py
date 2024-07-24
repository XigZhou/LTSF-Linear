import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MLPBlock(nn.Module):
    def __init__(self, input_size, patch_len, dim, hidden_size, patch_num, selective_channel_dim, con1d_stride, con1d_kernel_size, con1d_padding):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

        self.mlp_mix = nn.Sequential(
            nn.Linear(input_size * patch_len, dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(dim, input_size * patch_len),
            nn.Linear(dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)

        )

        self.patch_mix = nn.Sequential(
            nn.Conv1d(in_channels=patch_num, out_channels=selective_channel_dim,
                      stride=con1d_stride, kernel_size=con1d_kernel_size, padding=con1d_padding),
            nn.Conv1d(in_channels=selective_channel_dim, out_channels=patch_num,
                      stride=con1d_stride, kernel_size=con1d_kernel_size, padding=con1d_padding),

        )



    def forward(self, x):

        # x = x + self.mlp_mix(x)
        # x = self.norm(x)
        # x = x + self.patch_mix(x)
        # x = self.norm(x)
        # return x
        # x = self.norm(x+self.mlp_mix(x))

        x = self.norm(x + self.patch_mix(x))
        return x


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        return self.fn(self.norm(x)) + x
class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self,config):
        super(Model, self).__init__()
        # 16,6,16,7 ----> 6,16,7
        self.input_size = config.input_size

        self.dim = config.dim
        self.hidden_size = config.hidden_size
        self.depth = config.depth

        self.fc = nn.Linear(self.hidden_size, self.input_size)
        # Decompsition Kernel Size
        self.kernel_size = config.kernel_size

        self.con1d_kernel_size = config.con1d_kernel_size
        self.con1d_stride = config.con1d_stride
        self.con1d_padding = config.con1d_padding

        self.decompsition = series_decomp(self.kernel_size)
        self.patch_num = config.patch_num
        self.patch_len = config.patch_len

        self.selective_channel_dim =self.patch_len*2

        self.mixer_blocks = nn.ModuleList([])
        self.patch_blocks = nn.ModuleList([])

        # for _ in range(self.depth):
        #     self.mixer_blocks.append(MixerBlock(self.input_size, self.patch_len, self.dim, self.hidden_size))
        for _ in range(self.depth):
            self.mixer_blocks.append(MLPBlock(self.input_size, self.patch_len, self.dim, self.hidden_size,self.patch_num, self.selective_channel_dim, self.con1d_stride, self.con1d_kernel_size,
                           self.con1d_padding))

        for _ in range(self.depth):
            self.patch_blocks.append(MLPBlock(self.input_size, self.patch_len, self.dim, self.hidden_size,self.patch_num, self.selective_channel_dim, self.con1d_stride, self.con1d_kernel_size,
                           self.con1d_padding))

        self.predictBlock = nn.Sequential(
            nn.Linear(self.input_size * self.patch_len, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.input_size * self.patch_len)
        )
        self.predictFN = nn.Linear(self.hidden_size, self.input_size * self.patch_len)
        self.mlp_mix = nn.Sequential(
            nn.Linear(self.input_size * self.patch_len, self.dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(dim, input_size * patch_len),
            nn.Linear(self.dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)

        )
        self.mlp_mix1 = nn.Sequential(
            nn.Linear(self.input_size * self.patch_len, self.dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(dim, input_size * patch_len),
            nn.Linear(self.dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)

        )


        self.rnn = nn.RNN(self.input_size , self.hidden_size, batch_first=True)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # global seasonal_init_row_connect, trend_init_row_connect, mul_connection, diff_row_connection
        seasonal_init, trend_init = self.decompsition(x)
        #切片

        x_s = seasonal_init.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        x_s = x_s.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]
        x_s = x_s.reshape(x_s.size(0), self.patch_num, self.input_size * self.patch_len)
        #切片

        x_t = trend_init.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        x_t = x_t.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]
        x_t = x_t.reshape(x_t.size(0),self.patch_num,self.input_size*self.patch_len)

        x_s = self.mlp_mix(x_s)
        x_t = self.mlp_mix1(x_t)

        for mixer_block in self.mixer_blocks:
            x_s = mixer_block(x_s)

        for patch_block in self.patch_blocks:

            x_t = patch_block(x_t)

        x = x_s+x_t
        # x = seasonal_init
        x = self.predictFN(x)

        # a, b, c, d = x.shape
        x = x.reshape(x.size(0),self.patch_num*self.patch_len,self.input_size)

        return x  # xigzhou read log : [batch_size,pre_dict_len,number of values]


if __name__ == '__main__':
    con1d = nn.Conv1d(in_channels=6, out_channels=12,
              stride=1, kernel_size=3,padding=1)
    a = torch.rand(16, 6, 256)
    print(con1d(a).shape)
