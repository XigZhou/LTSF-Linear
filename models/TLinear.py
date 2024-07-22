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


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self,config):
        super(Model, self).__init__()
        # 16,6,16,7 ----> 6,16,7
        self.input_size = 7

        self.dim = 28
        self.hidden_size = 2 * self.input_size

        self.fc = nn.Linear(self.hidden_size, self.input_size)
        # Decompsition Kernel Size
        self.kernel_size = 25

        self.con1d_kernel_size = 3
        self.con1d_stride = 1
        self.con1d_padding = 1

        self.decompsition = series_decomp(self.kernel_size)
        self.patch_num = 6
        self.patch_len = 16

        self.selective_channel_dim =self.patch_len*2


        self.MLP_block = nn.Sequential(
                    nn.Linear(self.input_size, self.dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.dim, self.hidden_size),
                    nn.Dropout(0.1)
                )

        self.n1 = nn.Conv1d(in_channels=self.patch_len, out_channels=self.selective_channel_dim,
                  stride=self.con1d_stride, kernel_size=self.con1d_kernel_size, padding=self.con1d_padding)
        self.n2 = nn.Conv1d(in_channels=self.selective_channel_dim, out_channels=self.patch_len,
                  stride=self.con1d_stride, kernel_size=self.con1d_kernel_size, padding=self.con1d_padding)
        self.n3 = nn.Linear(self.hidden_size, self.input_size)
        self.n4 = nn.ReLU()
        self.n5 = nn.Dropout(0.1)


        self.rnn = nn.RNN(self.input_size , self.hidden_size, batch_first=True)

    def handle(self,x):



        # for i in range(x.size(0)):
        #     # 提取第 i 个切片在第一个维度的所有数据

        slice_input = x


        slice_input = self.n2(self.n1(x))



        # slice_input = self.n3(slice_input)
        # print('n3=', slice_input.shape)
        # slice_input = self.n4(slice_input)
        # print('n4=', slice_input.shape)
        # slice_input = self.n5(slice_input)
        # print('n5=', slice_input.shape)
        # x = self.selective_patch_info(x)

        return x+slice_input  # xigzhou read log : [batch_size,pre_dict_len,number of values]

    # def forward(self, x):
    #     # x: [Batch, Input length, Channel]
    #     # global seasonal_init_row_connect, trend_init_row_connect, mul_connection, diff_row_connection
    #     seasonal_init, trend_init = self.decompsition(x)
    #     #切片
    #     x_s = self.MLP_block(seasonal_init)
    #     print('x_s=',x_s.shape)
    #     x_s = x_s.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
    #     # z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
    #     print('x_s2=', x_s.shape)
    #     x_s = x_s.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]
    #     print('x_s=', x_s.shape)
    #     #切片
    #     x_t = self.MLP_block(trend_init)
    #     x_t = x_t.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
    #     x_t = x_t.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]
    #
    #     for i in range(x_s.size(0)):
    #
    #         x_s[i] = self.handle(x_s[i])
    #     for i in range(x_t.size(0)):
    #         x_t[i] = self.handle(x_t[i])
    #
    #     x = x_s+x_t
    #     # x = seasonal_init
    #
    #     print('reshape v before,', x.shape)
    #     a, b, c, d = x.shape
    #     x = x.reshape(a, b * c, d)
    #     print('reshape before,',x.shape)
    #     x = self.n3(x)
    #     print('reshape after,', x.shape)
    #     return x  # xigzhou read log : [batch_size,pre_dict_len,number of values]
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # global seasonal_init_row_connect, trend_init_row_connect, mul_connection, diff_row_connection
        seasonal_init, trend_init = self.decompsition(x)
        #切片
        x_s = self.MLP_block(seasonal_init)
        print('x_s=',x_s.shape)
        x_s = x_s.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        # z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        print('x_s2=', x_s.shape)
        x_s = x_s.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]
        print('x_s=', x_s.shape)
        #切片
        x_t = self.MLP_block(trend_init)
        x_t = x_t.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        x_t = x_t.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]

        for i in range(x_s.size(0)):

            x_s[i] = self.handle(x_s[i])
        for i in range(x_t.size(0)):
            x_t[i] = self.handle(x_t[i])

        x = x_s+x_t
        # x = seasonal_init

        print('reshape v before,', x.shape)
        a, b, c, d = x.shape
        x = x.reshape(a, b * c, d)
        print('reshape before,',x.shape)
        x = self.n3(x)
        print('reshape after,', x.shape)
        return x  # xigzhou read log : [batch_size,pre_dict_len,number of values]
