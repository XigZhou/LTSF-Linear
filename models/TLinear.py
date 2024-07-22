import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.hidden_size = 2 * self.input_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        # Decompsition Kernel Size
        self.kernel_size = 25
        self.decompsition = series_decomp(self.kernel_size)
        self.batch_norm = nn.BatchNorm1d(self.input_size)

    def handle(self,x):
        x = x.unfold(dimension=1, size=16, step=16)
        x = x.permute(0, 1, 3, 2)  # x: [Batch, Input length, Channel] -----> [Batch, Patch num, Patch len, Channel]

        # 定义线性变换
        linear1 = nn.Linear(7, 28)  # 16,6,16,7
        linear2 = nn.Linear(28, 7)  # 16,6,16,7

        # 定义激活函数（例如ReLU）
        activation = nn.ReLU()
        drop_out = nn.Dropout(0.3)
        # print(x.shape)
        # 对最后一维做线性变换并应用激活函数
        x = activation(linear1(x))
        x = drop_out(x)
        x = activation(linear2(x))
        x = drop_out(x)

        for i in range(x.size(0)):
            # 提取第 i 个切片在第一个维度的所有数据
            slice_tensor = x[i]
            output, _ = self.rnn(slice_tensor)
            #             print('output.shape=',output.shape)#output.shape= torch.Size([6, 16, 6])
            output = self.fc(output)
            # print('output.shape=',output.shape)#output.shape= torch.Size([6, 16, 7])

            # 将更新后的 output 存回 x
            # output = output.view(-1, 7)
            # output = self.batch_norm(output)
            # output = output.view(6, 16, 7)
            x[i] = output

        # print('after handle x=', x.shape)

        # 获取张量的原始维度
        a, b, c, d = x.shape
        x = x.reshape(a, b * c, d)

        return x  # xigzhou read log : [batch_size,pre_dict_len,number of values]

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # global seasonal_init_row_connect, trend_init_row_connect, mul_connection, diff_row_connection
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init_x = self.handle(seasonal_init)
        trend_init_x = self.handle(trend_init)

        for i in range(10):
            seasonal_init_x = self.handle(seasonal_init+seasonal_init_x)
            trend_init_x = self.handle(trend_init+trend_init_x)




        x = seasonal_init_x+trend_init_x
        # x = seasonal_init

        return x  # xigzhou read log : [batch_size,pre_dict_len,number of values]
