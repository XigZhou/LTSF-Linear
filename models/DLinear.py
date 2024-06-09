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

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.len_of_value = 7  # hard code only for quick test by xigzhou

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            #
            # author method
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            """ test1: same predict loss
                        self.Linear_Seasonal = nn.Sequential(nn.Linear(self.seq_len, self.seq_len*2),
                                      nn.ReLU(),
                                      nn.Linear(self.seq_len*2, self.pred_len),
                                      )

                        self.Linear_Trend = nn.Sequential(nn.Linear(self.seq_len, self.seq_len*2),
                                      nn.ReLU(),
                                      nn.Linear(self.seq_len*2, self.pred_len),
                                      )
            """
            """ test 2 
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            #压缩每行成为一个变量
            self.Linear_values_connection =  nn.Linear(self.len_of_value,1)
            #把每行的变量，拼接成为新的一行，from, [batch_size，seq_len,1] to [batch_size,1,seq_len], to [batch_size,len_of_value,pred_len]
            self.Linear_rows_connection = nn.Linear(self.seq_len,self.pred_len)
            """

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

            """ test 3

            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # 压缩每行成为一个变量
            self.Linear_values_connection = nn.Linear(self.len_of_value, 1)
            # 把每行的变量，拼接成为新的一行，from, [batch_size，seq_len,1] to [batch_size,1,seq_len], to [batch_size,len_of_value,pred_len]
            self.Linear_rows_connection = nn.Linear(self.seq_len, self.channels)

            self.merge = nn.Linear(self.seq_len, self.pred_len)
            """

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # global seasonal_init_row_connect, trend_init_row_connect, mul_connection, diff_row_connection
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                       1)  # xigzhou read log : [batch_size,number of values,lookback windows]
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            value_connection = self.Linear_values_connection(x)
            row_connection = value_connection.permute(0, 2, 1)
            # print(torch.matmul(value_connection, row_connection).shape)
            mul_connection = self.Linear_rows_connection(torch.matmul(value_connection, row_connection))
            diff_row_connection = self.merge(mul_connection.permute(0, 2, 1))
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            """below new add for test2
            1)类似残差?
            value_connection = self.Linear_values_connection(x)
            # print(value_connection.shape)
            row_connect =  self.Linear_rows_connection(value_connection.permute(0,2,1))
            # print(row_connect.shape)

             x = seasonal_output + trend_output +row_connect # change for test2
            2)
            seasonal_init_value_connection = self.Linear_values_connection(seasonal_init.permute(0,2,1))
            trend_init_value_connection  = self.Linear_values_connection(trend_init.permute(0,2,1))

            seasonal_init_row_connect = self.Linear_rows_connection(seasonal_init_value_connection.permute(0, 2, 1))
            trend_init_row_connect  = self.Linear_rows_connection(trend_init_value_connection.permute(0, 2, 1))
            x = seasonal_output + trend_output + seasonal_init_row_connect + trend_init_row_connect# change for test2

            3)
            value_connection = self.Linear_values_connection(x)
            row_connection = value_connection.permute(0, 2, 1)
            # print(torch.matmul(value_connection, row_connection).shape)
            mul_connection = self.Linear_rows_connection(torch.matmul(value_connection, row_connection))
            diff_row_connection = self.merge(mul_connection.permute(0, 2, 1))

             # value_connection = self.Linear_values_connection(x)
            # row_connection = value_connection.permute(0, 2, 1)
            # # print(torch.matmul(value_connection, row_connection).shape)
            # mul_connection = self.Linear_rows_connection(torch.matmul(value_connection, row_connection))
            # diff_row_connection = self.merge(mul_connection.permute(0, 2, 1))
            """

        x = seasonal_output + trend_output
        # x = seasonal_output + trend_output + diff_row_connection
        return x.permute(0, 2, 1)  # xigzhou read log : [batch_size,pre_dict_len,number of values]
