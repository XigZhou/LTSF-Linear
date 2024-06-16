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
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2,1)  # xigzhou read log : [batch_size,number of values,lookback windows]
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id', type=str, required=False, default='ETTh1_91', help='model id')
    parser.add_argument('--model', type=str, required=False, default='MLPMixer',
                        help='model name, options: [Autoformer, Informer, Transformer,DLinear,MLPMixer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=91, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=91, help='prediction sequence length')



    # MLPMixer
    parser.add_argument('--look_back_windows', type=int, default=91,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--channels', type=int, default=3,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--number_of_values', type=int, default=7, help='decoder input size')
    parser.add_argument('--dim', type=int, default=128, help='output size')
    parser.add_argument('--depth', type=int, default=2, help='dimension of model')
    parser.add_argument('--predict_len', type=int, default=7 * 7, help='output size')
    parser.add_argument('--expansion_factor', type=int, default=4, help='dimension of model')
    parser.add_argument('--expansion_factor_token', type=int, default=0.5, help='output size')

    # TimeMLPMixer(look_back_windows=91, channels=3, number_of_values=7, dim=128, depth=1, predict_len=7 * 7,
    #              expansion_factor=4,
    #              # #              expansion_factor_token=0.5, dropout=0.)

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Formers
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    model = Model(args)
    print("now modle is ", model)
    print("now modle is ", type(model))
    # print("self.model.parameters() is ", self.model.parameters())
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)