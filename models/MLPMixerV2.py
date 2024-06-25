import argparse

import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from mlp_mixer_pytorch import MLPMixer

from functools import partial



class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormResidual, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    x = nn.Sequential(
        dense(dim, inner_dim),
        # nn.GELU(),
        nn.ReLU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )
    # print(x)
    return x

def splitTensor(x, look_back_windows, split_len):
    if look_back_windows % split_len != 0:
        print('error')
    else:
        cnt = look_back_windows / split_len
        start_index = 0
        end_index = start_index + split_len
        list = []
        while cnt > 0:
            one_patch = x[:, :, start_index:end_index]
            one_patch = torch.unsqueeze(one_patch, dim=1)
            list.append(one_patch)

            start_index = end_index
            end_index = start_index + split_len

            cnt = cnt - 1
            # print("cnt =",cnt)
            # print("one_patch.shape=",one_patch.shape)
        result = torch.cat(list, dim=1)
        last = nn.Flatten(-2, -1)
        result= last(result)
        # print("result.shape=",result.shape)
    return  result,list

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.look_back_windows = config.look_back_windows
        self.channels = config.channels
        self.number_of_values = config.number_of_values
        self.dim = config.dim
        self.depth = config.depth
        self.predict_len = config.predict_len
        self.expansion_factor = config.expansion_factor
        self.expansion_factor_token = config.expansion_factor_token

        self.split_len = config.split_len
        self.use_norm = config.use_norm
        self.dropout = 0.3

        self.pix = self.number_of_values * self.split_len
        self.num_patches = self.look_back_windows // self.split_len
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.framework = nn.Sequential(
            # Rearrange('b c s ss -> b c (s ss)'),

            # nn.Linear((number_of_values ** 2) * channels, dim),#共享权重
            nn.Linear(self.pix , self.dim),  # 共享权重，时间序列我没有channel
            *[nn.Sequential(

                PreNormResidual(self.dim, FeedForward(self.num_patches, self.expansion_factor, self.dropout, self.chan_first)),##  return self.fn(self.norm(x)) + x
                # PreNormResidual(self.dim, FeedForward(self.dim, self.expansion_factor_token, self.dropout, self.chan_last))

            ) for _ in range(self.depth)],
            nn.LayerNorm(self.dim),
            # Reduce('b n c -> b c', 'mean'),
            nn.Linear(self.dim, self.predict_len)
        )
        self.final_output = nn.Linear(self.num_patches,self.number_of_values)

    def forward(self, x):
        x = x.permute(0,2,1)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x,_= splitTensor(x,self.look_back_windows,self.split_len)

        x = self.framework(x)

        x = self.final_output(x.permute(0,2,1))

        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False,
                        help='perform training on full input dataset without validation and testing')
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
    parser.add_argument('--look_back_windows', type=int, default=96,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--channels', type=int, default=3,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--number_of_values', type=int, default=7, help='decoder input size')
    parser.add_argument('--dim', type=int, default=256, help='output size')

    parser.add_argument('--depth', type=int, default=4, help='dimension of model')
    parser.add_argument('--predict_len', type=int, default=96, help='output size')
    parser.add_argument('--expansion_factor', type=int, default=4, help='dimension of model')
    parser.add_argument('--expansion_factor_token', type=int, default=0.5, help='output size')

    parser.add_argument('--split_len', type=int, default=6, help='dimension of model')

    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')


    # TimeMLPMixer(look_back_windows=91, channels=3, number_of_values=7, dim=128, depth=1, predict_len=7 * 7,
    #              expansion_factor=4,
    #              # #              expansion_factor_token=0.5, dropout=0.)

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    # Formers
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=21,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
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

    print(args)

    matrix = torch.arange(1., 10753.).reshape(16, 7, 96)
    print("原始矩阵:")
    print(matrix.shape)

    mixer = Model(args)

    y = mixer(matrix)

    print(y.shape)