import torch
from torch import nn as nn

# # 创建一个矩阵
# matrix = torch.arange(1, 49).reshape(2, 3, 8)
# print("原始矩阵:")
# print(matrix)
#
#
#
# # 使用索引分割矩阵
# # 分割中间的1x1部分
# print("\n使用索引分割:")
#
# a= matrix[:,:, 0:2]
# b= matrix[:,:, 2:4]
#
#
# flatten = nn.Flatten()
#
# print(flatten(a))
# print(flatten(b))
#
# z = torch.unsqueeze(a,dim=1)
# zz = torch.unsqueeze(b,dim=1)
#
# r = torch.cat((z,zz),dim=1)
#
# print(z.shape)
# print(zz.shape)
# print(r.shape)
#
# print(z)
# print(zz)
# print(r)
#
# print('>>>>>>>>>>>>>>>')
#
# last = nn.Flatten(-2,-1)
# print(last(r).shape)
# print('>>>>>>>>>>>>>>>')

# # 分割左上角的1x2部分
# print(matrix[:1, :2])
#
# # 分割右下角的2x2部分
# print(matrix[-2:, -2:])


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
            # print(one_patch)
        result = torch.cat(list, dim=1)
        last = nn.Flatten(-2, -1)
        result= last(result)

    return  result,list

if __name__ == '__main__':
    # 创建一个矩阵
    matrix = torch.arange(1, 10753).reshape(16, 7, 96)
    print("原始矩阵:")
    print(matrix)

    result,list=splitTensor(matrix,96,16)

    print(result.shape)
    # print(result)