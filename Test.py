import torch
if __name__ == '__main__':
    lst = [
       [[1., 2., 3.,4., 5., 6.,7., 8.], [1., 2., 3.,4., 5., 6.,7., 8.], [1., 2., 3.,4., 5., 6.,7., 8.]],
        [  [1., 2., 3.,4., 5., 6.,7., 8.], [1., 2., 3.,4., 5., 6.,7., 8.], [1., 2., 3.,4., 5., 6.,7., 8.]]
    ]
    tensor1 = torch.tensor(lst)

    x_enc = torch.tensor(lst)

    print(x_enc)
    y=torch.reshape(x_enc,[2,4,6])
    print(y.shape)
    print(y[:,:,2])

    print(torch.unsqueeze(x_enc))

    # print(x_enc)
    # means = x_enc.mean(1, keepdim=True).detach()
    # print(means.shape)
    # print(means)
    #
    # x_enc = x_enc - means
    # print(x_enc)
    # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    # print(torch.var(x_enc, dim=1, keepdim=True, unbiased=False))
    # print(torch.var(x_enc, dim=0, keepdim=True, unbiased=False))
    # x_enc /= stdev

    # torch.Size([2, 1])
    # torch.Size([1, 3])
    print('----------------------------------------------------------------------')


