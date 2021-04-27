import torch
import torch.nn as nn
from STTNS import STTNSNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
parameters:
A:邻接矩阵
in_channels:输入通道信息，只有速度信息，所以通道为1
embed_size:Transformer通道数
time_num:一天时间间隔数量
num_layers:堆叠层数
T_dim=12:输入时间维度。输入前一小时数据，所以60min/5min = 12
output_T_dim=3:输出时间维度。预测未来15,30,45min速度
heads = 1
"""



if __name__ == '__main__':
    """
    model = STTNS()
    criterion
    optimizer
    for i in rang(epochs):
        out, _ = model(args)
        
    
    """
    days = 10  # 用10天的数据进行训练
    val_day = 3  # 3天验证

    train_num = 288 * days  # 间隔5min一个数据， 一天288个5min
    val_num = 288 * days
    row_num = train_num + val_num

    # dataset
    v = pd.read_csv("PEMSD7/V_25.csv", nrows=row_num, header=None)
    adj = pd.read_csv("PEMSD7/W_25.csv", header=None)  # 邻接矩阵
    # print(v.shape) : [T, N]

    adj = np.array(adj)
    adj = torch.tensor(adj, dtype=torch.float32)

    v = np.array(v)
    v = v.T
    v = torch.tensor(v, dtype=torch.float32)
    # 最终 v shape:[N, T]。  N=25, T=row_num
    # print(v.shape)

    in_channels = 1
    embed_size = 64
    time_num = 288
    num_layers = 1
    T_dim = 12  # 12*5 = 60, 输入前一个小时数据
    output_T_dim = 3  # 预测后15min数据
    heads = 1
    epochs = 50
    dropout = 0
    forward_expansion = 4

    model = STTNSNet(adj, in_channels, embed_size, time_num, num_layers,
                     T_dim, output_T_dim, heads, dropout, forward_expansion)
    criterion = nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    """
    for i in range(epochs):
        pass
        
    """
    #   ----训练部分----
    # t表示遍历到的具体时间
    pltx = []
    plty = []

    for t in range(train_num - 21):
        x = v[:, t:t + 12]
        x = x.unsqueeze(0)
        y = v[:, t + 14:t + 21:3]
        # x shape:[1, N, T_dim]
        # y shape:[N, output_T_dim]

        out = model(x, t)
        loss = criterion(out, y)

        if t % 100 == 0:
            print("MAE loss:", loss)

        # 常规操作
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pltx.append(t)
        plty.append(loss.detach().numpy())

    plt.plot(pltx, plty, label="STTN train")
    plt.title("ST-Transformer train")
    plt.xlabel("t")
    plt.ylabel("MAE loss")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model, "model.pth")

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
