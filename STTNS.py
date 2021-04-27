import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder
import numpy as np


# model input shape:[1, N, T]
# model output shape:[N, T]
class STTNSNet(nn.Module):
    def __init__(self, adj, in_channels, embed_size, time_num,
                 num_layers, T_dim, output_T_dim, heads, dropout, forward_expansion):
        self.num_layers = num_layers
        super(STTNSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(embed_size, heads, adj, time_num, dropout, forward_expansion)
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, in_channels, 1)

    def forward(self, x, t):
        # input x:[C, N, T]
        x = x.unsqueeze(0)  # [1, C, N, T]

        # 通道变换
        x = self.conv1(x)  # [1, embed_size, N, T]

        x = x.squeeze(0)  # [embed_size, N, T]
        x = x.permute(1, 2, 0)  # [N, T, embed_size]
        x = self.transformer(x, x, x, t, self.num_layers)  # [N, T, embed_size]

        # 预测时间T_dim，转换时间维数
        x = x.unsqueeze(0)  # [1, N, T, C], C = embed_size
        x = x.permute(0, 2, 1, 3)  # [1, T, N, C]
        x = self.conv2(x)  # [1, out_T_dim, N, C]

        # 将通道降为in_channels
        x = x.permute(0, 3, 2, 1)  # [1, C, N, out_T_dim]
        x = self.conv3(x)  # [1, in_channels, N, out_T_dim]
        out = x.unsqueeze(0).unsqueeze(0)

        return out


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.sttnblock = STTNSNetBlock(embed_size, heads, adj, time_num, dropout, forward_expansion)

    def forward(self, query, key, value, t, num_layers):
        q, k, v = query, key, value
        for i in range(num_layers):
            out = self.sttnblock(q, k, v, t)
            q, k, v = out, out, out
        return out


# model input:[N, T, C]
# model output[N, T, C]
class STTNSNetBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(STTNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.TemporalTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, t):
        out1 = self.norm1(self.SpatialTansformer(query, key, value) + query)
        out2 = self.dropout(self.norm2(self.TemporalTransformer(out1, out1, out1, t) + out1))

        return out2


# model input:[N, T, C]
# model output:[N, T, C]
class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.adj = adj
        self.D_S = nn.Parameter(adj)
        self.embed_linear = nn.Linear(adj.shape[0], embed_size)
        self.attention = SSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN
        self.gcn = GCN(embed_size, embed_size * 2, embed_size, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Spatial Embedding 部分
        N, T, C = query.shape
        D_S = self.embed_linear((self.D_S))
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)

        # GCN 部分
        X_G = torch.Tensor(query.shape[0], 0, query.shape[2])
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)

        for t in range(query.shape[1]):
            o = self.gcn(query[:, t, :], self.adj)
            o = o.unsqueeze(1)  # shape [N, 1, C]
            X_G = torch.cat((X_G, o), dim=1)

        # spatial transformer
        query = query + D_S
        value = value + D_S
        key = key + D_S
        attn = self.attention(value, key, query)  # [N, T, C]
        M_s = self.dropout(self.norm1(attn + query))
        feedforward = self.feed_forward(M_s)
        U_s = self.dropout(self.norm2(feedforward + M_s))

        # 融合
        g = torch.sigmoid(self.fs(U_s) + self.fg(X_G))
        out = g * U_s + (1 - g) * X_G

        return out


# model input:[N,T,C]
# model output:[N,T,C]
class SSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N, T, C = query.shape
        query = values.reshape(N, T, self.heads, self.per_dim)
        keys = keys.reshape(N, T, self.heads, self.per_dim)
        values = values.reshape(N, T, self.heads, self.per_dim)

        # q, k, v:[N, T, heads, per_dim]
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        # spatial self-attention
        attn = torch.einsum("qthd, kthd->qkth", (queries, keys))  # [N, N, T, heads]
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.einsum("qkth,kthd->qthd", (attention, values))  # [N, T, heads, per_dim]
        out = out.reshape(N, T, self.heads * self.per_dim)  # [N, T, C]

        out = self.fc(out)

        return out


# input[N, T, C]
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.attention = TSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, t):
        # q, k, v：[N, T, C]
        N, T, C = query.shape

        D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(torch.arange(0, T))  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(N, T, C)

        # TTransformer
        x = D_T + query
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))

        out = U_t + x + M_t

        return out


class TSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[N, T, C]
        N, T, C = query.shape

        # q, k, v:[N,T,heads, per_dim]
        keys = key.reshape(N, T, self.heads, self.per_dim)
        queries = query.reshape(N, T, self.heads, self.per_dim)
        values = value.reshape(N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # compute temperal self-attention
        attnscore = torch.einsum("nqhd, nkhd->nqkh", (queries, keys))  # [N, T, T, heads]
        attention = torch.softmax(attnscore / (self.embed_size ** (1/2)), dim=2)

        out = torch.einsum("nqkh, nkhd->nqhd", (attention, values)) # [N, T, heads, per_dim]
        out = out.reshape(N, T, self.embed_size)
        out = self.fc(out)

        return out


