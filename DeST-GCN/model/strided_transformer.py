import torch
import torch.nn as nn
from einops import rearrange
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce


# 添加图卷积相关的类
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency.to(input_feature.device), support)
        if self.use_bias:
            output += self.bias
        return output


class FeedForwardGCN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.gcn1 = GraphConvolution(dim, mult * dim)
        self.ln1 = nn.LayerNorm(mult * dim)
        self.gcn2 = GraphConvolution(mult * dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dp = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, feature, adj):
        residual = feature
        x = self.gcn1(feature, adj)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.gcn2(x, adj)
        x = self.ln2(x)
        x = self.gelu(x)
        out = residual + x
        return out


# DGFormer 和 Strided Transformer 融合后的模型
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * args.n_joints, args.channel, kernel_size=1),
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        # 图卷积层，用于捕捉关节之间的空间结构信息
        self.gcn = FeedForwardGCN(args.channel, dropout=0.1)

        # Transformer和Strided Transformer部分
        self.Transformer = Transformer(args.layers, args.channel, args.d_hid, length=args.frames)
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, \
                                                     length=args.frames, stride_num=args.stride_num)

        # 输出FCN层
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x, adjacency):
        """
        :param x: 输入数据，形状为 (B, F, J, C)
        :param adjacency: 邻接矩阵，描述关节之间的空间关系，形状为 (J, J)
        """
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        # Encoder部分处理
        x = self.encoder(x)
        x = x.permute(0, 2, 1).contiguous()

        # 图卷积处理，建模关节之间的空间关系
        x = self.gcn(x, adjacency)

        # 通过Strided Transformer减少序列长度
        x_VTE = self.Transformer_reduce(x)

        # 通过标准Transformer进一步处理时序信息
        x = self.Transformer(x)

        # FCN输出部分
        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, x_VTE


