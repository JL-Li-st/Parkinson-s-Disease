
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_ALPHA = 0.1


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_ALPHA, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(TemporalConv, self).__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1),
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1, pad=True):
        super().__init__()

        self.window_size = window_size
        self.window_dilation = window_dilation
        self.window_stride = window_stride
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2 if pad else 0

        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, -1, V)
        x = x.transpose(2, 3).contiguous()
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                pos_list.append(j_id)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.time_len * self.joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe.to(x.dtype)[:, :, :x.size(2)]
        return x


class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(ST_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)

        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        weights = self.A.to(v.dtype)

        # if weights.shape[-1] != v.shape[-1]:
        #     weights = weights.expand(-1, -1, v.shape[-1])

        # 解决报错的方法
        # 如果 weights 的最后一维和 v 的最后一维不一致，进行适当的调整
        if weights.shape[-1] != v.shape[-1]:
            # 如果 weights 的最后一维是1，可以扩展为 v 的最后一维大小
            if weights.shape[-1] == 1:
                weights = weights.expand(-1, -1, v.shape[-1])
            elif weights.shape[-1] > v.shape[-1]:
                # 如果 weights 的最后一维比 v 的要大，则截断
                weights = weights[:, :, :v.shape[-1]]
            else:
                # 如果 weights 的最后一维比 v 的要小，则用 repeat 扩展
                weights = weights.repeat(1, 1, v.shape[-1] // weights.shape[-1])

        x = torch.einsum('hvu,nhctu->nctv', weights, v)
        x = self.bn(x)
        return x


class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1):
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale

        rel_channels = in_channels // 8 if in_channels != 3 else 8

        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x, A=None, alpha=1):
        N, C, T, V = x.size()
        res = x
        q, k, v = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x).view(N, self.num_scale, self.Nh, -1, T,

                                                                                      V)
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)
        weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)

        # 解决报错的方法
        # 检查 weights 的最后一维是否与 v 的最后一维匹配
        if weights.shape[-1] != v.shape[-1]:
            # 如果 weights 的最后一维是 1，可以扩展为 v 的最后一维大小
            if weights.shape[-1] == 1:
                weights = weights.expand(-1, -1, -1, -1, v.shape[-1])
            elif weights.shape[-1] > v.shape[-1]:
                # 如果 weights 的最后一维比 v 的大，则进行截断
                weights = weights[:, :, :, :, :v.shape[-1]]
            else:
                # 如果 weights 的最后一维比 v 的小，则用 repeat 扩展
                weights = weights.repeat(1, 1, 1, 1, v.shape[-1] // weights.shape[-1])

        x = torch.einsum('ngacvu, ngactu->ngctv', weights, v).contiguous().view(N, -1, T, V)
        x = self.bn(x)
        return x


class DeSGC(nn.Module):
    '''
    Note: This module is not included in the open-source release due to subsequent research and development.
    It will be made available in future updates after the completion of related studies.
    '''


    def __init__(self, in_channels, out_channels, A, k, num_scale=4, num_frame=64, num_joint=25):
        super(DeSGC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(A)

        self.num_scale = num_scale
        self.k = k
        self.delta = 10

        rel_channels = in_channels // 8 if in_channels != 3 else 8
        self.factor = rel_channels // num_scale

        self.pe = PositionalEncoding(in_channels, num_joint, num_frame)
        self.conv = PointWiseTCN(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.convQK = nn.Conv2d(in_channels, 2 * rel_channels * self.Nh, 1, groups=num_scale)
        self.convW = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1, 1, self.Nh, 1, 1, 1))
        self.bn = nn.BatchNorm2d(out_channels)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        v = self.relu(self.conv(x)).view(N, self.num_scale, self.Nh, -1, T, V)
        dtype, device = v.dtype, v.device

        # calculate score
        # ...

        # calculate weight
        # ...

        # convert to onehot
        # ...

        # sampling & aggregation
        # ...

        return x

class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0, dilation=1,
                 num_scale=1, num_frame=64):
        super(DeTGC, self).__init__()

        self.ks, self.stride, self.dilation = kernel_size, stride, dilation
        self.T = num_frame
        self.num_scale = num_scale

        self.eta = eta
        ref = (self.ks + (self.ks - 1) * (self.dilation - 1) - 1) // 2
        tr = torch.linspace(-min(ref, self.T - 1), min(ref, self.T - 1), self.eta)
        self.tr = nn.Parameter(tr)

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype

        # learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        tr = self.tr.to(dtype)

        # Expand dimensions for broadcasting
        t0 = t0.view(1, 1, -1).expand(-1, self.eta, -1)  # Shape: (1, eta, Tout)
        tr = tr.view(1, self.eta, 1)  # Shape: (1, eta, 1)

        # Calculate the time sampling locations
        t = t0 + tr
        t = t.clamp(0, T - 1)  # Clamp values of t to the range [0, T-1]
        t = t.view(1, 1, -1, 1)  # Shape: (1, 1, eta*Tout, 1)

        # Calculate indices for floor and ceil values
        tdn = torch.floor(t).long()  # Floor indices (rounded down)
        tup = torch.ceil(t).long()  # Ceil indices (rounded up)

        # Clamp indices to be within valid range
        tdn = torch.clamp(tdn, 0, T - 1)  # Ensure tdn values are within [0, T-1]
        tup = torch.clamp(tup, 0, T - 1)  # Ensure tup values are within [0, T-1]

        # Expand indices for batch and channel dimensions
        index1 = tdn.expand(N, C, -1, V)  # Shape: (N, C, eta*Tout, V)
        index2 = tup.expand(N, C, -1, V)  # Shape: (N, C, eta*Tout, V)

        # Sampling
        alpha = tup - t  # Compute interpolation factor
        alpha = alpha.expand(N, C, -1, V)  # Shape: (N, C, eta*Tout, V)

        # Gather values based on indices and perform linear interpolation
        x1 = x.gather(-2, index=index1)  # Shape: (N, C, eta*Tout, V)
        x2 = x.gather(-2, index=index2)  # Shape: (N, C, eta*Tout, V)

        # Ensure alpha is clamped between 0 and 1 for valid interpolation
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Linear interpolation
        x = x1 * (1 - alpha) + x2 * alpha  # Shape: (N, C, eta*Tout, V)

        # print(f"x.shape before view: {x.shape}")
        # print(f"Target shape: (N={N}, C={C}, eta={self.eta}, Tout={Tout}, V={V})")

        # Reshape x to match the output format
        try:
            x = x.view(N, C, self.eta, Tout, V)
        except RuntimeError as e:
            # Handle shape mismatch by adjusting Tout dynamically
            expected_size = N * C * self.eta * Tout * V
            actual_size = x.numel()
            if expected_size != actual_size:
                Tout = actual_size // (N * C * self.eta * V)
                if N * C * self.eta * Tout * V != actual_size:
                    raise ValueError(f"Cannot reshape input of size {actual_size} into target shape with Tout={Tout}")
            x = x.view(N, C, self.eta, Tout, V)

        # print(f"x.shape after view: {x.shape}")
        # print(f"Target shape: (N={N}, C={C}, eta={self.eta}, Tout={Tout}, V={V})")

        # Apply convolution and squeeze the eta dimension
        x = self.conv_out(x).squeeze(2)  # Shape: (N, C, Tout, V)

        return x




class MultiScale_TemporalModeling(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=5, stride=1, dilations=1,
                 num_scale=1, num_frame=64):
        super(MultiScale_TemporalModeling, self).__init__()

        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1

        self.tcn1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels,
                  scale_channels,
                  eta,
                  kernel_size=5,
                  stride=stride,
                  dilation=1,
                  num_scale=num_scale,
                  num_frame=num_frame)
        )

        self.tcn2 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels,
                  scale_channels,
                  eta,
                  kernel_size=5,
                  stride=stride,
                  dilation=2,
                  num_scale=num_scale,
                  num_frame=num_frame)
        )

        self.maxpool3x1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(scale_channels)
        )
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        x = torch.cat([self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_ALPHA = 0.1

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, k, eta, kernel_size=5, stride=1, dilations=2,
                 num_frame=64, num_joint=25, residual=True):
        super(Basic_Block, self).__init__()

        num_scale = 4
        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels != 3 else 1

        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            self.gcn = CTR_GC(in_channels, out_channels, A, self.num_scale)

        self.tcn = MultiScale_TemporalModeling(out_channels, out_channels, eta,
                                               stride=stride, num_scale=num_scale, num_frame=num_frame)

        # 修改残差路径 residual1，使其确保在维度不匹配的情况下也能匹配主分支的输出
        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),  # 调整通道数和步幅
            nn.BatchNorm2d(out_channels)
        ) if (in_channels != out_channels or stride != 1) else None

        self.residual2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        ) if not residual else None

        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())

    def forward(self, x):
        res = x
        # print(f"Input shape before GCN: {x.shape}")
        x = self.gcn(x)
        # print(f"Shape after GCN: {x.shape}")

        # 通过 residual1 调整残差 res 的形状以匹配 x 的形状
        if self.residual1 is not None:
            res = self.residual1(res)
            # print(f"Shape of adjusted residual: {res.shape}")

        # 如果形状不匹配，动态调整残差的尺寸
        if res.shape[2:] != x.shape[2:]:
            res = F.interpolate(res, size=(x.shape[2], x.shape[3]), mode='nearest')

        # 进行残差连接
        x = self.relu(x + res)
        # print(f"Shape after residual addition: {x.shape}")

        x = self.tcn(x)
        # print(f"Shape after TCN: {x.shape}")

        # 通过 residual2 调整残差以匹配 TCN 输出
        res_adjusted_2 = x  # 因为 self.residual2 只应用于第一次的残差，第二次使用的是经过 GCN + TCN 后的输出
        if self.residual2 is not None:
            res_adjusted_2 = self.residual2(x)
            # print(f"Shape of adjusted residual after TCN: {res_adjusted_2.shape}")

            # 如果形状不匹配，动态调整残差的尺寸
            if res_adjusted_2.shape[2:] != x.shape[2:]:
                res_adjusted_2 = F.interpolate(res_adjusted_2, size=(x.shape[2], x.shape[3]), mode='nearest')

        x = self.relu(x + res_adjusted_2)
        return x

