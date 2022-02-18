import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
# from .deform_conv_3d import DeformConv3D, DeformConv3D_alternative
from modulated_deform_conv import *

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        # d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class PatchAttn(nn.Module):
    """" Patch Attention Module """

    def __init__(self, in_dim, block_size):
        super(PatchAttn, self).__init__()
        self.chanel_in = in_dim
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.s2d = SpaceToDepth(block_size)
        self.d2s = DepthToSpace(block_size)

    def forward(self, x):
        x = self.s2d(x).contiguous()
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        out = self.d2s(out).contiguous()

        return out


class Cross_Packing_Attention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):

        super(Cross_Packing_Attention, self).__init__()

        self.kernel_size = kernel_size
        self.groups = groups

        self.ref_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * self.kernel_size * self.kernel_size, 3, stride=1, padding=1, bias=bias),
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size % 2,
            groups=groups,
            bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.rgb_conv = nn.Conv2d(in_channels, 3, 3, stride=1, padding=1, bias=bias)

        self.offset_conv.apply(self.init_0)

    def init_0(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # n, c, h, w = x.size()

        ref_x = self.ref_conv(torch.cat([x,
                                         torch.flip(x, [2]),
                                         torch.flip(x, [3]),
                                         torch.flip(x, [2, 3])], dim=1))
        offset = self.offset_conv(ref_x)

        # offset = offset.repeat(1, self.kernel_size * self.kernel_size, 1, 1)  # reduce computation
        x = self.relu(self.deform_conv(x, offset))
        # rgb = self.rgb_conv(x)
        return x  # , offset

 # 3D Cross_Packing_Attention
class SpaceToChannel_3d(nn.Module):
    def __init__(self, block_size):
        super(SpaceToChannel_3d, self).__init__()
        self.block_size = block_size
        self.block_size_cubic = block_size * block_size * block_size

    def forward(self, input):  # (B, C, W, H, D)
        output = input.permute(0, 2, 3, 4, 1)   # (B, W, H, D, C) = (1 60 36 60 256)
        (batch_size, s_width, s_height, s_depth, s_channel) = output.size()  #
        d_channel = s_channel * self.block_size_cubic
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        # d_depth = int(s_depth / self.block_size)
        t_1 = output.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_width, d_height, d_channel) for t_t in t_1]  # 1 10 6 55296
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 3, 1, 4)
        output = output.permute(0, 4, 1, 2, 3)
        return output

class ChannelToSpace_3d(nn.Module):
    def __init__(self, block_size):
        super(ChannelToSpace_3d, self).__init__()
        self.block_size = block_size
        self.block_size_cubic = block_size * block_size * block_size

    def forward(self, input):  # torch.Size([1, 432, 10, 6, 10])
        output = input.permute(0, 2, 3, 4, 1)  # torch.Size([1, 10, 6, 10, 432])
        (batch_size, d_width, d_height, d_depth, d_channel) = output.size()
        s_channel = int(d_channel / self.block_size_cubic)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        s_depth = int(d_depth * self.block_size)
        # print(batch_size, s_width, s_height, s_depth, s_channel)  # 1 30 18 30 16
        t_1 = output.reshape(batch_size, d_width, d_height, d_depth, self.block_size_cubic, s_channel)
        # print(t_1.shape)   # torch.Size([1, 10, 6, 10, 27, 16])
        spl = t_1.split(self.block_size, 4)  # show the number of split
        # spl = t_1.split([self.block_size * self.block_size, self.block_size * self.block_size, self.block_size * self.block_size], 4)
        # print(spl[0].shape)  # torch.Size([1, 10, 6, 10, 9, 16])  torch.Size([1, 10, 6, 10, 6, 64])
        stack = [t_t.reshape(batch_size, d_width, d_height, s_depth, s_channel) for t_t in spl]

        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4, 5).reshape(batch_size, s_width, s_height, s_depth, s_channel)

        output = output.permute(0, 4, 1, 2, 3)
        return output

class PatchAttn_3d(nn.Module):
    """" Patch Attention Module """

    def __init__(self, in_dim, block_size):
        super(PatchAttn_3d, self).__init__()
        self.chanel_in = in_dim
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.s2d = SpaceToChannel_3d(block_size)
        self.d2s = ChannelToSpace_3d(block_size)

    def forward(self, x):
        x = self.s2d(x).contiguous()
        m_batchsize, C, width, height, depth = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, width, height, depth)

        out = self.gamma * out + x

        out = self.d2s(out).contiguous()

        return out


class Cross_Packing_Attention_3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):
        super(Cross_Packing_Attention_3d, self).__init__()

        self.kernel_size = kernel_size
        self.groups = groups

        self.ref_conv = nn.Sequential(
            nn.Conv3d(in_channels * 4, in_channels, 3, stride=1, padding=1, bias=bias),  # 3 * 4
            nn.ReLU(inplace=True)
        )

        self.offset_conv_3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn_3d(in_channels, 6),  # block_size
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn_3d(in_channels, 6),  # block_size
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 3 * self.kernel_size * self.kernel_size * self.kernel_size, 3, stride=1, padding=1, bias=bias),
        )
        # self.deform_conv = DeformConv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride=kernel_size // 2,
        #     padding=kernel_size % 2,
        #     groups=groups,
        #     bias=bias)
        self.deform_conv_3d = DeformConv3d(in_channels, out_channels, kernel_size, padding=kernel_size % 2)  # ModulatedDeformConv3d, DeformConv3d
        self.relu = nn.ReLU(inplace=True)

        self.rgb_conv = nn.Conv3d(in_channels, 3, 3, stride=1, padding=1, bias=bias)



    def init_0(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # n, c, h, w = x.size()

        ref_x = self.ref_conv(torch.cat([x,
                                         torch.flip(x, [2]),
                                         torch.flip(x, [3]),
                                         torch.flip(x, [4])], dim=1))  # [2, 3]
        offset = self.offset_conv_3d(ref_x)  # torch.Size([1, 81, 30, 18, 30])
        # offset = offset.repeat(1, self.kernel_size * self.kernel_size, 1, 1)  # reduce computation
        x = self.relu(self.deform_conv_3d(x, offset=offset))
        # rgb = self.rgb_conv(x)
        return x  # , offset


# add deform
class ProjectedSS_Conv(nn.Module):
    def __init__(self, inplanes, planes, conv_list):
        super(ProjectedSS_Conv, self).__init__()
        self.conv_list = conv_list
        # self.conv0 = nn.Conv3d(1, planes, kernel_size=3, padding=1, dilation=1, bias=False)  # inplanes
        # self.bn0 = nn.BatchNorm3d(inplanes)
        # self.relu0 = nn.ReLU(inplace=True)

        # # deformable Block
        self.offsets = nn.ModuleList([nn.Conv3d(inplanes, 81, kernel_size=3, padding=1) for dil in conv_list])   # Sampling offsets
        self.conv = nn.ModuleList([DeformConv3d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in conv_list])  # DeformConv3d, ModulatedDeformConv3d
        self.bn = nn.ModuleList([nn.BatchNorm3d(planes) for dil in conv_list])
        self.relu = nn.ModuleList([nn.ReLU(inplace=True) for dil in conv_list])

        self.offsets.apply(self.init_0)

    def init_0(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # offsets = self.offsets(x)
        # y = self.relu0(self.bn0(self.conv(x, offset=offsets)))
        y = 0

        for i in range(0, len(self.conv_list)):
            offsets = self.offsets[i](x)
            y = y + self.relu[i](self.bn[i](self.conv[i](x, offset=offsets)))
            # y = self.relu[i](self.bn[i](self.conv[i](x+y)))
        # x = self.relu(y)
        return y

#
# # standard conv
# class ProjectedSS_Conv(nn.Module):
#     def __init__(self, inplanes, planes, conv_list):
#         super(ProjectedSS_Conv, self).__init__()
#         self.conv_list = conv_list
#         self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, dilation=1, bias=False)
#         self.bn0 = nn.BatchNorm3d(inplanes)
#         self.relu0 = nn.ReLU(inplace=True)
#
#         self.conv = nn.ModuleList([nn.Conv3d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in conv_list])
#         self.bn = nn.ModuleList([nn.BatchNorm3d(planes) for dil in conv_list])
#         self.relu = nn.ModuleList([nn.ReLU(inplace=True) for dil in conv_list])
#
#
#     def forward(self, x):
#         y = self.relu0(self.bn0(self.conv0(x)))
#         for i in range(1, len(self.conv_list)):
#             # y = y + self.relu[i](self.bn[i](self.conv[i](x)))
#             y = self.relu[i](self.bn[i](self.conv[i](x+y)))
#         # x = self.relu(y)
#         return y
