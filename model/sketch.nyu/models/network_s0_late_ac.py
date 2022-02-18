# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

class STAGE1(nn.Module):
    def __init__(self, class_num, norm_layer,feature, bn_momentum):
        super(STAGE1, self).__init__()
        self.business_layer = []
        self.feature = feature
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)
        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)

        self.oper_rgb = nn.Sequential(
            nn.Conv3d(12, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )

        self.business_layer.append(self.oper_rgb)

    def forward(self,rgb):
        edge_rgb = self.oper_rgb(rgb)
        seg_fea = edge_rgb
        semantic1 = self.semantic_layer1(seg_fea)
        semantic2 = self.semantic_layer2(semantic1)

        return semantic1,semantic2


class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer,feature, bn_momentum):
        super(STAGE2, self).__init__()
        self.business_layer = []
        self.feature = feature
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)
        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)

        self.oper_tsdf = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )

        self.business_layer.append(self.oper_tsdf)

    def forward(self, tsdf):
        edge_tsdf = self.oper_tsdf(tsdf)
        seg_fea = edge_tsdf
        semantic1 = self.semantic_layer1(seg_fea) + F.interpolate(seg_fea, size=[30, 18, 30])
        semantic2 = self.semantic_layer2(semantic1)


        return semantic1,semantic2

def channel_attention(num_channel):
    # todo add convolution here
    pool = nn.AdaptiveAvgPool3d(1)
    conv = nn.Conv3d(num_channel, num_channel, kernel_size=1)
    # bn = nn.BatchNorm2d(num_channel)
    activation = nn.Sigmoid()  # todo modify the activation function

    return nn.Sequential(*[pool, conv, activation])


class STAGE3(nn.Module):
    def __init__(self,class_num, norm_layer, feature, bn_momentum):

        super(STAGE3, self).__init__()
        self.business_layer = []
        self.rgb_acm_1=channel_attention(feature)
        self.tsdf_acm_1=channel_attention(feature)
        self.rgb_acm_2=channel_attention(feature*2)
        self.tsdf_acm_2=channel_attention(feature*2)

        self.pool_middle = nn.Sequential(nn.Conv3d(feature,feature*2,kernel_size=1),
                                  nn.AvgPool3d(kernel_size=3, padding=1, stride=2),
                                          norm_layer(feature*2, momentum=bn_momentum))
        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            ),]
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.classify_semantic)
        self.business_layer.append(self.rgb_acm_1)
        self.business_layer.append(self.tsdf_acm_1)
        self.business_layer.append(self.rgb_acm_2)
        self.business_layer.append(self.tsdf_acm_2)
        self.business_layer.append(self.pool_middle)
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.final_conv)

    def forward(self,rgbs,tsdfs ):
        rgb_fea = rgbs[0]
        tsdf_fea = tsdfs[0]
        rgb_fea_attn = self.rgb_acm_1(rgb_fea)
        tsdf_fea_attn = self.tsdf_acm_1(tsdf_fea)
        fuse_fea_1 = rgb_fea.mul(rgb_fea_attn) + tsdf_fea.mul(tsdf_fea_attn)

        rgb_fea = rgbs[1]
        tsdf_fea = tsdfs[1]
        rgb_fea_attn = self.rgb_acm_2(rgb_fea)
        tsdf_fea_attn = self.tsdf_acm_2(tsdf_fea)
        middle_down = self.pool_middle(fuse_fea_1)
        fuse_fea_2 = rgb_fea.mul(rgb_fea_attn) + tsdf_fea.mul(tsdf_fea_attn) + middle_down

        pred_semantic = self.classify_semantic[0](fuse_fea_2)+self.conv1(fuse_fea_1)
        pred_semantic = self.classify_semantic[1](pred_semantic)
        pred_semantic = self.final_conv(pred_semantic)
        pred_semantic = self.classify_semantic[2](pred_semantic)

        return pred_semantic




class Network_s0_late_ac(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network_s0_late_ac, self).__init__()
        self.business_layer = []


        self.stage1 = STAGE1(class_num, norm_layer, feature=feature, bn_momentum=bn_momentum)
        self.stage2 = STAGE2(class_num, norm_layer, feature=feature, bn_momentum=bn_momentum)
        self.stage3 = STAGE3(class_num, norm_layer, feature=feature, bn_momentum=bn_momentum)
        self.business_layer += self.stage1.business_layer
        self.business_layer += self.stage2.business_layer
        self.business_layer += self.stage3.business_layer

    def forward(self,img, depth_mapping_3d, tsdf, sketch_gt,seg_2d):
        rgb_semantic1,rgb_semantic2 = self.stage1(seg_2d)
        tsdf_semantic1, tsdf_semantic2 = self.stage2(tsdf)
        rgbs=[rgb_semantic1,rgb_semantic2]
        tsdfs=[tsdf_semantic1,tsdf_semantic2]
        pred_semantic = self.stage3(rgbs,tsdfs)
        results={'pred_semantic':pred_semantic}
        return results




if __name__ == '__main__':
    pass
