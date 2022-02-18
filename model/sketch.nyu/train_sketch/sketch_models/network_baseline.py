# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models.DDR import *

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


'''
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
'''




class STAGE1(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False, sketch_gt=False):
        super(STAGE1, self).__init__()
        self.business_layer = []
        self.feature = feature
        self.oper1 = nn.Sequential(
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
        self.business_layer.append(self.oper1)



        self.oper2 = nn.Sequential(
            nn.Conv3d(feature, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 2, kernel_size=3, padding=1, bias=False),
            norm_layer(2, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper2)

        self.completion1 = nn.Sequential(
            BottleneckDDR3d(c_in=feature, c=feature//4, c_out=feature, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature, c=feature//4, c_out=feature, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature, c=feature//4, c_out=feature, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature, c=feature//4, c_out=feature, kernel=3, dilation=2, residual=True),
        )
        self.pool11 = DownsampleBlock3d(c_in=feature, c_out=feature*2)
        self.pool12 = DownsampleBlock3d(c_in=feature, c_out=feature*2)

        self.completion2 = nn.Sequential(
            BottleneckDDR3d(c_in=feature*2, c=32, c_out=feature * 2, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature * 2, c=32, c_out=feature * 2, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature * 2, c=32, c_out=feature * 2, kernel=3, dilation=2, residual=True),
            BottleneckDDR3d(c_in=feature * 2, c=32, c_out=feature * 2, kernel=3, dilation=2, residual=True),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(feature*2, feature*2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(feature*2, momentum=bn_momentum),
            nn.ReLU(inplace=False)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(feature*2, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(feature*2, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False)
        )
        self.business_layer.append(self.pool11)
        self.business_layer.append(self.pool12)

        self.business_layer.append(self.completion1)
        self.business_layer.append(self.completion2)
        self.business_layer.append(self.deconv1)
        self.business_layer.append(self.deconv2)
        self.business_layer.append(self.deconv3)



    def forward(self, tsdf, depth_mapping_3d, sketch_gt=None):
        '''
        extract 3D feature
        '''

        raw_tsdf = self.oper1(tsdf)
        y1 = self.pool11(self.completion1(raw_tsdf))+self.pool12(raw_tsdf)
        y2 = self.pool2(self.completion2(y1))
        y3 = self.deconv1(y2)+y1
        y4 = self.deconv2(y3)+self.deconv3(y3)
        pred_sketch_refine = self.oper2(y4)

        return pred_sketch_refine








'''
main network2d
'''


class Network_baseline(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network_baseline, self).__init__()
        self.business_layer = []
        self.dilate = 2


        self.stage1 = STAGE1(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage1.business_layer

    def forward(self, rgb, depth_mapping_3d, tsdf, sketch_gt=None):

        h, w = rgb.size(2), rgb.size(3)

        pred_sketch_refine = self.stage1(tsdf,depth_mapping_3d)
        results={'pred_sketch_refine':pred_sketch_refine}
        return results

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    model = Network_baseline(class_num=2, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)