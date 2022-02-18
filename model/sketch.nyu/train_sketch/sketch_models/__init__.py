
from .network_unet import Network_unet
from .network_baseline import Network_baseline
from .network_s0 import Network_s0
from .network_s0_no_seg_2d import Network_s0_no_seg_2d
from .network_s0_sketch_from_2d import Network_s0_sketch_from_2d

from .network_warp_refine import Network_warp_refine
from .network_deform import Network_deform

from sketch_config import config_sketch

import os
from shutil import copy


import torch.nn as nn


def make_model(norm_layer,modelname,eval):

    if eval:
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        else:
            print('norm layer here wrong')
            exit()


    if modelname=='network_unet':
        net = Network_unet(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_baseline':
        net = Network_baseline(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0':
        net = Network_s0(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_no_seg_2d':
        net = Network_s0_no_seg_2d(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_sketch_from_2d':
        net = Network_s0_sketch_from_2d(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_warp_refine':
        net = Network_warp_refine(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_deform':
        net = Network_deform(class_num=config_sketch.num_classes, feature=128, bn_momentum=config_sketch.bn_momentum,
                                              pretrained_model=config_sketch.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    else:
        net = None
        print('no such model')
        exit()

    name = type(net).__name__
    # print(name)
    # print(modelname)
    assert name.lower()==modelname
    copy('./sketch_models/{}.py'.format(name.lower()), config_sketch.log_dir)
    return net





__all__ = ["make_model"]