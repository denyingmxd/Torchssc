from .network import Network
from .network_sobel import Network_sobel
from .network_no_cvae import Network_no_cvae
from .network_mapping import Network_mapping
from .network_tsdf_and_rgb_to_sketch import Network_tsdf_and_rgb_to_sketch
from .network_tsdf_and_rgb_to_sketch_explicit import Network_tsdf_and_rgb_to_sketch_explicit
from .network_tsdf_and_rgb_to_sketch_only_refine import Network_tsdf_and_rgb_to_sketch_only_refine
from .network_tsdf_and_rgb_to_sketch_explicit_explicit import Network_tsdf_and_rgb_to_sketch_explicit_explicit
from .network_tsdf_and_rgb_to_sketch_no_segres import Network_tsdf_and_rgb_to_sketch_no_segres
from .network_tsdf_and_rgb_to_sketch_early import Network_tsdf_and_rgb_to_sketch_early
from .network_tsdf_and_rgb_to_sketch_only_precision import Network_tsdf_and_rgb_to_sketch_only_precision
from .network_tsdf_and_rgb_to_sketch_late import Network_tsdf_and_rgb_to_sketch_late
from .network_tsdf_and_rgb_to_sketch_post_loss import Network_tsdf_and_rgb_to_sketch_post_loss
from .network_tsdf_and_rgb_to_sketch_explicit_post_loss  import Network_tsdf_and_rgb_to_sketch_explicit_post_loss
from .network_unet import Network_unet
from .network_unet_predict_sketch import Network_unet_predict_sketch
from .network_baseline import Network_baseline
from .network_s0_post_sketch import Network_s0_post_sketch
from .network_s0_tsdf_to_sketch import Network_s0_tsdf_to_sketch
from .network_warp import Network_warp
from .network_tsdf import Network_tsdf
from .network_seg_2d import Network_seg_2d
from .network_s_0_late import Network_s_0_late
from .network_s_0_late_multi_supervision import Network_s_0_late_multi_supervision
from .network_s0_late_ac import Network_s0_late_ac
from .network_s0_late_gated import Network_s0_late_gated
from .network_s0_late_ac2 import Network_s0_late_ac2
from .network_s0_late_ac3 import Network_s0_late_ac3
from .network_s0_late_ac4 import Network_s0_late_ac4
from .network_s0_late_ac5 import Network_s0_late_ac5
from .network_s0_late_ac6 import Network_s0_late_ac6
from .network_s0_late_ac7 import Network_s0_late_ac7
from .network_s0_late_ac8 import Network_s0_late_ac8
from .network_s0_late_ac9 import Network_s0_late_ac9

from .network_s0_rgb import Network_s0_rgb
from .network_deeplabv3 import Network_deeplabv3
from .network_resnet50 import Network_resnet50


from config import config
import os
from shutil import copy
from .network_s0 import Network_s0


import torch.nn as nn


def make_model(norm_layer,modelname,eval):

    if eval:
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        else:
            print('norm layer here wrong')
            exit()

    if modelname == 'network2d':
         net = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                    pretrained_model=config.pretrained_model,
                    norm_layer=norm_layer,eval=eval)


    elif modelname == 'network_no_cvae':
        net = Network_no_cvae(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                norm_layer=norm_layer,pretrained_model=config.pretrained_model, eval=eval)

    elif modelname == 'network_sobel':
        net = Network_sobel(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                pretrained_model=config.pretrained_model,
                norm_layer=norm_layer, eval=eval)

    elif modelname=='network_mapping':
        net = Network_mapping(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                pretrained_model=config.pretrained_model,
                norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch':
        net = Network_tsdf_and_rgb_to_sketch(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                pretrained_model=config.pretrained_model,
                norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_explicit':

        net= Network_tsdf_and_rgb_to_sketch_explicit(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_only_refine':
        net = Network_tsdf_and_rgb_to_sketch_only_refine(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_explicit_explicit':
        net = Network_tsdf_and_rgb_to_sketch_explicit_explicit(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_no_segres':
        net = Network_tsdf_and_rgb_to_sketch_no_segres(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_only_precision':
        net = Network_tsdf_and_rgb_to_sketch_only_precision(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_early':
        net = Network_tsdf_and_rgb_to_sketch_early(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                                   pretrained_model=config.pretrained_model,
                                                   norm_layer=norm_layer, eval=eval)

    elif modelname=='network_tsdf_and_rgb_to_sketch_late':
        net = Network_tsdf_and_rgb_to_sketch_late(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                                   pretrained_model=config.pretrained_model,
                                                   norm_layer=norm_layer, eval=eval)
    elif modelname=='network_tsdf_and_rgb_to_sketch_post_loss':
        net = Network_tsdf_and_rgb_to_sketch_post_loss(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                                   pretrained_model=config.pretrained_model,
                                                   norm_layer=norm_layer, eval=eval)

    elif modelname =='network_tsdf_and_rgb_to_sketch_explicit_post_loss':
        net = Network_tsdf_and_rgb_to_sketch_explicit_post_loss(class_num=config.num_classes, feature=128,
                                                       bn_momentum=config.bn_momentum,
                                                       pretrained_model=config.pretrained_model,
                                                       norm_layer=norm_layer, eval=eval)

    elif modelname=='network_unet':
        net = Network_unet(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_unet_predict_sketch':
        net = Network_unet_predict_sketch(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_baseline':
        net = Network_baseline(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0':
        net = Network_s0(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s0_post_sketch':
        net = Network_s0_post_sketch(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s0_tsdf_to_sketch':
        net = Network_s0_tsdf_to_sketch(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_warp':
        net = Network_warp(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_tsdf':
        net = Network_tsdf(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_seg_2d':
        net = Network_seg_2d(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s_0_late':
        net = Network_s_0_late(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s_0_late_multi_supervision':
        net = Network_s_0_late_multi_supervision(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s0_late_ac':
        net = Network_s0_late_ac(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_gated':
        net = Network_s0_late_gated(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac2':
        net = Network_s0_late_ac2(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac3':
        net = Network_s0_late_ac3(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac4':
        net = Network_s0_late_ac4(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac5':
        net = Network_s0_late_ac5(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac6':
        net = Network_s0_late_ac6(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_s0_late_ac7':
        net = Network_s0_late_ac7(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac8':
        net = Network_s0_late_ac8(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_s0_late_ac9':
        net = Network_s0_late_ac9(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)



    elif modelname=='network_s0_rgb':
        net = Network_s0_rgb(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)

    elif modelname=='network_deeplabv3':
        return Network_deeplabv3(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)
    elif modelname=='network_resnet50':
        return Network_resnet50(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                                              pretrained_model=config.pretrained_model,
                                              norm_layer=norm_layer, eval=eval)



    else:
        net = None
        print('no such model')
        exit()

    name = type(net).__name__
    assert name.lower()==modelname

    return net





__all__ = ["make_model"]