import cv2
import torch
import numpy as np
from torch.utils import data
import random
from sketch_config import config_sketch
from utils.img_utils import  normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, hha):
        img = normalize(img, self.img_mean, self.img_std)
        hha = normalize(hha, self.img_mean, self.img_std)

        p_img = img.transpose(2, 0, 1)
        p_hha = hha.transpose(2, 0, 1)

        extra_dict = {'hha_img': p_hha}

        return p_img, extra_dict
class ValPre(object):
    def __call__(self, img, hha):
        extra_dict = {'hha_img': hha}
        return img, extra_dict


def get_train_loader(engine, dataset, s3client=None):
    data_setting = {'img_root': config_sketch.img_root_folder,
                    'gt_root': config_sketch.gt_root_folder,
                    'hha_root':config_sketch.hha_root_folder,
                    'mapping_root': config_sketch.mapping_root_folder,
                    'train_source': config_sketch.train_source,
                    'eval_source': config_sketch.eval_source,
                    'seg_2d_sketch_root_folder': config_sketch.seg_2d_sketch_root_folder,
                    'seg_2d_root_folder': config_sketch.seg_2d_root_folder,
    }
    train_preprocess = TrainPre(config_sketch.image_mean, config_sketch.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config_sketch.batch_size * config_sketch.niters_per_epoch, s3client=s3client)

    train_sampler = None
    is_shuffle = True
    batch_size = config_sketch.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config_sketch.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config_sketch.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
