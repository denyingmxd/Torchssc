from __future__ import division
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from config import config
from dataloader import get_train_loader
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from models import make_model
from shutil import copy
from losses.multi_label_loss1 import MultiLabelLoss1
from losses.multi_label_loss2 import MultiLabelLoss2
from losses.multi_label_loss3 import MultiLabelLoss3
from losses.multi_label_loss4 import MultiLabelLoss4

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

setting_path = config.log_dir+'/setting.txt'
import json
if not os.path.isdir(config.log_dir):
    os.makedirs(config.log_dir)
with open(setting_path, "w") as outfile:
    CC = config.copy()
    for k,v in CC.items():
        if type(v)==np.ndarray:
            # print(type(v))
            CC[k]=v.tolist()
    json.dump(CC, outfile,indent=4)



parser = argparse.ArgumentParser()

port = str(int(float(time.time())) % 20)
os.environ['MASTER_PORT'] = str(10097 + int(port))

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config.model = args.model
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, NYUv2)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network2d and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)
    norm_layer = BatchNorm2d
    if engine.distributed:
        norm_layer = SyncBatchNorm

    model = make_model(norm_layer, config.model_name, False)
    name = type(model).__name__
    copy('./models/{}.py'.format(name.lower()), config.log_dir)

    init_weight(model.business_layer, nn.init.kaiming_normal_,
                norm_layer, config.bn_eps, config.bn_momentum,
                mode='fan_in')  # , nonlinearity='relu')

    if config.use_resnet_pretrained:
        print('Loading ResNet-50 pretrained weights from {}'.format(config.pretrained_model))
        state_dict = torch.load(config.pretrained_model)  # ['state_dict']
        transformed_state_dict = {}
        for k, v in state_dict.items():
            transformed_state_dict[k.replace('.bn.', '.')] = v

        model.backbone.load_state_dict(transformed_state_dict, strict=False)

        ''' fix the weight of resnet'''
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print('be careful, not using resnet pretrained weights')

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr  # * engine.world_size


    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, norm_layer,
                                   base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            print('use cuda')
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_sem = 0
        sum_com = 0
        sum_rest = 0
        sum_sem_tsdf =0
        sum_sem_rgb =0
        sum_multi_label_loss = 0

        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            img = minibatch['data']
            hha = minibatch['hha_img']
            label = minibatch['label']
            label_weight = minibatch['label_weight']
            tsdf = minibatch['tsdf']
            depth_mapping_3d = minibatch['depth_mapping_3d']
            seg_2d = minibatch['seg_2d']
            sketch_gt = minibatch['sketch_gt']
            label_multi = minibatch['label_multi']


            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            label_multi = label_multi.cuda(non_blocking=True)
            hha = hha.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)
            label_weight = label_weight.cuda(non_blocking=True)
            depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True)
            sketch_gt = sketch_gt.cuda(non_blocking=True)
            seg_2d = seg_2d.cuda(non_blocking=True)


            results = model(img, depth_mapping_3d, tsdf, sketch_gt,seg_2d)

            cri_weights = torch.FloatTensor([config.empty_loss_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()

            '''
            semantic loss
            '''
            output = results.get('pred_semantic')
            pred_semantic_rgb = results.get('pred_semantic_rgb')
            pred_semantic_tsdf = results.get('pred_semantic_tsdf')
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1)
            filterLabel = torch.index_select(label.view(-1), 0, selectindex)
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)
            if pred_semantic_tsdf is not None:
                pred_semantic_tsdf = pred_semantic_tsdf.permute(
                    0, 2, 3, 4, 1).contiguous().view(-1, 12)
                filterOutput_tsdf = torch.index_select(pred_semantic_tsdf, 0, selectindex)
                loss_semantic_tsdf = criterion(filterOutput_tsdf, filterLabel)
                loss_semantic_tsdf = torch.mean(loss_semantic_tsdf)
            if pred_semantic_rgb is not None:
                pred_semantic_rgb = pred_semantic_rgb.permute(
                    0, 2, 3, 4, 1).contiguous().view(-1, 12)
                filterOutput_rgb = torch.index_select(pred_semantic_rgb, 0, selectindex)
                loss_semantic_rgb = criterion(filterOutput_rgb, filterLabel)
                loss_semantic_rgb = torch.mean(loss_semantic_rgb)
            if config.use_label_multi_loss:
                if config.use_label_multi_loss1:
                    MultiLabelLoss = MultiLabelLoss1
                if config.use_label_multi_loss2:
                    MultiLabelLoss = MultiLabelLoss2
                if config.use_label_multi_loss3:
                    MultiLabelLoss = MultiLabelLoss3
                if config.use_label_multi_loss4:
                    MultiLabelLoss = MultiLabelLoss4
                label_multi_cri = MultiLabelLoss(num_classes=12)
                loss_semantic_multi =label_multi_cri(output,label_multi,label_weight)




            # reduce the whole loss over multi-gpu
            if engine.distributed:
                dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                loss_semantic = loss_semantic / engine.world_size
                loss = loss_semantic
                if pred_semantic_tsdf is not None:
                    dist.all_reduce(loss_semantic_tsdf, dist.ReduceOp.SUM)
                    loss_semantic_tsdf = loss_semantic_tsdf / engine.world_size
                    loss=loss+loss_semantic_tsdf
                if pred_semantic_rgb is not None:
                    dist.all_reduce(loss_semantic_rgb, dist.ReduceOp.SUM)
                    loss_semantic_rgb = loss_semantic_rgb / engine.world_size
                    loss=loss+loss_semantic_rgb
                if config.use_label_multi_loss:
                    dist.all_reduce(loss_semantic_multi, dist.ReduceOp.SUM)
                    loss_semantic_multi = loss_semantic_multi / engine.world_size
                    loss = loss + loss_semantic_multi

            else:
                loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr


            loss.backward()


            sum_loss += loss.item()
            sum_sem += loss_semantic.item()

            if pred_semantic_tsdf is not None:
                sum_sem_tsdf += loss_semantic_tsdf.item()
            if pred_semantic_rgb is not None:
                sum_sem_rgb += loss_semantic_rgb.item()
            if config.use_label_multi_loss:
                sum_multi_label_loss += loss_semantic_multi.item()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (sum_loss / (idx + 1)) \
                        + ' sem=%.5f' % (sum_sem / (idx + 1)) \
                        + ' sem_tsdf=%.5f' % (sum_sem_tsdf / (idx + 1)) \
                        + ' sem_rgb=%.5f' % (sum_sem_rgb / (idx + 1)) \
                        + ' sum_multi_label_loss=%.5f' % (sum_multi_label_loss / (idx + 1)) \


            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss/tot', sum_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/semantic', sum_sem / len(pbar), epoch)

            logger.add_scalar('train_loss/semantic_tsdf', sum_sem_tsdf / len(pbar), epoch)

            logger.add_scalar('train_loss/semantic_rgb', sum_sem_rgb / len(pbar), epoch)
            logger.add_scalar('train_loss/sum_multi_label_loss', sum_multi_label_loss / len(pbar), epoch)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > config.nepochs // 4) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

