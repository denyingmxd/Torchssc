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

from config import config
from dataloader import get_train_loader
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from models import make_model
from utils.Sobel import Sobel3D
from losses.edge_ce_loss import Edge_CE_Loss
from losses.Body_loss_cri import Body_Loss_Cri
from losses.local_around_edge_loss import Local_around_edge_loss
parser = argparse.ArgumentParser()

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


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

    norm_layer = BatchNorm2d
    if engine.distributed:
        norm_layer = SyncBatchNorm

    model = make_model(norm_layer,config.model_name,False)
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
        # exit()
    # group weight and config optimizer
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
        sum_sketch = 0
        sum_sketch_gsnn = 0
        sum_kld = 0
        sum_sketch_raw = 0
        sum_sketch_from_pred=0
        sum_edge_ce_loss=0
        sum_local_around_edge_loss=0
        sum_sem_final=0
        sum_body_loss=0
        cri_weights = torch.FloatTensor([config.empty_loss_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            img = minibatch['data']
            hha = minibatch['hha_img']
            label = minibatch['label']
            label_weight = minibatch['label_weight']
            tsdf = minibatch['tsdf']
            depth_mapping_3d = minibatch['depth_mapping_3d']

            sketch_gt = minibatch['sketch_gt']
            seg_2d = minibatch['seg_2d']

            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            hha = hha.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)
            label_weight = label_weight.cuda(non_blocking=True)
            depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True)
            sketch_gt = sketch_gt.cuda(non_blocking=True)
            seg_2d = seg_2d.cuda(non_blocking=True)

            results = model(img, depth_mapping_3d, tsdf, sketch_gt,seg_2d)

            if config.use_edge_ce_loss:
                edge_ce_loss_criterion = Edge_CE_Loss(weight=cri_weights).cuda()
            if config.use_local_around_edge_loss:
                local_around_edge_loss_criterion = Local_around_edge_loss().cuda()
            if config.use_body_loss:
                body_loss_criterion = Body_Loss_Cri().cuda()



            output = results.get('pred_semantic')
            pred_sketch_raw = results.get('pred_sketch_raw')
            pred_sketch = results.get('pred_sketch_refine')
            sketch_from_pred = results.get('sketch_from_pred')
            final_output = results.get('pred_semantic_final')
            pred_body = results.get('pred_body')
            '''
            semantic loss
            '''

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1)
            filterLabel = torch.index_select(label.view(-1), 0, selectindex)
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)

            if final_output is not None:
                filterFinalOutput = torch.index_select(final_output.permute(
                    0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
                loss_semantic_final = criterion(filterFinalOutput, filterLabel)
                loss_semantic_final = torch.mean(loss_semantic_final)
            if  pred_body is not None:
                loss_body = body_loss_criterion(pred_body,label,label_weight,sketch_gt)
                loss_body = torch.mean(loss_body)

            '''
            sketch loss
            '''
            filter_sketch_gt = torch.index_select(sketch_gt.view(-1), 0, selectindex)




            if pred_sketch_raw is not None:
                filtersketch_raw = torch.index_select(pred_sketch_raw.permute(0, 2, 3, 4, 1).contiguous()
                                                  .view(-1, 2), 0, selectindex)
            if pred_sketch is not None:
                filtersketch = torch.index_select(pred_sketch.permute(0, 2, 3, 4, 1).contiguous()
                                                .view(-1, 2), 0, selectindex)
            if sketch_from_pred is not None:
                filter_sketch_from_pred = torch.index_select(sketch_from_pred.permute(0, 2, 3, 4, 1).contiguous()
                                                          .view(-1, 2), 0, selectindex)

            if config.use_sketch_gt or config.model_name in ['network_sobel']:
                if engine.distributed:
                    dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                    loss_semantic = loss_semantic / engine.world_size
                else:
                    loss = Reduce.apply(*loss) / len(loss)



            else:
                criterion_sketch = nn.CrossEntropyLoss(ignore_index=255,reduction='none').cuda()


                if pred_sketch_raw is not None:
                    loss_sketch_raw = criterion_sketch(filtersketch_raw, filter_sketch_gt)
                    loss_sketch_raw = torch.mean(loss_sketch_raw)
                if pred_sketch is not None:
                    loss_sketch = criterion_sketch(filtersketch, filter_sketch_gt)
                    loss_sketch = torch.mean(loss_sketch)
                if sketch_from_pred is not None:
                    loss_sketch_from_pred = criterion_sketch(filter_sketch_from_pred, filter_sketch_gt)
                    loss_sketch_from_pred = torch.mean(loss_sketch_from_pred)
                    if config.use_edge_ce_loss:
                        edge_ce_loss,flag = edge_ce_loss_criterion(output, label, label_weight, sketch_from_pred)
                    if config.use_local_around_edge_loss:
                        local_around_edge_loss = local_around_edge_loss_criterion(output, label, label_weight, sketch_from_pred)

                if engine.distributed:
                    dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                    loss_semantic = loss_semantic / engine.world_size
                    if final_output is not None:
                        dist.all_reduce(loss_semantic_final, dist.ReduceOp.SUM)
                        loss_semantic_final = loss_semantic_final / engine.world_size
                    if pred_body is not None:
                        dist.all_reduce(loss_body, dist.ReduceOp.SUM)
                        loss_body = loss_body / engine.world_size
                    if pred_sketch is not None:
                        dist.all_reduce(loss_sketch, dist.ReduceOp.SUM)
                        loss_sketch = loss_sketch / engine.world_size



                    if pred_sketch_raw is not None:
                        dist.all_reduce(loss_sketch_raw, dist.ReduceOp.SUM)
                        loss_sketch_raw = loss_sketch_raw / engine.world_size

                    if sketch_from_pred is not None:
                        dist.all_reduce(loss_sketch_from_pred, dist.ReduceOp.SUM)
                        loss_sketch_from_pred = loss_sketch_from_pred / engine.world_size

                        if config.use_edge_ce_loss:
                            dist.all_reduce(edge_ce_loss, dist.ReduceOp.SUM)
                            edge_ce_loss = edge_ce_loss / engine.world_size
                        if config.use_local_around_edge_loss:
                            # print('b4',local_around_edge_loss)
                            dist.all_reduce(local_around_edge_loss, dist.ReduceOp.SUM)
                            # print('local_around_edge_loss',local_around_edge_loss)
                            local_around_edge_loss = local_around_edge_loss / engine.world_size


                else:
                    loss = Reduce.apply(*loss) / len(loss)

                current_idx = epoch * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                optimizer.param_groups[0]['lr'] = lr
                # optimizer.param_groups[1]['lr'] = lr
                for i in range(1, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr
                    optimizer.param_groups[i]['lr'] = lr
                loss = loss_semantic*config.semantic_loss_weight

                if final_output is not None:
                    loss += loss_semantic_final
                if pred_body is not None:
                    loss += loss_body
                if pred_sketch_raw is not None:
                    # print(123)
                    loss += (loss_sketch_raw) * config.sketch_weight
                if pred_sketch is not None:
                    loss += (loss_sketch) * config.sketch_weight
                if sketch_from_pred is not None:
                    loss += (loss_sketch_from_pred) * config.post_sobel_loss_weight
                    if config.use_edge_ce_loss:
                        if flag:
                            loss += (edge_ce_loss) * config.edge_ce_loss_weight
                    if config.use_local_around_edge_loss:
                        loss += (local_around_edge_loss) * config.local_around_edge_loss_weight




                sum_loss += loss.item()
                sum_sem += loss_semantic.item()
                if final_output is not None:
                    sum_sem_final += loss_semantic_final.item()
                if pred_body is not None:
                    sum_body_loss += loss_body.item()


                if pred_sketch_raw is not None:
                    sum_sketch_raw += loss_sketch_raw.item()
                if pred_sketch is not None:
                    sum_sketch += loss_sketch.item()
                if sketch_from_pred is not None:
                    sum_sketch_from_pred += loss_sketch_from_pred.item()
                    if config.use_edge_ce_loss:
                        if flag:
                            sum_edge_ce_loss += edge_ce_loss.item()
                    if config.use_local_around_edge_loss:
                        sum_local_around_edge_loss += local_around_edge_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (sum_loss / (idx + 1)) \
                        + ' sketch_loss=%.5f' % (sum_sketch / (idx + 1)) \
                        + ' raw_sketch_loss=%.5f'% (sum_sketch_raw / (idx + 1)) \
                        + ' edge_ce_loss=%.5f'% (sum_edge_ce_loss / (idx + 1)) \
                        + ' semantic_loss=%.5f' % (sum_sem / (idx + 1))\
                        + ' sketch_from_pred_loss=%.5f' % (sum_sketch_from_pred / (idx + 1))\
                        + ' local_around_edge_loss=%.5f' % (sum_local_around_edge_loss / (idx + 1))\
                        + ' semantic_final_loss=%.5f' % (sum_sem_final / (idx + 1))\
                        + ' sum_body_loss=%.5f' % (sum_body_loss / (idx+1))

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss/tot', sum_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/semantic', sum_sem / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch', sum_sketch / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_raw', sum_sketch_raw / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_gsnn', sum_sketch_gsnn / len(pbar), epoch)
            logger.add_scalar('train_loss/post_sketch_loss', sum_sketch_from_pred / len(pbar), epoch)
            logger.add_scalar('train_loss/edge_ce_loss', sum_edge_ce_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/local_around_edge_loss', sum_local_around_edge_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/KLD', sum_kld / len(pbar), epoch)
            logger.add_scalar('train_loss/semantic_final', sum_sem_final / len(pbar), epoch)
            logger.add_scalar('train_loss/body_loss', sum_body_loss / len(pbar), epoch)
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
