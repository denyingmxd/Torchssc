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

from sketch_config import config_sketch
from sketch_dataset import get_train_loader
import sys
sys.path.append('../')
from nyu_sketch import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from sketch_models import make_model
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
    config_sketch.model = args.model
    cudnn.benchmark = True
    seed = config_sketch.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, NYUv2)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config_sketch.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config_sketch.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    norm_layer = BatchNorm2d
    if engine.distributed:
        norm_layer = SyncBatchNorm

    model = make_model(norm_layer,config_sketch.model_name,False)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                norm_layer, config_sketch.bn_eps, config_sketch.bn_momentum,
                mode='fan_in')  # , nonlinearity='relu')




    # group weight and config optimizer
    base_lr = config_sketch.lr
    if engine.distributed:
        base_lr = config_sketch.lr  # * engine.world_size

    ''' fix the weight of resnet'''


    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, norm_layer,
                                   base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config_sketch.momentum,
                                weight_decay=config_sketch.weight_decay)


    # config lr policy
    total_iteration = config_sketch.nepochs * config_sketch.niters_per_epoch
    lr_policy = PolyLR(base_lr, config_sketch.lr_power, total_iteration)

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



    for epoch in range(engine.state.epoch, config_sketch.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config_sketch.niters_per_epoch), file=sys.stdout,
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
        sum_sketch_warped = 0



        for idx in pbar:
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()
            img = minibatch['data']
            hha = minibatch['hha_img']
            label = minibatch['label']
            label_weight = minibatch['label_weight']
            tsdf = minibatch['tsdf']
            depth_mapping_3d = minibatch['depth_mapping_3d']
            seg_2d = minibatch['seg_2d']
            seg_2d_sketch = minibatch['seg_2d_sketch']
            sketch_gt = minibatch['sketch_gt']

            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            hha = hha.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)
            label_weight = label_weight.cuda(non_blocking=True)
            depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True)
            sketch_gt = sketch_gt.cuda(non_blocking=True)
            seg_2d = seg_2d.cuda(non_blocking=True)
            seg_2d_sketch = seg_2d_sketch.cuda(non_blocking=True)

            results = model(img, depth_mapping_3d, tsdf, sketch_gt,seg_2d,seg_2d_sketch)




            pred_sketch_raw = results.get('pred_sketch_raw')
            pred_sketch = results.get('pred_sketch_refine')
            pred_sketch_warped = results.get('pred_sketch_warped')
            '''
            semantic loss
            '''
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1)


            '''
            sketch loss
            '''
            filter_sketch_gt = torch.index_select(sketch_gt.view(-1), 0, selectindex)




            if pred_sketch_raw is not None:
                assert pred_sketch_raw.shape[1]==2
                filtersketch_raw = torch.index_select(pred_sketch_raw.permute(0, 2, 3, 4, 1).contiguous()
                                                  .view(-1, 2), 0, selectindex)
            if pred_sketch is not None:
                assert pred_sketch.shape[1] == 2
                filtersketch = torch.index_select(pred_sketch.permute(0, 2, 3, 4, 1).contiguous()
                                                .view(-1, 2), 0, selectindex)
            if pred_sketch_warped is not None:
                assert pred_sketch_warped.shape[1] == 2
                filterpred_sketch_warped = torch.index_select(pred_sketch_warped.permute(0, 2, 3, 4, 1).contiguous()
                                                  .view(-1, 2), 0, selectindex)



            criterion_sketch = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()


            if pred_sketch_raw is not None:
                loss_sketch_raw = criterion_sketch(filtersketch_raw, filter_sketch_gt)
                loss_sketch_raw = torch.mean(loss_sketch_raw)
            if pred_sketch is not None:
                loss_sketch = criterion_sketch(filtersketch, filter_sketch_gt)
                loss_sketch = torch.mean(loss_sketch)
            if pred_sketch_warped is not None:
                loss_sketch_warped = criterion_sketch(filterpred_sketch_warped,filter_sketch_gt)
                loss_sketch_warped = torch.mean(loss_sketch_warped)


            if engine.distributed:
                if pred_sketch is not None:
                    dist.all_reduce(loss_sketch, dist.ReduceOp.SUM)
                    loss_sketch = loss_sketch / engine.world_size

                if pred_sketch_raw is not None:
                    dist.all_reduce(loss_sketch_raw, dist.ReduceOp.SUM)
                    loss_sketch_raw = loss_sketch_raw / engine.world_size

                if pred_sketch_warped is not None:
                    dist.all_reduce(loss_sketch_warped, dist.ReduceOp.SUM)
                    loss_sketch_warped = loss_sketch_warped / engine.world_size



            else:
                loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config_sketch.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            loss = torch.tensor(0.,requires_grad=True).cuda()


            if pred_sketch_raw is not None:
                loss += (loss_sketch_raw) * config_sketch.sketch_weight
            if pred_sketch is not None:
                loss += (loss_sketch) * config_sketch.sketch_weight
            if pred_sketch_warped is not None:
                loss += (loss_sketch_warped) * config_sketch.sketch_weight
            if loss.item()==0:
                print('something wrong')
                exit()




            sum_loss += loss.item()


            if pred_sketch_raw is not None:
                sum_sketch_raw += loss_sketch_raw.item()
            if pred_sketch is not None:
                sum_sketch += loss_sketch.item()
            if pred_sketch_warped is not None:
                sum_sketch_warped += loss_sketch_warped.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config_sketch.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config_sketch.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (sum_loss / (idx + 1)) \
                        + ' sketch_loss=%.5f' % (sum_sketch / (idx + 1)) \
                        + ' raw_sketch_loss=%.5f'% (sum_sketch_raw / (idx + 1)) \
                        + ' sketch_warped=%.5f'% (sum_sketch_warped / (idx + 1)) \

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss/tot', sum_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/semantic', sum_sem / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch', sum_sketch / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_raw', sum_sketch_raw / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_warped', sum_sketch_warped / len(pbar), epoch)
            logger.add_scalar('train_loss/post_sketch_loss', sum_sketch_from_pred / len(pbar), epoch)
            logger.add_scalar('train_loss/KLD', sum_kld / len(pbar), epoch)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > config_sketch.nepochs // 4) and (epoch % config_sketch.snapshot_iter == 0) or (epoch == config_sketch.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config_sketch.snapshot_dir,
                                                config_sketch.log_dir,
                                                config_sketch.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config_sketch.snapshot_dir,
                                                config_sketch.log_dir,
                                                config_sketch.log_dir_link)
