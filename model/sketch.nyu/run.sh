#!/usr/bin/env bash
export NGPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=$NGPUS train3.py -p 10097
python eval.py -e 100-250 -d 0-3 --save_path /ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_s0_multi_label_loss3/\
                --snapshot_dir /ssd/lyding/SSC/TorchSSC/model/sketch.nyu/log/network_s0_multi_label_loss3/snapshot/ \
                --modelname  network_s0