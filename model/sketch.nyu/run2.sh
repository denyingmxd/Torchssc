#!/usr/bin/env bash
export NGPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=$NGPUS train3.py -p 10098
python eval.py -e 100-250 -d 0-3 --save_path /ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_s0_multi_label_loss4/\
                --snapshot_dir /ssd/lyding/SSC/TorchSSC/model/sketch.nyu/log/network_s0_multi_label_loss4/ \
                --modelname  network_s0