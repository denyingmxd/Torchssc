#!/usr/bin/env bash
export NGPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_sketch.py -p 10098
python eval_sketch.py -e 100-250 -d 0-3 --save_path /ssd/lyding/SSC/TorchSSC/model/sketch.nyu/train_sketch/results/network_deform/