import numpy as np
import os
import sys

import torch

sys.path.append('../../furnace')
from utils.ply_utils import *
from utils.Sobel import Sobel3D

def compare_target_label(target,ddr_label):
    print(target.shape)
    print(ddr_label.shape)
    print(np.sum(target==255))
    print(np.sum(ddr_label==255))
    print(np.sum(ddr_label==0))
    print(np.sum(target==0))
    target[np.logical_or(target==255,target==0)] = -1
    ddr_label[np.logical_or(ddr_label==255,ddr_label==0)] = -1
    target[target>0] = 1
    ddr_label[ddr_label>0] = 1



def main():
    # data_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/original/epoch-249/0028.npy'
    tsdf_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/0001.npz'
    target_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/0001.npz'
    ddr_label_path = '/ssd/lyding/datasets/SSC/NYUtest_npz/NYU0001_0000_voxels.npz'
    # data = np.load(data_path)
    tsdf = np.load(tsdf_path)['arr_1']
    target = np.load(target_path)['arr_0']
    ddr_label = np.load(ddr_label_path)['target_lr']
    ddr_tsdf = np.load(ddr_label_path)['tsdf_lr']
    other_tsdf = np.load(tsdf_path)['arr_0']
    print(np.unique(other_tsdf))
    print(np.sum(other_tsdf==-1))
    print(np.sum(other_tsdf==0))
    print(np.sum(other_tsdf==1))
    print(np.sum(tsdf))
    print(np.sum(np.multiply(tsdf==1,target==255)))
    print(np.unique(ddr_tsdf))
    print(np.sum(ddr_tsdf<0))
    print(np.sum(ddr_tsdf==0.001))
    print(np.sum(ddr_tsdf==1))
    print(np.sum(target==255))
    print(np.sum(ddr_label==255))
    # print(np.all(target[target<255]==ddr_label[target<255]))
    # exit()
    # data = np.ravel(data)
    # data *= target != 255
    # data *= tsdf == 1
    # data = data.reshape(60,36,60)
    tsdf_0 = np.logical_not(tsdf).astype(int).reshape(60,36,60)
    tsdf_1 = tsdf.copy().reshape(60,36,60).astype(int)
    label_255 = target.copy()
    # label_255[label_255 != 255] = 0
    # label_255[label_255 == 255] = 1
    # label_255=label_255.reshape(60,36,60)
    # target=target.reshape(60,36,60)
    # compare_target_label(target,ddr_label)
    voxel_complete_ply(tsdf_0,'./test_cases/tsdf_1.ply')
    voxel_complete_ply(tsdf_1,'./test_cases/tsdf_0.ply')
    # voxel_complete_ply(label_255,'./test_cases/label_255.ply')
    # voxel_complete_ply(ddr_label,'./test_cases/ddr_label.ply')
    # voxel_complete_ply(ddr_label,'./test_cases/ddr_label.ply')
    # voxel_complete_ply(target,'./test_cases/target.ply')
    # voxel_complete_ply(data,'./out.ply')
    # voxel_complete_ply(data,'./out.ply')

main()
###torchssc and ddrnet have the same target but different 0 and 255
###but torchssc exludes the target 255 and also the tsdf where it is 1
###while ddrnet only exclude those where it is 255


def main2():
    #satnet and torchssc
    data_path = '/ssd/jenny/SUNCG/SATNet_datasets/nyu_selected_val/000000.npz'
    # new_data_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/SATNet_Mapping/000000.npz"
    loaddata = np.load(data_path)
    # new_loaddata = np.load(new_data_path)
    # print(loaddata.files)
    # satnet_mapping = new_loaddata['arr_4']
    # old_mapping = old_loaddata['arr_3']
    # print(np.sum(satnet_mapping==old_mapping))
    # print(satnet_mapping.shape)
    #satnet0 rgb
    #sarnet1 gt
    #satnet2 mapping
    #satnet3
    # exit()
    label = torch.LongTensor(loaddata['arr_1'].astype(np.int64))
    label_weight = torch.FloatTensor(loaddata['arr_2'].astype(np.float32))
    mapping = loaddata['arr_3'].astype(np.int64)
    print(np.sum(mapping>0))
    exit()
    mapping1 = np.ones((8294400), dtype=np.int64)
    mapping1[:] = -1
    ind, = np.where(mapping >= 0)
    mapping1[mapping[ind]] = ind
    mapping2 = torch.autograd.Variable(torch.FloatTensor(mapping1.reshape((1, 1, 240, 144, 240)).astype(np.float32)))
    mapping2 = torch.nn.MaxPool3d(4, 4)(mapping2).data.view(-1).numpy()
    mapping2[mapping2 < 0] = 307200
    depth_mapping_3d = mapping2.astype(np.int64)

    #torchssc
    torchssc_mapping = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0001.npz')['arr_0']
    print(len(depth_mapping_3d[depth_mapping_3d<307200]))
    print(len(torchssc_mapping[torchssc_mapping<307200]))
    print(np.sum((depth_mapping_3d<307200)&(torchssc_mapping<307200)))



def cal_prec_recall_iou(left,right,label_weight):
    nonefree = np.where((label_weight > 0))
    left=left[nonefree]
    right=right[nonefree]
    tp_occ = ((left > 0) & (right > 0)).astype(np.int8).sum()
    fp_occ = ((left == 0) & (right > 0)).astype(np.int8).sum()
    fn_occ = ((left > 0) & (right == 0)).astype(np.int8).sum()

    union = ((left > 0) | (right > 0)).astype(np.int8).sum()
    intersection = ((left > 0) & (right > 0)).astype(np.int8).sum()
    IOU_sc = intersection / union
    precision_sc = tp_occ / (tp_occ + fp_occ)
    recall_sc = tp_occ / (tp_occ + fn_occ)
    return [IOU_sc,precision_sc,recall_sc]

# print(123)
# tsdf_sketch = np.ravel(tsdf_sketch)
# tsdf_sketch *= target != 255
# tsdf_sketch *= label_weight == 1
# tsdf_sketch = tsdf_sketch.reshape(60, 36, 60)
# print(tsdf_sketch.shape)
# voxel_complete_ply(tsdf_sketch,'tsdf_sketch.ply')
# exit()


def main3():
    precision_sum=0
    iou_sum =0
    recall_sum =0
    count=0
    sobel = Sobel3D(thresh=1.6).cuda()
    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/test.txt','r') as f:
        test_lines=[i.strip() for i in f.readlines()]
    for i in range(1,1449+1):
        if '{:0>4d}'.format(i) in  test_lines:
            # print(i)
            label_weight_path='/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{:0>4d}.npz'.format(i)
            mapping_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/{:0>4d}.npz'.format(i)
            sketch_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D/{:0>4d}.npy'.format(i)
            target_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/{:0>4d}.npz'.format(i)
            sketch = np.load(sketch_path).reshape(-1,)
            target = np.load(target_path)['arr_0'].astype(np.int64)
            tsdf = np.load(label_weight_path)['arr_0'].astype(np.float32)
            label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)
            depth_mapping_3d = np.load(mapping_path)['arr_0'].astype(np.int64)
            depth_mapping_3d[depth_mapping_3d<307200]=1
            depth_mapping_3d[depth_mapping_3d==307200]=0
            mapping_and_sktech = cal_prec_recall_iou(sketch,depth_mapping_3d,label_weight)
            tsdf=tsdf.reshape(1, 1, 60, 36, 60)
            tsdf[tsdf<=0]=0
            tsdf[tsdf>0]=1
            tsdf_sketch = sobel(torch.tensor(tsdf).cuda())

            ##chaneg some here
            pred_sketch_raw_1 = tsdf_sketch.int()
            pred_sketch_raw_0 = torch.logical_not(pred_sketch_raw_1).float()
            pred_sketch_raw_1 = pred_sketch_raw_1.float()
            pred_sketch_raw = torch.cat([pred_sketch_raw_0, pred_sketch_raw_1], dim=1)
            tsdf_sketch = pred_sketch_raw.int().cpu().numpy().squeeze(0)
            tsdf_sketch = tsdf_sketch.argmax(0)
            # print(tsdf_sketch.shape)
            # exit()
            tsdf_and_sketch = cal_prec_recall_iou(sketch,tsdf_sketch.reshape(-1,),label_weight)
            print(i,'mapping and sketch', mapping_and_sktech)
            print(i,'tsdf skecth and sketch',tsdf_and_sketch)
            count+=1
            iou_sum +=mapping_and_sktech[0]
            precision_sum += mapping_and_sktech[1]
            recall_sum += mapping_and_sktech[2]
            break

    print('overall iou is',iou_sum/count)
    print('overall precision is', precision_sum/count)
    print('overall recall is', recall_sum/count)

###!!!!!! use label weight or tsdf to sobel is a question
# main3()





