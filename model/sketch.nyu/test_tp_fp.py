import numpy as np
import torch
import os
from utils.Sobel import Sobel3D
from config import config
from utils.ply_utils import voxel_complete_ply_torchssc
def cal_prec_recall_iou(left,right,label_weight):
    nonefree = np.where((label_weight > 0))
    left=left[nonefree]
    right=right[nonefree]
    tp_occ = ((left > 0) & (right > 0)).astype(np.int8).sum()
    fp_occ = ((left == 0) & (right > 0)).astype(np.int8).sum()
    fn_occ = ((left > 0) & (right == 0)).astype(np.int8).sum()

    union = ((left > 0) | (right > 0)).astype(np.int8).sum()
    intersection = ((left > 0) & (right > 0)).astype(np.int8).sum()
    # IOU_sc = intersection / union
    # precision_sc = tp_occ / (tp_occ + fp_occ)
    # recall_sc = tp_occ / (tp_occ + fn_occ)
    return np.array([tp_occ,fp_occ,fn_occ,intersection,union])


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # exclude 255
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), correct, labeled
def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


def print_iou(iou):
    tp_occ, fp_occ, fn_occ, intersection, union = iou
    IOU_sc = intersection / union
    precision_sc = tp_occ / (tp_occ + fp_occ)
    recall_sc = tp_occ / (tp_occ + fn_occ)
    return  [IOU_sc,precision_sc,recall_sc]

def main():
    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/test.txt', 'r') as f:
        test_lines = [i.strip() for i in f.readlines()]
    print(test_lines)
    raw_sketch_iou=np.array([0,0,0,0,0],dtype=float)
    sketch_iou=np.array([0,0,0,0,0],dtype=float)
    pred_sketch_iou=np.array([0,0,0,0,0],dtype=float)
    combine_sketch_iou = np.array([0, 0, 0, 0, 0], dtype=float)
    tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0
    hist_ssc = np.zeros((config.num_classes, config.num_classes))
    correct_ssc = 0
    labeled_ssc = 0
    sobel = Sobel3D()
    model = 'network_s0_Deeplabv3'
    epoch = 249
    save_path ='/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/{}/{}.csv'.format(model,epoch)

    raw_sketch_outs=[]
    sketch_outs=[]
    ssc_outs=[]
    pred_sketch_outs=[]

    for i in range(1, 2000):
        if '{:0>4d}'.format(i) in test_lines:
            case=i
            print('{:0>4d}'.format(i))


            # pred_sketch_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/{}/results/epoch-{}_sketch/{:0>4d}.npy'.format(model,epoch,case)
            # raw_sketch_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/{}/results/epoch-{}_raw_sketch/{:0>4d}.npy'.format(model,epoch,case)
            gt_sketch_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D/{:0>4d}.npy'.format(case)
            pred_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/{}/results/epoch-{}/{:0>4d}.npy'.format(model,epoch,case)
            gt_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/{:0>4d}.npz'.format(case)
            label_weight_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{:0>4d}.npz'.format(case)

            # pred_sketch = np.load(pred_sketch_path).reshape(-1,)
            # raw_sketch = np.load(raw_sketch_path).reshape(-1,)
            gt_sketch = np.load(gt_sketch_path).reshape(-1,)
            pred = np.load(pred_path).reshape(-1,)

            target = np.load(gt_path)['arr_0'].astype(np.int64)
            label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)

            nonefree_pred = pred[label_weight ==1 ]
            nonefree_label = target[label_weight==1]
            h_ssc, c_ssc, l_ssc = hist_info(config.num_classes, nonefree_pred, nonefree_label)
            hist_ssc += h_ssc
            correct_ssc += c_ssc
            labeled_ssc += l_ssc

            # raw_sketch_iou+=cal_prec_recall_iou(gt_sketch,raw_sketch,label_weight)
            # sketch_iou+=cal_prec_recall_iou(gt_sketch,pred_sketch,label_weight)
            sketch_from_pred = sobel(torch.from_numpy(pred.reshape(60,36,60)[None,None,:,:,:]))

            sketch_from_pred = sketch_from_pred.int().numpy().squeeze(0).squeeze(0).reshape(-1,)
            pred_sketch_iou+=cal_prec_recall_iou(gt_sketch,sketch_from_pred,label_weight)

            # print('pred_sketch',cal_prec_recall_iou(gt_sketch,sketch_from_pred,label_weight))

            # new_sketch = np.logical_and(raw_sketch, pred_sketch)
            # combine_sketch_iou += cal_prec_recall_iou(gt_sketch, new_sketch, label_weight)
            # break
            # print(gt_sketch.shape)
            # print(sketch_from_pred.shape)
            # raw_sketch_outs.append(cal_prec_recall_iou(gt_sketch,raw_sketch,label_weight))
            # sketch_outs.append(cal_prec_recall_iou(gt_sketch,pred_sketch,label_weight))
            ssc_outs.append([h_ssc, c_ssc, l_ssc])
            pred_sketch_outs.append(cal_prec_recall_iou(gt_sketch,sketch_from_pred,label_weight))
            # break
            # exit()

    if raw_sketch_outs==[]:
        raw_sketch_outs=np.zeros((len(test_lines),5))
    if sketch_outs == []:
        sketch_outs = np.zeros((len(test_lines), 5))



    print(print_iou(raw_sketch_iou))
    print(print_iou(sketch_iou))
    print(print_iou(pred_sketch_iou))
    print(print_iou(combine_sketch_iou))
    # score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)
    # print(score_ssc)
    write_csv(save_path,raw_sketch_outs,sketch_outs,ssc_outs,pred_sketch_outs,test_lines)


import csv
def write_csv(save_path,raw_sketch_outs,sketch_outs,ssc_outs,pred_skecth_outs,test_lines):
    with open(save_path,'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['num','ceil','floor','wall',
                    'win','chari','bed','sofa','table','tvs','furn','objs','avgs',
                     'raw_sketch_iou','raw_sketch_precision','raw_sketch_recall',
                     'sketch_iou','sketch_precision','sketch_recall','pred_sketch_iou','pred_sketch_precision','pred_sketch_recall'])
        for i in range(len(ssc_outs)):
            raw_sketch = raw_sketch_outs[i]
            raw_sketch=print_iou(raw_sketch)
            sketch = sketch_outs[i]
            sketch=print_iou(sketch)
            skecth_from_pred = pred_skecth_outs[i]
            skecth_from_pred=print_iou(skecth_from_pred)
            ssc = ssc_outs[i]
            ssc = compute_score(ssc[0], ssc[1], ssc[2])
            kk = ssc[0].tolist()
            for j in range(len(kk)):
                if np.isnan(kk[j]):
                    kk[j]=0
                # print(kk[i])
            # exit()
            csvwriter.writerow([test_lines[i], kk[1],
                      kk[2], kk[3], kk[4], kk[5], kk[6], kk[7], kk[8], kk[9], kk[10], kk[11], ssc[2],
                      raw_sketch[0], raw_sketch[1], raw_sketch[2],
                      sketch[0], sketch[1], sketch[2],
                                skecth_from_pred[0],skecth_from_pred[1],skecth_from_pred[2]])
import pandas
import matplotlib.pyplot as plt
def read_csv(model,epoch):
    save_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/{}/{}.csv'.format(model, epoch)

    df = pandas.read_csv(save_path)
    mious = df['avgs']
    sketch_iou = df['sketch_iou']
    plt.scatter(sketch_iou,mious)
    plt.show()





def read():
    model = 'original'
    epoch = 249
    read_csv(model,epoch)
main()
# read()
