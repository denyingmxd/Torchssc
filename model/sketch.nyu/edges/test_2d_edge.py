import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from generate_mapping_and_sketch import visualize_mapping
import torch
from utils.ply_utils import *
import cv2
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

def example():
    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/test.txt', 'r') as f:
        test_lines = [i.strip() for i in f.readlines()]
    for i,line in enumerate(test_lines):
        if i==9:
            img = Image.open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Edge2D_HHA/eval_results/imgs_epoch_019/NYU{}_colors.png'.format(line))
            img = np.array(img)
            # rgb = cv2.imread('/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU{}_colors.png'.format(line))
            # gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # sobelx = cv2.convertScaleAbs(sobelx)  # 转回uint8
            # sobely = cv2.convertScaleAbs(sobely)
            # sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

            # print(np.unique(img))
            # plt.hist(img.ravel(), bins=256,  fc='k', ec='k')
            # plt.show()
            mapping = np.load('/ssd/jenny/TorchSSC/DATA/NYU/Mapping/{}.npz'.format(line))['arr_0']
            thresh=100
            img[img<thresh]=0
            img[img>=thresh]=1
            plt.imshow(img)
            plt.show()
            img = torch.from_numpy(img.reshape(1,1,img.shape[0],img.shape[1])).float()
            x3d = visualize_mapping(mapping, img)[:,:,:,0].numpy()
            x3d[x3d<0]=0
            # print(x3d.shape)
            # print(torch.unique(x3d))
            voxel_complete_ply(x3d, './sadvfav.ply')



def main():
    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/test.txt', 'r') as f:
        test_lines = [i.strip() for i in f.readlines()]
    sketch_2d_iou = np.array([0, 0, 0, 0, 0], dtype=float)
    thresh = 100
    for i,line in enumerate(test_lines):
        print(i)
        img = Image.open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Edge2D_HHA/eval_results/imgs_epoch_019/NYU{}_colors.png'.format(line))
        img = np.array(img)
        mapping = np.load('/ssd/jenny/TorchSSC/DATA/NYU/Mapping/{}.npz'.format(line))['arr_0']
        label_weight_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}.npz'.format(line)
        label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)


        rgb = cv2.imread('/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU{}_colors.png'.format(line))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)  # 转回uint8
        sobely = cv2.convertScaleAbs(sobely)
        sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        img = sobelxy
        img[img<thresh]=0
        img[img>=thresh]=1
        img = torch.from_numpy(img.reshape(1,1,img.shape[0],img.shape[1])).float()
        x3d = visualize_mapping(mapping, img)[:,:,:,0].numpy().reshape(-1,)
        x3d[x3d<0]=0
        gt_sketch_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D/{}.npy'.format(line)
        gt_sketch = np.load(gt_sketch_path).reshape(-1,)




        sketch_2d_iou += cal_prec_recall_iou(gt_sketch, x3d, label_weight)

    print(print_iou(sketch_2d_iou))





if __name__ == '__main__':
    # example()
    main()


