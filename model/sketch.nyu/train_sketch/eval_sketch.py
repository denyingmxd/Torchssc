#!/usr/bin/env python3
# encoding: utf-8
import os
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
import csv
import sys
sys.path.append('../')
from sketch_config import config_sketch
from utils.pyt_utils import ensure_dir, parse_devices
from sketch_evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import compute_score
from nyu_sketch import NYUv2
from sketch_models import make_model
from sketch_dataset import ValPre
from torch.utils.data import dataloader
from multiprocessing.reduction import ForkingPickler

logger = get_logger()


default_collate_func = dataloader.default_collate


def cal_prec_recall_iou(left,right,label_weight):
    nonefree = np.where((label_weight > 0))
    left=left[nonefree]
    right=right[nonefree]
    tp_occ = ((left > 0) & (right > 0)).astype(np.int8).sum()
    fp_occ = ((left == 0) & (right > 0)).astype(np.int8).sum()
    fn_occ = ((left > 0) & (right == 0)).astype(np.int8).sum()

    union = ((left > 0) | (right > 0)).astype(np.int8).sum()
    intersection = ((left > 0) & (right > 0)).astype(np.int8).sum()

    return  np.array([tp_occ,fp_occ,fn_occ,intersection,union])


def print_iou(iou):
    tp_occ, fp_occ, fn_occ, intersection, union = iou
    IOU_sc = intersection / union
    precision_sc = tp_occ / (tp_occ + fp_occ)
    recall_sc = tp_occ / (tp_occ + fn_occ)
    return [IOU_sc, precision_sc, recall_sc]


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, model):
        img = data['data']
        label = data['label']
        hha = data['hha_img']
        tsdf = data['tsdf']
        label_weight = data['label_weight']
        depth_mapping_3d = data['depth_mapping_3d']
        seg_2d = data['seg_2d']
        seg_2d_sketch = data['seg_2d_sketch']
        name = data['fn']
        sketch_gt = data['sketch_gt']
        results = self.eval_ssc(img, hha, tsdf, depth_mapping_3d, sketch_gt,seg_2d, seg_2d_sketch,device)


        pred  = results.get('pred_semantic')
        pred_sketch_refine = results.get('pred_sketch_refine')
        pred_sketch_raw = results.get('pred_sketch_raw')
        sketch_from_pred = results.get('sketch_from_pred')

        results_dict = {'pred':pred, 'label':label, 'label_weight':label_weight,
                        'name':name, 'mapping':depth_mapping_3d,'pred_sketch_refine':pred_sketch_refine, 'sketch_gt':sketch_gt,'pred_sketch_raw':pred_sketch_raw,
                        'sketch_from_pred':sketch_from_pred}
        self.save_real_path = os.path.join(self.save_path,'results',model.split('/')[-1].split('.')[0])

        # exit()
        if self.save_path is not None:
            fn = name + '.npy'
            if pred_sketch_refine is not None:
                ensure_dir(self.save_real_path + '_sketch')
                np.save(os.path.join(self.save_real_path + '_sketch', fn), pred_sketch_refine)
            if sketch_from_pred is not None:
                ensure_dir(self.save_real_path + '_sketch_from_pred')
                np.save(os.path.join(self.save_real_path + '_sketch_from_pred', fn), sketch_from_pred)
            if pred_sketch_raw is not None:
                ensure_dir(self.save_real_path + '_raw_sketch')
                np.save(os.path.join(self.save_real_path + '_raw_sketch', fn), pred_sketch_raw)
            if pred is not None:
                ensure_dir(self.save_real_path)
                np.save(os.path.join(self.save_real_path, fn), pred)

            logger.info('Save the pred npz ' + fn)

        return results_dict

    def hist_info(self, n_cl, pred, gt):
        assert (pred.shape == gt.shape)
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                           minlength=n_cl ** 2).reshape(n_cl,
                                                        n_cl), correct, labeled

    def compute_metric(self, results,i):
        sketch_iou = np.array([0,0,0,0,0],dtype=float)
        raw_sketch_iou = np.array([0,0,0,0,0],dtype=float)
        sketch_from_pred_iou =np.array([0,0,0,0,0],dtype=float)


        for d in results:
            label = d['label'].astype(np.int64)
            label_weight = d['label_weight'].astype(np.float32)
            mapping = d['mapping'].astype(np.int64).reshape(-1)
            pred_sketch = d.get('pred_sketch_refine')
            sketch_gt = d.get('sketch_gt').ravel()
            raw_sketch = d.get('pred_sketch_raw')
            sketch_from_pred = d.get('sketch_from_pred')
            # Calculate the sketch iou
            if pred_sketch is not None:
                pred_sketch = pred_sketch.astype(np.int64)
                sketch_iou += cal_prec_recall_iou(sketch_gt, pred_sketch.ravel(), label_weight)
            if raw_sketch is not None:
                raw_sketch = raw_sketch.astype(np.int64)
                raw_sketch_iou += cal_prec_recall_iou(sketch_gt, raw_sketch.ravel(), label_weight)
            if sketch_from_pred is not None:
                sketch_from_pred = sketch_from_pred.astype(np.int64)
                sketch_from_pred_iou += cal_prec_recall_iou(sketch_gt, sketch_from_pred.ravel(), label_weight)







        raw_sketch_score = print_iou(raw_sketch_iou)
        sketch_score = print_iou(sketch_iou)
        sketch_from_pred_score = print_iou(sketch_from_pred_iou)

        result_line = self.print_ssc_iou(raw_sketch_score, sketch_score,sketch_from_pred_score,i)
        return result_line

    def eval_ssc(self, img, disp, tsdf, depth_mapping_3d, sketch_gt,seg_2d, seg_2d_sketch, device=None):
        ori_rows, ori_cols, c = img.shape
        input_data, input_disp = self.process_image_rgbd(img, disp, crop_size=None)
        results = self.val_func_process_ssc(input_data, input_disp, tsdf, depth_mapping_3d, sketch_gt, seg_2d, seg_2d_sketch,device)
        ssc_score = results.get('pred_semantic')
        sketch_score = results.get('pred_sketch_refine')
        raw_sketch_score = results.get('pred_sketch_raw')
        sketch_from_pred_score = results.get('sketch_from_pred')
        if ssc_score is not None:
            ssc_score = ssc_score.permute(1, 2, 3, 0)
            ssc_score = ssc_score.cpu().numpy()
            ssc_score = ssc_score.argmax(3)
        if sketch_score is not None:
            sketch_score = sketch_score.permute(1, 2, 3, 0)
            sketch_score = sketch_score.cpu().numpy()
            sketch_score = sketch_score.argmax(3)
        if raw_sketch_score is not None:
            raw_sketch_score = raw_sketch_score.permute(1, 2, 3, 0)
            raw_sketch_score = raw_sketch_score.cpu().numpy()
            raw_sketch_score = raw_sketch_score.argmax(3)
        if sketch_from_pred_score is not None:
            sketch_from_pred_score = sketch_from_pred_score.permute(1, 2, 3, 0)
            sketch_from_pred_score = sketch_from_pred_score.cpu().numpy()
            sketch_from_pred_score = sketch_from_pred_score.argmax(3)
        results={'pred_semantic':ssc_score, 'pred_sketch_refine':sketch_score, 'pred_sketch_raw':raw_sketch_score, 'sketch_from_pred':sketch_from_pred_score}
        return results

    def val_func_process_ssc(self, input_data, input_disp, tsdf, input_mapping, sketch_gt, seg_2d, seg_2d_sketch, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        input_disp = np.ascontiguousarray(input_disp[None, :, :, :], dtype=np.float32)
        input_disp = torch.FloatTensor(input_disp).cuda(device)

        # print(input_mapping.shape, 'hhhhhhhh')
        input_mapping = np.ascontiguousarray(input_mapping[None, :], dtype=np.int32)
        input_mapping = torch.LongTensor(input_mapping).cuda(device)

        tsdf = np.ascontiguousarray(tsdf[None, :], dtype=np.float32)
        tsdf = torch.FloatTensor(tsdf).cuda(device)

        sketch_gt = np.ascontiguousarray(sketch_gt[None, :], dtype=np.int32)
        sketch_gt = torch.LongTensor(sketch_gt).cuda(device)

        seg_2d = np.ascontiguousarray(seg_2d[None, :], dtype=np.float)
        seg_2d = torch.FloatTensor(seg_2d).cuda(device)

        seg_2d_sketch = np.ascontiguousarray(seg_2d_sketch[None, :], dtype=np.float)
        seg_2d_sketch = torch.FloatTensor(seg_2d_sketch).cuda(device)


        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                results = self.val_func(input_data, input_mapping, tsdf,sketch_gt,seg_2d,seg_2d_sketch)

                score = results.get('pred_semantic')

                bin_score = results.get('pred_sketch_raw')
                sketch_score = results.get('pred_sketch_refine')
                sketch_from_pred = results.get('sketch_from_pred')
                if score is not None:
                    score = score[0]
                    score = torch.exp(score)
                if sketch_score is not None:
                    sketch_score = sketch_score[0]
                if bin_score is not None:
                    bin_score = bin_score[0]
                if sketch_from_pred is not None:
                    sketch_from_pred = sketch_from_pred[0]
                # if self.is_flip:
                #     input_data = input_data.flip(-1)
                #     input_disp = input_disp.flip(-1)
                #     score_flip = self.val_func(input_data, input_disp)
                #     score_flip = score_flip[0]
                #     score += score_flip.flip(-1)

                results = {'pred_semantic': score, 'pred_sketch_raw': bin_score, 'pred_sketch_refine': sketch_score,
                           'sketch_from_pred':sketch_from_pred}
                return results

    def print_ssc_iou(self, raw_sketch_score_sc, sketch_score_sc,sketch_from_pred_iou,i):
        lines = []


        lines.append('--*-- Sketch Completion --*--\n')
        lines.append('IOU: %f\n' % sketch_score_sc[0])
        lines.append('pixel-accuracy: %f\n' % sketch_score_sc[1])  # 0 和 1 类的IOU
        lines.append('recall: %f\n' % sketch_score_sc[2])

        lines.append('--*-- Raw Sketch Completion --*--\n')
        lines.append('IOU: %f\n' % raw_sketch_score_sc[0])
        lines.append('pixel-accuracy: %f\n' % raw_sketch_score_sc[1])  # 0 和 1 类的IOU
        lines.append('recall: %f\n' % raw_sketch_score_sc[2])

        lines.append('--*-- Sketch from pred Completion --*--\n')
        lines.append('IOU: %f\n' % sketch_from_pred_iou[0])
        lines.append('pixel-accuracy: %f\n' % sketch_from_pred_iou[1])  # 0 和 1 类的IOU
        lines.append('recall: %f\n' % sketch_from_pred_iou[2])
        line = "\n".join(lines)
        print(line)
        with open(self.csv_out, 'a') as f:
            csvwriter = csv.writer(f)

            fileds = ['nyu', self.models[i].split('-')[-1],
                      sketch_score_sc[0], sketch_score_sc[1], sketch_score_sc[2],
                      raw_sketch_score_sc[0], raw_sketch_score_sc[1], raw_sketch_score_sc[2],
                      sketch_from_pred_iou[0], sketch_from_pred_iou[1], sketch_from_pred_iou[2]]
            csvwriter.writerow(fileds)

        return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('--save_path', '-p', default='results')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = make_model(None,config_sketch.model_name,True)
    data_setting = {'img_root': config_sketch.img_root_folder,
                    'gt_root': config_sketch.gt_root_folder,
                    'hha_root':config_sketch.hha_root_folder,
                    'mapping_root': config_sketch.mapping_root_folder,
                    'train_source': config_sketch.train_source,
                    'eval_source': config_sketch.eval_source,
                    'seg_2d_root_folder': config_sketch.seg_2d_root_folder,
                    'seg_2d_sketch_root_folder': config_sketch.seg_2d_sketch_root_folder,
                    }
    val_pre = ValPre()
    dataset = NYUv2(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config_sketch.num_classes, config_sketch.image_mean,
                                 config_sketch.image_std, network,
                                 config_sketch.eval_scale_array, config_sketch.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config_sketch.snapshot_dir, args.epochs, config_sketch.val_log_file,
                      config_sketch.link_val_log_file)
