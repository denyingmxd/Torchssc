import numpy as np
import sys,os
sys.path.append('../../furnace')
from utils.ply_utils import from_npz_to_ply
import argparse
import threading
from concurrent.futures import TimeoutError
import concurrent.futures

from tqdm import tqdm
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

def generate_ply(args):
    # print(npy_dir_names)
    # print(ply_dir_names)
    print(args)
    label_pred_dir = os.path.join(args.npy_path, 'results/epoch-{}'.format(args.epoch))
    sketch_pred_dir = os.path.join(args.npy_path, 'results/epoch-{}_sketch'.format(args.epoch))
    label_save_dir = os.path.join(args.npy_path, 'results-ply/epoch-{}'.format(args.epoch))
    sketch_save_dir =os.path.join(args.npy_path, 'results-ply/epoch-{}_sketch'.format(args.epoch))
    tsdf_dir='/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/'
    target_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/'

    # print(label_pred_dir)
    # print(label_save_dir)
    # exit()
    if not os.path.isdir(label_save_dir):
        os.mkdir(label_save_dir)
    if not os.path.isdir(sketch_save_dir):
        os.mkdir(sketch_save_dir)

    sketch_save_paths=[]
    label_save_paths=[]
    label_pred_paths=[]
    sketch_pred_paths=[]
    tsdf_paths=[]
    target_paths=[]



    if os.path.isdir(label_pred_dir):
        for file in os.listdir(label_pred_dir):
            if file.endswith('.npy'):
                label_pred_path = os.path.join(label_pred_dir, file)
                label_save_path = os.path.join(label_save_dir, file.replace('.npy', '.ply'))
                label_save_paths.append(label_save_path)
                label_pred_paths.append(label_pred_path)
                tsdf_paths.append(os.path.join(tsdf_dir,file.replace('npy','npz')))
                target_paths.append(os.path.join(target_dir,file.replace('npy','npz')))

    if os.path.isdir(sketch_pred_dir):
        for file in os.listdir(sketch_pred_dir):
            if file.endswith('.npy'):
                sketch_pred_path = os.path.join(sketch_pred_dir, file)
                sketch_save_path = os.path.join(sketch_save_dir, file.replace('npy', 'ply'))
                sketch_save_paths.append(sketch_save_path)
                sketch_pred_paths.append(sketch_pred_path)
    else:
        sketch_save_paths=[None]*len(label_save_paths)
        sketch_pred_paths=[None]*len(label_save_paths)


    if len(label_save_paths)!=len(sketch_save_paths):
        print(len(label_save_paths),len(sketch_save_paths))
        print('length not match')
        exit()
    # if len(os.listdir(label_pred_dir))!=len(label_save_paths):
    #     print('not enough labels')
    #     exit()
    # if len(os.listdir(sketch_pred_dir))!=len(sketch_save_paths):
    #     print('not enough labels')
    #     exit()
    import time
    a=time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_work, label_pred_paths[i], label_save_paths[i],
                                   tsdf_paths[i], target_paths[i],sketch_pred_paths[i], sketch_save_paths[i],i)
                   for i in range(len(label_pred_paths))]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    b=time.time()


    print(b-a)

def process_work(label_pred_path,label_save_path,tsdf_path,target_path,sketch_pred_path,sketch_save_path,i):
    from_npz_to_ply(label_pred_path, label_save_path, tsdf_path, target_path)
    # from_npz_to_ply(sketch_pred_path, sketch_save_path, tsdf_path, target_path)


def visualize_sktech_and_label_gt():
    label_paths=[]
    sktech_gt_paths=[]
    label_save_paths=[]
    sktech_gt_save_paths=[]
    train_list = []
    test_list = []

    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/train.txt','r') as f:
        train_list=[line.strip() for line in f.readlines()]
    with open('/ssd/lyding/SSC/TorchSSC/DATA/NYU/test.txt','r') as f:
        test_list=[line.strip() for line in f.readlines()]

    if not os.path.isdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label-ply'):
        os.mkdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label-ply')
    if not os.path.isdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D-ply'):
        os.mkdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D-ply')

    print(len(train_list))
    print(len(test_list))
    # exit()
    for file in sorted(os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label')):
        label_paths.append(os.path.join('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label',file))
        label_save_paths.append(os.path.join('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label-ply',file.replace('npz','ply')))

    for file in sorted(os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D')):
        sktech_gt_paths.append(os.path.join('/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D',file))
        sktech_gt_save_paths.append(os.path.join('/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D-ply', file.replace('npy', 'ply')))

    from multiprocessing import Pool

    pool = Pool(processes=12)
    for i in range(1, 1450):
        if '{:0>4d}'.format(i) in train_list or '{:0>4d}'.format(i) in test_list:
            result = pool.apply_async( process_work,(label_paths[i-1],label_save_paths[i-1],label_paths[i-1].replace('Label','TSDF'),label_paths[i-1],sktech_gt_paths[i-1],sktech_gt_save_paths[i-1],i))
        else:
            print('some file missing, {}'.format(i))
            exit()
    pool.close()
    pool.join()
    if result.successful():
        print('done')





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', default=None, type=str)
    parser.add_argument('--epoch', default=249, type=int)
    args = parser.parse_args()

    if args.npy_path is None:
        print('no npy_path selected, stop')
        exit()
    if args.epoch is None:
        print('no epoch selected, stop')
        exit()
    generate_ply(args)

main()
# visualize_sktech_and_label_gt()