import os

import numpy as np
import imageio
import torch
import sys
import torch.nn as nn
sys.path.append('../../furnace')
from utils.ply_utils import *
from utils.projection_layer import *
import matplotlib.pyplot as plt
def read_depth(depth_filename):
    depth = imageio.imread(depth_filename)  # numpy.float64
            # assert depth.shape == (img_h, img_w), 'incorrect default size'
    depth1 = np.asarray(depth,dtype=np.float64)
    # depth2 = np.asarray(depth)
    # print(depth1.dtype)
    # print(depth2.dtype)
    # print(np.all(depth1==depth2))
    return depth1

def read_rgb(rgb_filename):
    rgb = imageio.imread(rgb_filename)
    return rgb

def read_rle(rle_filename):  # 0.0005s
    r"""Read RLE compression data
    Return:
        vox_origin,
        cam_pose,
        vox_rle, voxel label data from file
    Shape:
        vox_rle, (240, 144, 240)
    """
    fid = open(rle_filename, 'rb')
    vox_origin = np.fromfile(fid, np.float32, 3).T  # Read voxel origin in world coordinates
    cam_pose = np.fromfile(fid, np.float32, 16).reshape((4, 4))  # Read camera pose
    vox_rle = np.fromfile(fid, np.uint32).reshape((-1, 1)).T  # Read voxel label data from file
    vox_rle = np.squeeze(vox_rle)  # 2d array: (1 x N), to 1d array: (N , )
    fid.close()
    return vox_origin, cam_pose, vox_rle

def proj_2d_to_3d(depth,cam_pose,vox_origin,K):
    H, W = depth.shape
    gx, gy = np.meshgrid(range(W), range(H))
    base=np.stack([gx,gy,np.ones_like(gx,dtype=np.float64)],axis=-1)
    pt = np.dot(base,np.linalg.inv(K).T)
    pt=np.multiply(pt,depth[:,:,np.newaxis])
    bb=np.concatenate([pt,np.ones((H,W,1))],axis=-1)
    ppt=np.dot(bb,cam_pose.T)
    ppt=ppt[:,:,:3]
    ppt=ppt-vox_origin
    return ppt/0.08

def modify(points,method):
    points=points.astype(int)
    return points



def proj_3d_to_2d(depth,cam_pose,vox_origin,K):
    gx, gy, gz = np.meshgrid(range(60), range(36), range(60), indexing='ij')
    gx = gx * 0.08
    gy = gy * 0.08
    gz = gz * 0.08

    gx += vox_origin[0]
    gy += vox_origin[1]
    gz += vox_origin[2]
    base = np.transpose(np.stack([gx, gy, gz, np.ones_like(gx, dtype=np.float64)], axis=0),(1,2,3,0))  # (60,36,60,4)
    cam_ex = np.linalg.inv(cam_pose)
    cam_ex = cam_ex[:3, :]
    points = np.dot(base,cam_ex.T)
    points = np.dot(points, K.T)

    points[:,:,:, 0] /= points[:,:,:, 2]
    points[:,:,:, 1] /= points[:,:,:, 2]
    # points=np.floor(points).astype(int)
    return points


def points_to_mapping(point):
    if point[0]<0 or point[0]>=480 or point[1]<0 or point[1]>=640:
        return 307200
    if np.isnan(point[0]) or np.isnan(point[0]):
        # print(point)
        return 307200
    else:
        # print(point)
        return point[0]*640+point[1]




def visualize_mapping(depth_mapping_3d,feature2d=None):
    if feature2d is None:
        feature2d = torch.ones((1,3,480,640))
    depth_mapping_3d=torch.from_numpy(depth_mapping_3d).long()
    b, c, h, w = feature2d.shape
    feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

    zerosVec = -1*torch.ones(b, 1,
                           c)  # for voxels that could not be projected from the depth map, we assign them zero vector
    # print(feature2d.shape)
    # print(zerosVec.shape)
    segVec = torch.cat((feature2d, zerosVec), 1)

    segres = [torch.index_select(segVec[i], 0, depth_mapping_3d) for i in range(b)]
    segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60
    segres=segres.squeeze(0).permute(1,2,3,0).int()
    # print(segres.shape)
    # print(segres[0,0,0])
    return segres

def visualize_position(position,x2d=None):
    if x2d is None:
        x2d = torch.ones((1,1,480,640))
        print(x2d.dtype)
    idx = torch.from_numpy(position).long()
    voxel_from_position =Project2Dto3D()(x2d,idx)
    # print(voxel_from_position.shape)
    low_voxel_from_position = nn.MaxPool3d(4, stride=4)(voxel_from_position)[0][0]
    # low_voxel_from_position = _downsample_label(voxel_from_position[0][0].numpy())
    # print(low_voxel_from_position.shape)
    # print(np.sum(low_voxel_from_position))
    # print(torch.sum(low_voxel_from_position))
    return low_voxel_from_position.int()
    # return low_voxel_from_position.astype(int)


def mapping_from_position(position):
    position_voxel = visualize_position(position,x2d=None)
    new_mapping = np.zeros((60, 36, 60))
    for x in range(60):
        for y in range(36):
            for z in range(60):
                if position_voxel[x][y][z]>0:
                    new_mapping[x][y][z]=position_voxel[x][y][z]
                else:
                    new_mapping[x][y][z]=307200
    return new_mapping.reshape(-1,)






def get_mapping_from_satnet(npz_file):
    loaddata = np.load(npz_file)
    mapping = loaddata['arr_3'].astype(np.int64)
    mapping1 = np.ones((8294400), dtype=np.int64)
    mapping1[:] = -1
    ind, = np.where(mapping >= 0)
    mapping1[mapping[ind]] = ind
    mapping2 =torch.FloatTensor(mapping1.reshape((1, 1, 240, 144, 240)).astype(np.float32))
    mapping2 = torch.nn.MaxPool3d(4, 4)(mapping2).data.view(-1).numpy()
    mapping2[mapping2 < 0] = 307200
    depth_mapping_3d = torch.LongTensor(mapping2.astype(np.int64))
    print(depth_mapping_3d.shape)
    return depth_mapping_3d


def mapping_from_depth(depth,cam_pose,vox_origin,K):
    vox_points = proj_2d_to_3d(depth, cam_pose, vox_origin, K)
    vox_points = vox_points.reshape(-1, 3)
    # new_mapping=np.ones((240*144*240,))*-1
    new_mapping=np.ones((129600,))*307200
    for i,point in enumerate(vox_points):
        z,x,y=point
        x=int(np.floor(x))
        y=int(np.floor(y))
        z=int(np.floor(z))
        # if (x >= 0 and x < 240 and  y >= 0 and y < 144 and z >= 0 and z < 240):
        if (x >= 0 and x < 60 and  y >= 0 and y < 36 and z >= 0 and z < 60):
            vox_id = 60*36*z+60*y+x
            # print()
            # vox_id =240*144*z+144*y+x
            new_mapping[vox_id] = i
            # new_mapping[i] = vox_id
    return new_mapping.astype(np.int64)

def compare_torchssc_my_mapping(mapping,new_mapping):
    new_mapping = visualize_mapping(new_mapping).numpy()
    new_mapping = new_mapping[:, :, :, 0]
    new_mapping[new_mapping<0]=0
    original_voxel_from_mapping = visualize_mapping(mapping).numpy()
    original_voxel_from_mapping = original_voxel_from_mapping[:, :, :, 0]
    original_voxel_from_mapping[original_voxel_from_mapping < 0] = 0
    # print(np.unique(new_mapping))
    # print(np.unique(original_voxel_from_mapping))
    # exit()
    print(np.sum(new_mapping&original_voxel_from_mapping))

    # satnet_mapping = np.load("/ssd/lyding/SSC/TorchSSC/DATA/NYU/SATNet_Mapping/000000.npz")['arr_4']
    # print(satnet_mapping.shape)
    # print(np.sum(satnet_mapping==new_mapping))
    # print(np.unique(satnet_mapping))
    # print(np.sum(satnet_mapping>-1))
    # print(np.sum(new_mapping>-1))

def mapping_from_3d(depth,cam_pose,vox_origin,K):
    points=proj_3d_to_2d(depth,cam_pose,vox_origin,K)
    mapping = np.ones((60, 36, 60))*307200
    depth=depth
    print(depth.shape)
    for z in range(60):
        for y in range(36):
            for x in range(60):
                u,v,d = points[x][y][z]
                u=int(np.floor(u))
                v=int(np.floor(v))

                if (u >= 0 and u < 480 and v >= 0 and v < 640 and d > 0.5 and d < 8):
                    if d-depth[u][v]>0:
                        # mapping[x][y][z] = 1
                        mapping[z][y][x] = v*480+u
    print(np.sum(mapping<307200))
    return mapping.reshape(-1,)




def main():
    K = np.array([[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
              [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
              [0, 0, 1]]) # fx = K(1,1); fy = K(2,2);
    # exit()
    ddr_npz_filename = './test_cases/NYU0028_0000_voxels.npz'
    ddr_npz = np.load(ddr_npz_filename)
    bin_filename='/ssd/jenny/augdepth_SSC/dataloaders/sscnet.cs.princeton.edu/sscnet_release/data/depthbin/NYUtest/NYU0028_0000.bin'
    depth = ddr_npz['depth'][0]
    vox_origin, cam_pose, vox_rle=read_rle(bin_filename)
    mapping = np.load('/ssd/jenny/TorchSSC/DATA/NYU/Mapping/0001.npz')['arr_0']
    position = ddr_npz['position']
    new_mapping = mapping_from_position(position)
    new_mapping = mapping_from_depth(depth,cam_pose,vox_origin,K)
    new_mapping=mapping
    new_mapping = visualize_mapping(new_mapping).numpy()
    new_mapping = new_mapping[:, :, :, 0]
    new_mapping[new_mapping < 0] = 0
    voxel_complete_ply(new_mapping, './test_cases/blabla.ply')
    print(len(np.floor(np.unique(position))))

    print(len(mapping[mapping<307200]))
    exit()

    # print(np.sum(new_mapping<307200))
    # compare_torchssc_my_mapping(mapping,new_mapping)

    exit()




    # points=proj_2d_to_3d(depth,cam_pose,vox_origin,K)
    # new_mapping=proj_3d_to_2d(depth,cam_pose,vox_origin,K,ddr_npz)
    # print(np.unique(new_mapping))
    original_voxel_from_mapping = visualize_mapping(mapping).numpy()
    original_voxel_from_mapping=original_voxel_from_mapping[:,:,:,0]
    original_voxel_from_mapping[original_voxel_from_mapping<0]=0
    # print(np.unique(original_voxel_from_mapping))
    # print(np.unique(voxel_from_position))

    # print(np.multiply(voxel_from_position,original_voxel_from_mapping).sum())

    # voxel_complete_ply(original_voxel_from_mapping,'./test_cases/original_mapping.ply')
    new_mapping = visualize_mapping(new_mapping).numpy()
    new_mapping = new_mapping[:, :, :, 0]
    voxel_complete_ply(new_mapping,'./test_cases/new_mapping.ply')
    # voxel_complete_ply(voxel_from_position,'./test_cases/position_nn_downsample.ply')
    exit()
    new_voxel_from_new_mapping = visualize_mapping(new_mapping).numpy()
    new_voxel_from_new_mapping = new_voxel_from_new_mapping[:, :, :, 0]
    new_voxel_from_new_mapping[new_voxel_from_new_mapping < 0] = 0
    # voxel_complete_ply(new_voxel_from_new_mapping, './test_cases/new_mapping_nn_downsample.ply')




def main2():
    K = np.array([[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
                  [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
                  [0, 0, 1]])  # fx = K(1,1); fy = K(2,2);
    ddr_npz_filename = './test_cases/NYU0028_0000_voxels.npz'
    ddr_npz = np.load(ddr_npz_filename)
    bin_filename = '/ssd/jenny/augdepth_SSC/dataloaders/sscnet.cs.princeton.edu/sscnet_release/data/depthbin/NYUtest/NYU0028_0000.bin'
    depth = ddr_npz['depth']
    vox_origin, cam_pose, vox_rle = read_rle(bin_filename)

    mapping = np.load('/ssd/jenny/TorchSSC/DATA/NYU/Mapping/0028.npz')['arr_0']
    # print(mapping.shape)
    # new_mapping = proj_3d_to_2d_c2(depth,cam_pose,vox_origin,K,ddr_npz)
    new_mapping = mapping_from_position(ddr_npz['position']).reshape(-1,)

    new_mapping = visualize_mapping(new_mapping).numpy()
    new_mapping = new_mapping[:, :, :, 0]
    new_mapping[new_mapping < 0] = 0
    new_mapping=new_mapping
    voxel_complete_ply(new_mapping,'./test_cases/new_mapping.ply')
    
def main3():
    base = '/ssd/lyding/datasets/SSC/NYUtest_npz/'
    out_base='/ssd/lyding/SSC/TorchSSC/DATA/NYU/My_NYU_Mapping/'
    import os
    for i in range(1,1450):
        print(i)
        name = base+'NYU{:>04d}_0000_voxels.npz'.format(i)
        print(name)
        if os.path.isfile(name):
            out_name = out_base+'{:>04d}.npz'.format(i)
            position = np.load(name)['position']
            new_mapping = mapping_from_position(position.reshape(-1, ))
            np.savez(out_name,new_mapping)

        # print(out_name)
        #     exit()




# main()
# my_mapping = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/My_NYU_Mapping/0028.npz')['arr_0'].reshape(-1,)
# print(my_mapping.shape)
# mapping = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0028.npz')['arr_0']
# print(mapping.shape)
# a=visualize_mapping(my_mapping)
# for i,file in enumerate(sorted(os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/My_NYU_Mapping/'))):
#     if int(file.split('.')[0])-1>i:

        # print(i,file)
        # exit()