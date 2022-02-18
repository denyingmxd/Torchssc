import numpy as np
import torch.nn as nn
import torch
from nyu_dataset import NYUDataset
import glob
import pickle
import sys
sys.path.append('../../../furnace')
from utils.ply_utils import *
import hydra
from flosp import *
from omegaconf import DictConfig


def visualize_mapping(depth_mapping_3d):
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




@hydra.main(config_name="./monoscene.yaml")
def main(config: DictConfig):
    full_scene_size = (60, 36, 60)
    project_scale=1

    dataset = NYUDataset(
                split="test",
                preprocess_root=config.NYU_preprocess_root,
                n_relations=config.n_relations,
                root=config.NYU_root,
                fliplr=0.5,
                frustum_size=config.frustum_size,
                color_jitter=(0.4, 0.4, 0.4),
            )
    for i in range(len(dataset)):
        data = dataset[i]
        name = data["name"]

        if name[3:7]=='0028':
            print(name)
            projected_pix_1 = data["projected_pix_1"]
            fov_mask = data["fov_mask_1"]

            # pix_x, pix_y = projected_pix_1[:, 0], projected_pix_1[:, 1]
            # new_mapping = pix_y * 640 + pix_x
            # new_mapping[~fov_mask]=640*480
            # mapping = np.load('/ssd/jenny/TorchSSC/DATA/NYU/Mapping/0028.npz')['arr_0']
            # print(mapping.shape)
            # print(new_mapping.shape)
            # new_mapping = visualize_mapping(new_mapping).numpy()
            x2d = torch.ones((1,480,640))

            # new_mapping = new_mapping[:, :, :, 0]
            flosp = FLoSP(full_scene_size, project_scale=project_scale, dataset="NYU")
            new_mapping = flosp(x2d, torch.from_numpy(projected_pix_1), torch.from_numpy(fov_mask)).numpy()[0].astype(np.int64)

            # new_mapping = new_mapping.reshape(60,36,60)
            # new_mapping[new_mapping ==307200] = 0
            # voxel_complete_ply(new_mapping, './new_mapping.ply')
            voxel_complete_ply(new_mapping, '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/from_mono_scene/123.ply')
            exit()
if __name__ == '__main__':
    main()