import torch
import torch.nn as nn
import os
import numpy as np
from utils.ply_utils import *
colorMap = np.array([[22, 191, 206],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling red tp
                     [43, 160, 4],      # 2 floor green fp
                     [158, 216, 229],   # 3 wall blue fn
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     [153, 153, 153],     # 12 Accessible area, or label==255, ignore
                     ]).astype(np.int32)

def print_iou(iou):
    tp_occ, fp_occ, fn_occ, intersection, union = iou
    IOU_sc = intersection / union
    precision_sc = tp_occ / (tp_occ + fp_occ)
    recall_sc = tp_occ / (tp_occ + fn_occ)
    return  [IOU_sc,precision_sc,recall_sc]


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



class Projection(nn.Module):
    def __init__(self,):
        super(Projection, self).__init__()


    def forward(self,feature2d,depth_mapping_3d):
        b, c, h, w = feature2d.shape
        feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

        zerosVec = torch.zeros(b, 1,c).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
        # zerosVec = (-1*torch.ones(b, 1,c)).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
        segVec = torch.cat((feature2d, zerosVec), 1)

        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60

        return segres



def test_projection_and_one_hot():

    seg_2d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/2DSSGT_npz/0001.npz'
    depth_mapping_3d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0001.npz'
    seg_2d=torch.from_numpy(np.load(seg_2d_path)['gt2d'].astype(np.int64)).cuda()
    seg_2d = seg_2d.view(1,1,seg_2d.shape[0],seg_2d.shape[1])

    depth_mapping_3d=torch.from_numpy(np.load(depth_mapping_3d_path)['arr_0'].astype(np.int64)).view(1,-1).cuda()
    print('a',depth_mapping_3d.shape)
    print('b',seg_2d.shape)
    seg_2d+=1
    seg_2d[seg_2d == 256] = 0
    seg_2d = seg_2d.float()
    b = seg_2d.clone()
    projection = Projection().cuda()
    seg_2d_one_hot_1 = projection(seg_2d, depth_mapping_3d)
    print('c',seg_2d_one_hot_1.shape)
    print('ff',(b==seg_2d).all())
    # exit()
    seg_2d_one_hot = nn.functional.one_hot(seg_2d_one_hot_1.long(), num_classes=12)[0].float().permute(0,4,1,2,3)
    print('dd',seg_2d.shape)
    seg_3d = nn.functional.one_hot(seg_2d.long(),num_classes=12).float()
    print('d',seg_3d.shape)
    seg_3d=seg_3d[0].permute(0,3,1,2)
    print('ddd', seg_3d.shape)
    seg_3d_one_hot = projection(seg_3d, depth_mapping_3d)
    print('e',seg_3d_one_hot.shape)

    ind1 = torch.argmax(seg_2d_one_hot, dim=1)
    ind2 = torch.argmax(seg_3d_one_hot, dim=1)
    print(ind1.shape)
    print(seg_2d_one_hot_1.shape)
    print((ind1[0]==seg_2d_one_hot_1[0][0]).all())
    exit()
    # print(ind1.shape)
    # print(ind2.shape)
    # print((ind1==ind2).sum())
    #
    #
    # print((seg_2d_one_hot==seg_3d_one_hot).sum())





    # voxel_complete_ply(ind1[0].cpu().numpy(), './0001.ply')
    # voxel_complete_ply(ind2[0].cpu().numpy(), './00001.ply')



##be careful using predicted result or gt specially 0 or 255
def save_seg_2d_to_3d_one_hot_npz():
    out_base='/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_30000_npz/{}'
    projection = Projection().cuda()
    files = os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_30000/')
    files.sort()
    for file in files:
        print(file)
        # exit()
        seg_2d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_30000/{}'.format(file)
        depth_mapping_3d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/{}'.format(file)
        seg_2d = torch.from_numpy(np.load(seg_2d_path)['arr_0']).float().cuda()
        seg_2d = seg_2d.view(1, 1, seg_2d.shape[0], seg_2d.shape[1])
        depth_mapping_3d = torch.from_numpy(np.load(depth_mapping_3d_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()

        # seg_2d += 1


        # seg_2d[seg_2d == 256] = 0
        # seg_2d = seg_2d.float()
        # print(seg_2d.shape)
        # print(torch.unique(seg_2d))
        # exit()
        # print(torch.max(seg_2d))
        # continue
        seg_2d_one_hot = projection(seg_2d, depth_mapping_3d)
        # ind2 = seg_2d_one_hot.clone().int()
        # ind2 = ind2[0][0].cpu().numpy()
        # print(np.unique(ind2))
        # exit()
        seg_2d_one_hot = nn.functional.one_hot(seg_2d_one_hot.long(), num_classes=12)[0].float().permute(0, 4, 1, 2, 3)
        # print(torch.unique(seg_2d_one_hot))
        seg_2d_one_hot=seg_2d_one_hot.cpu().numpy()[0].astype(np.uint8)
        # ind =
        # voxel_complete_ply(ind2,'0001.ply')
        np.savez(out_base.format(file),seg_2d_one_hot)

        # exit()
#

def from_3d_one_hot_npz_to_3d_sketch():
    out_base ='/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_sketch_npz/{}'
    from utils.Sobel import Sobel3D
    sobel = Sobel3D().cuda()
    for file in os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg2d_to_3d_one_hot/'):
        print(file)
        data = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg2d_to_3d_one_hot/{}'.format(file))['arr_0']
        data = torch.from_numpy(data).cuda()
        data=torch.argmax(data,dim=0)
        # voxel_complete_ply(data.cpu().numpy(), '0001.ply')
        data = data.view(1,1,60,36,60).float()
        sketch = sobel(data).int()[0][0].cpu().numpy().astype(np.uint8)
        # voxel_complete_ply(sketch,'sketch_0001.ply')
        np.savez(out_base.format(file),sketch)



            # exit()



def test_3d_sketch_from_2d_seg():
    sketch_iou = np.array([0, 0, 0, 0, 0], dtype=float)
    sketch_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_sketch_npz/'

    for file in os.listdir(sketch_dir):
        print(file)
        sketch = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_sketch_npz/{}'.format(file))['arr_0'].reshape(-1,)
        gt_sketch_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/sketch3D/{}'.format(file.replace('npz','npy'))
        gt_sketch = np.load(gt_sketch_path).reshape(-1, )
        label_weight_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}'.format(file)
        label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)
        sketch_iou+=cal_prec_recall_iou(gt_sketch, sketch, label_weight)
    print(print_iou(sketch_iou))




def save_2d_seg_to_sketch_to_ply():
    out_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_sketch_Deeplabv3_ply/{}'
    projection = Projection().cuda()
    files = os.listdir('/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d/')
    files.sort()
    for file in files:
        print(file)
        # exit()
        seg_2d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d/{}'.format(file)
        depth_mapping_3d_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/{}'.format(file)
        seg_2d = torch.from_numpy(np.load(seg_2d_path)['arr_0']).float().cuda()
        seg_2d = seg_2d.view(1, 1, seg_2d.shape[0], seg_2d.shape[1])
        depth_mapping_3d = torch.from_numpy(np.load(depth_mapping_3d_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
        seg_2d += 1
        seg_2d_to_3d = projection(seg_2d, depth_mapping_3d)
        seg_2d_to_3d=seg_2d_to_3d[0][0].cpu().numpy().astype(np.uint8)
        voxel_complete_ply(seg_2d_to_3d,out_dir.format(file.replace('npz','ply')))
        # np.savez(out_base.format(file), seg_2d_one_hot)
        # exit()


# def test_iou_on_edges():




def test_tsdf_and_S_0_together_ssc():
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
    epoch=249
    network_tsdf_path='/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_tsdf/results/epoch-{}'.format(epoch)
    S_0_Deeplabv3_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_s0_Deeplabv3/results/epoch-{}'.format(epoch)
    files=os.listdir(network_tsdf_path)
    files.sort()
    hist_ssc = np.zeros((12,12))
    correct_ssc = 0
    labeled_ssc = 0
    for file in files:
        print(file)
        network_tsdf_res = np.load(os.path.join(network_tsdf_path.format(epoch),file)).reshape(-1,)
        S_0_Deeplabv3_res = np.load(os.path.join(S_0_Deeplabv3_path.format(epoch),file)).reshape(-1,)
        target = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/{}'.format(file.replace('npy','npz')))['arr_0']
        label_weight = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}'.\
                               format(file.replace('npy','npz')))['arr_1'].astype(np.float32)

        pred = np.zeros_like(target)
        for i in range(len(pred)):
            if network_tsdf_res[i]==target[i]:
                pred[i]=target[i]
                continue
            if S_0_Deeplabv3_res[i]==target[i]:
                pred[i]=target[i]
                continue
            # if network_tsdf_res[i]==S_0_Deeplabv3_res[i]:
            #     pred[i]=S_0_Deeplabv3_res[i]
            # pred[i]=network_tsdf_res[i]

        nonefree_pred = pred[label_weight == 1]
        nonefree_label = target[label_weight == 1]
        h_ssc, c_ssc, l_ssc = hist_info(12, nonefree_pred, nonefree_label)
        hist_ssc += h_ssc
        correct_ssc += c_ssc
        labeled_ssc += l_ssc

        # break
    score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)

    print(score_ssc)


def test_occupied_voxels_tsdf_and_S_0():
    epoch = 249
    network_tsdf_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_tsdf/results/epoch-{}'.format(epoch)
    S_0_Deeplabv3_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_s0_Deeplabv3/results/epoch-{}'.format(
        epoch)
    files = os.listdir(network_tsdf_path)
    files.sort()
    nonempty_tsdf=0
    nonempty_s0=0
    for file in files:
        print(file)
        network_tsdf_res = np.load(os.path.join(network_tsdf_path.format(epoch), file)).reshape(-1, )
        S_0_Deeplabv3_res = np.load(os.path.join(S_0_Deeplabv3_path.format(epoch), file)).reshape(-1, )
        target = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/{}'.format(file.replace('npy', 'npz')))['arr_0']
        label_weight = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}'. \
                               format(file.replace('npy', 'npz')))['arr_1'].astype(np.float32)
        network_tsdf_res=network_tsdf_res[(label_weight == 1) & (target < 255)]
        S_0_Deeplabv3_res = S_0_Deeplabv3_res[(label_weight == 1) & (target < 255)]

        nonempty_s0+=np.sum(S_0_Deeplabv3_res>0)
        nonempty_tsdf+=np.sum(network_tsdf_res>0)
    print(nonempty_tsdf,nonempty_s0)


def rgb_complete_ply(vox_rgb, ply_filename,my_color_map,depth_mapping_3d):
    def _get_xyz(size):
        ####attention
        ####sometimes you need to change the z dierction by multiply -1
        """x 姘村钩 y楂樹綆  z娣卞害"""
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h  # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w  # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d  # z, front-back flip
        return _x, _y, _z
    #this receive voxel containging value ranging from 0 to 255*255*255 of size 60x36x60
    new_vertices = []
    num_vertices = 0
    new_faces = []
    num_faces = 0
    size = vox_rgb.shape
    vox_labeled = vox_rgb.flatten()
    _x, _y, _z = _get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    # vox_labeled[vox_labeled == 255] = 0

    _rgb = my_color_map[vox_labeled[:]]
    xyz_rgb = np.stack((_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2]), axis=-1)

    xyz_rgb = np.array(xyz_rgb)
    ply_data = xyz_rgb[np.where(vox_labeled >= 0)]
    # bb=time.time()

    nnn = torch.sum(depth_mapping_3d<307200).item()
    if len(ply_data)!=nnn:
        print('some wrong here')
        exit()

    for i in range(len(ply_data)):
        num_of_line = i
        z = float(ply_data[i][0])
        y = float(ply_data[i][1])
        x = float(ply_data[i][2])
        r = ply_data[i][3]
        g = ply_data[i][4]
        b = ply_data[i][5]
        vertice_one = [x, y, z, r, g, b]
        vertice_two = [x, y, z + 1, r, g, b]
        vertice_three = [x, y + 1, z + 1, r, g, b]
        vertice_four = [x, y + 1, z, r, g, b]
        vertice_five = [x + 1, y, z, r, g, b]
        vertice_six = [x + 1, y, z + 1, r, g, b]
        vertice_seven = [x + 1, y + 1, z + 1, r, g, b]
        vertice_eight = [x + 1, y + 1, z, r, g, b]
        vertices = [vertice_one, vertice_two, vertice_three, vertice_four,
                    vertice_five, vertice_six, vertice_seven, vertice_eight]
        new_vertices.append(vertices)
        num_vertices += len(vertices)

        base = 8 * num_of_line
        face_one = [4, 0 + base, 1 + base, 2 + base, 3 + base]
        face_two = [4, 7 + base, 6 + base, 5 + base, 4 + base]
        face_three = [4, 0 + base, 4 + base, 5 + base, 1 + base]
        face_four = [4, 1 + base, 5 + base, 6 + base, 2 + base]
        face_five = [4, 2 + base, 6 + base, 7 + base, 3 + base]
        face_six = [4, 3 + base, 7 + base, 4 + base, 0 + base]
        faces = [face_one, face_two, face_three, face_four,
                 face_five, face_six]
        new_faces.append(faces)
        num_faces += len(faces)
    # print(time.time()-bb)
    # cc = time.time()
    new_vertices = np.array(new_vertices).reshape(-1, 6)
    new_faces = np.array(new_faces).reshape(-1, 5)
    if len(ply_data) == 0:
        raise Exception("Oops!  That was no valid ply data.")
    ply_head = 'ply\n' \
               'format ascii 1.0\n' \
               'element vertex {}\n' \
               'property float x\n' \
               'property float y\n' \
               'property float z\n' \
               'property uchar red\n' \
               'property uchar green\n' \
               'property uchar blue\n' \
               'element   face   {}\n' \
               'property   list   uint8   int32   vertex_indices\n' \
               'end_header'.format(num_vertices, num_faces)
    # ---- Save ply data to disk
    # if not os.path.exists('/'.join(ply_filename.split('/')[:-1])):
    #     os.mkdir('/'.join(ply_filename.split('/')[:-1]))
    np.savetxt(ply_filename, new_vertices, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
    with open(ply_filename, 'ab') as f:
        np.savetxt(f, new_faces, fmt="%d %d %d %d %d", comments='')  # It takes 20s
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head

def test_rgb_proj_to_3d():

    rgb_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/'
    label_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/'
    tsdf_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/'
    mapping_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/'
    files = os.listdir(rgb_dir)
    files.sort()
    import cv2
    projection = Projection().cuda()
    for file in files:
        num = file[3:7]
        rgb =  torch.from_numpy(np.array(cv2.imread(os.path.join(rgb_dir,file))[:, :, ::-1])).float()
        h,w,c = rgb.shape
        rgb = rgb.view(1,c,h,w).cuda()
        label_path=os.path.join(label_dir,'{}.npz'.format(num))
        tsdf_path =os.path.join(tsdf_dir,'{}.npz'.format(num))
        mapping_path = os.path.join(mapping_dir,'{}.npz'.format(num))
        label = np.load(label_path)['arr_0'].astype(np.int64)
        label_weight = np.load(tsdf_path)['arr_1'].astype(np.float32)
        depth_mapping_3d = torch.from_numpy(np.load(mapping_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
        rrgb = torch.zeros((480,640)).cuda()
        for i in range(480):
            for j in range(640):
                r,g,b = rgb[0,:,i,j]
                rrgb[i,j] = r*1000*1000+g*1000+b
        rrgb = rrgb.view(1,1,480,640)
        rgb = rrgb
        #
        # rrgb_3d = projection(rrgb,depth_mapping_3d)
        # rrgb_3d = rrgb_3d[0].cpu().numpy()
        # for i in range(60):
        #     for j in range(36):
        #         for k in range(60):
        #             num = rrgb_3d[0,0,i,j,k]
        #             r,g,b=num//1000//1000,(num//1000)%1000,num%1000



        rgb_3d = projection(rgb, depth_mapping_3d)
        rgb_3d = rgb_3d[0].cpu().numpy()
        rgb_complete_ply(rgb_3d[0],'rgb3d.ply')

        break

def test_2d_label_proj_to_3d():
    # label2d_path = '/ssd/lyding/SSC/repositories/DeepLabV3Plus-Pytorch-master/results-latest_deeplabv3_resnet50_nyu_os16-13300/0001_pred_rgb.png'
    label2d_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU0001_colors.png"
    mapping_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0001.npz'
    import cv2
    projection = Projection().cuda()
    label2d = np.array(cv2.imread(label2d_path)[:, :, ::-1])
    color_label_2d = np.zeros((480, 640))
    for i in range(480):
        for j in range(640):
            r,g,b = label2d[i,j]
            color_label_2d[i,j]=r*256*256+g*256+b

    color_label_2d = torch.from_numpy(color_label_2d).float()
    color_label_2d = color_label_2d.view(1,1,480,640).cuda()
    depth_mapping_3d = torch.from_numpy(np.load(mapping_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
    label2d_3d = projection(color_label_2d,depth_mapping_3d)
    my_color_map = np.array([[i//256//256,(i//256)%256,i%256] for i in range(256**3)])
    label2d_3d = label2d_3d[0][0].cpu().numpy().astype(np.int64)


    rgb_complete_ply(label2d_3d,'rgb2d_3d.ply',my_color_map,depth_mapping_3d)


def save_2d_rgb_to_3d_ply():
    import cv2
    projection = Projection().cuda()
    my_color_map = np.array([[i // 256 // 256, (i // 256) % 256, i % 256] for i in range(256 ** 3)])
    out_base = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB3D_ply/{}.ply"
    for file in range(1,1450):
        num = '{:0>4d}'.format(file)
        print(num)
        label2d_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU{}_colors.png".format(num)
        mapping_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/{}.npz'.format(num)
        label2d = np.array(cv2.imread(label2d_path)[:, :, ::-1])
        color_label_2d = np.zeros((480, 640))
        for i in range(480):
            for j in range(640):
                r, g, b = label2d[i, j]
                color_label_2d[i, j] = r * 256 * 256 + g * 256 + b
        color_label_2d = torch.from_numpy(color_label_2d).float()
        color_label_2d = color_label_2d.view(1, 1, 480, 640).cuda()
        depth_mapping_3d = torch.from_numpy(np.load(mapping_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
        label2d_3d = projection(color_label_2d, depth_mapping_3d)
        label2d_3d = label2d_3d[0][0].cpu().numpy().astype(np.int64)
        rgb_complete_ply(label2d_3d, out_base.format(num), my_color_map, depth_mapping_3d)
        # break


def test_0_in_rgb():
    import cv2
    label2d_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU0001_colors.png"
    label2d = np.array(cv2.imread(label2d_path)[:, :, ::-1])
    mapping_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0001.npz'
    depth_mapping_3d = torch.from_numpy(np.load(mapping_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
    print(depth_mapping_3d)

def test_save_2d_rgb_normed():
    label2d_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/RGB/NYU0001_colors.png"
    mapping_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Mapping/0001.npz'
    import cv2
    projection = Projection().cuda()
    label2d = np.array(cv2.imread(label2d_path)[:, :, ::-1])
    color_label_2d = np.zeros((480, 640))
    for i in range(480):
        for j in range(640):
            r, g, b = label2d[i, j]
            color_label_2d[i, j] = r * 256 * 256 + g * 256 + b

    color_label_2d = torch.from_numpy(color_label_2d).float()
    color_label_2d = color_label_2d.view(1, 1, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.load(mapping_path)['arr_0'].astype(np.int64)).view(1, -1).cuda()
    label2d_3d = projection(color_label_2d, depth_mapping_3d)



def test_some_small_thing():
    from utils.ply_utils import voxel_complete_ply_torchssc
    one_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d/0001.npz"
    two_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg2d_to_3d_one_hot/0001.npz"
    three_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/seg_2d_to_3d_sketch_Deeplabv3_npz/0001.npz"
    label_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/0001.npz"
    tsdf_path = "/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/0001.npz"
    label = np.load(label_path)['arr_0'].astype(np.int64)
    label_weight = np.load(tsdf_path)['arr_1'].astype(np.float32)

    a = np.load(one_path)['arr_0']
    b = np.load(two_path)['arr_0']
    c = np.load(three_path)['arr_0']
    b = np.argmax(b,axis=0)
    c = np.argmax(c,axis=0)


    voxel_complete_ply_torchssc(b,'b.ply',label_weight,label)
    voxel_complete_ply_torchssc(c,'c.ply',label_weight,label)
    print()


def test_0_in_depth():
    base='/ssd/lyding/datasets/SSC/{}'
    case='NYUtrain_npz'
    files = os.listdir(base.format(case))
    files.sort()
    zero_num = 0
    for file in files:
        print(file)
        data = np.load(os.path.join(base.format(case),file))
        depth = data['depth'][0]
        zero_num+=(depth==0.0).sum()
    ratio = zero_num/(len(files)*depth.shape[0]*depth.shape[1])
    print(ratio)


def visualize_tsdf():
    file='0001'
    label_weight_path = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}.npz'.format(file)
    label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)
    tsdf = np.load(label_weight_path)['arr_0'].astype(np.float32)
    from utils.ply_utils import voxel_complete_ply

    tsdf_minus1 = tsdf==-1
    tsdf_zero = tsdf==0
    tsdf_1 = tsdf==1
    tsdf_neg = (tsdf<0) & (tsdf>-0.2)
    tsdf_pos =(tsdf>0) & (tsdf<1)
    tsdf_minus1 = tsdf_minus1.astype(int).reshape(60,36,60)
    tsdf_zero = tsdf_zero.astype(int).reshape(60,36,60)
    tsdf_1 = tsdf_1.astype(int).reshape(60,36,60)
    tsdf_neg = tsdf_neg.astype(int).reshape(60,36,60)
    tsdf_pos = tsdf_pos.astype(int).reshape(60,36,60)
    voxel_complete_ply(tsdf_minus1,'tsdf_minus_one.ply')
    voxel_complete_ply(tsdf_zero,'tsdf_zero.ply')
    voxel_complete_ply(tsdf_1,'tsdf_one.ply')
    voxel_complete_ply(tsdf_neg,'tsdf_neg.ply')
    voxel_complete_ply(tsdf_pos,'tsdf_pos.ply')
    return

def _downsample_label_my(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros( (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
        12
    ) , dtype=np.float)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            classes, cnts = np.unique(label_i_s, return_counts=True)
            class_counts = np.zeros(12)
            class_counts[classes] = cnts
            target_classes = class_counts / np.sum(class_counts)


            label_downscale[x, y, z] = target_classes
    return label_downscale


def downsample_high_res_gt():
    data_dir = '/ssd/lyding/datasets/SSC/NYUCADtest_npz/'
    out_dir = '/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label_multi/'
    files = os.listdir(data_dir)
    files.sort()
    from utils.ply_utils import voxel_complete_ply

    for file in files:
        print(file)

        data = np.load(os.path.join(data_dir,file))
        target_lr = data['target_lr']
        target_hr = data['target_hr']
        print(np.sum((target_lr>0)&(target_lr<255)))
        # voxel_complete_ply(target_lr,'low_1.ply')
        # voxel_complete_ply(target_hr,'high_1.ply')
        target_hr = torch.from_numpy(target_hr)
        down_from_high = _downsample_label_my(target_hr)
        # down_from_high = down_from_high.transpose(3,0,1,2)
        # np.savez(os.path.join(out_dir,num+'.npz'),down_from_high)
        abc = np.zeros((60,36,60),dtype=np.uint8)
        abc[(down_from_high.sum(axis=-1)==1)&(down_from_high.max(axis=-1)<1.)]=1
        # abc[(down_from_high.sum(axis=-1)==1)]=1
        print(np.sum(abc))
        # xxx=0
        # abc=np.zeros((60,36,60),dtype=np.uint8)
        # for x in range(60):
        #     for y in range(36):
        #         for z in range(60):
        #             probs = down_from_high[:,x,y,z]
        #             if np.sum(probs)==1 and np.max(probs)<1:
        #                 xxx+=1
        #                 abc[x,y,z]=1
        voxel_complete_ply(abc, 'abc.ply')
        print(12)
        # down_from_high = np.argmax(down_from_high,axis=0)
        # voxel_complete_ply(down_from_high,'down_from_high_1.ply')
        break

def test_tsdf_divided_acc():
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
    epoch=249
    network_tsdf_path='/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_tsdf/results/epoch-{}'.format(epoch)
    # S_0_Deeplabv3_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_seg_2d/results/epoch-{}'.format(epoch)
    S_0_Deeplabv3_path = '/ssd/lyding/SSC/TorchSSC/model/sketch.nyu/results/network_s0_multi_label_loss1/results/epoch-{}'.format(epoch)
    files=os.listdir(network_tsdf_path)
    files.sort()
    hist_ssc = np.zeros((12,12))
    correct_ssc = 0
    labeled_ssc = 0

    a, b =-0.2,0
    for file in files:
        print(file)
        network_tsdf_res = np.load(os.path.join(network_tsdf_path.format(epoch),file)).reshape(-1,)
        S_0_Deeplabv3_res = np.load(os.path.join(S_0_Deeplabv3_path.format(epoch),file)).reshape(-1,)
        label_multi = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label_multi/{}'.format(file.replace('.npy','.npz')))['arr_0']
        label_multi = ((label_multi.sum(axis=0)==1)&(label_multi.max(axis=0)<1)).flatten()
        target = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/Label/{}'.format(file.replace('npy','npz')))['arr_0']
        label_weight = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}'.\
                               format(file.replace('npy','npz')))['arr_1'].astype(np.float32)
        tsdf = np.load('/ssd/lyding/SSC/TorchSSC/DATA/NYU/TSDF/{}'.\
                               format(file.replace('npy','npz')))['arr_0'].astype(np.float32)
        # pred = np.zeros_like(target)
        pred = S_0_Deeplabv3_res
        # for i in range(len(pred)):
        #     if tsdf[i]>a and tsdf[i]<b:
        #         pred[i] = network_tsdf_res[i]
        #         continue
        # pred[(tsdf>a) & (tsdf<b)]=network_tsdf_res[(tsdf>a) & (tsdf<b)]

        cond = (label_weight == 1) & (label_multi == 1)
        # cond = (label_weight == 1)
        nonefree_pred = pred[cond]
        nonefree_label = target[cond]
        h_ssc, c_ssc, l_ssc = hist_info(12, nonefree_pred, nonefree_label)
        hist_ssc += h_ssc
        correct_ssc += c_ssc
        labeled_ssc += l_ssc

        # break
    score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)

    print(score_ssc)







# test_projection_and_one_hot()
# save_seg_2d_to_3d_one_hot_npz()
# from_3d_one_hot_npz_to_3d_sketch()
# test_3d_sketch_from_2d_seg()
# save_2d_seg_to_sketch_to_ply()
# test_rgb_proj_to_3d()
# test_2d_label_proj_to_3d()
# test_0_in_rgb()
# save_2d_rgb_to_3d_ply()
# test_tsdf_and_S_0_together_ssc()
# test_occupied_voxels_tsdf_and_S_0()
# test_some_small_thing()
# test_0_in_depth()
# visualize_tsdf()
# downsample_high_res_gt()
test_tsdf_divided_acc()