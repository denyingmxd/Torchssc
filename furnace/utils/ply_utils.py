import numpy as np
import os

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

def voxel_complete_ply(vox_labeled, ply_filename):

    if type(vox_labeled) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    new_vertices=[]
    num_vertices=0
    new_faces = []
    num_faces = 0
    size = vox_labeled.shape
    vox_labeled = vox_labeled.flatten()
    _x, _y, _z = _get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    vox_labeled[vox_labeled == 255] = 0
    _rgb = colorMap[vox_labeled[:]]
    xyz_rgb = np.stack((_x,_y,_z,_rgb[:, 0], _rgb[:, 1], _rgb[:, 2]),axis=-1)

    xyz_rgb = np.array(xyz_rgb)
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]
    # bb=time.time()

    for i in range(len(ply_data)):
        num_of_line=i
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
    new_vertices = np.array(new_vertices).reshape(-1,6)
    new_faces = np.array(new_faces).reshape(-1,5)
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
                'property   list   uint8   int32   vertex_indices\n'\
               'end_header'.format(num_vertices, num_faces)
    # ---- Save ply data to disk
    # if not os.path.exists('/'.join(ply_filename.split('/')[:-1])):
    #     os.mkdir('/'.join(ply_filename.split('/')[:-1]))
    np.savetxt(ply_filename, new_vertices, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
    with open(ply_filename,'ab') as f:
        np.savetxt(f, new_faces, fmt="%d %d %d %d %d", comments='')  # It takes 20s
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head


def voxel_complete_ply_all_class2(vox_labeled, ply_filename,out_path):
    if type(vox_labeled) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    size = vox_labeled.shape
    # vox_labeled = vox_labeled.flatten()
    for i in range(1,12):
        cls_voxel=np.zeros_like(vox_labeled)
        cls_voxel[vox_labeled==i]=i
        if np.max(cls_voxel)==0:
            # print('none')
            continue
        name = out_path + ply_filename[0].split('/')[-1][3:7] + '_{}.ply'.format(i)
        voxel_complete_ply(cls_voxel,name)

def voxel_complete_ply_no_color(vox_labeled, ply_filename,color):
    if type(vox_labeled) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    new_vertices=[]
    num_vertices=0
    new_faces = []
    num_faces = 0
    size = vox_labeled.shape
    vox_labeled = vox_labeled.flatten()
    _x, _y, _z = _get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    vox_labeled[vox_labeled == 255] = 0
    vox_labeled[vox_labeled == 1] = color
    _rgb = colorMap[vox_labeled[:]]
    ###attention here, define color here
    xyz_rgb = np.stack((_x,_y,_z,_rgb[:, 0], _rgb[:, 1], _rgb[:, 2]),axis=-1)

    xyz_rgb = np.array(xyz_rgb)
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]
    # bb=time.time()

    for i in range(len(ply_data)):
        num_of_line=i
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
    new_vertices = np.array(new_vertices).reshape(-1,6)
    new_faces = np.array(new_faces).reshape(-1,5)
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
                'property   list   uint8   int32   vertex_indices\n'\
               'end_header'.format(num_vertices, num_faces)
    # ---- Save ply data to disk
    # if not os.path.exists('/'.join(ply_filename.split('/')[:-1])):
    #     os.mkdir('/'.join(ply_filename.split('/')[:-1]))
    np.savetxt(ply_filename, new_vertices, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
    with open(ply_filename,'ab') as f:
        np.savetxt(f, new_faces, fmt="%d %d %d %d %d", comments='')  # It takes 20s
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        # z, y, x = np.unravel_index(i, small_size)  # 閫熷害鏇存參浜?
        # print(x, y, z)

        label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
        label_bin = label_i.flatten()  # faltten 杩斿洖鐨勬槸鐪熷疄鐨勬暟缁勶紝闇€瑕佸垎閰嶆柊鐨勫唴瀛樼┖闂?
        # label_bin = label_i.ravel()  # 灏嗗缁存暟缁勫彉鎴?1缁存暟缁勶紝鑰宺avel 杩斿洖鐨勬槸鏁扮粍鐨勮鍥?

        # zero_count_0 = np.sum(label_bin == 0)
        # zero_count_255 = np.sum(label_bin == 255)
        zero_count_0 = np.array(np.where(label_bin == 0)).size  # 瑕佹瘮sum鏇村揩
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
            label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def from_npz_to_ply(npz_file,ply_file,tsdf_file, target_file):
    print(npz_file)
    if 'npy' in npz_file:
        data = np.load(npz_file).astype(int)
    elif 'npz' in npz_file:
        data = np.load(npz_file)['arr_0'] ### be careful here
    else:
        print('not supported')
        exit()
    # print(data.files)
    tsdf = np.load(tsdf_file)['arr_1']
    target = np.load(target_file)['arr_0']
    data = np.ravel(data)
    data *= target != 255
    data *= tsdf == 1
    data = data.reshape(60, 36, 60)
    print(ply_file)

    if '_sketch' in npz_file:
        color=np.array([[22, 191, 206],
                        [0,128,255]])
        voxel_complete_ply(data,ply_file)
    else:
        voxel_complete_ply(data, ply_file)
        pass
    # print('aaa',npz_file,ply_file)
    return 0


def voxel_complete_ply_torchssc(vox_labeled, ply_filename,label_weight,target):
    vox_labeled = vox_labeled
    vox_labeled = np.ravel(vox_labeled)
    vox_labeled *= target != 255
    vox_labeled *= label_weight == 1
    vox_labeled = vox_labeled.reshape(60, 36, 60)
    if type(vox_labeled) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    new_vertices=[]
    num_vertices=0
    new_faces = []
    num_faces = 0
    size = vox_labeled.shape
    vox_labeled = vox_labeled.flatten()
    _x, _y, _z = _get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    vox_labeled[vox_labeled == 255] = 0
    _rgb = colorMap[vox_labeled[:]]
    xyz_rgb = np.stack((_x,_y,_z,_rgb[:, 0], _rgb[:, 1], _rgb[:, 2]),axis=-1)

    xyz_rgb = np.array(xyz_rgb)
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]
    # bb=time.time()

    for i in range(len(ply_data)):
        num_of_line=i
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
    new_vertices = np.array(new_vertices).reshape(-1,6)
    new_faces = np.array(new_faces).reshape(-1,5)
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
                'property   list   uint8   int32   vertex_indices\n'\
               'end_header'.format(num_vertices, num_faces)
    # ---- Save ply data to disk
    # if not os.path.exists('/'.join(ply_filename.split('/')[:-1])):
    #     os.mkdir('/'.join(ply_filename.split('/')[:-1]))
    np.savetxt(ply_filename, new_vertices, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
    with open(ply_filename,'ab') as f:
        np.savetxt(f, new_faces, fmt="%d %d %d %d %d", comments='')  # It takes 20s
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head

