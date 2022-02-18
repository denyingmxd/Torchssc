import torch
import torch.nn as nn
import torch.functional as F
class Sobel3D(nn.Module):
    def __init__(self,use_cuda=False,thresh=1):
        super(Sobel3D, self).__init__()
        self.kernel_x = [
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1], ],
            [[2, 0, -2],
             [2, 0, -2],
             [2, 0, -2], ],
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1], ]]

        self.kernel_y = [
            [[1, 2, 1],
             [1, 2, 1],
             [1, 2, 1], ],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0], ],
            [[-1, -2, -1],
             [-1, -2, -1],
             [-1, -2, -1], ]]

        self.kernel_z = [
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1], ]]

        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0)
        self.kernel_z = torch.FloatTensor(self.kernel_z).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=self.kernel_x, requires_grad=False)#.cuda()
        self.weight_y = nn.Parameter(data=self.kernel_y, requires_grad=False)#.cuda()
        self.weight_z = nn.Parameter(data=self.kernel_z, requires_grad=False)#.cuda()
        self.thresh=thresh



        self.conv_x = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight = self.weight_x
        self.conv_y = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y.weight = self.weight_y
        self.conv_z = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_z.weight = self.weight_z



    def forward(self, data):
        data[data == 255] = 0
        # print(self.conv_x.weight)

        data = data.float()

        dx = self.conv_x(data)
        dy = self.conv_y(data)
        dz = self.conv_z(data)
        dx[abs(dx) > 0] = 1
        dy[abs(dy) > 0] = 1
        dz[abs(dz) > 0] = 1
        cat = torch.cat([dx, dy, dz], dim=1)
        norm = torch.norm(cat, dim=1, keepdim=True)
        norm[norm <= self.thresh] = 0
        norm[norm > self.thresh] = 1
        return norm

