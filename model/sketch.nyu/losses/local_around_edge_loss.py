import torch
import torch.nn as nn
from utils.Sobel import Sobel3D
import torch.nn.functional as F
class Local_around_edge_loss(nn.Module):
    def __init__(self,s=4):
        super(Local_around_edge_loss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum')
        self.s = s
        self.makeup = torch.tensor(0.).cuda()

    def forward(self, output, label,label_weight,sketch_from_pred):
        sketch_from_pred = torch.argmax(sketch_from_pred,dim=1,keepdim=True).int()
        label = label.reshape_as(sketch_from_pred)
        label_weight_mask = label_weight.int().reshape_as(sketch_from_pred)


        around_edge_loss = 0
        b, c, h, w, d = output.shape
        output = F.softmax(output, dim=1)
        count = 0

        sh = h//self.s
        sw = w//self.s
        sd = d//self.s
        # self.makeup.zero_()
        for q in range(b*h*w*d//(self.s**3)):
            p,i,j,k = q//(sh*sw*sd),(q//(sw*sd))%sh,(q//sd)%sw,q%sd
            # print(p,i,j,k)
            edge_here = torch.sum(sketch_from_pred[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s])
            # print(edge_here)
            valid_here = torch.sum(label_weight_mask[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s])
            # print(valid_here)
            output_here = torch.sum(label[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s]<255)
            # print(valid_here,edge_here,output_here)
            if edge_here.item()>0 and valid_here.item()>0 and output_here.item()==self.s**3:
                # print('a',p, i, j, k)
                count+=1
                pred_frustum = output[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s]
                target_frustum = label[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s]
                frustum_mask = label_weight_mask[p,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s,k*self.s:(k+1)*self.s]
                pred_frustum = pred_frustum[:,frustum_mask[0]>0]
                pred_frustum = torch.sum(pred_frustum,dim=1)
                target_frustum = target_frustum[:,frustum_mask[0]>0].reshape(-1,)
                classes, cnts = torch.unique(target_frustum, return_counts=True)
                class_counts = torch.zeros(12).cuda()
                class_counts[classes.long()] = cnts.float()
                target_classes = class_counts/torch.sum(class_counts)
                pred_classes = pred_frustum/torch.sum(pred_frustum)

                around_edge_loss += self.criterion(pred_classes.log(),target_classes)
                # print('aaa', pred_classes, target_classes, pred_classes.log(), self.criterion(pred_classes.log(),target_classes))
        if count==0:
            # print('bbb',self.makeup)
            return self.makeup
        # print('count:',count)
        # print('around_edge_loss:',around_edge_loss)
        return around_edge_loss/count


