import torch
import torch.nn as nn


class Body_Loss_Cri(nn.Module):
    def __init__(self):
        super(Body_Loss_Cri, self).__init__()
        self.criterion =nn.CrossEntropyLoss(ignore_index=255, reduction='none')


    def forward(self, pred_body,label,label_weight,sketch_gt):
        sketch_gt = sketch_gt.view(label_weight.shape[0],-1).int()
        sketch_gt = torch.logical_not(sketch_gt).int()
        label_weight = label_weight.int()
        edges = sketch_gt & label_weight
        selectindex = torch.nonzero(edges.view(-1)).view(-1)
        filterLabel = torch.index_select(label.view(-1), 0, selectindex)
        filterOutput = torch.index_select(pred_body.permute(
            0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
        body_loss = self.criterion(filterOutput, filterLabel)
        body_loss = torch.mean(body_loss)
        return body_loss
