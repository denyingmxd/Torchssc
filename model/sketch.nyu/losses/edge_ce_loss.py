import torch
import torch.nn as nn


class Edge_CE_Loss(nn.Module):
    def __init__(self,weight):
        super(Edge_CE_Loss, self).__init__()
        self.criterion =nn.CrossEntropyLoss(ignore_index=255, weight=weight,reduction='none')
        self.makeup = torch.tensor(0.).cuda()

        self.threshold=1

    def forward(self, output, label,label_weight,sketch_from_pred):
        sketch_from_pred = torch.argmax(sketch_from_pred,dim=1,keepdim=True).int().view(label_weight.shape[0],-1)
        # print(self.makeup,self.makeup.requires_grad)
        label_weight = label_weight.int()
        # print(torch.unique(label_weight))
        # print(sketch_from_pred.dtype)
        # print(label_weight.dtype)
        # exit()
        edges = sketch_from_pred & label_weight
        # selectindex1 = torch.nonzero(edges).view(-1)
        selectindex = torch.nonzero(edges.view(-1)).view(-1)

        # print(selectindex.shape)
        # print(selectindex1.shape)
        # exit()
        # print(selectindex.numel())
        if selectindex.numel()<self.threshold:
            # print(123)
            return self.makeup,False
        filterLabel = torch.index_select(label.view(-1), 0, selectindex)
        filterOutput = torch.index_select(output.permute(
            0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
        edge_loss = self.criterion(filterOutput, filterLabel)
        edge_loss = torch.mean(edge_loss)
        # print(edge_loss)
        return edge_loss,True
