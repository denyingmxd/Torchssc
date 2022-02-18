import torch
import torch.nn as nn
class MultiLabelLoss4(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelLoss4, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.KLDivLoss(reduction='batchmean')
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target,label_weight):
        # pred = pred.view(-1, self.num_classes)
        b = label_weight.shape[0]
        loss=0
        for i in range(b):
            label_weight_b = label_weight[i]
            target_b = target[i]
            pred_b = pred[i]
            target_weight_b = (target_b.sum(dim=0)==1).view(label_weight_b.shape[0])
            target_weight_b = target_weight_b.int()
            label_weight_b = label_weight_b.int()
            weight_b = label_weight_b & target_weight_b
            selectindex = torch.nonzero(weight_b.view(-1)).view(-1)
            target_b = target_b.permute(1,2,3,0)
            filterLabel = torch.index_select(target_b.view(-1,12), 0, selectindex)
            filterOutput = torch.index_select(pred_b.permute(1,2,3, 0).contiguous().view(-1, 12), 0, selectindex)
            filterOutput = torch.log_softmax(filterOutput, dim=1)
            loss_b = self.get_loss(filterOutput, filterLabel)
            if len(selectindex)==0:
                continue
            loss = loss+loss_b
        loss/=b
        return loss

    def get_loss(self, output, target):
        return self.loss(output, target)