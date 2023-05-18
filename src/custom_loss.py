import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, worst_classes_l):
        super(CustomLoss, self).__init__()
        self.worst_classes = worst_classes_l

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = torch.Tensor([False]*target.shape[0]).to("cuda")
        for i in range(mask.shape[0]):
            if target[i] in self.worst_classes:
                mask[i] = True
        high_cost = (loss * mask.float()).mean()
        return loss + high_cost