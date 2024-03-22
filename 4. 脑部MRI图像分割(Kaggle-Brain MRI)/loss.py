import torch.nn as nn
import torch

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


class IOULoss(nn.Module):

    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = intersection / ((y_pred + y_true).sum() + 1e-16)
        return 1. - dsc

class FocalTverskyLoss(nn.Module):

    def __init__(self, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = 0.75
        self.smooth = 1.0
        self.alpha = 0.7

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = intersection / ((y_pred + y_true).sum() + 1e-16)

        true_pos = (y_true * y_pred).sum()
        false_neg = (y_true * (1-y_pred)).sum()
        false_pos = ((1-y_true)*y_pred).sum()
        pt_1 = (true_pos + self.smooth)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + self.smooth)
        dsc = torch.pow((1-pt_1), self.gamma)
        
        return dsc