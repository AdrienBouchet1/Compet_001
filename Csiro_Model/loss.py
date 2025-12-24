import torch.nn as nn 
import torch


class WeightedMSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, y_pred, y_true):
        # y_pred, y_true : [B, 5]
        w = self.weights.to(y_pred.device)
        loss = (y_pred - y_true) ** 2
        loss = loss * w
        return loss.mean()



