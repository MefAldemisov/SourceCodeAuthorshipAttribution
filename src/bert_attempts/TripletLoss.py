import torch
from torch import nn

'''
with reference to https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
'''


class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def calc_euclidean(x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        # strainge idea to use ReLU instead of max
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
