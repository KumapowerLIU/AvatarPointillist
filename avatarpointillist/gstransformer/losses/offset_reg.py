import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['OffsetReg']


class OffsetReg(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = sigma
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_offset):
        r"""
        Args:
            pred_offset: (B, N, 3)

        Returns:
            loss
        """
        target = torch.full_like(pred_offset, self.sigma, device=pred_offset.device)
        return self.mse_loss(pred_offset, target)


class ScaleReg(nn.Module):
    def __init__(self, threshold_scale=1.0):
        self.threshold_scale = threshold_scale
        super().__init__()
 

    def forward(self, scalings, visibility_filter, opt):
        r"""
        Args:
            pred_offset: (B, N, 3)

        Returns:
            loss
        """
        scale_loss = F.relu(torch.exp(scalings[visibility_filter]) - self.threshold_scale).norm(dim=1).mean()  
        return scale_loss