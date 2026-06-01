import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceloss(nn.Module):
    """Focal Loss + weighted Dice Loss.

    Focal Loss is applied per-pixel as described in Eq. 17 of the paper:
        L_focal = -Σ_i (1-p_i)^γ log(p_i)
    with γ = 2.
    """

    def __init__(self, weight=20, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        targets = targets.view(-1).float()
        inputs = inputs.view(-1).float()

        # Per-pixel Focal Loss (Eq. 17)
        bce_per_pixel = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_per_pixel)
        focal_loss = ((1 - pt) ** self.gamma * bce_per_pixel).mean()

        # Dice Loss
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return focal_loss, dice_loss
