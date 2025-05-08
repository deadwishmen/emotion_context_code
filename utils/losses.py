import torch
import torch.nn as nn
from torch.nn import functional as F


class DiscreteLoss(nn.Module):
  ''' Class to measure loss between categorical emotion predictions and labels.'''
  def __init__(self, weight_type='mean', device=torch.device('cpu')):
    super(DiscreteLoss, self).__init__()
    self.weight_type = weight_type
    self.device = device
    if self.weight_type == 'mean':
      self.weights = torch.ones((1,26))/26.0
      self.weights = self.weights.to(self.device)
    elif self.weight_type == 'static':
      self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
         0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
         0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
      self.weights = self.weights.to(self.device)

  def forward(self, pred, target):
    if self.weight_type == 'dynamic':
      self.weights = self.prepare_dynamic_weights(target)
      self.weights = self.weights.to(self.device)
    loss = (((pred - target)**2) * self.weights)
    return loss.sum()

  def prepare_dynamic_weights(self, target):
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
    weights = torch.zeros((1,26))
    weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001
    return weights




class BCEWithLogitsLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels using BCEWithLogitsLoss.'''
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        
        # Initialize weights based on specified type
        if self.weight_type == 'mean':
            self.weights = torch.ones((1,26))/26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)
            
        # Define BCEWithLogitsLoss without weight first (will be updated in forward)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        # Update weights if using dynamic weighting
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
            
        # Calculate BCE loss (without reduction)
        loss = self.bce_loss(pred, target)
        
        # Apply weights to the loss
        weighted_loss = loss * self.weights
        
        # Return sum of weighted losses
        return weighted_loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1,26))
        weights[target_stats != 0] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


    



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
