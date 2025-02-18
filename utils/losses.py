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


class BCEWithLogitsLossWeighted(nn.Module):
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.weight_type = weight_type
        self.device = device
        
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ]).unsqueeze(0)
        else:
            self.weights = None  # Will be computed dynamically
        
        if self.weights is not None:
            self.weights = self.weights.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target).to(self.device)
        
        loss = self.criterion(pred, target)
        
        if self.weights is not None:
            loss = loss * self.weights
        
        return loss.mean()  # Trả về giá trị trung bình loss

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0] + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


class CrossEtropyLoss(nn.Module):
    '''Class to measure loss between categorical emotion predictions and labels with weighted binary cross-entropy loss.'''

    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device

        # Define weights based on the weight_type
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ]).unsqueeze(0)
        else:
            self.weights = None  # Initialize weights as None for dynamic type

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            # Calculate dynamic weights based on the target
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)

        # Apply log-sigmoid function
        log_sigmoid_pos = F.logsigmoid(pred)  # Equivalent to log(sigmoid(pred))
        log_sigmoid_neg = F.logsigmoid(-pred)  # Equivalent to log(1 - sigmoid(pred))

        # Compute weighted binary cross-entropy loss
        loss = -(target * log_sigmoid_pos + (1 - target) * log_sigmoid_neg)

        # Apply weights
        if self.weights is not None:
            loss = loss * self.weights

        return loss.mean()

    def prepare_dynamic_weights(self, target):
        # Calculate frequency-based dynamic weights
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0] + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights