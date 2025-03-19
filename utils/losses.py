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
    



class FocalLoss(nn.Module):
    '''Focal Loss for multi-label classification'''
    def __init__(self, gamma=2.0, alpha=None, weight_type='mean', device=torch.device('cpu')):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight_type = weight_type
        self.device = device

        # Khởi tạo trọng số alpha (nếu có)
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha).unsqueeze(0).to(self.device)
        else:
            self.alpha = None  # Không dùng alpha nếu None
        
        # Thiết lập trọng số theo cách của DiscreteLoss
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ]).unsqueeze(0)

        self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        # Dùng dynamic weights nếu cần
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target).to(self.device)

        # Chuyển đổi dự đoán thành xác suất
        pred = torch.sigmoid(pred)  # Vì đây là multi-label classification
        
        # Tính Focal Loss
        pt = (pred * target) + ((1 - pred) * (1 - target))  # p_t = p nếu y=1, 1-p nếu y=0
        focal_weight = (1 - pt) ** self.gamma  # (1 - p_t)^gamma

        # Áp dụng trọng số alpha nếu có
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        
        # Tính Loss
        loss = - (target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        loss = loss * focal_weight * self.weights

        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights
