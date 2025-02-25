import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "Feature dimension must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = self.head_dim ** -0.5
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x_source, x_target):
        batch_size = x_source.size(0)
        
        Q = self.query(x_source)  # (batch_size, feature_dim)
        K = self.key(x_target)    # (batch_size, feature_dim)
        V = self.value(x_target)  # (batch_size, feature_dim)
        
        Q = Q.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        K = K.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        V = V.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        
        energy = torch.bmm(Q, K.transpose(-1, -2)) * self.scale
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(attention, V)
        
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.feature_dim)
        out = self.fc_out(out)
        return out