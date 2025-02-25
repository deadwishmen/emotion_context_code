import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "Feature dimension must be divisible by num_heads"
        
        # Các tầng tuyến tính để tạo Query, Key, Value
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = self.head_dim ** -0.5
        
        # Tầng fully-connected để tổng hợp đầu ra
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x_source, x_target):
        batch_size = x_source.size(0)
        
        # Tạo Query, Key, Value
        Q = self.query(x_source)  # (batch_size, feature_dim)
        K = self.key(x_target)    # (batch_size, feature_dim)
        V = self.value(x_target)  # (batch_size, feature_dim)
        
        # Reshape cho multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)  # (batch_size, num_heads, head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)  # (batch_size, num_heads, head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)  # (batch_size, num_heads, head_dim)
        
        # Tính attention scores
        energy = torch.bmm(Q, K.transpose(-1, -2)) * self.scale  # (batch_size, num_heads, head_dim, head_dim)
        attention = F.softmax(energy, dim=-1)                    # (batch_size, num_heads, head_dim, head_dim)
        
        # Áp dụng attention vào Value
        out = torch.bmm(attention, V)  # (batch_size, num_heads, head_dim)
        out = out.permute(0, 2, 1).contiguous()  # (batch_size, head_dim, num_heads)
        out = out.view(batch_size, self.feature_dim)  # (batch_size, feature_dim)
        
        # Tầng fully-connected cuối cùng
        out = self.fc_out(out)
        return out