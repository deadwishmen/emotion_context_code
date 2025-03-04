import torch
import torch.nn as nn
import math

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x):
        attn_out, _ = self.self_attention(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        mlp_out = self.mlp(self.ln_2(x))
        return x + mlp_out

class VisionTransformer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, 
                 hidden_dim: int, mlp_dim: int, dropout: float = 0.0, attention_dropout: float = 0.0, 
                 num_classes: int = 1000):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        seq_length = (image_size // patch_size) ** 2 + 1  # +1 for class token
        
        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.Sequential(*[EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout) 
                                     for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Khởi tạo trọng số
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _process_input(self, x):
        n, c, h, w = x.shape
        x = self.conv_proj(x)  # (n, hidden_dim, n_h, n_w)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)  # (n, seq_length, hidden_dim)
        return x

    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.ln(x)[:, 0]  # Lấy class token
        return self.head(x)

def vit_b_16(pretrained: bool = False, progress: bool = True, **kwargs):
    model = VisionTransformer(
        image_size=224, patch_size=16, num_layers=12, num_heads=12, 
        hidden_dim=768, mlp_dim=3072, **kwargs
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vit_b_16-c867db91.pth', 
                                                        progress=progress, check_hash=True)
        model.load_state_dict(state_dict)
    return model