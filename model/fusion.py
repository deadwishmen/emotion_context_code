import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import CrossAttention

from model.qformer import Qformer
from model.FAN import FANLayer
from density_adaptive_attention import DensityBlock




class FusionModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, conbine = 'concat', isSwinT = True):
        super(FusionModel, self).__init__()
        self.x = torch.tensor(num_face_features)
        self.dk = torch.sqrt(self.x)
        self.softmax = nn.Sequential(nn.Softmax(dim=1))
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.conbine = conbine

        self.isSwinT = isSwinT

        if isSwinT:
            self.num_features = num_body_features
            self.fc_context = nn.Linear(num_context_features, self.num_features)
        else:
            self.num_features = num_face_features
            self.fc_context = nn.Linear(num_context_features, self.num_features)
            self.fc_body = nn.Linear(num_body_features, self.num_features)


        self.fc_att = nn.Sequential(nn.Linear(self.num_features, 128),
                                    nn.BatchNorm1d(128),
                                    nn.GELU(),
                                    nn.Linear(128, 1),
                                   )

        self.fc1 = nn.Linear((self.num_features*3), 256)
        self.fc2 = nn.Linear(768, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, x_context, x_body, x_face):
        # Shape: (batch_size, num_context_features), (batch_size, num_body_features), (batch_size, num_face_features)

        

        
        context_features = x_context.view(-1, self.num_context_features) # context_features shape(B, num_context_features)
        body_features = x_body.view(-1, self.num_body_features) # body_features shape(B, num_body_features)
        face_features = x_face.view(-1, self.num_face_features) # face_features shape(B, num_face_features)


        if self.isSwinT:
            body_vec_repeat = face_features.unsqueeze(2).repeat(1, 1, int(self.num_body_features/self.num_face_features))
            face_features = body_vec_repeat.view(-1, self.num_body_features)
            context_features = self.fc_context(context_features)
        else:
            context_features = self.fc_context(context_features)
            body_features = self.fc_body(body_features)



        context_features = context_features.view(-1, self.num_features, 1)
        body_features = body_features.view(-1, self.num_features, 1)
        face_features = face_features.view(-1, self.num_features, 1)

        ###################### Cross Attention #######################
        # Attention for context features
        context_body = (context_features.transpose(1, 2) @ body_features)
        context_face = (context_features.transpose(1, 2) @ face_features)
        context_in = torch.cat((context_body, context_face), 1)
        context_atten = self.softmax(context_in)


        # Attention for body features
        body_context = (body_features.transpose(1, 2) @ context_features)
        body_face = (body_features.transpose(1, 2) @ face_features)
        body_in = torch.cat((body_context, body_face), 1)
        body_atten = self.softmax(body_in)

        # Attention for face features
        face_context = (face_features.transpose(1, 2) @ context_features)
        face_body = (face_features.transpose(1, 2) @ body_features)
        face_in = torch.cat((face_context, face_body), 1)
        face_atten = self.softmax(face_in)

        # Weighted sum of features using attention scores
        context_features = torch.sum(context_features@context_atten.transpose(1, 2), dim=2, keepdim=True)
        body_features = torch.sum(body_features @ body_atten.transpose(1, 2) , dim=2, keepdim=True)
        face_features = torch.sum(face_features @ face_atten.transpose(1, 2) , dim=2, keepdim=True)


        ########################## Self Attention ##################################

        context_features = context_features.view(-1, self.num_features)
        body_features = body_features.view(-1, self.num_features)
        face_features = face_features.view(-1, self.num_features)



        score_context = self.fc_att(context_features)
        score_body = self.fc_att(body_features)
        score_face = self.fc_att(face_features)

        score = torch.cat((score_context, score_body, score_face), 1) # shape socre (B, 1, 3)

        score = self.softmax(score) # shape score (B, 1, 3)

        score = score.view(-1, score.shape[1], 1) # shape (B, 3)

        context_features = context_features*score[:,0]
        body_features = body_features*score[:,1]
        face_features = face_features*score[:,2]

        # Concatenate features
        if self.conbine == 'concat':
          fuse_features = torch.cat((context_features, body_features, face_features), 1) # shape(B, 3*256)
          fuse_out = self.fc1(fuse_features)
        elif self.conbine == 'sum':
          fuse_features = torch.sum(torch.stack((context_features, body_features, face_features), dim = 1), dim = 1) # shape(B, 768)
          fuse_out = self.fc2(fuse_features)
        elif self.conbine == 'avg':
          fuse_features = torch.mean(torch.stack((context_features, body_features, face_features), dim = 1), dim = 1) # shape(B, 768)
          fuse_out = self.fc2(fuse_features)

        # Feed-forward through the rest of the network

        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)


        return cat_out

import torch
import torch.nn as nn

import torch
import torch.nn as nn

# Placeholder DensityBlock implementation


class FusionConcatModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features):
        super(FusionConcatModel, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_text_features = num_text_features

        # DensityBlock configuration
        norm_axes = [1, 1, 1]
        num_heads = [4, 4, 4]
        num_densities = [5, 5, 5]
        num_layers = 3
        padding_value = None
        eps = 1e-8

        # Linear layers for feature transformation
        self.fc_context = nn.Linear(num_context_features, 256)
        self.fc_body = nn.Linear(num_body_features, 256)
        self.fc_face = nn.Linear(num_face_features, 256)
        self.fc_text = nn.Linear(num_text_features, 256)
    
        # DensityBlock for text features and fused features
        self.attention_text = DensityBlock(norm_axes, num_heads, num_densities, num_layers, padding_value, eps)
        self.attention_fuse = DensityBlock(norm_axes, num_heads, num_densities, num_layers, padding_value, eps)

        # Output layers
        self.fc2 = nn.Linear(256, 26)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face, x_text):
        # Shape: (batch_size, num_context_features), (batch_size, num_body_features), (batch_size, num_face_features)
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)
        text_features = x_text.view(-1, self.num_text_features)

        # Transform features to 256 dimensions
        context_features = self.fc_context(context_features)
        body_features = self.fc_body(body_features)
        face_features = self.fc_face(face_features)
        text_features = self.fc_text(text_features)

        # Concatenate features
        fuse_features = torch.cat((context_features, body_features, face_features, text_features), 1)
        # Apply DensityBlock to fused features
        fuse_out = self.attention_text(fuse_features)



        # Feed-forward through the rest of the network
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc2(fuse_out)

        return cat_out









#######################Cross Attention Model######################

class FusionAttentionModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features):
        super(FusionAttentionModel, self).__init__()
        feature_dim = 256
        self.fc_context = nn.Linear(num_context_features, feature_dim)
        self.fc_body = nn.Linear(num_body_features, feature_dim)
        self.fc_face = nn.Linear(num_face_features, feature_dim)
        self.fc_text = nn.Linear(num_text_features, feature_dim)

        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 26)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.attention_layer = nn.MultiheadAttention(feature_dim, 8)  # Số heads là 8 thay vì 64 cho hợp lý

        self.gate_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def attention_self_cross(self, features):
        """
        features: (batch_size, 4, feature_dim) - Tập hợp đặc trưng từ các nguồn khác nhau
        """
        batch_size, num_modalities, feature_dim = features.shape
        
        # Chuyển về dạng (num_modalities, batch_size, feature_dim) để phù hợp với MultiheadAttention
        features = features.permute(1, 0, 2)  # (num_modalities, batch_size, feature_dim)
        
        # Self-Attention với Residual Connection
        self_outputs = []
        for i in range(num_modalities):
            query = key = value = features[i].unsqueeze(0)  # (1, batch_size, feature_dim)
            attended_output, _ = self.attention_layer(query, key, value)  # (1, batch_size, feature_dim)
            # Thêm residual connection: cộng đầu ra với query
            attended_output = attended_output + query  # Residual connection
            self_outputs.append(attended_output.squeeze(0))  # (batch_size, feature_dim)
        
        features_self = torch.stack(self_outputs, dim=1)  # (batch_size, num_modalities, feature_dim)

        # Cross-Attention với Residual Connection
        cross_outputs = []
        for i in range(num_modalities):
            query = features[i].unsqueeze(0)  # (1, batch_size, feature_dim)
            for j in range(num_modalities):
                if j != i:
                    key = value = features[num_modalities - j - 1].unsqueeze(0)  # Đảo ngược key-value
                    attended_output, _ = self.attention_layer(query, key, value)  # (1, batch_size, feature_dim)
                    # Thêm residual connection: cộng đầu ra với query
                    attended_output = attended_output + query  # Residual connection
                    cross_outputs.append(attended_output.squeeze(0))  # (batch_size, feature_dim)
        
        features_cross = torch.stack(cross_outputs, dim=1)  # (batch_size, num_modalities * (num_modalities-1), feature_dim)

        return features_self, features_cross

    def forward(self, x_context, x_body, x_face, x_text):
        # Chuyển đổi về feature_dim
        context_features = self.fc_context(x_context)
        body_features = self.fc_body(x_body)
        face_features = self.fc_face(x_face)
        text_features = self.fc_text(x_text)

        # Ghép nối thành tensor (batch_size, 4, 256)
        features = torch.stack([context_features, body_features, face_features, text_features], dim=1)

        # Self-Attention & Cross-Attention
        features_self, features_cross = self.attention_self_cross(features)

        # Chuyển danh sách thành Tensor
        features_self = torch.mean(features_self, dim=1)  # (batch_size, feature_dim)
        features_cross = torch.mean(features_cross, dim=1)  # (batch_size, feature_dim)

        # Cross-Gating Mechanism
        gate = self.gate_layer(features_self)  # (batch_size, 1)
        features_combined = gate * features_self + (1 - gate) * features_cross  # (batch_size, feature_dim)

        # Fully Connected Layers
        fuse_out = self.fc1(features_combined)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc2(fuse_out)

        return cat_out


#############Trasformer Fusion Model####################


class TransformerFusionModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features, 
                 feature_dim=256, num_heads=4, num_layers=16):
        super(TransformerFusionModel, self).__init__()
        self.feature_dim = feature_dim

        # Chiếu đặc trưng về cùng không gian
        self.fc_context = nn.Linear(num_context_features, feature_dim)
        self.fc_body = nn.Linear(num_body_features, feature_dim)
        self.fc_face = nn.Linear(num_face_features, feature_dim)
        self.fc_text = nn.Linear(num_text_features, feature_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim * 4,  # FFN size
            dropout=0.5,
            activation='GELU'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional Encoding (tùy chọn, để đánh dấu thứ tự các đặc trưng)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4, feature_dim))  # 4 là số đặc trưng

        # Tổng hợp và đầu ra
        self.fc1 = nn.Linear(feature_dim * 4, 256)  # Có thể thay bằng feature_dim nếu dùng mean
        self.fc2 = nn.Linear(256, 26)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face, x_text):
        # Chiếu đặc trưng
        context = self.fc_context(x_context.view(-1, x_context.size(-1)))  # (batch_size, feature_dim)
        body = self.fc_body(x_body.view(-1, x_body.size(-1)))             # (batch_size, feature_dim)
        face = self.fc_face(x_face.view(-1, x_face.size(-1)))             # (batch_size, feature_dim)
        text = self.fc_text(x_text.view(-1, x_text.size(-1)))             # (batch_size, feature_dim)

        # Gộp thành chuỗi đặc trưng
        features = torch.stack([context, body, face, text], dim=1)  # (batch_size, 4, feature_dim)

        # Thêm positional encoding
        features = features + self.pos_encoding  # (batch_size, 4, feature_dim)

        # Transformer yêu cầu input dạng (seq_len, batch_size, feature_dim)
        features = features.permute(1, 0, 2)  # (4, batch_size, feature_dim)
        transformer_out = self.transformer(features)  # (4, batch_size, feature_dim)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, 4, feature_dim)

        # Tổng hợp đặc trưng
        # Cách 1: Concatenate (như trước)
        fuse_features = transformer_out.reshape(-1, self.feature_dim * 4)  # (batch_size, 256 * 4)

        # Cách 2: Lấy trung bình (commented, bạn có thể thử)
        # fuse_features = transformer_out.mean(dim=1)  # (batch_size, feature_dim)

        # Đầu ra
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc2(fuse_out)
        return cat_out






class FusionAttnModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features):
        super(FusionAttnModel, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_text_features = num_text_features

        self.fc_context = nn.Linear(num_context_features, 256)
        self.fc_body = nn.Linear(num_body_features, 256)
        self.fc_face = nn.Linear(num_face_features, 256)
        self.fc_text = nn.Linear(num_text_features, 256)

        # Linear layer to compute attention weights
        self.attention_weights = nn.Sequential(nn.Linear(256 * 4, 512),
                                               nn.BatchNorm1d(512),
                                               nn.ReLU(),
                                               nn.Linear(512, 4))
        self.fc1 = nn.Linear(256 * 4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 26)

    def forward(self, x_context, x_body, x_face, x_text):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)
        text_features = x_text.view(-1, self.num_text_features)

        context_features = self.fc_context(context_features)
        body_features = self.fc_body(body_features)
        face_features = self.fc_face(face_features)
        text_features = self.fc_text(text_features)

        # Concatenate features to compute attention weights
        fuse_features = torch.cat((context_features, body_features, face_features, text_features), 1)
        attention_scores = self.attention_weights(fuse_features)  # (batch_size, 4)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, 4)

        # Stack the feature vectors
        feature_tensor = torch.stack([context_features, body_features, face_features, text_features], dim=1)  # (batch_size, 4, 256)

        # Compute attended features
        # attended_features = (feature_tensor * attention_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, 256)
        attended_features = (feature_tensor * attention_weights.unsqueeze(-1)).view(-1, 256 * 4)
        # Pass through the rest of the network
        out = self.fc1(attended_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.d1(out)
        cat_out = self.fc2(out)

        return cat_out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = head_dim * num_heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q, k, v):
        # q, k, v shapes: [batch_size, seq_len, dim]
        batch_size, q_len, _ = q.shape
        _, k_len, _ = k.shape
        
        # Project to multi-head queries, keys, values
        q = self.to_q(q).reshape(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(k).reshape(batch_size, k_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(v).reshape(batch_size, k_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights
        out = torch.matmul(attn, v)
        
        # Reshape and project back
        out = out.permute(0, 2, 1, 3).reshape(batch_size, q_len, -1)
        return self.to_out(out)

class AttentionBlock(nn.Module):
    def __init__(self, feature_dim, prompt_dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        
        # Self-attention components
        self.self_attention = MultiHeadAttention(feature_dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        
        # Cross-attention components
        self.cross_attention = MultiHeadAttention(feature_dim, num_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(feature_dim)
        
        # Projection for prompt embedding if dimensions don't match
        self.prompt_proj = None
        if prompt_dim != feature_dim:
            self.prompt_proj = nn.Linear(prompt_dim, feature_dim)
    
    def forward(self, features, prompt_embedding):
        # features shape: [batch_size, seq_len_img, feature_dim]
        # prompt_embedding shape: [batch_size, seq_len_prompt, prompt_dim]
        
        # Project prompt embeddings if needed
        if self.prompt_proj is not None:
            prompt_embedding = self.prompt_proj(prompt_embedding)
        
        # Self-attention: Image features attending to themselves
        residual = features
        features = self.norm1(features)
        features = residual + self.self_attention(features, features, features)
        
        # Cross-attention: Image features attending to text
        residual = features
        features = self.norm2(features)
        features = residual + self.cross_attention(features, prompt_embedding, prompt_embedding)
        
        # Feed-forward network
        residual = features
        features = self.norm3(features)
        features = residual + self.ffn(features)
        
        return features

class DualPathAttentionFusion(nn.Module):
    def __init__(
        self, 
        num_context_features, 
        num_body_features, 
        num_face_features, 
        num_text_features,
        hidden_dim=256,
        num_heads=8,
        head_dim=32,
        dropout=0.1,
        num_layers=2
    ):
        super().__init__()
        
        # Feature projection layers
        self.context_proj = nn.Linear(num_context_features, hidden_dim)
        self.body_proj = nn.Linear(num_body_features, hidden_dim)
        self.face_proj = nn.Linear(num_face_features, hidden_dim)
        self.text_proj = nn.Linear(num_text_features, hidden_dim)
        
        # Context as query to other modalities
        self.context_attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, hidden_dim, num_heads, head_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 26)
        )
        
    def forward(self, x_context, x_body, x_face, x_text):
        # Project features to common dimension
        context_features = self.context_proj(x_context)
        body_features = self.body_proj(x_body)
        face_features = self.face_proj(x_face)
        text_features = self.text_proj(x_text)
        
        # Add sequence dimension if not present
        if context_features.dim() == 2:
            context_features = context_features.unsqueeze(1)
        if body_features.dim() == 2:
            body_features = body_features.unsqueeze(1)
        if face_features.dim() == 2:
            face_features = face_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # Apply attention layers - context attends to other modalities
        enhanced_context_body = context_features
        enhanced_context_face = context_features
        enhanced_context_text = context_features
        
        for layer in self.context_attention_layers:
            enhanced_context_body = layer(enhanced_context_body, body_features)
            enhanced_context_face = layer(enhanced_context_face, face_features)
            enhanced_context_text = layer(enhanced_context_text, text_features)
        
        # Squeeze back if necessary (batch_size, hidden_dim)
        enhanced_context_body = enhanced_context_body.squeeze(1)
        enhanced_context_face = enhanced_context_face.squeeze(1)
        enhanced_context_text = enhanced_context_text.squeeze(1)
        context_features = context_features.squeeze(1)
        
        # Concatenate enhanced contexts with original context
        fused_features = torch.cat([
            context_features,
            enhanced_context_body,
            enhanced_context_face,
            enhanced_context_text
        ], dim=1)
        
        # Output classification
        output = self.fusion(fused_features)
        
        return output
    



# class QFormer(nn.Module):
#     def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features, embed_dim=256, num_heads=8, num_layers=6, num_queries=32):
#         super().__init__()


#         self.num_context_features = num_context_features
#         self.num_text_features = num_text_features

#         self.fc_context = nn.Linear(num_context_features, 256)
#         self.fc_text = nn.Linear(num_text_features, 256)
        
#         # Query Tokens (Học thông tin từ ảnh và văn bản)
#         self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
#         # Transformer Encoder (Giống BERT nhưng có Cross-Attention)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Cross-Attention giữa Query Tokens & Ảnh
#         self.image_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
#         # Cross-Attention giữa Query Tokens & Văn bản
#         self.text_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
#         # Feedforward Layer
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 26)
#         )

#     def forward(self, image_features, x_body, x_face, text_embeddings):
#         """
#         image_features: (batch_size, num_patches, embed_dim)  # Đặc trưng ảnh từ ViT
#         text_embeddings: (batch_size, num_tokens, embed_dim)  # Đặc trưng văn bản từ BERT
#         """


#         batch_size = image_features.shape[0]
        
#         # Query Tokens (Nhân bản cho mỗi batch)
#         queries = self.query_tokens.expand(batch_size, -1, -1)
        
#         # Self-Attention giữa các Query Tokens
#         queries = self.self_attn(queries)
        
#         # Cross-Attention giữa Query Tokens và Ảnh
#         queries, _ = self.image_cross_attn(queries, image_features, image_features)
        
#         # Cross-Attention giữa Query Tokens và Văn bản
#         queries, _ = self.text_cross_attn(queries, text_embeddings, text_embeddings)
        
#         # MLP để xử lý đầu ra
#         output = self.mlp(queries)
        
#         return output  # (batch_size, num_queries, embed_dim)



class QFormer(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features, embed_dim=768, num_heads=8, num_layers=6, num_queries=32):
        super().__init__()


        self.qformer = Qformer()
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(768, 26)

    def forward(self, x_context, x_body, x_face, text_features):
        """
        image_features: (batch_size, num_patches, embed_dim)  # Đặc trưng ảnh từ ViT
        text_embeddings: (batch_size, num_tokens, embed_dim)  # Đặc trưng văn bản từ BERT
        """
        x_body = x_body.reshape(-1, 7*7, 768)

        combined_visual = torch.cat([x_context, x_body], dim=1) # shape (batch, path_size, 768)
        emotion_logits, pooled_features, loss_NCE = self.qformer(
            combined_visual, 
            text_features, # shape (batch, seq_len, 768)
        )
        print(emotion_logits.shape)
        # x_context = self.flatten(x_context) 
        # x_body = self.flatten(x_body)


        
        # loss_NCE.backward(retain_graph=True)
        
        emotion_logits = self.fc(emotion_logits)

        return emotion_logits, loss_NCE