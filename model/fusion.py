import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import CrossAttention





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

class FusionConcatModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features):
        super(FusionConcatModel, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_text_features = num_text_features

        self.fc_context = nn.Linear(num_context_features, 256)
        self.fc_body = nn.Linear(num_body_features, 256)
        self.fc_face = nn.Linear(num_face_features, 256)
        self.fc_text = nn.Linear(num_text_features, 256)

        self.fc1 = nn.Linear(256*4, 256)
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

        context_features = self.fc_context(context_features)
        body_features = self.fc_body(body_features)
        face_features = self.fc_face(face_features)
        text_features = self.fc_text(text_features)

        # Concatenate features
        fuse_features = torch.cat((context_features, body_features, face_features, text_features), 1)
        fuse_out = self.fc1(fuse_features)

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




class AdaptiveAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(AdaptiveAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # Mạng phụ để tính trọng số thích nghi
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),  # Đầu vào là tất cả đặc trưng gộp lại
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # Đầu ra là trọng số cho 4 đặc trưng
            nn.Sigmoid()  # Trọng số trong khoảng [0, 1]
        )

    def forward(self, context, body, face, text):
        # context, body, face, text: (batch_size, feature_dim)
        batch_size = context.size(0)
        
        # Gộp tất cả đặc trưng để đưa vào gating network
        all_features = torch.cat([context, body, face, text], dim=1)  # (batch_size, feature_dim * 4)
        
        # Tính trọng số thích nghi
        weights = self.gating_network(all_features)  # (batch_size, 4)
        weights = weights.unsqueeze(2)  # (batch_size, 4, 1) - để nhân với đặc trưng
        
        # Chuẩn bị đặc trưng để nhân với trọng số
        features = torch.stack([context, body, face, text], dim=1)  # (batch_size, 4, feature_dim)
        
        # Áp dụng trọng số thích nghi
        weighted_features = features * weights  # (batch_size, 4, feature_dim)
        
        # Tổng hợp bằng cách lấy trung bình hoặc cộng (ở đây dùng trung bình)
        fused_features = weighted_features.view(-1, self.feature_dim*4) # (batch_size, feature_dim * 4)
        return fused_features

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = self.head_dim ** -0.5
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim).permute(0, 1, 2)
        
        energy = torch.bmm(Q, K.transpose(-1, -2)) * self.scale
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(attention, V).permute(0, 2, 1).contiguous().view(batch_size, self.feature_dim)
        return self.fc_out(out)

class AdaptiveFusionModelWithSelfAttention(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features, 
                 feature_dim=256):
        super(AdaptiveFusionModelWithSelfAttention, self).__init__()
        self.feature_dim = feature_dim

        # Chiếu đặc trưng
        self.fc_context = nn.Linear(num_context_features, feature_dim)
        self.fc_body = nn.Linear(num_body_features, feature_dim)
        self.fc_face = nn.Linear(num_face_features, feature_dim)
        self.fc_text = nn.Linear(num_text_features, feature_dim)

        # Self-Attention cho từng đặc trưng
        self.self_att = SelfAttention(feature_dim=feature_dim)

        # Adaptive Attention
        self.adaptive_att = AdaptiveAttention(feature_dim=feature_dim)

        # Tầng đầu ra
        self.fc1 = nn.Linear(feature_dim*4, 256)
        self.fc2 = nn.Linear(256, 26)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face, x_text):
        # Chiếu đặc trưng
        context = self.fc_context(x_context.view(-1, x_context.size(-1)))
        body = self.fc_body(x_body.view(-1, x_body.size(-1)))
        face = self.fc_face(x_face.view(-1, x_face.size(-1)))
        text = self.fc_text(x_text.view(-1, x_text.size(-1)))

        # Áp dụng Self-Attention
        context = self.self_att(context)
        body = self.self_att(body)
        face = self.self_att(face)
        text = self.self_att(text)

        # Áp dụng Adaptive Attention
        fused_features = self.adaptive_att(context, body, face, text)

        # Đầu ra
        fuse_out = self.fc1(fused_features)
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
        self.d1 = nn.dropout(p=0.5)
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