import torch
import torch.nn as nn
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
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features, conbine = 'concat', isSwinT = True):
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

class FusionFullCrossAttentionModel(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_text_features):
        super(FusionFullCrossAttentionModel, self).__init__()
        self.fc_context = nn.Linear(num_context_features, 256)
        self.fc_body = nn.Linear(num_body_features, 256)
        self.fc_face = nn.Linear(num_face_features, 256)
        self.fc_text = nn.Linear(num_text_features, 256)

        # Cross-Attention cho tất cả đặc trưng
        self.cross_attention = CrossAttention(feature_dim=256)

        self.fc1 = nn.Linear(256 * 4, 256)  # Tổng hợp 4 đặc trưng
        self.fc2 = nn.Linear(256, 26)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face, x_text):
        context_features = self.fc_context(x_context.view(-1, x_context.size(-1)))
        body_features = self.fc_body(x_body.view(-1, x_body.size(-1)))
        face_features = self.fc_face(x_face.view(-1, x_face.size(-1)))
        text_features = self.fc_text(x_text.view(-1, x_text.size(-1)))

        # Gộp tất cả đặc trưng thành một tensor
        all_features = torch.stack([context_features, body_features, face_features, text_features], dim=1)  # (batch_size, 4, 256)

        # Áp dụng Cross-Attention giữa tất cả
        batch_size = all_features.size(0)
        fused_features = []
        for i in range(4):
            source = all_features[:, i, :]  # (batch_size, 256)
            target = all_features.mean(dim=1)  # (batch_size, 256) - trung bình tất cả đặc trưng làm target
            out = self.cross_attention(source, target)
            fused_features.append(out)
        
        # Tổng hợp
        fuse_features = torch.cat(fused_features, dim=1)  # (batch_size, 256*4)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc2(fuse_out)
        return cat_out
