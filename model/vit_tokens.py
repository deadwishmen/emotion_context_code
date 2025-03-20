import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer, Encoder

class ViTForVisualTokens(nn.Module):
    def __init__(self, pretrained=True, model_name='vit_b_16'):
        """
        Khởi tạo Vision Transformer để trích xuất visual tokens
        
        Args:
            pretrained (bool): Sử dụng pretrained weights từ ImageNet nếu True
            model_name (str): Tên mô hình ViT ('vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14')
        """
        super().__init__()
        
        # Lấy mô hình ViT từ torchvision
        if model_name == 'vit_b_16':
            base_model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            base_model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            base_model = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_l_32':
            base_model = models.vit_l_32(pretrained=pretrained)
        elif model_name == 'vit_h_14':
            base_model = models.vit_h_14(pretrained=pretrained)
        else:
            raise ValueError(f"Không hỗ trợ model_name: {model_name}")
        
        # Lấy các thành phần của mô hình
        self.patch_size = base_model.patch_size
        self.image_size = base_model.image_size
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        self.encoder = base_model.encoder
        self.pos_embedding = base_model.pos_embedding
        
        # Bỏ đi lớp cuối (head) vì chúng ta chỉ quan tâm tới các token
        self.heads = None  # Không cần lớp phân loại
    
    def forward(self, x):
        """
        Forward pass qua mô hình để trích xuất visual tokens
        
        Args:
            x (torch.Tensor): Batch ảnh đầu vào [B, C, H, W]
            
        Returns:
            torch.Tensor: Visual tokens [B, N+1, D] (bao gồm CLS token)
        """
        # Chuyển ảnh thành patch embeddings
        batch_size = x.shape[0]
        x = self.conv_proj(x)
        
        # Reshape lại kích thước patches
        x = x.flatten(2).transpose(1, 2)
        
        # Thêm class token
        cls_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Thêm positional embeddings
        x = x + self.pos_embedding
        
        # Đưa qua encoder transformer
        x = self.encoder(x)
        
        # Trả về tất cả tokens (bao gồm cả class token)
        return x

def get_vit_for_tokens(model_name='vit_b_16', pretrained=True):
    """
    Tạo mô hình ViT từ torchvision để trích xuất visual tokens
    
    Args:
        model_name (str): Tên mô hình ViT ('vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14')
        pretrained (bool): Sử dụng pretrained weights từ ImageNet nếu True
        
    Returns:
        ViTForVisualTokens: Mô hình ViT được sửa đổi để trả về visual tokens
    """
    model = ViTForVisualTokens(pretrained=pretrained, model_name=model_name)
    return model

# Ví dụ sử dụng
def extract_visual_tokens(images, model_name='vit_b_16', pretrained=True):
    """
    Trích xuất visual tokens từ batch ảnh
    
    Args:
        images (torch.Tensor): Batch ảnh [B, 3, H, W], đã được chuẩn hóa
        model_name (str): Tên mô hình ViT
        pretrained (bool): Sử dụng pretrained weights
        
    Returns:
        tokens (torch.Tensor): Visual tokens [B, num_patches+1, embedding_dim]
    """
    device = images.device
    
    # Khởi tạo mô hình
    model = get_vit_for_tokens(model_name, pretrained).to(device)
    model.eval()
    
    # Trích xuất tokens
    with torch.no_grad():
        tokens = model(images)
    
    return tokens

# Ví dụ chi tiết về việc sử dụng
def example_usage():
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    # Tạo transformer để tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Đọc ảnh mẫu
    image = Image.open("sample_image.jpg").convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch
    
    # Trích xuất tokens
    model = get_vit_for_tokens('vit_b_16', pretrained=True)
    model.eval()
    
    with torch.no_grad():
        tokens = model(image_tensor)
    
    # Shape của tokens: [batch_size, num_patches + 1, embedding_dim]
    # tokens[:, 0] là CLS token
    # tokens[:, 1:] là các patch tokens
    print(f"Token shape: {tokens.shape}")
    
    # Làm việc với tokens
    cls_token = tokens[:, 0]  # Token đầu tiên là CLS token
    patch_tokens = tokens[:, 1:]  # Các token còn lại là patch tokens
    
    # Ví dụ: Tính cosine similarity giữa các patch token và CLS token
    from torch.nn.functional import cosine_similarity
    token_similarities = cosine_similarity(
        patch_tokens.view(-1, patch_tokens.size(-1)), 
        cls_token.view(-1, cls_token.size(-1))
    )
    
    # Tạo heatmap từ các giá trị similarity
    similarity_map = token_similarities.view(
        1, 
        int(np.sqrt(patch_tokens.size(1))), 
        int(np.sqrt(patch_tokens.size(1)))
    )
    print(f"Similarity map shape: {similarity_map.shape}")
    
    return tokens, similarity_map