import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
import torchvision.models as models
import torchvision.transforms as T # Use T for transforms alias
import logging
import math
from torch.nn import Parameter
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Hàm Helper ---

def load_hf_resources(model_name: str = "openai/clip-vit-base-patch32", cache_dir: Optional[str] = None) -> Tuple[AutoProcessor, AutoTokenizer]:
    """Tải processor và tokenizer từ Hugging Face."""
    try:
        # Use legacy=False for new PIL behavior if needed, but default should be fine
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        logging.info(f"Loaded processor and tokenizer for {model_name} successfully.")
        return processor, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Hugging Face resources ({model_name}): {e}")
        raise

def load_resnet50_places365(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Tải mô hình ResNet-50 đã pretrain trên Places365 và loại bỏ lớp cuối cùng."""
    try:
        model = models.resnet50(weights=None) # Load architecture without pretrained weights initially
        model.fc = nn.Linear(model.fc.in_features, 365) # Adjust final layer for Places365
        
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"ResNet checkpoint not found at: {checkpoint_path}")

        logging.info(f"Loading ResNet checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if saved with DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif isinstance(checkpoint, dict): # Check if it's a state_dict directly
             state_dict = checkpoint
        else:
             # Assume it's the model itself (less common for training checkpoints)
             model = checkpoint
             logging.warning("Loaded checkpoint directly as model. Assuming it contains the full model structure.")
             state_dict = None # Already loaded

        if state_dict is not None:
             # Adjust state_dict keys if necessary (e.g., if fc layer name differs)
             # Example: If saved model fc is named 'classifier', adapt it
             # state_dict = {k.replace('classifier', 'fc') if 'classifier' in k else k: v for k,v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False) # Use strict=False initially for flexibility

        model = nn.Sequential(*list(model.children())[:-1]) # Remove the final classification layer
        model.eval()
        logging.info("Loaded ResNet-50 Places365 successfully and removed final layer.")
        return model.to(device)
    except Exception as e:
        logging.error(f"Failed to load ResNet-50 Places365 from {checkpoint_path}: {e}")
        raise

# --- Biến đổi Ảnh ---
# Define transforms directly, avoid global scope for clarity if possible
def get_resnet_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Lớp Loss ---
class BCEWithLogitsLoss(nn.Module):
    """BCE Loss với các tùy chọn trọng số (mean, static, dynamic)."""
    def __init__(self, weight_type: str = 'mean', device: torch.device = torch.device('cpu'), num_classes: int = 26):
        super().__init__()
        self.weight_type = weight_type
        self.device = device
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.weights: Optional[torch.Tensor] = None

        if self.weight_type == 'mean':
            self.weights = torch.ones((1, self.num_classes), device=self.device) / self.num_classes
        elif self.weight_type == 'static':
            # Nguồn gốc/lý do của các trọng số này nên được ghi chú nếu có thể
            static_weights = torch.FloatTensor([
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ]).unsqueeze(0)
            assert static_weights.shape[1] == self.num_classes, f"Static weights must have {self.num_classes} elements."
            self.weights = static_weights.to(self.device)
        elif self.weight_type != 'dynamic':
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        current_weights = self.weights
        if self.weight_type == 'dynamic':
            current_weights = self._prepare_dynamic_weights(target).to(self.device)
        
        loss = self.bce_loss(pred, target)
        if current_weights is not None:
           weighted_loss = loss * current_weights
        else: # Should not happen if weight_type is valid
           weighted_loss = loss
           
        return weighted_loss.mean() # Mean over all elements (batch and classes)

    def _prepare_dynamic_weights(self, target: torch.Tensor) -> torch.Tensor:
        target_stats = torch.sum(target, dim=0, dtype=torch.float32).unsqueeze(dim=0) # Shape: [1, num_classes]
        # Avoid division by zero or log(1) = 0. Use a small epsilon for stability.
        weights = torch.zeros_like(target_stats)
        positive_counts = target_stats > 0
        # Clamp counts to avoid log(1.2) which is small but non-zero.
        # Log argument must be > 1 for inverse relation. Add 1.2 seems okay.
        weights[positive_counts] = 1.0 / torch.log(target_stats[positive_counts] + 1.2)
        # Assign a very small weight (or a default large weight) for classes with no positive samples
        weights[~positive_counts] = 1.0 / torch.log(torch.tensor(1.2)) # Or another suitable default, e.g. 0.0001 if that worked
        
        # Normalize weights (optional, but can help control magnitude)
        # weights = weights / weights.sum() * self.num_classes 

        return weights # Shape: [1, num_classes]

# --- Tokenization ---
def tokenize_text(path_dataset: str, tokenizer: AutoTokenizer, max_length: int = 77) -> List[Dict[str, torch.Tensor]]:
    """Tokenize cột 'Output' từ file CSV."""
    try:
        if not os.path.exists(path_dataset):
             raise FileNotFoundError(f"CSV file not found: {path_dataset}")
        df = pd.read_csv(path_dataset)
        if 'Output' not in df.columns:
            raise ValueError(f"Column 'Output' not found in {path_dataset}")

        sentences = df['Output'].fillna("").tolist() # Handle potential NaNs
        logging.info(f"Tokenizing {len(sentences)} sentences from {path_dataset}...")
        # Batch tokenization is faster
        tokens = tokenizer(sentences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        # Convert to list of dicts for dataset
        tokenized_list = [{key: tokens[key][i] for key in tokens.keys()} for i in range(len(sentences))]
        logging.info(f"Tokenization complete for {path_dataset}.")
        return tokenized_list
    except Exception as e:
        logging.error(f"Error tokenizing text from {path_dataset}: {e}")
        raise

# --- Dataset ---
class EmoticDataset(Dataset):
    """Dataset cho EMOTIC, trả về ảnh thô và text tokens."""
    def __init__(self, context_path: str, body_path: str, cat_path: str, text_tokens: List[Dict[str, torch.Tensor]]):
        try:
            self.context_arr = np.load(context_path)
            self.body_arr = np.load(body_path)
            self.cat_arr = np.load(cat_path)
            self.text_tokens = text_tokens

            # Basic validation
            assert len(self.context_arr) == len(self.body_arr) == len(self.cat_arr) == len(self.text_tokens), \
                "Mismatch in data array lengths!"
            logging.info(f"Loaded EmoticDataset: {len(self)} samples.")
            logging.info(f"Sample context shape: {self.context_arr[0].shape}, dtype: {self.context_arr[0].dtype}")
            logging.info(f"Sample body shape: {self.body_arr[0].shape}, dtype: {self.body_arr[0].dtype}")

        except FileNotFoundError as e:
            logging.error(f"Error loading dataset arrays: {e}")
            raise
        except Exception as e:
            logging.error(f"Error initializing EmoticDataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.cat_arr)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor], np.ndarray]:
        """Trả về context (ảnh thô), body (ảnh thô), text tokens, và nhãn."""
        try:
            # Trả về dữ liệu NumPy thô, việc chuyển đổi sẽ diễn ra trong collate_fn
            context_img = self.context_arr[index]
            body_img = self.body_arr[index]
            
            # Đảm bảo ảnh có 3 kênh (ví dụ, loại bỏ kênh alpha nếu có)
            if context_img.shape[-1] == 4:
                context_img = context_img[..., :3]
            if body_img.shape[-1] == 4:
                body_img = body_img[..., :3]
                
            # Không cần transpose ở đây nếu đã chuẩn hóa định dạng [H, W, C] khi lưu .npy
            # If format is [C, H, W], transpose it here or in collate_fn
            # if context_img.shape[0] == 3:
            #     context_img = context_img.transpose(1, 2, 0)
            # if body_img.shape[0] == 3:
            #      body_img = body_img.transpose(1, 2, 0)

            token_text = self.text_tokens[index]
            cat_label = self.cat_arr[index]

            return context_img, body_img, token_text, cat_label

        except Exception as e:
            # Log lỗi nhưng trả về None để collate_fn có thể bỏ qua
            logging.error(f"Error processing dataset item {index}: {e}", exc_info=True)
            # Returning None might be complex with batching. A placeholder might be better,
            # but for now, let's re-raise to see the error during debugging.
            # For production, consider returning dummy data or using a flag.
            raise # Or return None, None, None, None

# --- Collate Function ---
def create_collate_fn(processor: AutoProcessor, resnet_transform: T.Compose, tokenizer: AutoTokenizer, device: torch.device) -> callable:
    """Tạo hàm collate_fn để xử lý batch dữ liệu."""
    def collate_fn(batch: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Optional[np.ndarray]]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        # Lọc các mẫu bị lỗi (None) từ __getitem__ nếu có
        # batch = [item for item in batch if item[0] is not None] # Simple filtering
        # If __getitem__ raises error instead of returning None, this isn't needed

        if not batch:
            return None # Return None if the entire batch is invalid

        try:
            context_imgs_np, body_imgs_np, text_tokens_list, cat_labels_np = zip(*batch)

            # Chuyển đổi ảnh NumPy arrays sang PIL Images hoặc tensors nếu cần
            # processor expects PIL Images or PyTorch tensors
            # Assume input numpy arrays are HWC, uint8 [0-255]
            context_pil = [Image.fromarray(img) for img in context_imgs_np]
            body_pil = [Image.fromarray(img) for img in body_imgs_np]

            # Áp dụng CLIP processor cho ảnh
            processed_context = processor(images=context_pil, return_tensors="pt", padding=True).to(device)
            processed_body = processor(images=body_pil, return_tensors="pt", padding=True).to(device)
            context_pixel_values = processed_context['pixel_values']
            body_pixel_values = processed_body['pixel_values']

            # Áp dụng ResNet transform cho context images (chuyển sang tensor trước)
            context_resnet_tensors = torch.stack([resnet_transform(img) for img in context_pil]).to(device)

            # Xử lý text tokens (đã là tensors, chỉ cần stack và chuyển device)
            # Pad attention masks explicitly if needed, although tokenizer usually handles it
            input_ids = torch.stack([t['input_ids'].squeeze(0) for t in text_tokens_list]).to(device) # Remove potential extra dim
            attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in text_tokens_list]).to(device)
            processed_tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}

            # Chuyển đổi nhãn sang tensor
            cat_labels = torch.tensor(np.array(cat_labels_np), dtype=torch.float32).to(device)

            return context_pixel_values, body_pixel_values, context_resnet_tensors, processed_tokens, cat_labels
        except Exception as e:
            logging.error(f"Error in collate_fn: {e}", exc_info=True)
            return None # Skip batch on error

    return collate_fn

# --- Tải dữ liệu ---
def load_data(data_src: str, batch_size: int, device: torch.device, processor: AutoProcessor, tokenizer: AutoTokenizer, resnet_transform: T.Compose, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tải dữ liệu và tạo DataLoaders."""
    try:
        required_suffixes = ['_context_arr.npy', '_body_arr.npy', '_cat_arr.npy', '.csv']
        sets = ['train', 'val', 'test']
        for s in sets:
            for suffix in required_suffixes:
                 fpath = os.path.join(data_src, f"{s}{suffix}")
                 if not os.path.exists(fpath):
                     raise FileNotFoundError(f"File {os.path.basename(fpath)} not found in {data_src}")

        logging.info("Tokenizing text data...")
        train_text = tokenize_text(os.path.join(data_src, 'train.csv'), tokenizer)
        val_text = tokenize_text(os.path.join(data_src, 'val.csv'), tokenizer)
        test_text = tokenize_text(os.path.join(data_src, 'test.csv'), tokenizer)

        logging.info("Creating datasets...")
        train_dataset = EmoticDataset(
            os.path.join(data_src, 'train_context_arr.npy'), os.path.join(data_src, 'train_body_arr.npy'),
            os.path.join(data_src, 'train_cat_arr.npy'), train_text
        )
        val_dataset = EmoticDataset(
            os.path.join(data_src, 'val_context_arr.npy'), os.path.join(data_src, 'val_body_arr.npy'),
            os.path.join(data_src, 'val_cat_arr.npy'), val_text
        )
        test_dataset = EmoticDataset(
            os.path.join(data_src, 'test_context_arr.npy'), os.path.join(data_src, 'test_body_arr.npy'),
            os.path.join(data_src, 'test_cat_arr.npy'), test_text
        )

        logging.info("Creating dataloaders...")
        collate_func = create_collate_fn(processor, resnet_transform, tokenizer, device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=collate_func, pin_memory=True if device.type=='cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_func, pin_memory=True if device.type=='cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_func, pin_memory=True if device.type=='cuda' else False)

        logging.info("Data loaders created successfully.")
        return train_loader, val_loader, test_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# --- Tính toán mAP ---
def calculate_map(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Tính Mean Average Precision (mAP)."""
    ap_scores = []
    num_classes = labels.shape[1]
    for i in range(num_classes):
        pred_i = predictions[:, i]
        label_i = labels[:, i]
        # Kiểm tra xem có nhãn dương nào không
        if np.any(label_i > 0): # Check for any positive labels
            try:
               ap = average_precision_score(label_i, pred_i)
               ap_scores.append(ap)
            except ValueError as e:
                logging.warning(f"Could not compute AP for class {i}. Error: {e}. Skipping.")
        else:
            # Log thay vì chỉ bỏ qua âm thầm nếu cần theo dõi
             logging.debug(f"Skipping AP for class {i}: no positive labels found in this set.")


    if not ap_scores:
         logging.warning("No AP scores calculated (perhaps no positive samples in any class?). Returning 0.0 mAP.")
         return 0.0

    mean_ap = np.mean(ap_scores)
    return float(mean_ap)

# --- CCIM Components ---
class DotProductIntervention(nn.Module):
    def __init__(self, con_feature_dim: int, fuse_size: int, attention_dim: int = 256):
        super().__init__()
        self.con_feature_dim = con_feature_dim
        self.fuse_size = fuse_size
        self.attention_dim = attention_dim
        self.query = nn.Linear(self.fuse_size, self.attention_dim, bias=False)
        self.key = nn.Linear(self.con_feature_dim, self.attention_dim, bias=False)
        self.scale = math.sqrt(self.attention_dim)

    def forward(self, confounder_set: torch.Tensor, fuse_rep: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        # confounder_set: (num_confounders, con_feature_dim)
        # fuse_rep: (batch_size, fuse_size)
        # probabilities: (num_confounders, 1) - Ensure shape [num_confounders, 1] for broadcasting
        
        query = self.query(fuse_rep)  # (batch_size, attention_dim)
        key = self.key(confounder_set)  # (num_confounders, attention_dim)
        
        # Calculate attention scores
        mid = torch.matmul(query, key.transpose(0, 1)) / self.scale # (batch_size, num_confounders)
        attention_weights = F.softmax(mid, dim=-1)  # (batch_size, num_confounders)
        attention_weights = attention_weights.unsqueeze(2)  # (batch_size, num_confounders, 1)
        
        # Expand probabilities for broadcasting: (1, num_confounders, 1)
        probabilities_exp = probabilities.transpose(0, 1).unsqueeze(0) # Correct shape if input is [N, 1]

        # Weighted sum: Attention * Confounder * Prior Probability
        # Confounder set needs expansion: (1, num_confounders, con_feature_dim)
        confounder_set_exp = confounder_set.unsqueeze(0)

        # Perform the weighted sum
        fin = (attention_weights * confounder_set_exp * probabilities_exp).sum(dim=1)  # (batch_size, con_feature_dim)
        
        return fin


class AdditiveIntervention(nn.Module):
    # Tương tự như code gốc, giữ nguyên cấu trúc
     def __init__(self, con_feature_dim, fuse_size, attention_dim=256):
        super(AdditiveIntervention, self).__init__()
        self.con_feature_dim = con_feature_dim
        self.fuse_size = fuse_size
        self.attention_dim = attention_dim
        self.Tan = nn.Tanh()
        self.query = nn.Linear(self.fuse_size, self.attention_dim, bias=False)
        self.key = nn.Linear(self.con_feature_dim, self.attention_dim, bias=False)
        self.w_t = nn.Linear(self.attention_dim, 1, bias=False)

     def forward(self, confounder_set: torch.Tensor, fuse_rep: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        query = self.query(fuse_rep) # (batch, att_dim)
        key = self.key(confounder_set) # (n_conf, att_dim)
        
        query_expand = query.unsqueeze(1) # (batch, 1, att_dim)
        key_expand = key.unsqueeze(0) # (1, n_conf, att_dim)

        fuse = query_expand + key_expand # (batch, n_conf, att_dim) - Broadcasting
        fuse = self.Tan(fuse)
        
        attention_scores = self.w_t(fuse).squeeze(-1) # (batch, n_conf)
        attention_weights = F.softmax(attention_scores, dim=1) # (batch, n_conf)
        attention_weights = attention_weights.unsqueeze(2) # (batch, n_conf, 1)

        # Expand probabilities and confounders
        probabilities_exp = probabilities.transpose(0, 1).unsqueeze(0) # (1, n_conf, 1)
        confounder_set_exp = confounder_set.unsqueeze(0) # (1, n_conf, conf_dim)

        fin = (attention_weights * confounder_set_exp * probabilities_exp).sum(dim=1) # (batch, conf_dim)
        return fin


class Classifier(nn.Module):
    # Giữ nguyên như code gốc
    def __init__(self, embed_dim: int = 128, dropout_rate: float = 0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.drop = nn.Dropout(p=dropout_rate)
        self.norm = nn.BatchNorm1d(embed_dim) # BatchNorm or LayerNorm

    def forward(self, out):
        residual = out
        out = self.norm(out)
        out = F.gelu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        # Consider LayerNorm instead of BatchNorm if batch size is small
        out = residual + out * 0.3 # Residual connection scaling
        return out


class CCIM(nn.Module):
    def __init__(self, num_joint_feature: int, con_feature_dim: int, strategy: str = 'dp_cause', final_embed_dim: int = 128):
        super().__init__()
        self.num_joint_feature = num_joint_feature
        self.con_feature_dim = con_feature_dim
        self.final_embed_dim = final_embed_dim

        if strategy == 'dp_cause':
            self.causal_intervention = DotProductIntervention(con_feature_dim, num_joint_feature)
        elif strategy == 'ad_cause':
            self.causal_intervention = AdditiveIntervention(con_feature_dim, num_joint_feature)
        else:
            raise ValueError(f"Do Not Exist This Strategy: {strategy}")

        self.w_h = Parameter(torch.Tensor(self.num_joint_feature, self.final_embed_dim))
        self.w_g = Parameter(torch.Tensor(self.con_feature_dim, self.final_embed_dim))
        self.classifier = Classifier(embed_dim=self.final_embed_dim) # Use configured embed_dim
        # Output layer removed, will be added in the main model

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    def forward(self, joint_feature: torch.Tensor, confounder_dictionary: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        # joint_feature: (batch_size, num_joint_feature)
        # confounder_dictionary: (num_confounders, con_feature_dim)
        # prior: (num_confounders, 1)

        if confounder_dictionary is None or prior is None:
            raise ValueError("Confounder dictionary and prior must be set in CCIM.")
        
        # Ensure prior has the correct shape [num_confounders, 1]
        if prior.dim() == 1:
            prior = prior.unsqueeze(1)
        elif prior.shape[1] != 1:
             # Assuming input is [1, N], transpose it
             if prior.shape[0] == 1 and prior.shape[1] == confounder_dictionary.shape[0]:
                 prior = prior.transpose(0, 1)
             else:
                  raise ValueError(f"Prior shape {prior.shape} is not compatible. Expected [{confounder_dictionary.shape[0]}, 1].")


        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior) # (batch_size, con_feature_dim)
        
        proj_h = torch.matmul(joint_feature, self.w_h) # (batch_size, final_embed_dim)
        proj_g_z = torch.matmul(g_z, self.w_g) # (batch_size, final_embed_dim)
        
        do_x = proj_h + proj_g_z # Combined feature after intervention
        
        out = self.classifier(do_x) # (batch_size, final_embed_dim)
        # Final FC layer for classification will be outside CCIM module
        return out

# --- Bayesian Linear Layer ---
class BayesianLinear(nn.Module):
    # Giữ nguyên như code gốc
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Register buffers for standard deviations (derived from log_sigma)
        # self.register_buffer('weight_sigma', torch.exp(self.weight_log_sigma)) # Cannot register buffer derived from Parameter
        # self.register_buffer('bias_sigma', torch.exp(self.bias_log_sigma))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization (can be experimented with)
        nn.init.normal_(self.weight_mu, 0, 0.1) # Kaiming or Xavier might also work
        nn.init.constant_(self.weight_log_sigma, -5) # Initialize variance to be small
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_log_sigma, -5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases from the learned distributions
        # Use reparameterization trick: sample from N(0,1) and scale/shift
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # Sample noise only during training? Or always for Bayesian effect? Typically always.
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Standard linear transformation with sampled parameters
        return F.linear(x, weight, bias)

    def kl_divergence(self, prior_mu: float = 0.0) -> torch.Tensor:
        """Calculates the KL divergence between the learned distribution and the prior N(prior_mu, prior_sigma^2)."""
        # KL divergence for Gaussian distributions q(w | mu, sigma^2) || p(w | prior_mu, prior_sigma^2)
        # KL = log(prior_sigma / sigma) + (sigma^2 + (mu - prior_mu)^2) / (2 * prior_sigma^2) - 0.5
        
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # Prior terms (assuming prior_mu=0 for simplicity, matching common practice)
        prior_log_sigma = math.log(self.prior_sigma)

        # Weight KL
        kl_weight = (prior_log_sigma - self.weight_log_sigma +
                     0.5 * (weight_sigma**2 + (self.weight_mu - prior_mu)**2) / (self.prior_sigma**2) - 0.5)
        
        # Bias KL
        kl_bias = (prior_log_sigma - self.bias_log_sigma +
                   0.5 * (bias_sigma**2 + (self.bias_mu - prior_mu)**2) / (self.prior_sigma**2) - 0.5)

        # Sum over all parameters
        return kl_weight.sum() + kl_bias.sum()


# --- Main Model ---
class CLIPEmoticModel(nn.Module):
    def __init__(self, 
                 clip_pretrained: str = "openai/clip-vit-base-patch32", 
                 num_cat: int = 26, 
                 hidden_dim: int = 512, 
                 embed_dim: int = 128, # Dimension before final classification
                 prior_sigma: float = 1.0, 
                 num_confounders: int = 256, # Should match k-means clusters
                 ccim_strategy: str = 'dp_cause',
                 resnet_feature_dim: int = 2048, # Output dim of ResNet before avgpool
                 freeze_clip: bool = True, # Flag to freeze CLIP weights
                 cache_dir: Optional[str] = None):
        super().__init__()
        self.num_cat = num_cat
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.freeze_clip = freeze_clip
        self.num_confounders = num_confounders # Store for reference if needed
        self.resnet_feature_dim = resnet_feature_dim

        try:
            logging.info(f"Loading CLIP model: {clip_pretrained}")
            self.clip = CLIPModel.from_pretrained(clip_pretrained, cache_dir=cache_dir, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1")
            self.clip_dim = self.clip.config.projection_dim # Usually 512 for base models
            logging.info(f"CLIP model loaded. Projection dim: {self.clip_dim}")
        except Exception as e:
            logging.error(f"Failed to load CLIP model '{clip_pretrained}': {e}")
            raise

        # Freeze CLIP parameters if requested
        if self.freeze_clip:
            logging.info("Freezing CLIP model parameters.")
            for param in self.clip.parameters():
                param.requires_grad = False
        else:
             logging.info("CLIP model parameters will be fine-tuned.")


        # Projection layers (Bayesian)
        self.context_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)
        self.body_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)
        self.text_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)

        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim) # Fuses projected features

        # CCIM Module
        self.ccim = CCIM(num_joint_feature=hidden_dim, # Input to CCIM is fused feature
                         con_feature_dim=resnet_feature_dim, # From ResNet confounder dict
                         strategy=ccim_strategy,
                         final_embed_dim=embed_dim) # CCIM outputs this dimension

        # Final classification layer (after CCIM)
        self.emotic_fc = nn.Linear(embed_dim, num_cat)

        # Prototypes for contrastive loss
        self.prototypes = nn.Parameter(torch.randn(num_cat, embed_dim)) # Should match CCIM output dim
        nn.init.xavier_uniform_(self.prototypes)

        # Placeholders for confounder dictionary and prior (set externally)
        self.register_buffer('confounder_dict', None, persistent=False)
        self.register_buffer('confounder_prior', None, persistent=False)


    def set_confounder_dict(self, confounder_dict: torch.Tensor, confounder_prior: torch.Tensor):
        """Sets the confounder dictionary and prior probabilities."""
        if confounder_dict.shape != (self.num_confounders, self.resnet_feature_dim):
             logging.warning(f"Confounder dict shape mismatch. Expected: ({self.num_confounders}, {self.resnet_feature_dim}), Got: {confounder_dict.shape}")
        if confounder_prior.shape[0] != self.num_confounders or confounder_prior.dim() > 2 or (confounder_prior.dim()==2 and confounder_prior.shape[1]!=1):
             logging.warning(f"Confounder prior shape mismatch. Expected: ({self.num_confounders}, 1) or ({self.num_confounders}), Got: {confounder_prior.shape}")
             # Attempt to reshape if possible
             if confounder_prior.numel() == self.num_confounders:
                 confounder_prior = confounder_prior.view(self.num_confounders, 1)
                 logging.warning(f"Reshaped prior to {confounder_prior.shape}")
             else:
                 raise ValueError("Cannot reshape prior to required dimensions.")

        # Ensure they are on the correct device (use register_buffer assignment)
        self.confounder_dict = confounder_dict.clone().detach()
        self.confounder_prior = confounder_prior.clone().detach()
        logging.info(f"Confounder dictionary (shape: {self.confounder_dict.shape}) and prior (shape: {self.confounder_prior.shape}) set.")

    def forward(self, context_images: torch.Tensor, body_images: torch.Tensor, text_tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get CLIP features
        # Use torch.no_grad() only if CLIP is truly frozen and we want absolute certainty
        # If freeze_clip=True and requires_grad=False was set, this isn't strictly necessary
        # but doesn't hurt. If fine-tuning (freeze_clip=False), this MUST be removed.
        clip_context = {} if self.freeze_clip else torch.enable_grad()
        with clip_context:
             # CLIP expects pixel_values directly for images
             context_features = self.clip.get_image_features(pixel_values=context_images)
             body_features = self.clip.get_image_features(pixel_values=body_images)
             # Text features use input_ids and attention_mask
             text_features = self.clip.get_text_features(input_ids=text_tokens['input_ids'],
                                                         attention_mask=text_tokens['attention_mask'])

        # Project features using Bayesian layers
        context_emb = self.context_proj(context_features)
        body_emb = self.body_proj(body_features)
        text_emb = self.text_proj(text_features)

        # Normalize embeddings (important for contrastive/similarity tasks)
        # Note: Normalizing *before* fusion might be different than normalizing *after*
        context_emb = F.normalize(context_emb, p=2, dim=1)
        body_emb = F.normalize(body_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)

        # Fuse features
        combined_features = torch.cat([context_emb, body_emb, text_emb], dim=1)
        fused_features = F.relu(self.fusion(combined_features)) # (batch_size, hidden_dim)

        # Apply Causal Intervention (CCIM)
        if self.confounder_dict is None or self.confounder_prior is None:
             # This shouldn't happen during normal training/eval if set_confounder_dict is called.
             raise RuntimeError("Confounder dictionary and prior are not set. Call set_confounder_dict() before the forward pass.")
            # Fallback (less ideal): Use zeros/ones - provide a warning
            # logging.warning("Confounder dict/prior not set! Using dummy values for CCIM.")
            # Ensure dummy tensors are on the correct device and have compatible (though likely incorrect) dimensions
            # dummy_confounder_dict = torch.zeros(self.num_confounders, self.resnet_feature_dim, device=fused_features.device)
            # dummy_prior = torch.ones(self.num_confounders, 1, device=fused_features.device) / self.num_confounders
            # intervened_features = self.ccim(fused_features, dummy_confounder_dict, dummy_prior)
        else:
            # Move confounder dict/prior to the correct device just in case
            conf_dict_dev = self.confounder_dict.to(fused_features.device)
            conf_prior_dev = self.confounder_prior.to(fused_features.device)
            intervened_features = self.ccim(fused_features, conf_dict_dev, conf_prior_dev) # (batch_size, embed_dim)

        # Final classification prediction
        cat_pred = self.emotic_fc(intervened_features) # (batch_size, num_cat)

        # Return embeddings for potential contrastive loss, predictions, and prototypes
        return context_emb, body_emb, text_emb, cat_pred, self.prototypes

    def kl_divergence(self) -> torch.Tensor:
        """Sums KL divergence from all BayesianLinear layers."""
        kl_total = torch.tensor(0.0, device=self.emotic_fc.weight.device) # Ensure KL starts on correct device
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_total = kl_total + module.kl_divergence()
        return kl_total

# --- Contrastive and Regularization Losses ---
def prototypical_contrastive_loss(embeddings: torch.Tensor, 
                                   cat_labels: torch.Tensor, 
                                   prototypes: torch.Tensor, 
                                   temperature: float = 0.07) -> torch.Tensor:
    """Prototypical Contrastive Loss (InfoNCE style)."""
    # Normalize embeddings and prototypes (defensive)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1) # Prototypes from the model
    
    # Calculate similarity matrix (batch_size, num_prototypes)
    sim_matrix = torch.matmul(embeddings, prototypes.t()) / temperature
    
    loss = torch.tensor(0.0, device=embeddings.device)
    valid_samples = 0

    for i in range(cat_labels.size(0)): # Iterate over samples in the batch
        pos_mask = cat_labels[i] == 1
        neg_mask = cat_labels[i] == 0
        
        # Check if there are both positive and negative prototypes for this sample
        if torch.any(pos_mask) and torch.any(neg_mask):
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            # Similarity to positive prototypes for sample i
            pos_sim = sim_matrix[i, pos_indices] # Shape: [num_pos]
            
            # Similarity to negative prototypes for sample i
            neg_sim = sim_matrix[i, neg_indices] # Shape: [num_neg]

            # Treat each positive prototype as an anchor
            for pos_idx in pos_indices:
                 # Create logits for InfoNCE: [sim_to_this_pos, sim_to_all_neg]
                 anchor_pos_sim = sim_matrix[i, pos_idx].unsqueeze(0) # Shape: [1]
                 logits = torch.cat([anchor_pos_sim, neg_sim], dim=0) # Shape: [1 + num_neg]
                 
                 # Labels for cross-entropy: 0 indicates the positive class
                 labels_nce = torch.zeros(1, dtype=torch.long, device=embeddings.device)
                 
                 loss = loss + F.cross_entropy(logits.unsqueeze(0), labels_nce, reduction='mean')

            valid_samples += len(pos_indices) # Count each positive anchor used

    return loss / max(valid_samples, 1) # Average loss over all valid positive anchors


def regularization_loss(prototypes: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Encourages prototypes to be separated."""
    prototypes = F.normalize(prototypes, p=2, dim=1) # Work with normalized prototypes
    num_prototypes = prototypes.size(0)
    loss = torch.tensor(0.0, device=prototypes.device)
    count = 0
    
    if num_prototypes < 2:
        return loss # No pairs to compare

    for i in range(num_prototypes):
        for j in range(i + 1, num_prototypes):
            # Cosine similarity between -1 and 1. We want low similarity (large angle).
            similarity = torch.dot(prototypes[i], prototypes[j])
            # Loss = max(0, similarity - margin) # Penalize if similarity > margin (too close)
            # OR using distance:
            dist = torch.norm(prototypes[i] - prototypes[j], p=2)
            loss += F.relu(margin - dist) # Penalize if distance < margin
            count += 1
            
    return loss / max(count, 1)

# --- Confounder Dictionary Creation ---
@torch.no_grad() # Ensure no gradients are computed here
def create_confounder_dict(train_loader: DataLoader, 
                             resnet_model: nn.Module, 
                             device: torch.device, 
                             num_confounders: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tạo dictionary các confounder bằng K-Means trên đặc trưng ResNet."""
    resnet_model.eval() # Ensure evaluation mode
    context_features_list = []
    logging.info("Starting confounder dictionary creation...")
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None: # Skip invalid batches from collate_fn
             logging.warning(f"Skipping batch {batch_idx} in confounder creation due to collate error.")
             continue

        # Extract ResNet context tensors from the collated batch
        # Assumes collate_fn puts resnet tensors at index 2
        _, _, context_resnet_tensors, _, _ = batch_data 
        context_resnet_tensors = context_resnet_tensors.to(device)

        # Pass through ResNet (already on the correct device)
        # ResNet model already removed the FC layer
        features = resnet_model(context_resnet_tensors) # Output shape (batch, 2048, 1, 1)
        features = features.squeeze() # Flatten: [batch_size, 2048]
        if features.dim() == 1: # Handle case of batch size 1
             features = features.unsqueeze(0)
        
        features = F.normalize(features, p=2, dim=1) # Normalize features
        context_features_list.append(features.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
             logging.info(f"Processed {batch_idx + 1}/{len(train_loader)} batches for confounder features.")

    if not context_features_list:
        raise ValueError("No context features extracted for K-Means. Check DataLoader and ResNet model.")
    
    context_features_all = np.concatenate(context_features_list, axis=0)
    logging.info(f"Total context features extracted: {context_features_all.shape}")

    if context_features_all.shape[0] < num_confounders:
        logging.warning(f"Number of samples ({context_features_all.shape[0]}) is less than num_confounders ({num_confounders}). Reducing clusters to number of samples.")
        num_confounders = context_features_all.shape[0]

    logging.info(f"Running KMeans with {num_confounders} clusters...")
    kmeans = KMeans(n_clusters=num_confounders, init='k-means++', random_state=42, n_init=10) # n_init added
    kmeans.fit(context_features_all)
    
    # Cluster centers are the confounder dictionary
    confounder_dict_np = kmeans.cluster_centers_
    # Normalize centers as well? Depends on the intervention mechanism expectation
    confounder_dict_np = confounder_dict_np / np.linalg.norm(confounder_dict_np, axis=1, keepdims=True)

    confounder_dict = torch.tensor(confounder_dict_np, dtype=torch.float32) # Shape: [num_confounders, feature_dim]
    
    # Calculate prior probability for each cluster (confounder)
    labels = kmeans.labels_
    prior = np.bincount(labels, minlength=num_confounders) / len(labels)
    confounder_prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(1) # Shape: [num_confounders, 1]
    
    logging.info(f"Confounder dictionary created (Shape: {confounder_dict.shape})")
    logging.info(f"Confounder prior created (Shape: {confounder_prior.shape})")

    return confounder_dict, confounder_prior # Return on CPU, move to device later

# --- Training Loop ---
def train_model(model: CLIPEmoticModel, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                cat_loss_fn: nn.Module,
                num_epochs: int, 
                device: torch.device,
                lambda_proto: float, 
                lambda_reg: float, 
                lambda_kl: float,
                temperature: float,
                prototype_update_alpha: float,
                checkpoint_path: str):
    """Train the model."""
    best_val_map = 0.0
    
    # --- Create and Set Confounder Dictionary ---
    # It's better to load the ResNet model outside the loop/function
    # resnet_model = load_resnet50_places365(resnet_checkpoint_path, device)
    # confounder_dict, confounder_prior = create_confounder_dict(train_loader, resnet_model, device, model.num_confounders)
    # del resnet_model # Free memory
    # torch.cuda.empty_cache()
    # model.set_confounder_dict(confounder_dict.to(device), confounder_prior.to(device)) # Move to device here
    # Moved this logic to main script before calling train_model

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_cls = 0.0
        train_loss_proto = 0.0
        train_loss_reg = 0.0
        train_loss_kl = 0.0
        num_batches_train = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None: # Skip problematic batches
                 logging.warning(f"Skipping training batch {batch_idx+1}/{num_batches_train} due to previous error.")
                 continue
                 
            context_images, body_images, _, text_tokens, cat_labels = batch_data
            # Data should already be on the correct device from collate_fn

            optimizer.zero_grad()

            # Forward pass
            context_emb, body_emb, text_emb, cat_pred, prototypes = model(context_images, body_images, text_tokens)

            # --- Update Prototypes (Momentum/EMA) ---
            with torch.no_grad():
                 # Ensure prototype update happens on the correct device
                 prototypes_device = model.prototypes.device 
                 current_prototypes = model.prototypes.clone().detach() # Work with a detached copy

                 for j in range(model.num_cat):
                     mask = cat_labels[:, j] == 1
                     if mask.sum() > 0:
                          # Combine context and body embeddings for prototype update
                          # Using only body or context might also be valid choices
                          mean_emb = (context_emb[mask].detach() + body_emb[mask].detach()).mean(dim=0) / 2
                          # Normalize the mean embedding before EMA update
                          mean_emb_norm = F.normalize(mean_emb, p=2, dim=0) 
                          
                          # EMA update
                          updated_proto_j = prototype_update_alpha * current_prototypes[j] + \
                                           (1 - prototype_update_alpha) * mean_emb_norm
                          
                          # Normalize the updated prototype
                          model.prototypes.data[j] = F.normalize(updated_proto_j, p=2, dim=0)

            # --- Calculate Losses ---
            cls_loss = cat_loss_fn(cat_pred, cat_labels)
            
            # Combine context and body embeddings for proto loss? Or separate?
            # Original code did separate, let's stick to that unless there's reason to change.
            proto_loss_context = prototypical_contrastive_loss(context_emb, cat_labels, prototypes, temperature)
            proto_loss_body = prototypical_contrastive_loss(body_emb, cat_labels, prototypes, temperature)
            proto_loss = (proto_loss_context + proto_loss_body) / 2
            
            reg_loss = regularization_loss(prototypes)
            
            kl_loss = model.kl_divergence()

            # Total loss
            total_loss = cls_loss + lambda_proto * proto_loss + lambda_reg * reg_loss + lambda_kl * kl_loss

            # Backpropagation
            total_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses for logging
            train_loss_total += total_loss.item()
            train_loss_cls += cls_loss.item()
            train_loss_proto += proto_loss.item() # Use the averaged value
            train_loss_reg += reg_loss.item()
            train_loss_kl += kl_loss.item()

            if (batch_idx + 1) % 100 == 0: # Log progress periodically
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{num_batches_train}], Loss: {total_loss.item():.4f}')

        # --- Validation Phase ---
        model.eval()
        val_loss_total = 0.0
        val_predictions_all = []
        val_labels_all = []
        num_batches_val = len(val_loader)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if batch_data is None:
                    logging.warning(f"Skipping validation batch {batch_idx+1}/{num_batches_val} due to previous error.")
                    continue

                context_images, body_images, _, text_tokens, cat_labels = batch_data
                
                # Forward pass (no prototype update during validation)
                context_emb, body_emb, text_emb, cat_pred, prototypes_val = model(context_images, body_images, text_tokens)

                # Calculate validation losses (optional, but good for monitoring)
                cls_loss_val = cat_loss_fn(cat_pred, cat_labels)
                proto_loss_ctx_val = prototypical_contrastive_loss(context_emb, cat_labels, prototypes_val, temperature)
                proto_loss_body_val = prototypical_contrastive_loss(body_emb, cat_labels, prototypes_val, temperature)
                proto_loss_val = (proto_loss_ctx_val + proto_loss_body_val) / 2
                reg_loss_val = regularization_loss(prototypes_val)
                kl_loss_val = model.kl_divergence() # Note: KL depends on sampled weights, will vary

                total_loss_val = cls_loss_val + lambda_proto * proto_loss_val + lambda_reg * reg_loss_val + lambda_kl * kl_loss_val
                val_loss_total += total_loss_val.item()

                # Store predictions and labels for mAP calculation
                pred_probs = torch.sigmoid(cat_pred)
                val_predictions_all.append(pred_probs.cpu().numpy())
                val_labels_all.append(cat_labels.cpu().numpy())

        # --- End of Epoch Logging & Evaluation ---
        avg_train_loss = train_loss_total / num_batches_train if num_batches_train > 0 else 0
        avg_val_loss = val_loss_total / num_batches_val if num_batches_val > 0 else 0
        
        avg_train_cls = train_loss_cls / num_batches_train if num_batches_train > 0 else 0
        avg_train_proto = train_loss_proto / num_batches_train if num_batches_train > 0 else 0
        avg_train_reg = train_loss_reg / num_batches_train if num_batches_train > 0 else 0
        avg_train_kl = train_loss_kl / num_batches_train if num_batches_train > 0 else 0

        # Calculate mAP on validation set
        val_map = 0.0
        if val_predictions_all:
            val_predictions_np = np.concatenate(val_predictions_all, axis=0)
            val_labels_np = np.concatenate(val_labels_all, axis=0)
            if val_predictions_np.shape[0] > 0: # Ensure there are samples
                 val_map = calculate_map(val_predictions_np, val_labels_np)
        
        logging.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        logging.info(f"Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_cls:.4f}, Proto: {avg_train_proto:.4f}, Reg: {avg_train_reg:.4f}, KL: {avg_train_kl:.4f})")
        logging.info(f"Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Val mAP: {val_map:.4f}")

        # Save the best model based on validation mAP
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"*** New best model saved to {checkpoint_path} with Val mAP: {best_val_map:.4f} ***")
            
        # Optional: Learning rate scheduling
        # scheduler.step() or scheduler.step(val_map) 

# --- Testing Function ---
@torch.no_grad()
def test_model(model: CLIPEmoticModel, 
               test_loader: DataLoader, 
               device: torch.device, 
               num_samples: int = 10): # Number of samples for Bayesian prediction averaging
    """Evaluate the model on the test set."""
    model.eval() # Set to evaluation mode
    all_predictions_avg = []
    all_labels = []
    num_batches_test = len(test_loader)

    logging.info(f"Starting testing with {num_samples} samples per input...")
    
    for batch_idx, batch_data in enumerate(test_loader):
        if batch_data is None:
            logging.warning(f"Skipping test batch {batch_idx+1}/{num_batches_test} due to previous error.")
            continue
            
        context_images, body_images, _, text_tokens, cat_labels = batch_data
        
        # Collect predictions from multiple forward passes (due to Bayesian layers)
        pred_probs_samples = []
        for _ in range(num_samples):
            # Forward pass - each pass samples different weights/biases in Bayesian layers
            _, _, _, cat_pred, _ = model(context_images, body_images, text_tokens)
            pred_probs = torch.sigmoid(cat_pred)
            pred_probs_samples.append(pred_probs) # List of [batch, num_cat] tensors

        # Average the probabilities across samples
        pred_probs_stack = torch.stack(pred_probs_samples, dim=0) # Shape: [num_samples, batch, num_cat]
        pred_probs_avg = torch.mean(pred_probs_stack, dim=0) # Shape: [batch, num_cat]
        
        all_predictions_avg.append(pred_probs_avg.cpu().numpy())
        all_labels.append(cat_labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
             logging.info(f"Processed {batch_idx + 1}/{num_batches_test} test batches.")


    if not all_predictions_avg:
         logging.error("No predictions generated during testing.")
         return 0.0

    # Concatenate results from all batches
    all_predictions_np = np.concatenate(all_predictions_avg, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    # Calculate mAP
    mean_ap = calculate_map(all_predictions_np, all_labels_np)
    logging.info(f"--- Test Results ---")
    logging.info(f"Test mAP ({num_samples} samples): {mean_ap:.4f}")

    # Calculate and print per-class AP scores
    ap_scores = []
    num_classes = all_labels_np.shape[1]
    print("\nPer-class AP scores:")
    for i in range(num_classes):
         pred_i = all_predictions_np[:, i]
         label_i = all_labels_np[:, i]
         if np.any(label_i > 0):
             try:
                 ap = average_precision_score(label_i, pred_i)
                 ap_scores.append(ap)
                 print(f"Class {i}: {ap:.4f}")
             except ValueError:
                 print(f"Class {i}: Error calculating AP (likely all labels are the same)")
                 ap_scores.append(0.0) # Or nan
         else:
              print(f"Class {i}: No positive labels, AP not calculated.")
              ap_scores.append(0.0) # Or nan

    # Sanity check
    recalculated_map = np.mean([score for score in ap_scores if not np.isnan(score) and score>0]) # Mean of calculated APs
    logging.info(f"Recalculated mAP (avg of above): {recalculated_map:.4f}")

    return mean_ap

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP-Emotic Model with CCIM")
    
    # Data and Paths
    parser.add_argument('--data_dir', type=str, default="/content/dataset_emotic", help="Directory containing train/val/test data (.npy, .csv)")
    parser.add_argument('--resnet_checkpoint', type=str, default="./resnet50_places365.pth.tar", help="Path to ResNet-50 Places365 checkpoint")
    parser.add_argument('--clip_model_name', type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model name")
    parser.add_argument('--cache_dir', type=str, default=None, help="Directory for Hugging Face cache")
    parser.add_argument('--output_checkpoint', type=str, default='best_clip_emotic_ccim_v2.pt', help="Path to save the best model checkpoint")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of DataLoader workers")
    
    # Model Hyperparameters
    parser.add_argument('--num_cat', type=int, default=26, help="Number of emotion categories")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for projections and fusion")
    parser.add_argument('--embed_dim', type=int, default=128, help="Final embedding dimension before classification (CCIM output)")
    parser.add_argument('--num_confounders', type=int, default=256, help="Number of confounders (K-Means clusters)")
    parser.add_argument('--ccim_strategy', type=str, default='dp_cause', choices=['dp_cause', 'ad_cause'], help="CCIM intervention strategy")
    parser.add_argument('--prior_sigma', type=float, default=1.0, help="Prior sigma for Bayesian layers")
    parser.add_argument('--freeze_clip', action='store_true', help="Freeze CLIP model weights during training") # Defaults to False unless specified
    parser.add_argument('--no_freeze_clip', action='store_false', dest='freeze_clip', help="Fine-tune CLIP model weights")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and evaluation") # Reduced default BS
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Optimizer learning rate") # Adjusted LR
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument('--loss_weight_type', type=str, default='dynamic', choices=['mean', 'static', 'dynamic'], help="Weighting type for BCE loss")
    parser.add_argument('--lambda_proto', type=float, default=0.1, help="Weight for prototypical contrastive loss")
    parser.add_argument('--lambda_reg', type=float, default=0.01, help="Weight for prototype regularization loss")
    parser.add_argument('--lambda_kl', type=float, default=1e-6, help="Weight for KL divergence loss (scaled down)") # Scaled down default KL weight
    parser.add_argument('--proto_temp', type=float, default=0.07, help="Temperature for prototypical contrastive loss")
    parser.add_argument('--proto_alpha', type=float, default=0.99, help="Momentum/EMA alpha for prototype updates (higher means slower updates)") # Adjusted alpha
    parser.add_argument('--reg_margin', type=float, default=0.8, help="Margin for prototype regularization loss") # Adjusted margin

    # Testing
    parser.add_argument('--test_only', action='store_true', help="Only run testing using the saved checkpoint")
    parser.add_argument('--test_samples', type=int, default=10, help="Number of sampling passes for Bayesian testing")

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Script arguments: {args}")
    
    # Load Hugging Face resources
    processor, tokenizer = load_hf_resources(args.clip_model_name, args.cache_dir)
    
    # Get ResNet transform
    resnet_transform = get_resnet_transform()
    
    # --- Load Data ---
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_src=args.data_dir,
        batch_size=args.batch_size,
        device=device, # Pass device for collate_fn
        processor=processor,
        tokenizer=tokenizer,
        resnet_transform=resnet_transform,
        num_workers=args.num_workers
    )
    
    # --- Initialize Model ---
    logging.info("Initializing model...")
    model = CLIPEmoticModel(
        clip_pretrained=args.clip_model_name,
        num_cat=args.num_cat,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        prior_sigma=args.prior_sigma,
        num_confounders=args.num_confounders,
        ccim_strategy=args.ccim_strategy,
        resnet_feature_dim=2048, # Standard for ResNet50 output before FC
        freeze_clip=args.freeze_clip,
        cache_dir=args.cache_dir
    ).to(device)

    # --- Training Phase ---
    if not args.test_only:
        # Create and set confounder dictionary *before* training starts
        logging.info("Preparing confounder dictionary using ResNet...")
        resnet_model = load_resnet50_places365(args.resnet_checkpoint, device)
        try:
            confounder_dict, confounder_prior = create_confounder_dict(
                 train_loader, resnet_model, device, args.num_confounders
             )
            # Set dictionary in the main model
            model.set_confounder_dict(confounder_dict.to(device), confounder_prior.to(device))
        except Exception as e:
            logging.error(f"Failed to create confounder dictionary: {e}. Training cannot proceed with CCIM.", exc_info=True)
            # Decide whether to exit or proceed without CCIM (if model supports it)
            exit()
        finally:
            # Clean up ResNet model to free GPU memory before training CLIP model
            del resnet_model
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            logging.info("ResNet model removed from memory.")


        # Optimizer (consider filtering parameters if fine-tuning CLIP with different LR)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Loss Function
        cat_loss_fn = BCEWithLogitsLoss(weight_type=args.loss_weight_type, device=device, num_classes=args.num_cat)

        logging.info("Starting training...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            cat_loss_fn=cat_loss_fn,
            num_epochs=args.epochs,
            device=device,
            lambda_proto=args.lambda_proto,
            lambda_reg=args.lambda_reg,
            lambda_kl=args.lambda_kl,
            temperature=args.proto_temp,
            prototype_update_alpha=args.proto_alpha,
            checkpoint_path=args.output_checkpoint
        )
        logging.info("Training finished.")

    # --- Testing Phase ---
    logging.info("\n--- Starting evaluation on test set ---")
    # Load the best checkpoint saved during training or specified for testing
    if os.path.exists(args.output_checkpoint):
         logging.info(f"Loading best model state dict from: {args.output_checkpoint}")
         # Ensure map_location handles CPU/GPU cases
         map_location = device if torch.cuda.is_available() else torch.device('cpu')
         try:
             model.load_state_dict(torch.load(args.output_checkpoint, map_location=map_location))
         except Exception as e:
              logging.error(f"Error loading state dict from {args.output_checkpoint}: {e}")
              logging.warning("Proceeding with the potentially untrained model for testing.")
    else:
        logging.warning(f"Checkpoint file not found at {args.output_checkpoint}. Testing with the current model state (likely untrained if --test_only).")

    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=args.test_samples
    )

    logging.info("Script finished.")