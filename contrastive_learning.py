import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from sklearn.metrics import average_precision_score
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm tải processor và tokenizer
def load_hf_resources(model_name="openai/clip-vit-base-patch32", cache_dir=None):
    try:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        logging.info("Loaded processor and tokenizer successfully.")
        return processor, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Hugging Face resources: {e}")
        raise

processor, tokenizer = load_hf_resources()

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ]).unsqueeze(0)
            self.weights = self.weights.to(self.device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = self.bce_loss(pred, target)
        weighted_loss = loss * self.weights
        return weighted_loss.mean()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0)
        weights = torch.zeros((1, 26), device=target_stats.device)
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights

def tokenize_text(path_dataset):
    try:
        df = pd.read_csv(path_dataset)
        if 'Output' not in df.columns:
            raise ValueError(f"Column 'Output' not found in {path_dataset}")
        sentences = df['Output'].tolist()
        tokens = tokenizer(sentences, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        tokenized_list = [{key: tokens[key][i] for key in tokens.keys()} for i in range(tokens['input_ids'].size(0))]
        return tokenized_list
    except Exception as e:
        logging.error(f"Error tokenizing text from {path_dataset}: {e}")
        raise

class EmoticDataset(Dataset):
    def __init__(self, x_context, x_body, x_text, y_cat, device="cpu"):
        self.x_context = x_context
        self.x_body = x_body
        self.x_text = x_text
        self.y_cat = y_cat
        self.device = device

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        try:
            # Xử lý context
            context = self.x_context[index]
            if context.shape[0] == 3:
                context = context.transpose(1, 2, 0)
            processed_context = processor(images=context, return_tensors="pt")['pixel_values'][0]

            # Xử lý body
            body = self.x_body[index]
            if body.shape[0] == 3:
                body = body.transpose(1, 2, 0)
            processed_body = processor(images=body, return_tensors="pt")['pixel_values'][0]

            # Xử lý text
            token_text = self.x_text[index]

            # Nhãn
            cat_label = torch.tensor(self.y_cat[index], dtype=torch.float32)
            return processed_context, processed_body, token_text, cat_label
        except Exception as e:
            logging.error(f"Error processing dataset item {index}: {e}")
            raise

def load_data(data_src, batch_size, device):
    try:
        required_files = [
            'train_context_arr.npy', 'train_body_arr.npy', 'train_cat_arr.npy', 'train.csv',
            'val_context_arr.npy', 'val_body_arr.npy', 'val_cat_arr.npy', 'val.csv',
            'test_context_arr.npy', 'test_body_arr.npy', 'test_cat_arr.npy', 'test.csv'
        ]
        for f in required_files:
            if not os.path.exists(os.path.join(data_src, f)):
                raise FileNotFoundError(f"File {f} not found in {data_src}")
        
        # Tải dữ liệu context
        train_context = np.load(os.path.join(data_src, 'train_context_arr.npy'))
        val_context = np.load(os.path.join(data_src, 'val_context_arr.npy'))
        test_context = np.load(os.path.join(data_src, 'test_context_arr.npy'))

        # Tải dữ liệu body
        train_body = np.load(os.path.join(data_src, 'train_body_arr.npy'))
        val_body = np.load(os.path.join(data_src, 'val_body_arr.npy'))
        test_body = np.load(os.path.join(data_src, 'test_body_arr.npy'))

        # Tải nhãn
        train_cat = np.load(os.path.join(data_src, 'train_cat_arr.npy'))
        val_cat = np.load(os.path.join(data_src, 'val_cat_arr.npy'))
        test_cat = np.load(os.path.join(data_src, 'test_cat_arr.npy'))

        # Tải văn bản
        train_text = tokenize_text(os.path.join(data_src, 'train.csv'))
        val_text = tokenize_text(os.path.join(data_src, 'val.csv'))
        test_text = tokenize_text(os.path.join(data_src, 'test.csv'))

        # Tạo dataset
        train_dataset = EmoticDataset(train_context, train_body, train_text, train_cat, device=device)
        val_dataset = EmoticDataset(val_context, val_body, val_text, val_cat, device=device)
        test_dataset = EmoticDataset(test_context, test_body, test_text, test_cat, device=device)

        # Tạo DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logging.info("Data loaders created successfully.")
        return train_loader, val_loader, test_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def calculate_map(predictions, labels):
    ap_scores = []
    for i in range(labels.shape[1]):
        if np.any(labels[:, i]):
            ap = average_precision_score(labels[:, i], predictions[:, i])
            ap_scores.append(ap)
        else:
            logging.warning(f"Skipping AP for class {i}: no positive labels.")
    return np.mean(ap_scores) if ap_scores else 0.0

# Lớp Bayesian Linear
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Tham số cho phân phối trọng số
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # Tham số cho phân phối bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.constant_(self.weight_log_sigma, -5)
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_log_sigma, -5)

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # Lấy mẫu trọng số và bias từ các phân phối
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # Tính KL divergence cho trọng số và bias
        kl_weight = 0.5 * (self.prior_sigma ** 2 / torch.exp(self.weight_log_sigma * 2) + (self.weight_mu / self.prior_sigma) ** 2 - 1 + 2 * self.weight_log_sigma)
        kl_bias = 0.5 * (self.prior_sigma ** 2 / torch.exp(self.bias_log_sigma * 2) + (self.bias_mu / self.prior_sigma) ** 2 - 1 + 2 * self.bias_log_sigma)
        return kl_weight.sum() + kl_bias.sum()

class CLIPEmoticModel(nn.Module):
    def __init__(self, clip_pretrained="openai/clip-vit-base-patch32", num_cat=26, hidden_dim=512, prior_sigma=1.0):
        super(CLIPEmoticModel, self).__init__()
        try:
            self.clip = CLIPModel.from_pretrained(clip_pretrained, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise
        self.clip_dim = self.clip.config.projection_dim
        self.context_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)
        self.body_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)
        self.text_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma)
        self.cat_head = nn.Sequential(
            BayesianLinear(hidden_dim * 3, hidden_dim, prior_sigma),  # *3 vì có context, body, text
            nn.ReLU(),
            nn.Dropout(0.3),
            BayesianLinear(hidden_dim, hidden_dim // 2, prior_sigma),
            nn.ReLU(),
            nn.Dropout(0.3),
            BayesianLinear(hidden_dim // 2, num_cat, prior_sigma)
        )
        self.num_cat = num_cat
        self.hidden_dim = hidden_dim
        self.prototypes = nn.Parameter(torch.randn(num_cat, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, context_images, body_images, text_tokens):
        # dùng CLIP để lấy đặc trưng ảnh và văn bản
        # context_images: (batch_size, 3, 224, 224) và body_images: (batch_size, 3, 224, 224)
        # text_tokens: dict chứa input_ids và attention_mask (batch_size, 77) bị cắt bớt 
        with torch.no_grad():
            context_features = self.clip.get_image_features(context_images) 
            body_features = self.clip.get_image_features(body_images)
            text_features = self.clip.get_text_features(**text_tokens)
        context_emb = self.context_proj(context_features)
        body_emb = self.body_proj(body_features)
        text_emb = self.text_proj(text_features)
        context_emb = F.normalize(context_emb, p=2, dim=1)
        body_emb = F.normalize(body_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        combined_features = torch.cat([context_emb, body_emb, text_emb], dim=1)
        cat_pred = self.cat_head(combined_features)
        return context_emb, body_emb, text_emb, cat_pred, self.prototypes

    def kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl

def prototypical_contrastive_loss(embeddings, cat_labels, prototypes, temperature=0.07):
    # Kết hợp context và body embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, prototypes.T) / temperature
    loss = 0.0
    valid_samples = 0
    for i in range(cat_labels.size(0)):
        pos_indices = torch.where(cat_labels[i] == 1)[0]
        neg_indices = torch.where(cat_labels[i] == 0)[0]
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue
        pos_logits = sim_matrix[i, pos_indices]
        neg_logits = sim_matrix[i, neg_indices]
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.zeros_like(logits)
        labels[:len(pos_logits)] = 1.0
        loss_i = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
        loss += loss_i
        valid_samples += 1
    return loss / max(valid_samples, 1)

def regularization_loss(prototypes, margin=1.0):
    num_prototypes = prototypes.size(0)
    loss = 0.0
    for i in range(num_prototypes):
        for j in range(i + 1, num_prototypes):
            dist = torch.norm(prototypes[i] - prototypes[j], p=2)
            loss += F.relu(margin - dist)
    return loss / max(1, num_prototypes * (num_prototypes - 1) / 2)

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=2e-4, device="cuda", alpha=0.9):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Sử dụng AdamW
    cat_loss_fn = BCEWithLogitsLoss(weight_type='dynamic', device=device)

    lambda_proto = 0.1
    lambda_reg = 0.01
    lambda_kl = 1e-5
    temperature = 0.1

    best_val_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_proto_loss = 0.0
        train_reg_loss = 0.0
        train_kl_loss = 0.0

        for batch_idx, (context_images, body_images, text_tokens, cat_labels) in enumerate(train_loader):
            context_images = context_images.to(device)
            body_images = body_images.to(device)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            cat_labels = cat_labels.to(device)

            optimizer.zero_grad()

            context_emb, body_emb, text_emb, cat_pred, prototypes = model(context_images, body_images, text_tokens)

            # Update prototypes with EMA
            with torch.no_grad():
                for j in range(model.num_cat):
                    mask = cat_labels[:, j] == 1
                    if mask.sum() > 0:
                        # Kết hợp context và body embeddings
                        mean_emb = (context_emb[mask] + body_emb[mask]).mean(dim=0) / 2
                        model.prototypes[j] = alpha * model.prototypes[j] + (1 - alpha) * mean_emb
                        model.prototypes[j] = F.normalize(model.prototypes[j], p=2, dim=0)

            cls_loss = cat_loss_fn(cat_pred, cat_labels)
            # Sử dụng cả context và body trong contrastive loss
            proto_loss_context = prototypical_contrastive_loss(context_emb, cat_labels, prototypes, temperature)
            proto_loss_body = prototypical_contrastive_loss(body_emb, cat_labels, prototypes, temperature)
            proto_loss = (proto_loss_context + proto_loss_body) / 2
            reg_loss = regularization_loss(prototypes)
            kl_loss = model.kl_divergence()

            total_loss = cls_loss + lambda_proto * proto_loss + lambda_reg * reg_loss + lambda_kl * kl_loss
            train_loss += total_loss.item()
            train_cls_loss += cls_loss.item()
            train_proto_loss += proto_loss.item()
            train_reg_loss += reg_loss.item()
            train_kl_loss += kl_loss.item()

            total_loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for context_images, body_images, text_tokens, cat_labels in val_loader:
                context_images = context_images.to(device)
                body_images = body_images.to(device)
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
                cat_labels = cat_labels.to(device)

                context_emb, body_emb, text_emb, cat_pred, prototypes = model(context_images, body_images, text_tokens)

                cls_loss = cat_loss_fn(cat_pred, cat_labels)
                proto_loss_context = prototypical_contrastive_loss(context_emb, cat_labels, prototypes, temperature)
                proto_loss_body = prototypical_contrastive_loss(body_emb, cat_labels, prototypes, temperature)
                proto_loss = (proto_loss_context + proto_loss_body) / 2
                reg_loss = regularization_loss(prototypes)
                kl_loss = model.kl_divergence()

                total_loss = cls_loss + lambda_proto * proto_loss + lambda_reg * reg_loss + lambda_kl * kl_loss
                val_loss += total_loss.item()

                pred_probs = torch.sigmoid(cat_pred)
                val_predictions.append(pred_probs.cpu().numpy())
                val_labels.append(cat_labels.cpu().numpy())

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_map = calculate_map(val_predictions, val_labels)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} (Cls: {train_cls_loss/len(train_loader):.4f}, "
              f"Proto: {train_proto_loss/len(train_loader):.4f}, Reg: {train_reg_loss/len(train_loader):.4f}, "
              f"KL: {train_kl_loss/len(train_loader):.4f})")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val mAP: {val_map:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), 'best_clip_emotic.pt')
            print(f"New best model saved with Val mAP: {best_val_map:.4f}")

def test_model(model, test_loader, device="cuda", num_samples=10):
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for context_images, body_images, text_tokens, cat_labels in test_loader:
            context_images = context_images.to(device)
            body_images = body_images.to(device)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            cat_labels = cat_labels.to(device)
            pred_probs_samples = []
            for _ in range(num_samples):
                _, _, _, cat_pred, _ = model(context_images, body_images, text_tokens)
                pred_probs = torch.sigmoid(cat_pred)
                pred_probs_samples.append(pred_probs)
            pred_probs_avg = torch.mean(torch.stack(pred_probs_samples), dim=0)
            all_predictions.append(pred_probs_avg.cpu().numpy())
            all_labels.append(cat_labels.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    ap_scores = []
    for i in range(all_labels.shape[1]):
        if np.any(all_labels[:, i]):
            ap = average_precision_score(all_labels[:, i], all_predictions[:, i])
            ap_scores.append(ap)
        else:
            logging.warning(f"Skipping AP for class {i} in test: no positive labels.")
    mean_ap = np.mean(ap_scores) if ap_scores else 0.0
    print(f"Test Results:")
    print(f"mAP: {mean_ap:.4f}")
    print("\nPer-class AP scores:")
    for i, ap in enumerate(ap_scores):
        print(f"Class {i}: {ap:.4f}")
    return mean_ap

if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_path = "/kaggle/input/emotion-torch/context_dataset/context_dataset"
        batch_size = 128
        num_epochs = 30
        logging.info(f"Using device: {device}")
        logging.info(f"Data path: {data_path}")
        train_loader, val_loader, test_loader = load_data(data_path, batch_size, device)
        model = CLIPEmoticModel(num_cat=26)
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
        model.load_state_dict(torch.load('best_clip_emotic.pt', map_location=device))
        print("\nEvaluating best model on test set:")
        mean_ap = test_model(model, test_loader, device=device)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise