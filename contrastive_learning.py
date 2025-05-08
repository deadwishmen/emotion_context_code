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
logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm tải processor và tokenizer
def load_hf_resources(model_name="openai/clip-vit-base-patch32", cache_dir=None):
    try:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1", cache_dir=cache_dir)
        print("Loaded processor and tokenizer successfully.")
        return processor, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Hugging Face resources: {e}")
        raise

processor, tokenizer = load_hf_resources()

# Lớp Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Hàm mã hóa văn bản
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

# Lớp Dataset tùy chỉnh
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
            context = self.x_context[index]
            if len(context.shape) != 3 or context.shape[2] not in [3, 4]:
                logging.warning(f"Invalid context shape at index {index}: {context.shape}")
                raise ValueError(f"Expected context shape [H, W, 3] or [3, H, W], got {context.shape}")
            if context.shape[0] == 3:
                context = context.transpose(1, 2, 0)
            processed_context = processor(images=context, return_tensors="pt")['pixel_values'][0]

            body = self.x_body[index]
            if body.shape[0] == 3:
                body = body.transpose(1, 2, 0)
            processed_body = processor(images=body, return_tensors="pt")['pixel_values'][0]

            token_text = self.x_text[index]
            cat_label = torch.tensor(self.y_cat[index], dtype=torch.float32)
            return processed_context, processed_body, token_text, cat_label, context
        except Exception as e:
            logging.error(f"Error processing dataset item {index}: {e}")
            raise

# Hàm tải dữ liệu
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
        
        train_context = np.load(os.path.join(data_src, 'train_context_arr.npy'))
        val_context = np.load(os.path.join(data_src, 'val_context_arr.npy'))
        test_context = np.load(os.path.join(data_src, 'test_context_arr.npy'))

        train_body = np.load(os.path.join(data_src, 'train_body_arr.npy'))
        val_body = np.load(os.path.join(data_src, 'val_body_arr.npy'))
        test_body = np.load(os.path.join(data_src, 'test_body_arr.npy'))

        train_cat = np.load(os.path.join(data_src, 'train_cat_arr.npy'))
        val_cat = np.load(os.path.join(data_src, 'val_cat_arr.npy'))
        test_cat = np.load(os.path.join(data_src, 'test_cat_arr.npy'))

        train_text = tokenize_text(os.path.join(data_src, 'train.csv'))
        val_text = tokenize_text(os.path.join(data_src, 'val.csv'))
        test_text = tokenize_text(os.path.join(data_src, 'test.csv'))

        train_dataset = EmoticDataset(train_context, train_body, train_text, train_cat, device=device)
        val_dataset = EmoticDataset(val_context, val_body, val_text, val_cat, device=device)
        test_dataset = EmoticDataset(test_context, test_body, test_text, test_cat, device=device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("Data loaders created successfully.")
        return train_loader, val_loader, test_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Hàm tính mean Average Precision
def calculate_map(predictions, labels):
    ap_scores = []
    for i in range(labels.shape[1]):
        if np.any(labels[:, i]):
            ap = average_precision_score(labels[:, i], predictions[:, i])
            ap_scores.append(ap)
        else:
            logging.warning(f"Skipping AP for class {i}: no positive labels.")
    return np.mean(ap_scores) if ap_scores else 0.0

# Bayesian Linear Layer cải tiến
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('prior_mu', torch.tensor(prior_mu))
        self.register_buffer('prior_sigma', torch.tensor(prior_sigma))

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity='linear')
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    def _softplus(self, x):
        return torch.log1p(torch.exp(x))

    def _sample(self, mu, rho):
        sigma = self._softplus(rho)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps, sigma

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight_mu, self.bias_mu)

        weight, weight_sigma = self._sample(self.weight_mu, self.weight_rho)
        bias, bias_sigma = self._sample(self.bias_mu, self.bias_rho)

        out = F.linear(x, weight, bias)

        self.kl = self._kl_divergence(self.weight_mu, weight_sigma) \
                  + self._kl_divergence(self.bias_mu, bias_sigma)

        return out

    def _kl_divergence(self, mu_q, sigma_q):
        mu_p = self.prior_mu
        sigma_p = self.prior_sigma

        term1 = torch.log(sigma_p / sigma_q)
        term2 = (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
        kl = term1 + term2 - 0.5
        return kl.sum()

    def kl_divergence(self):
        return self.kl

# Mô hình chính
class CLIPEmoticModel(nn.Module):
    def __init__(self, clip_pretrained="openai/clip-vit-base-patch32", num_cat=26, hidden_dim=512, prior_sigma=1.0):
        super(CLIPEmoticModel, self).__init__()
        try:
            self.clip = CLIPModel.from_pretrained(clip_pretrained, local_files_only=os.getenv("TRANSFORMERS_OFFLINE", "0") == "1")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise
        self.clip_dim = self.clip.config.projection_dim
        self.context_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma=prior_sigma)
        self.body_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma=prior_sigma)
        self.text_proj = BayesianLinear(self.clip_dim, hidden_dim, prior_sigma=prior_sigma)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.emotion_pred = BayesianLinear(hidden_dim, num_cat, prior_sigma=prior_sigma)

    def forward(self, context_images, body_images, text_tokens):
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
        fused_features = F.relu(self.fusion(combined_features))
        cat_pred = self.emotion_pred(fused_features)
        return cat_pred

    def kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl

# Hàm huấn luyện
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=2e-4, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    cat_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    lambda_kl = 1e-5

    best_val_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_kl_loss = 0.0

        for context_images, body_images, text_tokens, cat_labels, _ in train_loader:
            context_images = context_images.to(device)
            body_images = body_images.to(device)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            cat_labels = cat_labels.to(device)

            optimizer.zero_grad()
            cat_pred = model(context_images, body_images, text_tokens)
            cls_loss = cat_loss_fn(cat_pred, cat_labels)
            kl_loss = model.kl_divergence()
            total_loss = cls_loss + lambda_kl * kl_loss
            train_loss += total_loss.item()
            train_cls_loss += cls_loss.item()
            train_kl_loss += kl_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for context_images, body_images, text_tokens, cat_labels, _ in val_loader:
                context_images = context_images.to(device)
                body_images = body_images.to(device)
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
                cat_labels = cat_labels.to(device)
                cat_pred = model(context_images, body_images, text_tokens)
                cls_loss = cat_loss_fn(cat_pred, cat_labels)
                kl_loss = model.kl_divergence()
                total_loss = cls_loss + lambda_kl * kl_loss
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
        print(f"Train Loss: {avg_train_loss:.4f} (Cls: {train_cls_loss/len(train_loader):.4f}, KL: {train_kl_loss/len(train_loader):.4f})")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val mAP: {val_map:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), 'best_clip_emotic_bayesian.pt')
            print(f"New best model saved with Val mAP: {best_val_map:.4f}")

# Hàm kiểm tra
def test_model(model, test_loader, device="cuda", num_samples=10):
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for context_images, body_images, text_tokens, cat_labels, _ in test_loader:
            context_images = context_images.to(device)
            body_images = body_images.to(device)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            cat_labels = cat_labels.to(device)
            pred_probs_samples = []
            for _ in range(num_samples):
                cat_pred = model(context_images, body_images, text_tokens)
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

# Hàm chính
if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_path = "/kaggle/input/emotion-torch/context_dataset/context_dataset"
        batch_size = 128
        num_epochs = 20
        print(f"Using device: {device}")
        print(f"Data path: {data_path}")
        train_loader, val_loader, test_loader = load_data(data_path, batch_size, device)
        model = CLIPEmoticModel(num_cat=26)
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
        model.load_state_dict(torch.load('best_clip_emotic_bayesian.pt', map_location=device))
        print("\nEvaluating best model on test set:")
        mean_ap = test_model(model, test_loader, device=device)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise