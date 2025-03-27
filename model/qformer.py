import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel, BertModel
from infonce import SupervisedInfoNCE, InfoNCE
import torch.nn.functional as F

class Qformer(nn.Module):
    """
    Qformer module that processes visual features and combines them with text features
    for emotion recognition in context, with added Image-Text Contrastive Loss (ITC).
    """
    def __init__(
        self,
        vision_width=768,       # Visual feature dimension
        text_width=768,         # Text feature dimension
        num_query_tokens=32,    # Number of query tokens
        cross_attention_freq=2,
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_intermediate_size=3072,
        temperature=0.07,       # Temperature parameter for contrastive loss
    ):
        super().__init__()

        # Initialize Qformer configuration
        self.qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        self.qformer_config.encoder_width = vision_width
        self.qformer_config.add_cross_attention = True
        self.qformer_config.is_decoder = True
        self.qformer_config.cross_attention_freq = cross_attention_freq
        self.qformer_config.query_length = num_query_tokens
        self.qformer_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        self.qformer_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        self.qformer_config.intermediate_size = qformer_intermediate_size

        # Initialize BERT-based Qformer
        self.qformer = BertModel(self.qformer_config)

        # Query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.qformer_config.initializer_range)

        # Projection layers
        self.vision_proj = nn.Linear(vision_width, self.qformer_config.hidden_size) if vision_width != self.qformer_config.hidden_size else nn.Identity()
        self.text_proj = nn.Linear(text_width, self.qformer_config.hidden_size) if text_width != self.qformer_config.hidden_size else nn.Identity()

        # Cross-modal transformations
        self.cross_modal_text_transform = nn.Linear(self.qformer_config.hidden_size, self.qformer_config.hidden_size)
        self.cross_modal_image_transform = nn.Linear(self.qformer_config.hidden_size, self.qformer_config.hidden_size)

        # Normalization, pooling, and classification
        self.norm = nn.LayerNorm(self.qformer_config.hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.qformer_config.hidden_size, 26)  # Assuming 26 emotion classes

        # Contrastive loss parameters
        self.temp = nn.Parameter(torch.tensor(temperature))  # Learnable temperature

    def forward(self, image_features, text_features=None, text_attention_mask=None):
        """
        Process image and text features through Qformer with contrastive loss.

        Args:
            image_features: Visual features [batch_size, seq_len_i, vision_width]
            text_features: Text features [batch_size, seq_len_t, text_width]
            text_attention_mask: Attention mask for text [batch_size, seq_len_t]

        Returns:
            classification_output: Output for emotion classification
            combined_query: Combined query output
            loss_itc: Image-Text Contrastive Loss (if text_features is provided)
        """
        batch_size = image_features.shape[0]

        # Project image features
        image_features = self.vision_proj(image_features)

        # Expand query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Self-attention with image features
        outputs = self.qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            return_dict=True,
            output_hidden_states=True,
        )
        image_query_output = outputs.hidden_states[-1]  # [batch_size, num_query_tokens, hidden_size]
        query_image = self.cross_modal_image_transform(image_query_output)

        # Initialize loss_itc
        loss_itc = None

        # Process text if available
        if text_features is not None:
            # Project text features
            text_features = self.text_proj(text_features)

            # Cross-attention with text
            text_outputs = self.qformer(
                inputs_embeds=image_query_output,
                encoder_hidden_states=text_features,
                encoder_attention_mask=text_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            text_query_output = text_outputs.hidden_states[-1]  # [batch_size, num_query_tokens, hidden_size]
            query_text = self.cross_modal_text_transform(text_query_output)

            # Residual connections
            query_image = query_image + image_query_output
            query_text = query_text + text_query_output

            # Combine pathways
            combined_query = torch.cat([query_image, query_text], dim=1)

            # Contrastive Loss Calculation
            # Aggregate query tokens by taking the mean
            image_feats = query_image.mean(dim=1)  # [batch_size, hidden_size]
            text_feats = query_text.mean(dim=1)    # [batch_size, hidden_size]

            # Normalize features
            image_feats = F.normalize(image_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)

            # Compute similarity
            sim_i2t = torch.matmul(image_feats, text_feats.T) / self.temp  # [batch_size, batch_size]
            sim_t2i = sim_i2t.T  # Symmetric: [batch_size, batch_size]

            # Targets: diagonal elements are positive pairs
            targets = torch.arange(batch_size).to(image_feats.device)

            # Compute ITC loss
            loss_i2t = F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            loss_t2i = F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            loss_itc = (loss_i2t + loss_t2i) / 2

        else:
            # Only image pathway
            combined_query = query_image

        # Pooling and classification
        combined_query = combined_query.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        x = self.avgpool(combined_query)  # [batch_size, hidden_size, 1]
        x = nn.ReLU()(self.flatten(x))    # [batch_size, hidden_size]
        x = nn.Dropout(0.2)(x)
        classification_output = self.fc(x)  # [batch_size, 26]

        return classification_output, combined_query, loss_itc

# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo mô hình
    qformer = Qformer()

    # Dữ liệu giả lập
    batch_size = 4
    seq_len_i = 50
    seq_len_t = 20
    image_features = torch.randn(batch_size, seq_len_i, 768)
    text_features = torch.randn(batch_size, seq_len_t, 768)
    text_attention_mask = torch.ones(batch_size, seq_len_t)

    # Forward pass
    cls_output, combined_query, loss_itc = qformer(image_features, text_features, text_attention_mask)

    print("Classification Output:", cls_output.shape)  # [batch_size, 26]
    print("Combined Query:", combined_query.shape)     # [batch_size, hidden_size, seq_len]
    print("ITC Loss:", loss_itc.item() if loss_itc is not None else "N/A")