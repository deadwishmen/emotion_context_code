import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel, BertModel

class Qformer(nn.Module):
    """
    Qformer module that processes visual features and combines them with text features
    for emotion recognition in context.
    """
    def __init__(
        self,
        vision_width=768,       # Sửa thành 768 dựa trên lỗi
        text_width=768,         # Language model embedding dimension
        num_query_tokens=32,    # Number of query tokens
        cross_attention_freq=2,
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_intermediate_size=3072,
    ):
        super().__init__()
        

        
        # Initialize query tokens (Q in the diagram)
        # self.num_query_tokens = num_query_tokens
        # self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, vision_width))
        # nn.init.normal_(self.query_tokens, std=0.02)
        
        # Initialize Qformer configuration
        self.qformer_config = BertConfig.from_pretrained("bert-base-uncased") 
        self.qformer_config.encoder_width = vision_width
        
        # Set up cross-attention
        self.qformer_config.add_cross_attention = True
        self.qformer_config.is_decoder = True
        self.qformer_config.cross_attention_freq = cross_attention_freq
        self.qformer_config.query_length = num_query_tokens
        
        # Adjust dropout parameters
        self.qformer_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        self.qformer_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        self.qformer_config.intermediate_size = qformer_intermediate_size
        
        # Initialize BERT model with LM head for Qformer
        # self.qformer = BertLMHeadModel(self.qformer_config)
        self.qformer = BertModel(self.qformer_config)
        

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.qformer_config.initializer_range)


        # Projection layers shown in the diagram - Điều chỉnh kích thước đầu vào
        # Chỉ cần projection nếu kích thước khác với hidden_size của BERT
        if vision_width != self.qformer_config.hidden_size:
            self.vision_proj = nn.Linear(vision_width, self.qformer_config.hidden_size)
        else:
            self.vision_proj = nn.Identity()
            
        if text_width != self.qformer_config.hidden_size:
            self.text_proj = nn.Linear(text_width, self.qformer_config.hidden_size)
        else:
            self.text_proj = nn.Identity()
        
        # Cross-modal feed forward projections
        self.cross_modal_text_transform = nn.Linear(self.qformer_config.hidden_size, self.qformer_config.hidden_size)
        self.cross_modal_image_transform = nn.Linear(self.qformer_config.hidden_size, self.qformer_config.hidden_size)
        
        self.norm = nn.LayerNorm(self.qformer_config.hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        # Final output projection for emotion recognition
        self.fc = nn.Linear(self.qformer_config.hidden_size, 26)  # Assuming 7 emotion classes
        
    def forward(self, image_features, text_features=None, text_attention_mask=None):
        """
        Process image and text features through Qformer
        
        Args:
            image_features: Visual features [batch_size, seq_len_i, vision_width]
            text_features: Text features [batch_size, seq_len_t, text_width]
            text_attention_mask: Attention mask for text [batch_size, seq_len_t]
        """
        batch_size = image_features.shape[0]
        
        # Project image features if needed
        image_features = self.vision_proj(image_features)
        
        # Expand query tokens to batch dimension
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Self-attention stage
        outputs = self.qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,  # No mask for image features
            return_dict=True,
            output_hidden_states=True,
        )
        
        # Get query output from hidden states
        image_query_output = outputs.hidden_states[-1]
        
        # Apply cross-modal transform for image pathway
        query_image = self.cross_modal_image_transform(image_query_output)
        
        # Process with text if available
        if text_features is not None:
            # Project text features if needed
            text_features = self.text_proj(text_features)
            
            # Cross-attention with text
            text_outputs = self.qformer(
                inputs_embeds=image_query_output,  # Use output from previous stage
                encoder_hidden_states=text_features,
                encoder_attention_mask=text_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # Get text query output
            text_query_output = text_outputs.hidden_states[-1]
            
            # Apply cross-modal transform for text pathway
            query_text = self.cross_modal_text_transform(text_query_output)
            

            query_image = query_image + image_query_output  # Residual on image pathway
            query_text = query_text + text_query_output    # Residual on text pathway
            # Combine both pathways
            combined_query = torch.cat([query_image, query_text], dim = 1)
        else:
            # Only use image pathway if no text is provided
            combined_query = query_image
        combined_query = combined_query.transpose(1, 2)
        
        # Pooling and classification
        x = self.avgpool(combined_query)
        x = nn.ReLU()(self.flatten(x))
        x = nn.Dropout(0.2)(x)
        # Final classification
        emotion_logits = self.fc(x)
        
        return emotion_logits, combined_query