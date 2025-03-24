import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel, BertTokenizer

class Qformer(nn.Module):
    def __init__(
        self,
        num_query_tokens=32,
        cross_attention_freq=32,
        embed_dim=768,
        qformer_hidden_dropout_prob=0.5,
        qformer_attention_probs_dropout_prob=0.2,
        qformer_intermediate_size=3072,
    ):
        super().__init__()
        
        # Initialize the Qformer (BERT-based model with cross-attention)
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, embed_dim))
        self.qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        self.qformer_config.encoder_width = embed_dim
        
        # Add cross-attention layers
        self.qformer_config.add_cross_attention = True
        self.qformer_config.is_decoder = True
        self.qformer_config.cross_attention_freq = cross_attention_freq
        self.qformer_config.query_length = num_query_tokens
        
        # Adjust dropout rates
        self.qformer_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        self.qformer_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        self.qformer_config.intermediate_size = qformer_intermediate_size
        
        # Initialize BERT model with LM head for Qformer
        self.qformer = BertLMHeadModel(self.qformer_config)
        
        # Initialize query tokens with random values
        nn.init.normal_(self.query_tokens, std=0.02)
        
    def forward(self, image_embeds, text_embeds=None, text_attention_mask=None):
        """
        Args:
            image_embeds (torch.Tensor): Output từ vision encoder [batch_size, seq_len_i, embed_dim]
            text_embeds (torch.Tensor, optional): Output từ text encoder [batch_size, seq_len_t, embed_dim]
            text_attention_mask (torch.Tensor, optional): Attention mask cho text [batch_size, seq_len_t]
        """
        batch_size = image_embeds.shape[0]
        
        # Repeat query tokens for each example in batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Prepare inputs for Qformer
        outputs = self.qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            return_dict=True,
            output_hidden_states=True,  # Đảm bảo hidden_states được trả về
        )
        
        # Sửa lỗi: BertLMHeadModel trả về CausalLMOutputWithCrossAttentions
        # nên chúng ta cần truy cập hidden_states thay vì last_hidden_state
        query_output = outputs.hidden_states[-1]  # Lấy hidden state của layer cuối cùng
        
        # Nếu có text, thì thực hiện cross-attention với text
        if text_embeds is not None:
            # Cross-attention với text
            text_outputs = self.qformer(
                inputs_embeds=query_output,
                encoder_hidden_states=text_embeds,
                encoder_attention_mask=text_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # Final query features sau khi qua cả image và text
            query_output = text_outputs.hidden_states[-1]
            
        return query_output