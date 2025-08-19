from transformers import PretrainedConfig

class AlkoConfig(PretrainedConfig):
    model_type = "alko"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,     # GQA if < num_attention_heads
        intermediate_size=2048,       # MLP width
        max_position_embeddings=2048,
        pos_type="rope",              # "rope" or "learned"
        rope_theta=5e5,
        norm_type="rmsnorm",          # "rmsnorm" or "layernorm"
        mlp_type="swiglu",            # "swiglu" or "gelu"
        dropout=0.0,
        is_encoder_decoder=False,
        is_decoder=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads or num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.pos_type = str(pos_type)
        self.rope_theta = float(rope_theta)
        self.norm_type = str(norm_type)
        self.mlp_type = str(mlp_type)
        self.dropout = float(dropout)
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder
