import math, torch, torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_alko import AlkoConfig

# -------- Norms --------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def make_norm(kind, d):  # "rmsnorm" or "layernorm"
    return RMSNorm(d) if kind == "rmsnorm" else nn.LayerNorm(d, eps=1e-5)

# -------- RoPE (simple demo) --------
def apply_rope(q, k, theta):
    # q,k: [B,H,T,D] — compute in float32 for stability
    B, H, T, D = q.shape
    half = D // 2
    device = q.device

    # Use float32 for all RoPE computations to ensure stability
    freqs = torch.arange(half, device=device, dtype=torch.float32)
    # Use math.log(theta) formulation for stability
    log_theta = math.log(theta)
    inv_freq = torch.exp(-log_theta * freqs / half)  # [half]
    
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]  # [T,1]
    ang = pos * inv_freq[None, :]  # [T,half]
    cos = ang.cos()[None, None, :, :]  # [1,1,T,half]
    sin = ang.sin()[None, None, :, :]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rot(q), rot(k)  # keep float32; we'll cast later

# -------- MLPs --------
class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w12 = nn.Linear(d, 2*hidden, bias=False)
        self.out = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.out(torch.nn.functional.silu(a) * b)

class GeluMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down(self.act(self.up(x)))

def make_mlp(kind, d, hidden):  # "swiglu" or "gelu"
    return SwiGLU(d, hidden) if kind == "swiglu" else GeluMLP(d, hidden)




# -------- Attention (supports GQA) --------
class AlkoAttention(nn.Module):
    def __init__(self, cfg: AlkoConfig):
        super().__init__()
        self.h = cfg.num_attention_heads
        self.dh = cfg.hidden_size // self.h
        self.hk = cfg.num_key_value_heads
        self.use_rope = (cfg.pos_type == "rope")
        self.theta = cfg.rope_theta
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,D]
        k = self.k(x).view(B, T, self.hk, self.dh).transpose(1, 2)  # [B,HK,T,D]
        v = self.v(x).view(B, T, self.hk, self.dh).transpose(1, 2)  # [B,HK,T,D]

        # Cast to float32 for all attention math to ensure stability
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        # RoPE - apply in float32 for stability
        if self.use_rope:
            q, k = apply_rope(q, k, self.theta)

        # GQA broadcast
        if self.hk != self.h:
            rep = self.h // self.hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Attention computation in float32 for numerical stability
        att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.dh))  # [B,H,T,T]
        
        # Apply causal mask in float32
        if attn_mask is not None:
            # Ensure mask is in float32 and properly broadcasted
            m = attn_mask.to(torch.float32)
            att = att + m

        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        y = torch.matmul(att, v)  # [B,H,T,D] - still in float32
        
        # Cast back to model's native dtype at the end
        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, self.h * self.dh)
        return self.o(y)

# -------- Block --------
class AlkoBlock(nn.Module):
    def __init__(self, cfg: AlkoConfig):
        super().__init__()
        d, ff = cfg.hidden_size, cfg.intermediate_size
        self.norm1 = make_norm(cfg.norm_type, d)
        self.attn  = AlkoAttention(cfg)
        self.norm2 = make_norm(cfg.norm_type, d)
        self.mlp   = make_mlp(cfg.mlp_type, d, ff)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask):
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

# -------- Model --------
class AlkoLLM(PreTrainedModel):  # a.k.a. AlkoForCausalLM
    config_class = AlkoConfig

    def __init__(self, cfg: AlkoConfig):
        super().__init__(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.use_learned_pos = (cfg.pos_type == "learned")
        if self.use_learned_pos:
            self.pos = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([AlkoBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm_f = make_norm(cfg.norm_type, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.config.tie_word_embeddings = True
        
        # Initialize weights more conservatively
        self._init_weights_apply()
        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # Much more conservative initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Reduced from 0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            # Much more conservative initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Reduced from 0.02
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif hasattr(module, "weight") and module.__class__.__name__ == "RMSNorm":
            torch.nn.init.ones_(module.weight)

    def _init_weights_apply(self):
        """Apply weight initialization to all modules"""
        self.apply(self._init_weights)

    def _causal_mask(self, T, device):
        # Always create mask in float32 for numerical stability
        m = torch.full((T, T), float("-inf"), device=device, dtype=torch.float32)
        return torch.triu(m, diagonal=1)[None, None, :, :]  # [1,1,T,T]

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, new_embeddings):
        self.embed = new_embeddings
        self.config.vocab_size = new_embeddings.num_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def can_generate(self):  # transformers will call this
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        return_dict: bool = True,
        **kwargs
    ):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        
        # Safety check: clip extreme embedding values
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf in embeddings, clipping")
            x = torch.clamp(x, min=-10.0, max=10.0)
        
        if self.use_learned_pos:
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.pos(pos)[None, :, :]

        # Use float32 mask to prevent numerical instability
        attn_mask = self._causal_mask(T, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)
            # Safety check: clip extreme values after each block
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("WARNING: NaN/Inf after block, clipping")
                x = torch.clamp(x, min=-10.0, max=10.0)

        x = self.norm_f(x)
        logits = self.lm_head(self.dropout(x))
        
        # Safety check: clip extreme logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN/Inf in logits, clipping")
            logits = torch.clamp(logits, min=-100.0, max=100.0)

        loss = None 
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
            
            # Safety check: clip extreme loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {loss}, replacing with safe value")
                loss = torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
            elif loss > 100.0:
                print(f"WARNING: Loss {loss} too high, clipping to 100")
                loss = torch.clamp(loss, max=100.0)

        if not return_dict:
            return (logits,) if loss is None else (loss, logits)

        return CausalLMOutputWithPast(loss=loss, logits=logits)
