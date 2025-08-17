import math, torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):  # [B,T,C]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):  # hidden ≈ 2–3× d_model (not 4×)
        super().__init__()
        self.w12 = nn.Linear(d_model, 2*hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)
    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.proj(torch.nn.functional.silu(a) * b)

def apply_rope(q, k, theta=5e5):
    # q,k: [B,H,T,D] (D even). Broadcast cos/sin over B,H.
    B, H, T, D = q.shape
    half = D // 2
    device, dtype = q.device, q.dtype
    freqs = torch.arange(half, device=device, dtype=dtype)
    inv_freq = theta ** (-freqs / half)                 # [half]
    pos = torch.arange(T, device=device, dtype=dtype)[:, None]  # [T,1]
    ang = pos * inv_freq[None, :]                       # [T,half]
    cos = ang.cos()[None, None, :, :]                   # [1,1,T,half]
    sin = ang.sin()[None, None, :, :]
    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rot(q), rot(k)

def causal_mask(T, device, dtype):
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None, :, :]  # [1,1,T,T]

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, rope_theta=5e5, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.hk = n_kv_heads or n_heads   # GQA if hk < h; MQA if hk=1
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, self.h*self.dh, bias=False)
        self.k = nn.Linear(d_model, self.hk*self.dh, bias=False)
        self.v = nn.Linear(d_model, self.hk*self.dh, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.theta = rope_theta

    def forward(self, x, attn_mask=None):  # x: [B,T,C]
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h,  self.dh).transpose(1, 2)  # [B,H,T,D]
        k = self.k(x).view(B, T, self.hk, self.dh).transpose(1, 2)  # [B,HK,T,D]
        v = self.v(x).view(B, T, self.hk, self.dh).transpose(1, 2)  # [B,HK,T,D]

        # RoPE on Q/K (no learned pos emb needed)
        q, k = apply_rope(q, k, self.theta)

        # GQA: broadcast K/V groups to all Q heads
        if self.hk != self.h:
            repeat = self.h // self.hk
            k = k.repeat_interleave(repeat, dim=1)  # [B,H,T,D]
            v = v.repeat_interleave(repeat, dim=1)  # [B,H,T,D]

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)        # [B,H,T,T]
        if attn_mask is not None:
            att = att + attn_mask                                   # mask shape [1,1,T,T]
        w = att.softmax(dim=-1)
        w = self.drop(w)
        y = w @ v                                                   # [B,H,T,D]
        y = y.transpose(1, 2).reshape(B, T, C)                      # [B,T,C]
        return self.o(y)

class LLaMAStyleBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, ff_hidden=None, rope_theta=5e5, dropout=0.1):
        super().__init__()
        ff_hidden = ff_hidden or int(2.5 * d_model)  # common LLaMA-ish width
        self.norm1 = RMSNorm(d_model)
        self.attn  = GQAAttention(d_model, n_heads, n_kv_heads, rope_theta, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp   = SwiGLU(d_model, ff_hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))  # pre-norm
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x