import math, torch, torch.nn as nn
# x: [B,T,C], C divisible by H

class GPT2MLP(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.fc = nn.Linear(d_model, mult*d_model)
        self.act = nn.GELU()
        self.proj = nn.Linear(mult*d_model, d_model)
    def forward(self, x):  # [B,T,C]
        return self.proj(self.act(self.fc(x)))

class GPT2Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):  # x: [B,T,C]
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).transpose(1, 3)  # [B,H,T,3,D]
        q, k, v = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:]                # [B,H,T,D]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)              # [B,H,T,T]
        if attn_mask is not None:
            att = att + attn_mask
        w = att.softmax(dim=-1)
        y = w @ v                                                         # [B,H,T,D]
        y = y.transpose(1,2).reshape(B, T, C)                             # [B,T,C]
        return self.o(y)

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)   # LayerNorm
        self.attn = GPT2Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, mult)
    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2Positional(nn.Module):
    def __init__(self, max_pos, d_model):
        super().__init__()
        self.pos = nn.Embedding(max_pos, d_model)  # learned absolute positions
    def forward(self, tok_emb, positions):         # positions: [T]
        return tok_emb + self.pos(positions)[None,:,:]


def causal_mask(T, device, dtype):
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None, :, :]  # [1,1,T,T]