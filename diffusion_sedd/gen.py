import sys, math, torch, tiktoken
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
decode = lambda l: enc.decode(l)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size, n_embed, n_layers, n_heads, drop_rate = 128, 512, 6, 16, 0.0
lam_min, lam_max = 1e-3, 8.0
log_ratio = torch.log(torch.tensor(lam_max / lam_min))

def noise(t):
    lam = lam_min * (lam_max / lam_min) ** t
    return lam, lam * log_ratio.to(t.device)

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.n_heads = num_heads
        head_dim = n_embed // num_heads
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.dropout = nn.Dropout(drop_rate)
        half = head_dim // 2
        freqs = 1.0 / (1000 ** (torch.arange(half).float() / half))
        ang = torch.outer(torch.arange(block_size).float(), freqs)
        self.register_buffer('rope_cos', torch.cos(ang), persistent=False)
        self.register_buffer('rope_sin', torch.sin(ang), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embed, dim=2)
        q = self.q_norm(q.view(B, T, self.n_heads, C // self.n_heads)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.n_heads, C // self.n_heads)).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        cos, sin = self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_rate if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), nn.Dropout(drop_rate))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.adaLN = nn.Linear(n_embed, 4 * n_embed)
    def forward(self, x, temb):
        s1, c1, s2, c2 = self.adaLN(F.silu(temb))[:, None, :].chunk(4, dim=-1)
        x = x + self.sa(self.ln1(x) * (1 + c1) + s1)
        x = x + self.ffwd(self.ln2(x) * (1 + c2) + s2)
        return x

class BLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, num_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.time_mlp = nn.Sequential(nn.Linear(n_embed, n_embed), nn.SiLU(), nn.Linear(n_embed, n_embed))
    def forward(self, idx, t):
        temb = self.time_mlp(timestep_embedding(t, n_embed))
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x, temb)
        return self.lm_head(self.ln(x))

    @torch.no_grad()
    def generate(self, n_samples, steps=128):
        N = vocab_size
        x = torch.randint(N, (n_samples, block_size), device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(n_samples)
            dt = ts[i] - ts[i + 1]
            _, sigma = noise(t)
            s = torch.exp(self(x, t).clamp(max=20))
            rate = sigma[:, None, None] / N * s
            rate.scatter_(-1, x[..., None], 0.0)
            probs = (rate * dt).clamp(0, 1)
            stay = (1 - probs.sum(-1, keepdim=True)).clamp_min(0)
            probs.scatter_(-1, x[..., None], stay)
            x = torch.multinomial(probs.view(-1, N), 1).view(n_samples, block_size)
        return x

torch.manual_seed(0)
model = BLM().to(device)
ck = torch.load('ckpt_full.pt', map_location=device, weights_only=False)
model.load_state_dict(ck['model'])
print(f"loaded ckpt @ step {ck['epoch']}")
model.eval()

for steps in (128, 256):
    print(f"\n{'='*70}\n{steps} denoising steps\n{'='*70}")
    out = model.generate(n_samples=6, steps=steps)
    for j, row in enumerate(out.tolist()):
        print(f"\n--- sample {j+1} ({steps} steps) ---")
        print(decode(row))
