import math, time, json, statistics as st
import torch, tiktoken
import torch.nn as nn
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
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
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
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

def sync():
    if device == 'cuda':
        torch.cuda.synchronize()

@torch.no_grad()
def denoise_step(model, x, ts, i, n):
    t = ts[i].expand(n)
    dt = ts[i] - ts[i + 1]
    _, sigma = noise(t)
    s = torch.exp(model(x, t).clamp(max=20))
    rate = sigma[:, None, None] / vocab_size * s
    rate.scatter_(-1, x[..., None], 0.0)
    probs = (rate * dt).clamp(0, 1)
    stay = (1 - probs.sum(-1, keepdim=True)).clamp_min(0)
    probs.scatter_(-1, x[..., None], stay)
    return torch.multinomial(probs.view(-1, vocab_size), 1).view(n, block_size)

@torch.no_grad()
def timed_generate(model, n_samples, steps, track_mem=True):
    if track_mem and device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    x = torch.randint(vocab_size, (n_samples, block_size), device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    # prefill = first denoising step
    sync(); a = time.perf_counter()
    x = denoise_step(model, x, ts, 0, n_samples)
    sync(); prefill = (time.perf_counter() - a) * 1000
    # decode = remaining steps
    sync(); a = time.perf_counter()
    for i in range(1, steps):
        x = denoise_step(model, x, ts, i, n_samples)
    sync(); decode = (time.perf_counter() - a) * 1000
    peak_a = torch.cuda.max_memory_allocated() / 1e6 if device == 'cuda' else 0.0
    peak_r = torch.cuda.max_memory_reserved() / 1e6 if device == 'cuda' else 0.0
    return prefill, decode, peak_a, peak_r

N_RUNS, N_SAMPLES, STEPS = 20, 1, 128

torch.manual_seed(0)
model = BLM().to(device)
ck = torch.load('ckpt_full.pt', map_location=device, weights_only=False)
model.load_state_dict(ck['model'])
model.eval()
param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

# cold-start run (includes one-time CUDA kernel init), then 2 warmups
cold_pf, cold_dc, _, _ = timed_generate(model, N_SAMPLES, STEPS, track_mem=False)
cold_total = cold_pf + cold_dc
for _ in range(2):
    timed_generate(model, N_SAMPLES, STEPS, track_mem=False)

runs = []
for _ in range(N_RUNS):
    pf, dc, pa, pr = timed_generate(model, N_SAMPLES, STEPS)
    runs.append({"prefill_ms": pf, "decode_ms": dc, "total_ms": pf + dc,
                 "decode_per_step_ms": dc / (STEPS - 1),
                 "peak_mem_alloc_mb": pa, "peak_mem_reserved_mb": pr})

def summ(key):
    v = [r[key] for r in runs]
    return {"mean": st.mean(v), "std": st.pstdev(v), "min": min(v), "max": max(v)}

out = {
    "config": {"n_samples": N_SAMPLES, "denoising_steps": STEPS, "block_size": block_size,
               "device": device, "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else "cpu",
               "checkpoint_step": int(ck['epoch']), "model_param_mem_mb": round(param_mem, 1)},
    "note": ("Non-autoregressive diffusion: no prompt/KV-cache. Generation is `denoising_steps` "
             "full-sequence bidirectional forward passes. 'prefill' = first denoising step, "
             "'decode' = remaining steps (each the same cost as prefill here)."),
    "cold_start_total_ms": cold_total,
    "n_runs": N_RUNS,
    "summary": {k: summ(k) for k in
                ["prefill_ms", "decode_ms", "decode_per_step_ms", "total_ms",
                 "peak_mem_alloc_mb", "peak_mem_reserved_mb"]},
    "runs": runs,
}
json.dump(out, open('gen_timing.json', 'w'), indent=2)

s = out["summary"]
print(f"gpu: {out['config']['gpu']} | model params: {param_mem:.0f} MB")
print(f"cold-start total: {cold_total:.1f} ms")
print(f"prefill (1 step):   {s['prefill_ms']['mean']:.2f} +/- {s['prefill_ms']['std']:.2f} ms")
print(f"decode ({STEPS-1} steps): {s['decode_ms']['mean']:.1f} +/- {s['decode_ms']['std']:.1f} ms "
      f"({s['decode_per_step_ms']['mean']:.2f} ms/step)")
print(f"total  ({STEPS} steps): {s['total_ms']['mean']:.1f} +/- {s['total_ms']['std']:.1f} ms")
print(f"peak mem alloc: {s['peak_mem_alloc_mb']['mean']:.0f} MB | reserved: {s['peak_mem_reserved_mb']['mean']:.0f} MB")
print("saved gen_timing.json")
