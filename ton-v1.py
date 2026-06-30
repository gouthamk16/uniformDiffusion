import os
import sys
import math
import time
import numpy as np
import tiktoken

sys.stdout.reconfigure(encoding='utf-8') 
script_start = time.perf_counter()


def stamp(msg, t0):
    dt = time.perf_counter() - t0
    print(f"[{time.perf_counter() - script_start:8.2f}s] {msg}: {dt:.3f}s")
    return time.perf_counter()


# GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
decode = lambda l: enc.decode(l)

# Data: karpathy/tinystories-gpt4-clean, tokenized once and cached to disk.
CACHE = 'tinystories_gpt2.bin'
MAX_TOKENS = 100_000_000

def build_cache():
    from datasets import load_dataset
    t0 = time.perf_counter()
    ds = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    eot = enc.eot_token
    chunks, total, batch = [], 0, []
    for ex in ds:
        batch.append(ex['text'])
        if len(batch) == 1024:
            for ids in enc.encode_ordinary_batch(batch):
                ids.append(eot)
                chunks.append(np.array(ids, dtype=np.uint16))
                total += len(ids)
            batch = []
            if total >= MAX_TOKENS:
                break
    arr = np.concatenate(chunks)
    arr.tofile(CACHE)
    stamp(f"tokenized {len(arr):,} tokens -> {CACHE}", t0)
    return arr

t0 = time.perf_counter()
if os.path.exists(CACHE):
    arr = np.fromfile(CACHE, dtype=np.uint16)
    stamp(f"loaded cache {len(arr):,} tokens", t0)
else:
    arr = build_cache()

n = int(0.9 * len(arr))
train_data = arr[:n]
val_data = arr[n:]


import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


# Hyperparams
batch_size = 16
block_size = 128
n_embed = 512
n_layers = 8
n_heads = 8
drop_rate = 0.2
epochs = 20000
eval_interval = 1000
eval_iters = 100
lr = 3e-4           
min_lr = 3e-5        
warmup_steps = 300
gen_steps = 128
train_budget_s = 300  # fixed wall-clock training budget per run


def lr_at(step):
    # linear warmup then cosine decay to min_lr
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, epochs - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x0 = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
    return x0.to(device)


# Geometric noise schedule: cumulative noise lam(t) for t in [0, 1]
lam_min, lam_max = 1e-3, 8.0
log_ratio = torch.log(torch.tensor(lam_max / lam_min))

def noise(t):
    lam = lam_min * (lam_max / lam_min) ** t
    return lam, lam * log_ratio.to(t.device)

def corrupt(x0, t):
    lam, _ = noise(t)
    beta = torch.exp(-lam)
    replace = torch.rand(x0.shape, device=x0.device) < (1 - beta)[:, None]
    rand_tok = torch.randint(vocab_size, x0.shape, device=x0.device)
    return torch.where(replace, rand_tok, x0)


def dwdse_loss(model, x0, t):
    # Diffusion-weighted denoising score entropy (uniform graph). x0:(B,T), t:(B,)
    lam, dlam = noise(t)
    beta = torch.exp(-lam)
    a = beta + (1 - beta) / vocab_size  # stay prob
    b = (1 - beta) / vocab_size         # switch prob

    xt = corrupt(x0, t)
    log_s = model(xt, t).clamp(max=20)  # log of ratios s_theta
    s = torch.exp(log_s)                # (B, T, N)

    # target ratio R[b,i,y] = p(y|x0)/p(xt|x0); 'a/denom' at true token, 'b/denom' elsewhere
    denom = torch.where(xt == x0, a[:, None], b[:, None])      # (B, T)
    R = (b[:, None] / denom)[..., None].expand(-1, -1, vocab_size).clone()
    R.scatter_(-1, x0[..., None], (a[:, None] / denom)[..., None])

    K = R * (R.clamp_min(1e-9).log() - 1)
    term = s - R * log_s + K                # Bregman score entropy per candidate y
    term.scatter_(-1, xt[..., None], 0.0)   # drop y == xt (the current token)
    return (dlam[:, None] * term.sum(-1)).mean()


# Attention block
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        # bidirectional: no causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, n_embed),
            nn.SiLU(),
            nn.Linear(n_embed, n_embed),
        )

    def forward(self, idx, t):
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.positional_embedding_table(torch.arange(idx.shape[1], device=device))
        time_embeddings = self.time_mlp(t[:, None])
        x = token_embeddings + position_embeddings + time_embeddings[:, None, :]
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)  # log of ratios s_theta, (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, n_samples, steps=128):
        # Euler tau-leaping: start from pure noise (t=1), clean up down to t=0
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


t0 = time.perf_counter()
model = BLM().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossi = []
stamp(f"model built ({n_params/1e6:.1f}M params)", t0)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x0 = get_batch(split)
            t = torch.rand(x0.shape[0], device=device)
            losses[k] = dwdse_loss(model, x0, t).item()
        out[split] = losses.mean()
    model.train()
    return out


start_epoch = 0


# training loop
train_start = time.perf_counter()
timers = {'data': 0.0, 'loss': 0.0, 'backward': 0.0, 'step': 0.0}
n_timed = 0

for epoch in range(start_epoch, epochs):

    if time.perf_counter() - train_start > train_budget_s:
        break

    cur_lr = lr_at(epoch)
    for g in optimizer.param_groups:
        g['lr'] = cur_lr

    if epoch % eval_interval == 0:
        te = time.perf_counter()
        losses = estimate_loss()
        lossi.append(losses['val'])
        sync()
        eval_dt = time.perf_counter() - te
        if n_timed:
            per = {k: 1000 * v / n_timed for k, v in timers.items()}
            tot = sum(per.values())
            sps = 1000 / tot if tot else 0
            print(f"  timing/step: data {per['data']:.1f}ms | loss {per['loss']:.1f}ms | "
                  f"backward {per['backward']:.1f}ms | opt {per['step']:.1f}ms | "
                  f"{sps:.1f} steps/s | {sps*batch_size*block_size:,.0f} tok/s")
            timers = {k: 0.0 for k in timers}
            n_timed = 0
        mem = f" | gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == 'cuda' else ""
        print(f"Step {epoch}/{epochs} : train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| lr {cur_lr:.2e} | eval {eval_dt:.2f}s{mem}")

    sync(); t_a = time.perf_counter()
    x0 = get_batch('train')
    t = torch.rand(x0.shape[0], device=device)
    sync(); t_b = time.perf_counter()

    loss = dwdse_loss(model, x0, t)
    sync(); t_c = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    sync(); t_d = time.perf_counter()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    sync(); t_e = time.perf_counter()

    timers['data'] += t_b - t_a
    timers['loss'] += t_c - t_b
    timers['backward'] += t_d - t_c
    timers['step'] += t_e - t_d
    n_timed += 1

stamp("training done", train_start)

# final eval at end of budget so the last logged val reflects end-of-training
losses = estimate_loss()
lossi.append(losses['val'])
mem = f" | gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == 'cuda' else ""
print(f"Step {epoch}/{epochs} : train {losses['train']:.4f} | val {losses['val']:.4f} "
      f"| lr {lr_at(epoch):.2e} | eval --{mem}")

# Generate samples
tg = time.perf_counter()
sample = model.generate(n_samples=1, steps=gen_steps)
stamp(f"generation ({gen_steps} steps)", tg)
print("\n--- sample ---")
print(decode(sample[0].tolist()))

plt.plot([l.item() if torch.is_tensor(l) else l for l in lossi])
plt.savefig('loss.png')
stamp("total", script_start)
