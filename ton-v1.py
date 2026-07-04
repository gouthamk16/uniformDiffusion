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
mask_id = vocab_size          # MDLM absorbing [MASK] token (extra embedding row)
decode = lambda l: enc.decode([t for t in l if t < vocab_size])

# Data: karpathy/tinystories-gpt4-clean, tokenized once and cached to disk.
CACHE = 'tinystories_gpt2_full.bin'  # full dataset, no token cap

def build_cache():
    # stream the whole dataset to disk (low memory); atomic rename so a killed
    # build never leaves a partial cache that looks complete.
    from datasets import load_dataset
    t0 = time.perf_counter()
    ds = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    eot = enc.eot_token
    tmp, total, report, batch = CACHE + '.tmp', 0, 50_000_000, []
    with open(tmp, 'wb') as f:
        def flush(texts):
            nonlocal total
            for ids in enc.encode_ordinary_batch(texts):
                ids.append(eot)
                np.array(ids, dtype=np.uint16).tofile(f)
                total += len(ids)
        for ex in ds:
            batch.append(ex['text'])
            if len(batch) == 1024:
                flush(batch); batch = []
                if total >= report:
                    stamp(f"  tokenized {total:,} tokens...", t0); report += 50_000_000
        if batch:
            flush(batch)
    os.replace(tmp, CACHE)
    stamp(f"tokenized {total:,} tokens -> {CACHE}", t0)
    return np.fromfile(CACHE, dtype=np.uint16)

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


# Hyperparams
batch_size = 16
block_size = 128
n_embed = 512
n_layers = 6
n_heads = 16
drop_rate = 0.0
epochs = 100000
eval_interval = 1000
eval_iters = 40
lr = 3e-3
min_lr = 3e-5
warmup_steps = 100
gen_steps = 256
train_budget_s = int(os.environ.get('BUDGET', 600))  # 10-min autoresearch budget (fresh run)


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
    # MDLM absorbing forward: each token -> [MASK] independently with prob t (linear schedule)
    mask = torch.rand(x0.shape, device=x0.device) < t[:, None]
    return torch.where(mask, mask_id, x0), mask


def dwdse_loss(model, x0, t):
    # MDLM NELBO (linear schedule): (1/t)-weighted cross-entropy at masked positions only.
    xt, mask = corrupt(x0, t)
    logits = model(xt, t)                                    # (B, T, vocab)
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), x0.reshape(-1),
                         reduction='none').view(x0.shape)    # (B, T)
    per_sample = (ce * mask).sum(-1) / t.clamp_min(1e-3)     # (B,)
    return per_sample.mean()


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd); rotate halves
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def topk_sample(logits, k=20):
    # sample from the top-k tokens only (drops low-prob tail garbage)
    v = logits.topk(min(k, logits.size(-1)), dim=-1).values
    logits = logits.masked_fill(logits < v[..., -1:], float('-inf'))
    return torch.multinomial(F.softmax(logits, -1).view(-1, vocab_size), 1).view(logits.shape[:-1])


# Attention block: fused QKV + scaled_dot_product_attention (bidirectional)
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
        self.sa = MultiHeadAttention(num_heads)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.adaLN = nn.Linear(n_embed, 4 * n_embed)  # DiT-style time modulation (scale+shift)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, temb):
        s1, c1, s2, c2 = self.adaLN(F.silu(temb))[:, None, :].chunk(4, dim=-1)
        x = x + self.sa(self.ln1(x) * (1 + c1) + s1)
        x = x + self.ffwd(self.ln2(x) * (1 + c2) + s2)
        return x


class BLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size + 1, n_embed)  # +1 for [MASK]
        self.blocks = nn.ModuleList([Block(n_embed, num_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.SiLU(),
            nn.Linear(n_embed, n_embed),
        )

    def forward(self, idx, t):
        temb = self.time_mlp(timestep_embedding(t, n_embed))
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x, temb)
        x = self.ln(x)
        return self.lm_head(x)  # log of ratios s_theta, (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, n_samples, steps=128):
        # MDLM reverse process: start all [MASK], progressively unmask to predicted tokens.
        x = torch.full((n_samples, block_size), mask_id, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(n_samples)
            x0_hat = topk_sample(self(x, t))
            is_mask = x == mask_id
            unmask_p = (ts[i] - ts[i + 1]) / ts[i].clamp_min(1e-6)
            do = is_mask & (torch.rand(x.shape, device=device) < unmask_p)
            x = torch.where(do, x0_hat, x)
        is_mask = x == mask_id
        if is_mask.any():
            x0_hat = topk_sample(self(x, ts[-1].expand(n_samples)))
            x = torch.where(is_mask, x0_hat, x)
        return x


t0 = time.perf_counter()
model = BLM().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
lossi = []
stamp(f"model built ({n_params/1e6:.1f}M params)", t0)


@torch.no_grad()
def estimate_loss():
    # fixed-seed eval: identical (batch, t, corruption) draws every call, so val
    # depends only on weights and is comparable across runs. RNG state restored after.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == 'cuda' else None
    torch.manual_seed(1234)
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
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    return out


_gpt2 = None

@torch.no_grad()
def gen_quality(n_samples=8, steps=gen_steps):
    # loss-agnostic quality: generative perplexity under GPT-2 (our samples are GPT-2
    # tokens already) plus distinct-2 diversity to catch degenerate low-ppl output.
    global _gpt2
    if _gpt2 is None:
        import importlib.util  # hide the (broken) torchvision from transformers' lazy loader
        _orig = importlib.util.find_spec
        importlib.util.find_spec = lambda n, *a, **k: None if str(n).split('.')[0] == 'torchvision' else _orig(n, *a, **k)
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        _gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
    model.eval()
    x = torch.cat([model.generate(n_samples=16, steps=steps)
                   for _ in range((n_samples + 15) // 16)], 0)[:n_samples]
    model.train()
    nll, ntok = 0.0, 0
    for i in range(0, n_samples, 8):
        ids = x[i:i + 8]
        logits = _gpt2(ids).logits[:, :-1]
        tgt = ids[:, 1:]
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               reduction='sum').item()
        ntok += tgt.numel()
    ppl = math.exp(nll / ntok)
    bg, tot = set(), 0
    for r in x.tolist():
        for j in range(len(r) - 1):
            bg.add((r[j], r[j + 1])); tot += 1
    return ppl, len(bg) / max(1, tot), x


EXPCKPT = 'exp_ckpt.pt'  # per-experiment checkpoint so a kill doesn't waste training

def save_exp(epoch, acc_s):
    tmp = EXPCKPT + '.tmp'
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                'epoch': epoch, 'lossi': lossi, 'acc_s': acc_s}, tmp)
    os.replace(tmp, EXPCKPT)

start_epoch, accumulated_s = 0, 0.0
if os.path.exists(EXPCKPT):
    ck = torch.load(EXPCKPT, map_location=device)
    model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['opt'])
    start_epoch, lossi, accumulated_s = ck['epoch'] + 1, ck['lossi'], ck['acc_s']
    print(f"resumed experiment at step {start_epoch}, {accumulated_s:.0f}s trained so far")


# training loop (budget = accumulated training time across resumes)
train_start = time.perf_counter()
last_save = train_start
timers = {'data': 0.0, 'loss': 0.0, 'backward': 0.0, 'step': 0.0}
n_timed = 0

for epoch in range(start_epoch, epochs):

    if accumulated_s + (time.perf_counter() - train_start) > train_budget_s:
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
    B = x0.shape[0]
    t = (torch.arange(B, device=device) + torch.rand(B, device=device)) / B  # stratified
    sync(); t_b = time.perf_counter()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
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

    if time.perf_counter() - last_save > 30:
        save_exp(epoch, accumulated_s + (time.perf_counter() - train_start))
        last_save = time.perf_counter()

stamp("training done", train_start)

# end-of-budget metrics: GPT-2 gen-ppl (primary). val is the last periodic eval (secondary).
tg = time.perf_counter()
ppl, distinct2, samples = gen_quality()
stamp("quality eval", tg)
val = float(lossi[-1]) if lossi else float('nan')
mem = f" | gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == 'cuda' else ""
print(f"RESULT @ step {epoch} : val {val:.1f} | gen_ppl {ppl:.2f} "
      f"| distinct2 {distinct2:.3f}{mem}")
print("\n--- sample ---")
print(decode(samples[0].tolist()))

if os.path.exists(EXPCKPT):
    os.remove(EXPCKPT)  # experiment finished; next launch starts fresh

plt.plot([l.item() if torch.is_tensor(l) else l for l in lossi])
plt.savefig('loss.png')
stamp("total", script_start)
