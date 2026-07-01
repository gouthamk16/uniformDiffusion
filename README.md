# Uniform Text Diffusion Transformer

This is a small text diffusion model. It learns to write short children's stories. The first version is named `TON-v1`.

It is built on the SEDD idea (Score Entropy Discrete Diffusion, paper [2310.16834](https://arxiv.org/pdf/2310.16834)). It uses the **uniform** version, not the absorbing/mask version.

![SEDD](SEDD.png)

## What it does 

Will write up a detailed article on how the math works later. Following is a very rough idea. Normal language models write one word, then the next, then the next, left to right (autoregressive, causal attention). This model does not do that.

Instead it works like cleaning up a noisy picture:

1. Start with a line of pure random words. Total nonsense.
2. Look at the whole line at once and guess which words are wrong.
3. Replace some of the wrong words with better ones.
4. Repeat this many times.
5. After all the steps you are left with a real story.

So the model does not build a sentence from scratch. It starts with garbage and slowly fixes it until it reads like English.

## How the "uniform" part works

During training we take a real story and add noise to it. Adding noise means we randomly swap some of its words for completely random words. The more noise we add, the more words get swapped.

"Uniform" means any word can be swapped for any other word with equal chance. There is no special blank or mask token. This is the part that makes it different from the more common absorbing version.

The model is then trained to undo this swapping. It learns, for each spot in the text, how likely every other word is to be the correct one. That is what the loss function (denoising score entropy) measures.

## The model itself

- A Transformer, 6 layers, 512 hidden size, 16 attention heads. Around 77 million parameters.
- Attention is **bidirectional**, meaning every word can look at every other word, both left and right. A normal GPT can only look left.
- Positions are encoded with **RoPE** (rotary embeddings) instead of a learned position table.
- It takes the noise level as an input, fed into every block through **adaptive layer norm (adaLN)**, so each layer knows how messy the current text is.
- Vocabulary is the GPT-2 tokenizer, about 50,257 tokens. (Working on coming up with a tokenizer for this model, GPT-2 tokenizer is borderline overkill for this model)
- The layer count, head count, RoPE, adaLN, and most other choices here were found by the autoresearch loop (see below), not hand-picked.

## The data

It trains on `karpathy/tinystories-gpt4-clean`, a set of very simple short stories written for small children. The full corpus is about 540 million tokens, tokenized once and saved to `tinystories_gpt2_full.bin` so we do not have to redo it every run. (The autoresearch experiments used a 100-million-token slice for speed; the final full training run uses the whole thing.)

## Training details

- Runs on CUDA (tested on an RTX 4060 Laptop, 8 GB).
- Batch size 16, context length 128 tokens.
- Trains in bfloat16 with TF32 matmuls and a fused Adam optimizer for speed.
- Learning rate warms up for 100 steps then cools down on a cosine curve to a small final value.
- Saves a checkpoint to `ckpt_full.pt` every 1000 steps.
- If `ckpt_full.pt` already exists, the script picks up where it left off.
- It times every part of every step (data loading, loss, backward, optimizer) and prints it.

## Autoresearch

![autoresearch progress](ar_progress.png)

We didn't tune this model by hand; we let an agent do it. Following Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) idea, an autonomous loop repeatedly edits the training script, trains for a fixed 5-minute budget on an RTX 4060 laptop GPU, reads back the validation loss, and keeps the change only if the number went down (otherwise it reverts and tries something else). Over **52 experiments (22 kept)** it drove val loss from **325k to 198k, about a 39% improvement**, entirely on its own. Most of the early gains came from speed hacks that simply let the step-starved model train more in the fixed budget: TF32, fused flash-attention, a bf16 pass, and a closed-form rewrite of the score-entropy loss, followed by a run of small architecture wins (QK-norm, sinusoidal + adaLN time conditioning, 16 heads, 6 layers). The cliff near the end of the graph is the single biggest find: swapping the learned position embedding for RoPE cut the loss by ~28% in one shot. Every dead end it hit along the way (bigger models, weight tying, SwiGLU, importance sampling) is logged with a one-line reason in `results.tsv`.

And the best part: [check out some stories this model actually wrote](generated_samples.txt), produced from just ~20 minutes of total training on an RTX 4060 laptop GPU.

We also benchmarked generation on the same RTX 4060 laptop GPU: about 2.3 seconds and ~1.4 GB of peak memory to write one story (128 denoising steps). The full per-run timings (prefill, decode, and memory) are saved in [gen_timing.json](gen_timing.json).

## Files

- `ton-v1.py` is the whole thing: data, model, training, and sampling.
- `analyze_gemma.py` - code to analyze DiffusionGemma model by Google DeepMind
- `diffusionGemma_layers.json` - layers of the diffusionGemma model (only the diffusion transformer part)

## Commands

You need a venv with PyTorch (CUDA), tiktoken, datasets, numpy, and matplotlib installed.

Train (this also resumes automatically if `ckpt_full.pt` is present):

```
conda activate <env_name>
python ton-v1.py
```

Inference runs at the end of the same script: after training it starts from random noise, runs the denoising steps, and prints one generated story. The loss curve is saved to `loss.png`.

To generate stories from an already-trained checkpoint (`ckpt_full.pt`) without retraining:

```
python gen.py
```
