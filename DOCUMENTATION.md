# NanoChat: Complete Technical Documentation

> **The simplest end-to-end LLM harness.** Train a GPT-2 equivalent chatbot from scratch in ~3 hours on 8xH100 for ~$72.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [End-to-End Flow](#3-end-to-end-flow)
4. [Directory Structure](#4-directory-structure)
5. [Detailed Component Docs](#5-detailed-component-docs)
6. [The "Single Dial" Philosophy](#6-the-single-dial-philosophy)
7. [Data Flow Diagram](#7-data-flow-diagram)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. System Overview

NanoChat covers the **full lifecycle** of an LLM on a single GPU node:

```
┌─────────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│ Tokenizer   │───>│ Pretraining  │───>│  SFT    │───>│   RL    │───>│ Inference│
│ Training    │    │ (base model) │    │ (chat)  │    │ (math)  │    │ (serve)  │
└─────────────┘    └──────────────┘    └─────────┘    └─────────┘    └──────────┘
 tok_train.py       base_train.py      chat_sft.py    chat_rl.py     chat_web.py
                                                                      chat_cli.py
```

| Stage | What Happens | Data | Time (8xH100) |
|-------|-------------|------|----------------|
| **Tokenizer** | Train BPE tokenizer on FineWeb-Edu | ~2B characters | ~5 min |
| **Pretrain** | Next-token prediction on web text | FineWeb-Edu 100B | ~2.5 hours |
| **SFT** | Learn chat format & capabilities | SmolTalk, MMLU, GSM8K, SpellingBee | ~15 min |
| **RL** | Optimize math reasoning via GRPO | GSM8K (rewards) | ~15 min |
| **Inference** | Serve via web UI or CLI | - | - |

---

## 2. Architecture at a Glance

### Model (`nanochat/gpt.py`)
A modernized GPT-2/LLaMA hybrid Transformer:

```
Input Token IDs: (B, T)
        │
        ▼
┌─── Embedding ───┐    (wte: vocab → n_embd)
│   + RMSNorm     │
│   save as x0    │
└────────┬────────┘
         │
    ╔════╧════════════════════════════════════════╗
    ║  × N Transformer Blocks                     ║
    ║                                             ║
    ║  x = resid_lambda * x + x0_lambda * x0     ║
    ║         │                                   ║
    ║  ┌──────┴──────┐                            ║
    ║  │ RMSNorm     │                            ║
    ║  │ Attention   │  (RoPE, QK-Norm, GQA,     ║
    ║  │ + VE gate   │   Sliding Window, FA3)     ║
    ║  └──────┬──────┘                            ║
    ║  x = x + attn_out                           ║
    ║         │                                   ║
    ║  ┌──────┴──────┐                            ║
    ║  │ RMSNorm     │                            ║
    ║  │ MLP         │  (Linear → ReLU² → Linear) ║
    ║  └──────┬──────┘                            ║
    ║  x = x + mlp_out                            ║
    ╚════════╤════════════════════════════════════╝
         │
    ┌────┴────┐
    │ RMSNorm │
    │ LM Head │    (n_embd → vocab, untied from wte)
    │ Softcap │    (15 * tanh(logits / 15))
    └────┬────┘
         │
         ▼
Logits: (B, T, vocab_size)
```

### Key Architectural Features

| Feature | What | Why |
|---------|------|-----|
| **RoPE** | Rotary positional embeddings | Relative positions, no learned params, extrapolation |
| **QK Norm** | Normalize Q and K after RoPE | Training stability at scale |
| **GQA** | Multiple Q heads share K/V heads | Smaller KV cache, faster inference |
| **Sliding Window** | "SSSL" pattern per layer | Memory efficiency for long sequences |
| **Value Embeddings** | Mix token embeddings into V (alternating layers) | ResFormer-inspired, improves representations |
| **resid_lambdas** | Per-layer residual scaling (init 1.0) | Learnable residual connection strength |
| **x0_lambdas** | Blend initial embedding back in (init 0.1) | Gradient highway to first layer |
| **ReLU²** | Square after ReLU in MLP | Sparsity + amplification, simpler than SwiGLU |
| **Logit Softcap** | `15 * tanh(logits/15)` | Prevents extreme logits, stabilizes training |
| **No Bias** | All linear layers bias=False | Fewer parameters, cleaner optimization |
| **Zero-init projections** | `c_proj` and `mlp.c_proj` init to 0 | Blocks start as identity, gradually "turn on" |

---

## 3. End-to-End Flow

### Complete Pipeline (what `runs/speedrun.sh` does)

```bash
# Step 0: Train tokenizer (one-time)
python -m scripts.tok_train --vocab-size=32768

# Step 1: Pretrain base model
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26

# Step 2: Supervised fine-tuning
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag=d26

# Step 3: Reinforcement learning
torchrun --nproc_per_node=8 -m scripts.chat_rl --model-tag=d26

# Step 4: Evaluate
torchrun --nproc_per_node=8 -m scripts.chat_eval

# Step 5: Serve
python -m scripts.chat_web
```

### What Happens at Each Stage

#### Tokenizer Training
```
FineWeb-Edu text → [rustbpe BPE training] → merge rules → [tiktoken conversion] → tokenizer.pkl
```
Learns 32,768 token vocabulary using BPE. Exported to tiktoken for fast inference.

#### Pretraining
```
Parquet files → [DataLoader: tokenize + BOS-aligned packing] → (B, T) batches
    → [GPT forward] → cross-entropy loss → [MuonAdamW backward] → parameter update
```
Trains for ~10.5 × num_params tokens. Auto-configures batch size, LR, weight decay from depth.

#### SFT
```
Chat datasets → [TaskMixture blend] → [render_conversation: tokenize + mask] → (B, T) batches
    → [GPT forward] → masked cross-entropy (assistant tokens only) → [MuonAdamW backward]
```
Trains for 1 epoch over ~856K conversations. Only trains on assistant responses.

#### RL
```
GSM8K questions → [Engine: generate 16 samples] → [reward: correct=1, wrong=0]
    → [advantage = reward - mean] → [policy gradient loss] → [MuonAdamW backward]
```
Generates multiple solutions, reinforces correct ones, suppresses incorrect ones.

#### Inference
```
User prompt → [tokenize] → [Engine: prefill + decode with KV cache]
    → [tool use state machine: detect python blocks, execute, inject output]
    → [detokenize] → response
```

---

## 4. Directory Structure

```
nanochat/
├── nanochat/                    # Core library
│   ├── gpt.py                   # Transformer model          → docs/model.md
│   ├── optim.py                 # Muon + AdamW optimizer     → docs/optimizer.md
│   ├── engine.py                # Inference engine + KV cache → docs/engine.md
│   ├── tokenizer.py             # BPE tokenizer              → docs/tokenizer.md
│   ├── dataloader.py            # Data packing + batching    → docs/dataloader.md
│   ├── dataset.py               # Data download + iteration  → docs/dataloader.md
│   ├── flash_attention.py       # FA3/SDPA unified interface → docs/utilities.md
│   ├── execution.py             # Sandboxed code execution   → docs/utilities.md
│   ├── checkpoint_manager.py    # Save/load checkpoints      → docs/utilities.md
│   ├── common.py                # Distributed utils, logging → docs/utilities.md
│   ├── core_eval.py             # CORE benchmark evaluation  → docs/evaluation.md
│   ├── loss_eval.py             # Bits-per-byte metric       → docs/evaluation.md
│   └── report.py                # Training report generation
│
├── scripts/                     # Training & inference workflows
│   ├── base_train.py            # Pretraining                → docs/training.md
│   ├── chat_sft.py              # Supervised fine-tuning     → docs/training.md
│   ├── chat_rl.py               # Reinforcement learning     → docs/training.md
│   ├── chat_web.py              # Web UI server
│   ├── chat_cli.py              # CLI chat interface
│   ├── base_eval.py             # Base model evaluation      → docs/evaluation.md
│   ├── chat_eval.py             # Chat model evaluation      → docs/evaluation.md
│   ├── tok_train.py             # Tokenizer training         → docs/tokenizer.md
│   └── tok_eval.py              # Tokenizer evaluation       → docs/evaluation.md
│
├── tasks/                       # Evaluation & training datasets
│   ├── common.py                # TaskMixture, TaskSequence  → docs/evaluation.md
│   ├── gsm8k.py                 # Math reasoning (8K problems)
│   ├── mmlu.py                  # Multiple choice knowledge
│   ├── smoltalk.py              # General conversations
│   ├── spellingbee.py           # Letter counting / spelling
│   ├── arc.py                   # Science reasoning
│   ├── humaneval.py             # Python coding
│   └── customjson.py            # Custom JSONL datasets
│
├── runs/                        # Orchestration scripts
│   ├── speedrun.sh              # Full pipeline (tokenizer → train → SFT → RL → serve)
│   ├── miniseries.sh            # Train multiple model sizes
│   ├── scaling_laws.sh          # Scaling experiments
│   └── runcpu.sh                # CPU/MPS minimal example
│
├── tests/                       # Test suite
├── dev/                         # Development utilities
└── docs/                        # Detailed documentation (you are here)
```

---

## 5. Detailed Component Docs

Each component has its own deep-dive document covering every function, class, and design decision:

| Document | Covers | Key Topics |
|----------|--------|------------|
| [docs/model.md](docs/model.md) | `gpt.py` | GPTConfig, RoPE, attention, MLP, value embeddings, init, forward pass, generation |
| [docs/optimizer.md](docs/optimizer.md) | `optim.py` | AdamW fused step, Muon orthogonalization, Polar Express, NorMuon, distributed communication |
| [docs/engine.md](docs/engine.md) | `engine.py` | KV cache, prefill/decode, tool use state machine, batched generation |
| [docs/tokenizer.md](docs/tokenizer.md) | `tokenizer.py` | BPE, special tokens, chat templating, masking, rustbpe internals |
| [docs/dataloader.md](docs/dataloader.md) | `dataloader.py`, `dataset.py` | FineWeb-Edu, BOS-aligned best-fit packing, DDP sharding, resume |
| [docs/training.md](docs/training.md) | `base_train.py`, `chat_sft.py`, `chat_rl.py` | Scaling laws, LR schedules, gradient accumulation, GRPO, FP8 |
| [docs/evaluation.md](docs/evaluation.md) | `core_eval.py`, `loss_eval.py`, `tasks/` | BPB metric, CORE benchmark, task datasets, pass@k |
| [docs/utilities.md](docs/utilities.md) | `common.py`, `checkpoint_manager.py`, `flash_attention.py`, `execution.py` | Distributed setup, checkpointing, FA3/SDPA, sandboxed execution |

---

## 6. The "Single Dial" Philosophy

NanoChat's central design principle: **one parameter (`--depth`) controls everything**.

```
                    --depth=26
                        │
          ┌─────────────┼─────────────────────────────┐
          ▼             ▼                              ▼
    Model Config    Batch Size                   Training Horizon
    ┌──────────┐   ┌─────────────┐              ┌──────────────┐
    │dim=1664  │   │auto-computed│              │10.5 × params │
    │heads=13  │   │via Power    │              │= ~11B tokens │
    │layers=26 │   │Lines paper  │              └──────┬───────┘
    └──────────┘   │B ∝ D^0.383 │                     │
                   └──────┬──────┘                     │
                          │                            │
                    ┌─────┴──────┐              ┌──────┴───────┐
                    │LR Scaling  │              │Weight Decay  │
                    │η ∝ √(B/B₀)│              │λ∝√(B/B₀)·D₀/D│
                    └────────────┘              └──────────────┘
```

This means:
- `--depth=4`: Tiny model (~5M params) for CPU testing
- `--depth=12`: Small model (~130M params) for debugging
- `--depth=20`: Medium model (~520M params) for experimentation
- `--depth=26`: Large model (~1.1B params) for GPT-2 speedrun

All hyperparameters (model width, number of heads, batch size, learning rate, weight decay, training horizon) are **automatically derived** from depth using scaling laws calibrated at d12.

---

## 7. Data Flow Diagram

### Pretraining Data Flow

```
HuggingFace (remote)
    │
    │ download_single_file() [on demand, with retries]
    ▼
~/.cache/nanochat/base_data/shard_XXXXX.parquet
    │
    │ _document_batches() [infinite iterator, DDP-sharded]
    ▼
text_batch: ["doc1 text", "doc2 text", ...]
    │
    │ tokenizer.encode(text_batch, prepend=BOS)
    ▼
token_lists: [[BOS, 42, 17, ...], [BOS, 99, 3, ...], ...]
    │
    │ BOS-aligned best-fit packing
    ▼
row_buffer: (B, T+1) tensor on CPU
    │
    │ inputs = row[:, :-1], targets = row[:, 1:]
    │ single pin_memory → GPU transfer
    ▼
inputs: (B, T) on GPU    targets: (B, T) on GPU
```

### SFT Data Flow

```
Task Datasets (SmolTalk, MMLU, GSM8K, SpellingBee)
    │
    │ TaskMixture[idx] → conversation dict
    ▼
{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    │
    │ tokenizer.render_conversation() → (ids, mask)
    ▼
ids:  [BOS, user_start, ..., user_end, asst_start, ..., asst_end]
mask: [ 0,      0,      ...,    0,        0,       1...,   1    ]
    │
    │ BOS-aligned best-fit packing (pad, don't crop)
    ▼
inputs: (B, T)    targets: (B, T) with padding masked as -1
```

### RL Data Flow

```
GSM8K train question
    │
    │ tokenizer.render_for_completion() → prompt tokens
    ▼
[BOS, user_start, "What is 2+3?", user_end, assistant_start]
    │
    │ Engine.generate_batch(num_samples=16)
    ▼
16 completions with tool use (calculator)
    │
    │ task.reward() → 1 if correct, 0 if wrong
    ▼
advantages = rewards - mean(rewards)
    │
    │ model.forward(inputs, targets, loss_reduction='none')
    │ loss = -(logp * advantages).sum() / num_valid
    ▼
Policy gradient update
```

---

## 8. Key Design Decisions

### Why Muon instead of just AdamW?
Muon orthogonalizes weight updates, ensuring all directions in weight space are updated equally. This leads to **faster convergence** (fewer steps to reach the same loss), which is critical for the speedrun goal.

### Why BOS-aligned packing?
Every row starts with BOS, so every token can attend back to a clean document boundary. Naive concatenation creates confusing cross-document attention patterns.

### Why no DDP wrapper?
`DistMuonAdamW` handles gradient synchronization internally with ZeRO-2 style sharding. This is more memory-efficient than PyTorch DDP (which replicates the full model).

### Why ReLU² instead of SwiGLU?
Simpler (no gating, fewer params), and empirically competitive at this model scale. SwiGLU uses 50% more parameters in the MLP for the gating mechanism.

### Why untied embedding/head?
Tying forces the embedding and unembedding to share weights, which constrains the model. Untying allows them to specialize independently.

### Why logit soft-capping?
Without it, logits can grow unboundedly, leading to training instability (especially with bfloat16). The `15 * tanh(logits/15)` smoothly constrains them to [-15, 15].

### Why train a custom tokenizer?
The default GPT-2/GPT-4 tokenizers waste tokens on patterns not common in the training data. A custom tokenizer tuned on FineWeb-Edu gets better compression, meaning the model sees more content per context window.
