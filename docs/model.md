# NanoChat Component: GPT Model

**File:** `nanochat/gpt.py`

## 1. Overview

The GPT model is a modernized GPT-2/LLaMA hybrid Transformer. It takes token IDs as input and produces either logits (inference) or a loss (training).

```
Token IDs (B, T) --> [Embedding] --> [N x Transformer Blocks] --> [LM Head] --> Logits (B, T, V)
```

### Architecture Highlights
- **Rotary Positional Embeddings (RoPE)** instead of learned positional embeddings
- **QK Normalization** for training stability
- **Untied weights** for token embedding (`wte`) and language model head (`lm_head`)
- **ReLU^2 activation** in MLP (instead of GELU)
- **RMSNorm with no learnable parameters**
- **No bias** in any linear layer
- **Group Query Attention (GQA)** for efficient KV cache during inference
- **Sliding Window Attention** (configurable per-layer pattern)
- **Value Embeddings** (ResFormer-inspired) on alternating layers
- **Per-layer learnable scalars** (`resid_lambdas`, `x0_lambdas`) for residual stream control
- **Logit soft-capping** at 15 (tanh squashing)
- **Flash Attention 3** on Hopper GPUs, SDPA fallback elsewhere

---

## 2. Data Structures

### `GPTConfig` (dataclass)

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # Maximum context length
    vocab_size: int = 32768       # Number of tokens (32K)
    n_layer: int = 12             # Number of Transformer blocks (depth)
    n_head: int = 6               # Number of query heads
    n_kv_head: int = 6            # Number of key/value heads (GQA)
    n_embd: int = 768             # Model dimension (width)
    window_pattern: str = "SSSL"  # Sliding window attention pattern
```

**`window_pattern`**: A string tiled across layers. Each character controls the attention window for that layer:
- `L` = Long (full context, `sequence_len` tokens)
- `S` = Short (half context, `sequence_len / 2` tokens)
- The **final layer always gets L** regardless of pattern.

Example: `"SSSL"` with 12 layers -> `S, S, S, L, S, S, S, L, S, S, S, L`

---

## 3. Functions & Classes (in order of appearance)

### `norm(x)` (line 42)
```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```
Purely functional RMSNorm with **no learnable parameters** (no gamma/beta). Applied before attention and MLP in each block (Pre-Norm architecture).

**Why no learnable params?** Simplicity. The per-layer scalars (`resid_lambdas`, `x0_lambdas`) serve a similar purpose.

---

### `has_ve(layer_idx, n_layer)` (line 47)
```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```
Returns `True` if a layer should have a Value Embedding. Value embeddings are placed on **alternating layers**, with the **last layer always included**. This saves memory (only half the layers have VE) while ensuring the final layer benefits from it.

---

### `apply_rotary_emb(x, cos, sin)` (line 51)
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```
Applies Rotary Positional Embeddings (RoPE) to queries and keys.

**How RoPE works:**
1. Split the head dimension in half: `x1, x2`
2. Rotate pairs of dimensions using pre-computed `cos` and `sin` frequencies
3. The rotation angle depends on the **position in the sequence** (time step)
4. This gives the model relative position information: two tokens that are N positions apart will always have the same rotation difference, regardless of absolute position

**Why RoPE over learned embeddings?**
- Naturally handles sequences of any length (extrapolation)
- Encodes **relative** position (important for attention patterns)
- No learnable parameters needed

---

### `CausalSelfAttention` (line 59)

The attention mechanism. Takes input `x` and computes self-attention.

**Parameters:**
- `c_q`: Query projection (`n_embd` -> `n_head * head_dim`)
- `c_k`: Key projection (`n_embd` -> `n_kv_head * head_dim`)
- `c_v`: Value projection (`n_embd` -> `n_kv_head * head_dim`)
- `c_proj`: Output projection (`n_embd` -> `n_embd`)
- `ve_gate`: Optional gate for value embeddings (only on alternating layers)

**`forward(self, x, ve, cos_sin, window_size, kv_cache)`:**

Step-by-step:
1. **Project** input to Q, K, V: `(B, T, C)` -> `(B, T, H, D)`
2. **Value Residual** (if `ve` is provided): Mix value embeddings into V using an input-dependent gate per head. The gate uses only the first 32 channels of `x` (efficiency trick), applies sigmoid scaled to (0, 2), so at init it's ~1.0 (neutral).
3. **Apply RoPE** to Q and K
4. **QK Norm**: Normalize Q and K after RoPE for training stability
5. **Flash Attention**:
   - **Training** (`kv_cache is None`): Use `flash_attn_func` with causal mask and optional sliding window
   - **Inference** (`kv_cache` provided): Use `flash_attn_with_kvcache` which handles cache management in-place
6. **Output projection**: Reshape and project back to residual stream

**GQA (Group Query Attention):** When `n_kv_head < n_head`, multiple query heads share the same K/V head. This reduces KV cache size during inference without significantly hurting quality.

---

### `MLP` (line 121)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.c_proj(x)
        return x
```

Simple 2-layer feedforward network with **ReLU^2** activation (instead of GELU).

**Why ReLU^2?** It introduces sparsity (dead neurons) while the squaring operation amplifies strong activations. Empirically competitive with SwiGLU while being simpler (no gating mechanism, fewer parameters).

---

### `Block` (line 134)

A single Transformer block with Pre-Norm residual connections:

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
    x = x + self.mlp(norm(x))
    return x
```

Pattern: `x -> norm -> sublayer -> add` (Pre-Norm, not Post-Norm).

---

### `GPT` (line 146)

The full model. This is the main class you interact with.

#### `__init__(self, config, pad_vocab_size_to=64)`

**Important footgun:** This runs on `meta` device (shapes only, no data). All actual initialization happens in `init_weights()`.

Components created:
- `transformer.wte`: Token embedding (`padded_vocab_size` x `n_embd`)
- `transformer.h`: List of `Block` modules (the Transformer layers)
- `lm_head`: Language model head (`n_embd` x `padded_vocab_size`)
- `resid_lambdas`: Per-layer scalar that scales the residual stream (init: 1.0)
- `x0_lambdas`: Per-layer scalar that blends initial embedding back in (init: 0.1)
- `value_embeds`: Per-layer value embedding tables (alternating layers only)
- `cos`, `sin`: Pre-computed rotary embedding buffers (not saved to checkpoint)

**Vocab padding:** The vocab is padded to the nearest multiple of 64 for tensor core efficiency. The extra logits are sliced off in `forward()`.

---

#### `init_weights(self)` (line 188)

All initialization in one place for clarity:

| Component | Init | Std |
|-----------|------|-----|
| `wte` (embedding) | Normal | 1.0 |
| `lm_head` | Normal | 0.001 |
| `attn.c_q, c_k, c_v` | Uniform | `1/sqrt(n_embd)` |
| `attn.c_proj` | **Zeros** | 0 |
| `mlp.c_fc` | Uniform | `1/sqrt(n_embd)` |
| `mlp.c_proj` | **Zeros** | 0 |
| `resid_lambdas` | Fill | 1.0 |
| `x0_lambdas` | Fill | 0.1 |
| `ve_gate` | **Zeros** | 0 (so sigmoid(0)=0.5, scaled by 2 = 1.0, neutral) |

**Key insight:** Output projections (`c_proj`) are initialized to **zeros**. This means at initialization, each Transformer block is an identity function (the residual connection passes through unchanged). Training gradually "turns on" each layer.

**Why Uniform instead of Normal for weights?** The `sqrt(3)` multiplier ensures the same standard deviation as Normal, but Uniform avoids outlier weights that could destabilize early training.

---

#### `_precompute_rotary_embeddings(self, seq_len, head_dim, base=10000)` (line 243)

Pre-computes the `cos` and `sin` tables for RoPE. Over-allocates by 10x the sequence length.

The frequency formula: `inv_freq[i] = 1 / (base ^ (2i / head_dim))`

Lower dimensions rotate faster (high frequency), higher dimensions rotate slower (low frequency). This creates a spectrum of position encodings at different scales.

---

#### `_compute_window_sizes(self, config)` (line 260)

Converts the window pattern string (e.g., `"SSSL"`) into a list of `(left, right)` tuples for Flash Attention:
- `L` -> `(sequence_len, 0)` = full context, causal
- `S` -> `(sequence_len // 2, 0)` = half context, causal

The last layer always gets full context.

---

#### `estimate_flops(self)` (line 292)

Estimates FLOPs per token (forward + backward). Two components:
1. **MatMul FLOPs**: `6 * num_matmul_params` (2 for forward multiply+accumulate, 4 for backward)
2. **Attention FLOPs**: `12 * H * D * effective_seq_len` per layer (accounts for sliding window)

Excludes embedding lookups and softmax (tiny compared to matmuls).

---

#### `setup_optimizer(self, ...)` (line 348)

Creates the MuonAdamW optimizer with parameter groups:

| Parameters | Optimizer | Notes |
|-----------|-----------|-------|
| `lm_head` | AdamW | Low LR, no weight decay |
| `wte` (embedding) | AdamW | Higher LR, no weight decay |
| `value_embeds` | AdamW | Same LR as embedding |
| `resid_lambdas` | AdamW | Very low LR (0.01 * scalar_lr) |
| `x0_lambdas` | AdamW | Higher beta1 (0.96) for stability |
| Transformer matrices (by shape) | **Muon** | Grouped by shape for stacking |

The AdamW LRs are scaled by `1/sqrt(model_dim / 768)` (muP-style scaling).

---

#### `forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean')` (line 388)

The main forward pass:

1. **Rotary offset**: If KV cache exists, offset the rotary embeddings to the current position
2. **Embed**: `wte(idx)` -> normalize -> save as `x0`
3. **Transformer blocks**: For each block:
   - Scale residual: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
   - Look up value embedding if this layer has one
   - Run the block (attention + MLP)
4. **Final norm** on output
5. **LM head**: Project to vocab size, slice off padding, convert to float32
6. **Logit soft-cap**: `logits = 15 * tanh(logits / 15)` - smoothly caps logits to [-15, 15]
7. **Loss or logits**: If targets provided, compute cross-entropy loss. Otherwise return logits.

**The `x0` residual:** At each layer, a small fraction of the initial embedding is blended back in via `x0_lambdas`. This helps with gradient flow in deep networks (inspired by modded-nanogpt).

---

#### `generate(self, tokens, max_tokens, temperature, top_k, seed)` (line 425)

Naive autoregressive generation (no KV cache, re-processes entire sequence each step). This is the simple/slow reference implementation. For efficient generation, use `Engine.generate()` instead.

Yields one token at a time (streaming). Supports:
- **Temperature scaling**: Higher = more random, 0 = greedy
- **Top-k sampling**: Only sample from top k most likely tokens

---

## 4. Model Sizing via `--depth`

The `--depth` parameter controls everything:

```
model_dim = depth * aspect_ratio (default 64)
num_heads = model_dim / head_dim (default 128)
```

| Depth | Model Dim | Heads | Approx Params |
|-------|-----------|-------|---------------|
| 4 | 256 | 2 | ~5M |
| 12 | 768 | 6 | ~130M |
| 20 | 1280 | 10 | ~520M |
| 26 | 1664 | 13 | ~1.1B |

This "single dial" philosophy means all hyperparameters (LR, batch size, training horizon, weight decay) are automatically computed from depth via scaling laws.
