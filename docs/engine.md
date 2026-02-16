# NanoChat Component: Inference Engine

**File:** `nanochat/engine.py`

## 1. Overview

The Engine is the **inference runtime**. It wraps the GPT model and provides efficient batched generation with KV cache management and tool use (Python code execution).

```
Prompt Tokens --> [Engine.generate] --> Token Stream
                      |
                      ├── KV Cache (fast autoregressive decoding)
                      ├── Tool Use State Machine (Python REPL)
                      └── Multi-sample batched generation
```

The Engine knows nothing about text - it operates purely on token IDs. The tokenizer is only needed for tool use (decoding Python expressions and encoding results).

---

## 2. Calculator Tool Helpers

### `timeout(duration, formula)` (line 27)
Context manager that sets a UNIX alarm signal. If the code inside doesn't finish within `duration` seconds, it raises an exception.

### `eval_with_timeout(formula, max_time=3)` (line 36)
Safely evaluates a Python expression string with:
- 3-second timeout
- No builtins (sandboxed `eval`)
- Suppresses SyntaxWarnings

### `use_calculator(expr)` (line 47)
The lightweight calculator for inline math during generation. Two modes:

1. **Pure math**: If the expression contains only `0-9*+-/.() `, evaluate it directly (but disallow `**` to prevent DoS)
2. **String operations**: If the expression contains `.count(` and only safe characters, evaluate it (used for letter counting tasks like "how many r's in strawberry?")

Returns `None` if the expression is unsafe or fails.

---

## 3. KV Cache

### `KVCache` (line 83)

Pre-allocated key/value cache designed for Flash Attention 3's API.

**Constructor:**
```python
KVCache(batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype)
```

**Internal storage:**
```python
self.k_cache = torch.zeros(num_layers, B, T, H, D)  # All layers' key cache
self.v_cache = torch.zeros(num_layers, B, T, H, D)  # All layers' value cache
self.cache_seqlens = torch.zeros(B, dtype=int32)     # Current position per batch element
```

**Key difference from FA2-style cache:** Tensors are `(B, T, H, D)` not `(B, H, T, D)`. FA3 updates the cache **in-place** during `flash_attn_with_kvcache`.

**Methods:**

| Method | Purpose |
|--------|---------|
| `reset()` | Zero out `cache_seqlens` (reuse cache without reallocating) |
| `get_pos()` | Current sequence position (assumes uniform across batch) |
| `get_layer_cache(layer_idx)` | Returns `(k_cache, v_cache)` views for one layer |
| `advance(num_tokens)` | Move position forward by `num_tokens` |
| `prefill(other)` | Copy another cache's contents into this one (for multi-sample generation) |

---

## 4. Token Sampling

### `sample_next_token(logits, rng, temperature, top_k)` (line 136)

Takes logits of shape `(B, vocab_size)` and samples the next token for each batch element.

**Algorithm:**
1. If `temperature == 0`: Greedy (argmax)
2. If `top_k > 0`:
   - Keep only top-k logits, set rest to `-inf`
   - Divide by temperature
   - Softmax -> multinomial sample
3. Otherwise:
   - Divide by temperature
   - Softmax -> multinomial sample

**Why a separate `rng` generator?** For reproducibility. Each generation call creates its own `torch.Generator` seeded deterministically, so the same prompt always produces the same output.

---

## 5. Row State

### `RowState` (line 155)

Per-sample state during batched generation:

```python
class RowState:
    current_tokens: list     # Full token sequence so far
    forced_tokens: deque     # Queue of tokens to force-inject (tool outputs)
    in_python_block: bool    # Inside <|python_start|>...<|python_end|>?
    python_expr_tokens: list # Collecting tokens of current Python expression
    completed: bool          # Has this row finished generating?
```

---

## 6. Engine

### `Engine.__init__(self, model, tokenizer)` (line 166)

Simple wrapper. Stores model and tokenizer references.

### `Engine.generate(...)` (line 171)

The main generation method. This is a **streaming generator** that yields `(token_column, token_masks)` at each step.

**Signature:**
```python
def generate(self, tokens, num_samples=1, max_tokens=None,
             temperature=1.0, top_k=None, seed=42):
```

**Full flow:**

#### Step 1: Prefill (line 194-206)
```
tokens = [bos, user_start, Hello, user_end, assistant_start]
         └─────────── prompt tokens ──────────────────────┘
```
1. Create a **batch=1** KV cache sized to `len(tokens)`
2. Run the full prompt through the model in one forward pass (efficient!)
3. Get logits for the last position
4. Expand logits to `num_samples` (all samples share the same prompt)

#### Step 2: Replicate KV Cache (line 209-218)
1. Create a new KV cache with `batch_size=num_samples` and `seq_len=len(tokens)+max_tokens`
2. Copy the prefill cache into it via `kv_cache_decode.prefill(kv_cache_prefill)`
3. Delete the prefill cache (free memory)

**Why this two-step approach?** The prompt is the same for all samples, so we only process it once (batch=1) and then replicate. This saves `num_samples - 1` redundant prefill passes.

#### Step 3: Initialize Row States (line 221)
One `RowState` per sample, each starting with a copy of the prompt tokens.

#### Step 4: Main Generation Loop (line 224-275)

Each iteration:
1. **Check stop conditions**: max tokens reached, or all rows completed
2. **Sample** next token from logits: `sample_next_token(logits, rng, temperature, top_k)`
3. **For each row:**
   - If there are **forced tokens** in the queue, use the forced token instead of the sampled one
   - Track whether the token was sampled (`mask=1`) or forced (`mask=0`)
   - Append token to row state
   - If `<|assistant_end|>` or `<|bos|>` is generated, mark row as completed
   - **Tool use state machine:**
     - On `<|python_start|>`: Enter python block, start collecting expression tokens
     - On `<|python_end|>`: Exit python block, evaluate expression with `use_calculator`
       - If result is valid: Force-inject `<|output_start|> result_tokens <|output_end|>`
     - While in python block: Collect expression tokens
4. **Yield** `(token_column, token_masks)` - one token per row, one mask per row
5. **Prepare next logits**: Forward the token column through model with KV cache

#### Tool Use State Machine Diagram

```
Normal Generation
     │
     ├── Model generates <|python_start|>  ──→  Collecting Python Tokens
     │                                              │
     │                                              ├── Model generates <|python_end|>
     │                                              │        │
     │                                              │        ├── Evaluate expression
     │                                              │        ├── Force: <|output_start|>
     │                                              │        ├── Force: result tokens
     │                                              │        └── Force: <|output_end|>
     │                                              │               │
     │                                              │               └── Back to Normal
     │                                              │
     │                                              └── (keep collecting tokens)
     │
     ├── Model generates <|assistant_end|>  ──→  Row Completed
     └── Model generates <|bos|>            ──→  Row Completed
```

### `Engine.generate_batch(...)` (line 277)

Non-streaming wrapper around `generate()`. Collects all tokens into final sequences. Returns:
- `results`: List of token sequences (one per sample)
- `masks`: List of mask sequences (1=sampled, 0=forced/prompt)

Terminal tokens (`assistant_end`, `bos`) are **not** included in results.

---

## 7. Inline Test (`__main__` block, line 302)

When run directly (`python -m nanochat.engine`), it:
1. Loads a base model checkpoint
2. Generates tokens using the naive `model.generate()` (no KV cache)
3. Generates tokens using `Engine.generate()` (with KV cache)
4. Compares outputs to verify they match
5. Reports timing for both approaches

This validates that the KV cache implementation is correct by comparing against the reference (slow) implementation.

---

## 8. Generation Performance

| Approach | Speed | Memory | Use Case |
|----------|-------|--------|----------|
| `model.generate()` | Slow (O(T^2)) | Low | Reference, debugging |
| `Engine.generate()` | Fast (O(T)) | Higher (KV cache) | Production inference |

The KV cache avoids recomputing attention for all previous tokens at each step. Instead, only the new token's K/V are appended, and attention is computed against the cached K/V.

The `prefill + decode` pattern is standard in LLM inference:
- **Prefill**: Process all prompt tokens in parallel (one forward pass)
- **Decode**: Generate one token at a time using cached K/V (autoregressive)
