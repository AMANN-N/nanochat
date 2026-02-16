# NanoChat Component: Optimizer

**File:** `nanochat/optim.py`

## 1. Overview

NanoChat uses a **split optimization strategy**: different parameter types get different optimizers.

| Parameter Type | Optimizer | Examples |
|----------------|-----------|----------|
| 2D matrices (attention, MLP weights) | **Muon** | `c_q.weight`, `c_k.weight`, `c_v.weight`, `c_proj.weight`, `c_fc.weight`, `mlp.c_proj.weight` |
| Embeddings | **AdamW** | `wte.weight`, `value_embeds` |
| Unembedding | **AdamW** | `lm_head.weight` |
| Per-layer scalars | **AdamW** | `resid_lambdas`, `x0_lambdas` |

**Why split?** Muon is designed for 2D weight matrices where orthogonalization makes sense. Embeddings and scalars don't have a meaningful "orthogonal" structure, so AdamW is used instead.

Two implementations are provided:
- `MuonAdamW`: Single GPU (for reference/debugging)
- `DistMuonAdamW`: Multi-GPU with async communication overlap

---

## 2. AdamW

### `adamw_step_fused(...)` (line 20)

A `torch.compile`-optimized fused AdamW step. All operations in a single compiled graph to eliminate Python overhead.

**Algorithm:**
```
1. Weight decay:     p = p * (1 - lr * wd)          # Decoupled (before update)
2. First moment:     m = lerp(m, grad, 1-β1)        # Running mean of gradients
3. Second moment:    v = lerp(v, grad², 1-β2)        # Running mean of squared gradients
4. Bias correction:  m̂ = m / (1 - β1^t)
                     v̂ = v / (1 - β2^t)
5. Update:           p = p - lr * m̂ / (√v̂ + ε)
```

**0-D CPU Tensors trick:** Hyperparameters are stored as 0-D CPU tensors (e.g., `torch.tensor(0.0)`) rather than Python floats. This prevents `torch.compile` from recompiling the graph every time a hyperparameter changes (e.g., learning rate schedule).

---

## 3. Muon (MomentUm Orthogonalized by Newton-Schulz)

### Core Idea

Standard SGD/Adam updates can be "unbalanced" - some directions in weight space get much larger updates than others. Muon fixes this by **orthogonalizing** the update direction.

**Intuition:** After computing the gradient and applying momentum, Muon replaces the update with the **nearest orthogonal matrix**. This is like saying "move in this direction, but make sure all dimensions get updated equally."

Mathematically, if `G = USV^T` is the SVD of the gradient, the ideal orthogonal update is `UV^T` (strip away the singular values). Computing full SVD is expensive, so Muon approximates it.

### `muon_step_fused(...)` (line 90)

The complete Muon step in one compiled kernel:

**Step 1: Nesterov Momentum**
```python
momentum_buffer.lerp_(grad, 1 - momentum)           # Update running average
g = grad.lerp_(momentum_buffer, momentum)            # Nesterov lookahead
```
Nesterov momentum "looks ahead" by using the momentum buffer to anticipate where the gradient will be.

**Step 2: Polar Express (Orthogonalization)**
```python
X = g / (g.norm() * 1.02 + 1e-6)    # Normalize
for a, b, c in polar_express_coeffs:  # 5 iterations
    A = X.mT @ X                      # or X @ X.mT for wide matrices
    B = b * A + c * (A @ A)
    X = a * X + X @ B                 # or B @ X for wide matrices
```

This is the **Polar Express Sign Method** (replacing the original Newton-Schulz iteration). It iteratively converges toward the orthogonal polar factor of the gradient matrix.

**Why "Polar Express"?** The name comes from the "polar decomposition" of a matrix (A = UP where U is orthogonal, P is positive semidefinite). The iteration computes U, which is the closest orthogonal matrix to A.

The coefficients are pre-computed for 5 iterations with a safety factor. The result is approximately `US'V^T` where `S'` is diagonal with values ~Uniform(0.5, 1.5) - not exactly orthogonal, but empirically works just as well.

**Step 3: NorMuon Variance Reduction**
```python
v_mean = g.square().mean(dim=red_dim)        # Per-row or per-column variance
second_momentum_buffer.lerp_(v_mean, 1-β2)   # EMA of variance
step_size = rsqrt(second_momentum_buffer)     # Adaptive per-neuron LR
g = g * step_size * (v_norm / v_norm_new)     # Scale while preserving total norm
```

After orthogonalization, Muon's output has non-uniform scales across neurons. NorMuon normalizes these by tracking per-neuron variance (similar to how Adam adapts per-parameter).

The `red_dim` is chosen based on matrix shape:
- Tall matrices (`rows >= cols`): Reduce along columns (`red_dim = -1`)
- Wide matrices (`rows < cols`): Reduce along rows (`red_dim = -2`)

**Step 4: Cautious Update with Weight Decay**
```python
mask = (g * params) >= 0                      # Only decay where gradient aligns with weight
params -= lr * g + lr * wd * params * mask    # Cautious weight decay
```

**Cautious weight decay:** Only applies weight decay in directions where the gradient and the current weight agree in sign. This prevents the optimizer from fighting against itself.

---

## 4. Polar Express Coefficients (line 82)

```python
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    ...
]
```

These are pre-computed from the paper [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932) for `num_iters=5, safety_factor=2e-2, cushion=2`. Each tuple `(a, b, c)` defines one iteration of the orthogonalization.

---

## 5. Single GPU: `MuonAdamW` (line 152)

### `_step_adamw(self, group)` (line 194)
Processes each AdamW parameter individually. Lazy-inits state (step counter, first/second moments), fills 0-D tensors, calls `adamw_step_fused`.

### `_step_muon(self, group)` (line 229)
Processes all Muon parameters in a group **together** (stacked for efficiency). All params in a group must have the same shape.

1. **Stack** all gradients and parameters into single tensors
2. **Init** momentum buffer `(num_params, *shape)` and factored second momentum buffer
3. Call `muon_step_fused` on the stacked tensors
4. **Copy back** updated stacked params to original parameter tensors

### `step(self)` (line 283)
Iterates over param groups, dispatching to `_step_adamw` or `_step_muon`.

---

## 6. Distributed: `DistMuonAdamW` (line 297)

The distributed version adds communication overlap using a **3-phase async pattern**:

### Phase 1: Launch All Async Reduces

**`_reduce_adamw(group, world_size)`** (line 369):
- **Small params** (<1024 elements): `all_reduce` (replicated state, tiny tensors)
- **Large params**: `reduce_scatter` along dim 0 (each rank gets 1/N of the gradient)

**`_reduce_muon(group, world_size)`** (line 387):
- Stack all K gradients into a single tensor
- Zero-pad to `ceil(K/N) * N` for even division
- `reduce_scatter` so each rank gets its chunk of ~K/N gradients

### Phase 2: Wait for Reduces, Compute, Launch Gathers

**`_compute_adamw(group, info, gather_list, rank, world_size)`** (line 408):
- Wait for reduce to complete
- Compute AdamW update on this rank's slice
- Launch async `all_gather` to reassemble the full parameter

**`_compute_muon(group, info, gather_list, rank)`** (line 449):
- Wait for reduce to complete
- Compute Muon update only for params this rank owns
- Launch async `all_gather` to share updates

### Phase 3: Wait for Gathers, Copy Back

**`_finish_gathers(gather_list)`** (line 499):
- Wait for all async gathers
- For Muon: copy from stacked buffer back to individual parameter tensors

### Communication Pattern Diagram

```
Rank 0:  [reduce_scatter grad0] ─── wait ─── [compute update0] ─── [all_gather param0] ─── wait ─── [copy back]
Rank 1:  [reduce_scatter grad1] ─── wait ─── [compute update1] ─── [all_gather param1] ─── wait ─── [copy back]
                                  overlap!                          overlap!
```

The key optimization is that while one group's gather is running, the next group's compute can start.

### Memory Optimization (ZeRO-2 Style)

- Optimizer state is **sharded**: each rank only stores state for the params it owns
- For AdamW large params: `exp_avg` and `exp_avg_sq` are 1/N the size
- For Muon: `momentum_buffer` and `second_momentum_buffer` are ~1/N the size
- Buffer reuse: `stacked_grads` tensor is reused as `all_gather` output (no simultaneous allocation)

---

## 7. Learning Rate Scaling

The Muon learning rate has a shape-dependent correction:

```python
lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
```

For tall matrices (rows > cols), the LR is scaled up by `sqrt(rows/cols)`. This compensates for the orthogonalization, which has different scaling properties for rectangular matrices.

---

## 8. Why This Matters

Traditional Adam treats every parameter independently. Muon treats weight matrices **holistically** by considering their structure:
- It ensures updates are well-conditioned (no single direction dominates)
- This leads to faster convergence (fewer steps to reach the same loss)
- The orthogonalization acts as an implicit regularizer

The tradeoff is computational cost: Muon needs 5 matrix multiplications per parameter update (the Polar Express iterations). But since these run on GPU and overlap with communication, the wall-clock overhead is small.
