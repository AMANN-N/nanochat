# NanoChat Component: Training Pipeline

**Files:** `scripts/base_train.py`, `scripts/chat_sft.py`, `scripts/chat_rl.py`

## 1. The Three Training Stages

NanoChat follows the standard modern LLM training pipeline:

```
Stage 1: Pretraining          Stage 2: SFT                 Stage 3: RL
(base_train.py)               (chat_sft.py)                (chat_rl.py)
────────────────               ────────────                 ────────────
Data: FineWeb-Edu (web text)   Data: Chat datasets          Data: GSM8K (math)
Goal: Language modeling        Goal: Follow instructions     Goal: Correct reasoning
Loss: Cross-entropy on         Loss: Cross-entropy on        Loss: Policy gradient
      next-token prediction          assistant tokens only         (REINFORCE-style)
LR:   High (warmup + warmdown) LR:   Medium (ramp down)      LR:   Low (5% of SFT, ramp down)
Time: ~3 hours (8xH100)       Time: ~15 minutes             Time: ~15 minutes
```

Each stage loads the checkpoint from the previous stage and produces a new checkpoint.

---

## 2. Stage 1: Pretraining (`scripts/base_train.py`)

### Purpose
Train the base language model on raw web text. The model learns to predict the next token given all previous tokens. This teaches it grammar, facts, reasoning patterns, and world knowledge.

### Command
```bash
# Single GPU
python -m scripts.base_train --depth=12

# Multi-GPU (recommended)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26
```

### Key CLI Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--depth` | 20 | Model depth (controls everything via scaling laws) |
| `--aspect-ratio` | 64 | `model_dim = depth * aspect_ratio` |
| `--total-batch-size` | -1 (auto) | Total tokens per step (auto-computed from scaling laws) |
| `--target-param-data-ratio` | 10.5 | Chinchilla-style data:param ratio |
| `--fp8` | false | Enable FP8 training on H100+ |
| `--window-pattern` | "SSSL" | Sliding window pattern |
| `--warmdown-ratio` | 0.5 | Fraction of training for LR warmdown |

### Scaling Laws (Automatic Hyperparameter Selection)

The key innovation: **one parameter (`--depth`) controls everything**.

**Step 1: Training Horizon**
```python
target_tokens = target_param_data_ratio * num_scaling_params
# e.g., 10.5 * 100M params = 1.05B tokens
```
How many tokens to train on. Uses Chinchilla-style data:param ratio (default 10.5, derived experimentally).

**Step 2: Optimal Batch Size**
```python
# Power Lines paper: Bopt ∝ D^0.383
predicted_batch_size = B_REF * (target_tokens / D_REF) ** 0.383
total_batch_size = 2 ** round(log2(predicted_batch_size))  # nearest power of 2
```
The optimal batch size grows sublinearly with training horizon. Reference: B_REF=524,288 tokens at d12.

**Step 3: Learning Rate Correction**
```python
batch_lr_scale = (total_batch_size / B_REF) ** 0.5  # η ∝ √(B/B_ref)
```
Larger batch sizes allow proportionally higher learning rates (sqrt scaling for Adam/Muon).

**Step 4: Weight Decay Scaling**
```python
# T_epoch framework: λ = λ_ref * √(B/B_ref) * (D_ref/D)
weight_decay_scaled = wd * sqrt(B/B_ref) * (D_ref/D)
```
Weight decay is scaled to maintain a constant "T_epoch" across different batch sizes and training horizons.

### Model Initialization Flow

```python
model = build_model_meta(depth)    # 1. Create on meta device (shapes only, no data)
model.to_empty(device=device)       # 2. Allocate storage on GPU (garbage data)
model.init_weights()                # 3. Initialize all parameters properly
model = torch.compile(model)        # 4. Compile for performance
```

### Learning Rate Schedule

```
LR multiplier
1.0 ─────────────────────────────────╲
                                       ╲
                                        ╲
0.0 ────────────────────────────────────╲──
    |── warmup ──|───── constant ──|── warmdown ──|
    0%           0%                50%            100%
```

Default: No warmup, constant for first 50%, linear decay to 0 for last 50%.

### Muon Momentum Schedule
```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95  # Warm up from 0.85 to 0.95 over 300 steps
```

### Weight Decay Schedule
```python
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)  # Linear decay to 0
```

### Training Loop (line 395)

Each iteration:
1. **Evaluate** (if scheduled): val BPB, CORE metric, sample from model
2. **Save checkpoint** (if scheduled)
3. **Forward/backward** with gradient accumulation:
   ```python
   for micro_step in range(grad_accum_steps):
       loss = model(x, y)
       loss = loss / grad_accum_steps
       loss.backward()
       x, y, state = next(train_loader)  # Prefetch next batch!
   ```
4. **Update schedulers**: LR multiplier, Muon momentum, weight decay
5. **Optimizer step**: `optimizer.step()` then `model.zero_grad(set_to_none=True)`
6. **Logging**: EMA loss, tok/sec, MFU, ETA

### Gradient Accumulation

To achieve large effective batch sizes without OOMing:
```
total_batch_size = device_batch_size * max_seq_len * world_size * grad_accum_steps
```
Example: `32 * 2048 * 8 * 1 = 524,288 tokens` with 8 GPUs and no accumulation.

### FP8 Training (line 164)

Optional FP8 quantization for ~20% speedup on H100+:
1. Convert Linear layers to `Float8Linear` (from torchao)
2. Only layers with dimensions divisible by 16 (hardware requirement)
3. `disable_fp8()` context manager temporarily swaps back to BF16 for evaluation

### GC Management (line 552)

The garbage collector is manually managed for performance:
```python
gc.collect()   # Clean up setup garbage
gc.freeze()    # Freeze surviving objects (exclude from future GC scans)
gc.disable()   # Disable automatic GC entirely
# ... every 5000 steps: gc.collect() for safety
```
This prevents ~500ms GC pauses that would otherwise interrupt training.

### Output
Checkpoint saved to `~/.cache/nanochat/base_checkpoints/d{depth}/`

---

## 3. Stage 2: Supervised Fine-Tuning (`scripts/chat_sft.py`)

### Purpose
Teach the base model to follow instructions by training on conversation data. The model learns the chat format (user/assistant turns) and specific capabilities (math, spelling, general knowledge).

### Command
```bash
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag=d26
```

### Data Mixture

```python
train_dataset = TaskMixture([
    SmolTalk(split="train"),           # 460K general conversations
    MMLU(subset="auxiliary_train"),     # 100K multiple choice problems
    GSM8K(subset="main", split="train"),  # 8K math problems (×2 epochs)
    GSM8K(subset="main", split="train"),
    CustomJSON(identity_conversations),    # 1K identity conversations (×2)
    CustomJSON(identity_conversations),
    SimpleSpelling(size=200000),        # 200K "spell this word" tasks
    SpellingBee(size=80000),            # 80K "count this letter" tasks
])  # Total: ~856K rows
```

### Key Difference from Pretraining: Masking

During SFT, the model only trains on **assistant tokens**. User tokens, BOS, and system tokens are masked (loss = ignore_index = -1).

### SFT DataLoader: `sft_data_generator_bos_bestfit()` (line 127)

Similar to the pretraining dataloader but for conversations:

1. **Render** each conversation to token IDs using `tokenizer.render_conversation()`
2. **Pack** conversations into rows using best-fit algorithm (same as pretraining)
3. **Pad** (instead of crop!) when nothing fits - SFT never discards tokens
4. **Mask** padding positions with `targets[i, content_len-1:] = -1`

**Key difference from pretraining:** SFT pads instead of cropping. This ensures no training data is lost (SFT datasets are much smaller than pretraining data, so every example matters).

### LR Schedule (SFT)
```python
def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2
```
Constant for first 80%, linear decay to 0 over last 20%.

### Stopping Condition
SFT trains for exactly one epoch (full pass through the dataset) unless `--num-iterations` is specified.

### Output
Checkpoint saved to `~/.cache/nanochat/chatsft_checkpoints/d{depth}/`

---

## 4. Stage 3: Reinforcement Learning (`scripts/chat_rl.py`)

### Purpose
Further optimize the SFT model for **correct reasoning** on math problems. The model generates multiple solutions, correct ones are reinforced, incorrect ones are suppressed.

### Command
```bash
torchrun --nproc_per_node=8 -m scripts.chat_rl --model-tag=d26
```

### Algorithm: Simplified GRPO

NanoChat implements a simplified version of GRPO (Group Relative Policy Optimization) that reduces to REINFORCE:

1. **No trust region** (no KL regularization to reference model)
2. **On-policy** (no PPO ratio+clip needed)
3. **Token-level normalization** (DAPO style)
4. **Simple advantage**: `advantage = reward - mean_reward` (no z-score)

### Rollout Loop: `get_batch()` (line 91)

For each training example:

1. **Render** the conversation, removing the assistant's answer
2. **Generate** `num_samples` (default 16) completions using `Engine.generate_batch()`
   - Tool use is active (calculator can be used for math)
   - Each sample gets a different random seed
3. **Score** each completion: `reward = 1` if answer matches ground truth, `0` otherwise
4. **Compute advantages**: `advantages = rewards - mean(rewards)`
5. **Yield** `(sequences, inputs, targets, rewards, advantages)`

### Training Step (line 252)

For each optimization step:

1. **Collect rollouts** from `examples_per_rank` questions
2. For each question's rollouts:
   ```python
   logp = -model(inputs, targets, loss_reduction='none')  # Log probabilities
   pg_obj = (logp * advantages.unsqueeze(-1)).sum()        # Policy gradient objective
   loss = -pg_obj / (num_valid * num_passes * examples_per_rank)
   loss.backward()
   ```
3. **Optimizer step** with linearly decaying LR

**Key insight:** The loss is `-(logp * advantage)`. For correct answers (advantage > 0), this increases their probability. For incorrect answers (advantage < 0), this decreases their probability.

### Evaluation: `run_gsm8k_eval()` (line 156)

Periodically evaluates Pass@k on the GSM8K test set:
- Generate k samples per question
- Pass@k = fraction of questions where at least one of k samples is correct
- Distributed across ranks, results aggregated via `all_reduce`

### LR Schedule (RL)
```python
def get_lr_multiplier(it):
    return 1.0 - it / num_steps  # Linear decay from 1 to 0
```
The initial LR is already very low (5% of SFT LR by default).

### Output
Checkpoint saved to `~/.cache/nanochat/chatrl_checkpoints/d{depth}/`

---

## 5. End-to-End Pipeline

The complete pipeline is orchestrated by `runs/speedrun.sh`:

```bash
# Step 0: Train tokenizer (optional, only needed once)
python -m scripts.tok_train

# Step 1: Pretrain base model
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26

# Step 2: Supervised fine-tuning
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag=d26

# Step 3: Reinforcement learning
torchrun --nproc_per_node=8 -m scripts.chat_rl --model-tag=d26

# Step 4: Serve
python -m scripts.chat_web
```

### Checkpoint Directory Structure
```
~/.cache/nanochat/
├── tokenizer/
│   ├── tokenizer.pkl          # Trained BPE tokenizer
│   └── token_bytes.pt         # Token -> byte length mapping
├── base_data/
│   ├── shard_00000.parquet    # FineWeb-Edu data shards
│   ├── shard_00001.parquet
│   └── ...
├── base_checkpoints/
│   └── d26/
│       ├── model_005000.pt    # Model weights at step 5000
│       ├── optim_005000_rank0.pt  # Optimizer state (sharded per rank)
│       └── meta_005000.json   # Metadata (config, loop state, etc.)
├── chatsft_checkpoints/
│   └── d26/
│       ├── model_000500.pt
│       └── meta_000500.json
└── chatrl_checkpoints/
    └── d26/
        ├── model_000300.pt
        └── meta_000300.json
```

---

## 6. Distributed Training

All three stages support multi-GPU training via `torchrun`. The distributed setup uses:

- **No DDP wrapper** - NanoChat uses its own `DistMuonAdamW` optimizer which handles gradient synchronization internally (ZeRO-2 style)
- **NCCL backend** for GPU communication
- **Data sharding** - each rank reads different data (no duplication)
- **Gradient accumulation** - if total batch size requires it

The `compute_init()` function in `common.py` handles all distributed setup:
```python
ddp, rank, local_rank, world_size, device = compute_init("cuda")
```
