# NanoChat Component: Data Pipeline

**Files:** `nanochat/dataset.py`, `nanochat/dataloader.py`

## 1. Overview

The data pipeline handles downloading, reading, tokenizing, and batching pretraining data. It operates as a streaming pipeline - data is tokenized on-the-fly, never stored as token files on disk.

```
FineWeb-Edu (Parquet on HuggingFace)
    │
    ▼
dataset.py: Download shards on demand
    │
    ▼
dataloader.py: Read → Tokenize → Pack → Batch
    │
    ▼
(inputs, targets) tensors on GPU   shape: (B, T)
```

---

## 2. Dataset (`nanochat/dataset.py`)

### Constants

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822                    # Last shard: shard_01822.parquet
DATA_DIR = ~/.cache/nanochat/base_data/
```

The dataset is **FineWeb-Edu 100B** (shuffled), a curated subset of web text focused on educational content. It's stored as 1823 parquet files, each containing multiple "row groups" of text documents.

### Functions

#### `list_parquet_files(data_dir=None)` (line 33)
Returns sorted list of all `.parquet` file paths in the data directory. Excludes `.tmp` files (partial downloads).

#### `parquets_iter_batched(split, start=0, step=1)` (line 43)
Iterator over document batches from parquet files.
- `split="train"`: All files except the last one
- `split="val"`: Only the last file
- `start/step`: For DDP sharding (e.g., `start=rank, step=world_size`)

Reads one row group at a time, yields the `text` column as a list of strings.

#### `download_single_file(index)` (line 60)
Downloads a single parquet shard with retry logic:
- 5 attempts with exponential backoff (2, 4, 8, 16 seconds)
- Downloads to `.tmp` file first, then atomic rename
- Skips if file already exists

#### `__main__` block (line 112)
CLI for batch downloading: `python -m nanochat.dataset -n 100 -w 4` downloads 100 shards using 4 parallel workers.

---

## 3. DataLoader (`nanochat/dataloader.py`)

### The Problem: Packing Documents into Fixed-Length Sequences

LLMs train on fixed-length sequences of `T+1` tokens (T inputs + 1 target at the end). But documents vary wildly in length - from 10 tokens to 10,000+. How do you fit them into fixed rows?

**NanoChat's approach: BOS-Aligned Best-Fit Packing**

Rules:
1. Every row **starts with BOS** (beginning of sequence)
2. Pack as many complete documents as possible into each row
3. When nothing fits, **crop** a document to fill the remaining space
4. **No padding** - every token position is used (100% utilization)

### `_document_batches(split, resume_state_dict, tokenizer_batch_size)` (line 25)

**Infinite iterator** over document text batches from parquet files.

Features:
- **DDP sharding**: Each rank reads different row groups (`rg_idx = rank, rank + world_size, rank + 2*world_size, ...`)
- **Resume support**: Can restart from a specific `(pq_idx, rg_idx, epoch)` position
- **Multi-epoch**: Loops infinitely over the dataset, incrementing `epoch`

Yields: `(text_batch, (pq_idx, rg_idx, epoch))`

### `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)` (line 73)

The main dataloader. This is the function called by the training script.

**Signature:**
```python
def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,              # For encoding text -> tokens
    B,                      # Batch size (rows)
    T,                      # Sequence length
    split,                  # "train" or "val"
    tokenizer_threads=4,    # Parallel tokenization threads
    tokenizer_batch_size=128, # Docs per tokenization call
    device="cuda",
    resume_state_dict=None, # For resuming training
    buffer_size=1000        # Document buffer size for best-fit
)
```

**Algorithm (for each row in the batch):**

```
row_capacity = T + 1 (need T inputs + 1 target)
doc_buffer = [...1000 tokenized documents...]

For each position in the row:
    1. Ensure buffer has >= buffer_size documents
    2. remaining = row_capacity - current_position

    3. BEST-FIT: Find the LARGEST document that fits entirely
       - Scan all docs in buffer
       - Pick the one with length closest to (but not exceeding) remaining
       - Pop it from buffer, append to row

    4. If NO document fits:
       - CROP: Find the SHORTEST document in buffer
       - Take only `remaining` tokens from it (discard the rest)
       - This fills the row exactly to capacity
```

**Why best-fit instead of just greedy?**
Greedy (just take the next document) would waste space when a long document doesn't fit. Best-fit searches the buffer for a document that fills the gap well, reducing how many tokens get cropped. ~35% of tokens are cropped with this approach at T=2048.

**Why crop the shortest?**
When nothing fits, we must crop. Cropping the shortest document wastes the fewest tokens (the remaining uncropped portion is discarded).

**Memory Optimization:**

```python
# Pre-allocate ALL buffers once
row_buffer = torch.empty((B, row_capacity), dtype=torch.long)     # Build rows here
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)  # Staging area
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)    # On GPU

# Views into buffers (no allocation)
cpu_inputs  = cpu_buffer[:B*T].view(B, T)
cpu_targets = cpu_buffer[B*T:].view(B, T)
inputs  = gpu_buffer[:B*T].view(B, T)
targets = gpu_buffer[B*T:].view(B, T)
```

The layout `[inputs (B*T) | targets (B*T)]` in a single contiguous buffer enables a **single Host-to-Device transfer** per batch (instead of two).

**Data flow per batch:**
1. Fill `row_buffer` with packed documents (CPU)
2. Copy `row_buffer[:, :-1]` -> `cpu_inputs` and `row_buffer[:, 1:]` -> `cpu_targets`
3. Single `gpu_buffer.copy_(cpu_buffer)` with `non_blocking=True` (async HtoD)
4. Yield `(inputs, targets, state_dict)`

### `tokenizing_distributed_data_loader_bos_bestfit(...)` (line 162)
Simple wrapper that drops the `state_dict` from yields. Used for validation where resume isn't needed.

---

## 4. Visual Example

Given documents: `[BOS A A A]`, `[BOS B B]`, `[BOS C C C C C]`, `[BOS D]`

Packing into rows of capacity 8:

```
Row 0: [BOS C C C C C] [BOS D]     <- C fits best (5), then D fills remaining (2)
Row 1: [BOS A A A] [BOS B B] [X X] <- A (4) + B (3) = 7, need to crop 1 more from next doc
```
(X = cropped tokens from the next available document)

**Key property:** Every row starts with BOS, so every token can attend back to a clear document boundary. This is better than naive concatenation where a token might attend across unrelated documents.

---

## 5. Train/Val Split

The split is simple: **last parquet file = validation, everything else = training**.

```python
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

With 1823 files, this gives ~99.95% train and ~0.05% val. The val set is still large enough (millions of tokens) for reliable BPB evaluation.

---

## 6. Resume Support

The dataloader tracks its position via `state_dict`:
```python
{"pq_idx": 42, "rg_idx": 100, "epoch": 1}
```

On resume, it seeks to the correct parquet file and row group, advances by 1 to avoid repeating data, and continues from there. This enables mid-training restarts without data repetition.
