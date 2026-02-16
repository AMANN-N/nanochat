# NanoChat Component: Utilities

**Files:** `nanochat/common.py`, `nanochat/checkpoint_manager.py`, `nanochat/flash_attention.py`, `nanochat/execution.py`

---

## 1. Common Utilities (`nanochat/common.py`)

### Logging

#### `ColoredFormatter` (line 13)
Custom logging formatter with ANSI colors:
- DEBUG: Cyan, INFO: Green, WARNING: Yellow, ERROR: Red, CRITICAL: Magenta
- Highlights numbers, percentages, and shard identifiers in INFO messages

#### `setup_default_logging()` (line 39)
Configures the root logger with `ColoredFormatter`. Called at module import time.

### Path Management

#### `get_base_dir()` (line 50)
Returns the nanochat data directory. Priority:
1. `$NANOCHAT_BASE_DIR` environment variable
2. Default: `~/.cache/nanochat/`

All data (tokenizer, datasets, checkpoints) is stored here.

#### `download_file_with_lock(url, filename, postprocess_fn)` (line 61)
Downloads a file to the base directory with concurrency protection:
- Uses `FileLock` to prevent multiple ranks from downloading simultaneously
- Double-checks existence after acquiring lock (another rank may have finished)
- Optional `postprocess_fn` runs after download

### Distributed Helpers

#### `print0(s="", **kwargs)` (line 97)
Print only on rank 0. The most-used utility in the codebase - prevents duplicate output in multi-GPU training.

#### `is_ddp_requested()` (line 116)
Returns `True` if torchrun environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) are present.

#### `is_ddp_initialized()` (line 123)
Returns `True` if `torch.distributed` process group is initialized. Used at cleanup.

#### `get_dist_info()` (line 130)
Returns `(ddp, rank, local_rank, world_size)`. If not distributed, returns `(False, 0, 0, 1)`.

### Compute Initialization

#### `autodetect_device_type()` (line 142)
Prefers CUDA > MPS > CPU. Returns a string.

#### `compute_init(device_type)` (line 153)
The standard initialization function called at the start of every script:
1. Set random seeds (`torch.manual_seed(42)`)
2. Enable TF32 for CUDA matmuls
3. If DDP: Initialize process group with NCCL backend
4. Returns `(ddp, rank, local_rank, world_size, device)`

#### `compute_cleanup()` (line 190)
Destroys the distributed process group if initialized.

### GPU Performance

#### `DummyWandb` (line 195)
No-op wandb replacement. Used when wandb logging is disabled or on non-master ranks.

#### `get_peak_flops(device_name)` (line 207)
Hardcoded BF16 peak FLOPS for various GPUs. Used to compute MFU (Model FLOPS Utilization).

| GPU | Peak BF16 FLOPS |
|-----|-----------------|
| H100 SXM | 989 TFLOPS |
| H100 PCIe | 756 TFLOPS |
| A100 | 312 TFLOPS |
| RTX 4090 | 165 TFLOPS |

Returns `inf` for unknown GPUs (MFU shows as 0% rather than wrong).

---

## 2. Checkpoint Manager (`nanochat/checkpoint_manager.py`)

### Saving

#### `save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank)` (line 42)
Saves three files:
- `model_{step:06d}.pt` - Model state dict (rank 0 only)
- `meta_{step:06d}.json` - Metadata as JSON (rank 0 only)
- `optim_{step:06d}_rank{rank}.pt` - Optimizer state (**each rank saves its own**, because optimizer state is sharded in `DistMuonAdamW`)

### Loading

#### `load_checkpoint(checkpoint_dir, step, device, load_optimizer, rank)` (line 61)
Loads model, optimizer, and metadata from a checkpoint directory.

### Backward Compatibility

#### `_patch_missing_config_keys(model_config_kwargs)` (line 23)
Adds default values for config keys added after old checkpoints were saved. Currently patches:
- `window_pattern` -> `"L"` (old models used full context)

#### `_patch_missing_keys(model_data, model_config)` (line 30)
Adds default values for new parameters missing in old checkpoints:
- `resid_lambdas` -> all 1.0 (identity scaling)
- `x0_lambdas` -> all 0.0 (disabled)

### Model Building

#### `build_model(checkpoint_dir, step, device, phase)` (line 77)
The main model loading function:
1. Load checkpoint data
2. Fix `_orig_mod.` prefix from `torch.compile`
3. Patch missing config keys and parameters
4. Create model on meta device, then move to real device
5. Load state dict
6. Set train/eval mode
7. Load tokenizer and verify compatibility

#### `find_largest_model(checkpoints_dir)` (line 118)
Guesses the best model tag:
- First tries to parse `d<number>` tags, picks largest depth
- Falls back to most recently modified directory

#### `find_last_step(checkpoint_dir)` (line 138)
Finds the highest step number among `model_*.pt` files in a checkpoint directory.

### Convenience Functions

#### `load_model_from_dir(checkpoints_dir, device, phase, model_tag, step)` (line 149)
Auto-detects model tag and step if not provided. Calls `build_model`.

#### `load_model(source, *args, **kwargs)` (line 164)
Top-level API. Maps source name to directory:
- `"base"` -> `base_checkpoints/`
- `"sft"` -> `chatsft_checkpoints/`
- `"rl"` -> `chatrl_checkpoints/`

Usage: `model, tokenizer, meta = load_model("rl", device, phase="eval")`

---

## 3. Flash Attention (`nanochat/flash_attention.py`)

### Purpose

Provides a unified Flash Attention interface that automatically uses:
- **Flash Attention 3** on Hopper GPUs (sm90) - fastest
- **PyTorch SDPA fallback** everywhere else (Ampere, Ada, Blackwell, CPU, MPS)

### Detection

#### `_load_flash_attention_3()` (line 23)
Tries to load FA3:
1. Check CUDA availability
2. Check compute capability == 9 (Hopper only; Blackwell sm100 not yet supported)
3. Try importing via HuggingFace kernels
4. Returns the FA3 module or None

```python
_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
```

### SDPA Fallback

#### `_sdpa_attention(q, k, v, window_size, enable_gqa)` (line 61)
Implements sliding window attention using PyTorch's `scaled_dot_product_attention`:

Three code paths:
1. **Full context, same length** (training): Use `is_causal=True` (optimized)
2. **Single token** (decode step): Slice K/V to window, use `is_causal=False`
3. **General case** (chunk inference): Build explicit attention mask with window

### Public API

#### `flash_attn_func(q, k, v, causal, window_size)` (line 99)
Training path (no KV cache). Input: `(B, T, H, D)`.
- FA3: Direct call
- SDPA: Transpose to `(B, H, T, D)`, call SDPA, transpose back

#### `flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, causal, window_size)` (line 123)
Inference path (with KV cache). FA3 updates cache in-place.
- FA3: Direct call with cache management
- SDPA: Manually insert new K/V into cache, slice to current position, compute attention

### Export

```python
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
```
Used as: `from nanochat.flash_attention import flash_attn`

---

## 4. Sandboxed Execution (`nanochat/execution.py`)

### Purpose

Safely execute Python code generated by the LLM. Used for:
- HumanEval coding benchmarks
- Tool use in chat (Python REPL)

### `ExecutionResult` (line 38)
Dataclass returned by `execute_code()`:
```python
@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False
```

### Safety Mechanisms

#### `time_limit(seconds)` (line 64)
Context manager using `signal.ITIMER_REAL` to enforce execution timeout. Raises `TimeoutException` on expiry.

#### `capture_io()` (line 77)
Captures stdout/stderr and blocks stdin (prevents the code from reading user input).

#### `create_tempdir()` (line 89)
Creates a temporary directory and changes into it. Directory is deleted on exit.

#### `reliability_guard(maximum_memory_bytes)` (line 134)
Disables dangerous system functions:

| Category | Disabled Functions |
|----------|-------------------|
| Process | `os.kill`, `os.fork`, `os.system`, `subprocess.Popen` |
| Filesystem | `os.remove`, `os.rmdir`, `shutil.rmtree`, `os.rename` |
| Permissions | `os.chmod`, `os.chown`, `os.chroot` |
| Memory | `resource.setrlimit` (256MB default) |
| Misc | `builtins.exit`, `builtins.quit`, `faulthandler.disable()` |

Also blocks modules: `ipdb`, `joblib`, `resource`, `psutil`, `tkinter`.

**Limitations (NOT covered):**
- Network access (sockets can still be opened)
- Python dynamic features (ctypes could bypass restrictions)
- No kernel-level isolation (no seccomp, containers, or virtualization)

### `execute_code(code, timeout=5.0, maximum_memory_bytes=256MB)` (line 286)

The main API. Runs code in a **separate process** for full isolation:

1. Create a `multiprocessing.Manager().dict()` for IPC
2. Spawn a child process running `_unsafe_execute`
3. Wait for it to finish (with timeout + 1 second grace)
4. If still alive after timeout: `p.kill()`
5. Return `ExecutionResult` from the shared dict

The child process (`_unsafe_execute`):
1. Create temp directory, change into it
2. Save references to cleanup functions (before disabling them)
3. Apply `reliability_guard` (disables dangerous functions)
4. Execute code with `exec()` inside `capture_io()` and `time_limit()`
5. Write results to shared dict
6. Restore cleanup functions for temp directory removal
