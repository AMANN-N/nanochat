# NanoChat Technical Documentation

## 1. System Overview

NanoChat is a minimal, hackable, end-to-end LLM training and deployment harness. It is designed to be run on a single 8xH100 GPU node and covers the entire lifecycle of an LLM:

1.  **Tokenization**: BPE tokenizer training and inference.
2.  **Pretraining**: Training a base GPT model on FineWeb-Edu.
3.  **Supervised Fine-Tuning (SFT)**: Tuning the model on instruction datasets (SmolTalk, MMLU, GSM8K).
4.  **Reinforcement Learning (RL)**: Optimizing the model for math reasoning (GSM8K) using a simplified GRPO-like algorithm.
5.  **Inference & Deployment**: Serving the model via a CLI or a ChatGPT-like Web UI, with support for Python tool use.

The system is designed around **Scaling Laws**, where a single parameter (`--depth`) determines all other hyperparameters (width, learning rates, batch size, etc.) to ensure compute-optimal training.

---

## 2. Directory Structure & Key Files

### `nanochat/` (Core Library)
*   **`gpt.py`**: The Transformer model definition.
    *   Features: Rotary Embeddings, QK Norm, Untied Embedding/Head weights, ReLU^2 MLP, RMSNorm (no learnable params), No biases, GQA.
    *   Supports Flash Attention 3 on Hopper GPUs (via `flash_attn.py`), falls back to SDPA.
*   **`engine.py`**: High-performance inference engine.
    *   Manages KV Cache (`KVCache` class) for efficient generation.
    *   Handles "Tool Use" (detects `<|python_start|>` tokens, executes code, feeds output back).
*   **`optim.py`**: The **Muon** Optimizer.
    *   A fused implementation of AdamW (for embeddings/scalars) and Muon (for 2D matrix parameters).
    *   Muon orthogonalizes updates via Newton-Schulz iteration, allowing for faster convergence.
*   **`tokenizer.py`**: BPE Tokenizer.
    *   Hybrid approach: Uses `rustbpe` for fast training and `tiktoken` for fast inference.
    *   Vocabulary size: ~32k.
*   **`dataset.py`**: Data management.
    *   Downloads FineWeb-Edu parquet shards on demand.
*   **`dataloader.py`**: Distributed Data Loader.
    *   Implements "BOS-aligned bestfit packing": Documents are packed to minimize waste (no padding), and every sequence starts with a `<|bos|>` token.
*   **`execution.py`**: Sandboxed Python execution environment.
    *   Used for the "Calculator" / Python tool capabilities.
    *   Runs code in a temporary directory with restricted system calls.

### `scripts/` (Workflows)
*   **`base_train.py`**: Pretraining script.
    *   Trains the base model.
    *   Implements scaling laws logic to auto-configure hyperparameters based on model depth.
*   **`chat_sft.py`**: Supervised Fine-Tuning script.
    *   Trains on a mixture of datasets (SmolTalk, MMLU, GSM8K, SpellingBee).
*   **`chat_rl.py`**: Reinforcement Learning script.
    *   Optimizes the model on GSM8K using a simplified GRPO algorithm (Group Relative Policy Optimization).
    *   Generates multiple samples, scores them, and updates the policy without a critic model.
*   **`chat_web.py`**: Web Interface.
    *   Launches a FastAPI backend and serves `nanochat/ui.html` for a ChatGPT-like experience.

---

## 3. Detailed Component Analysis

### The Model (`gpt.py`)
The architecture is a modernized GPT-2/LLaMA hybrid:
*   **Positional Embeddings**: Rotary (RoPE), precomputed in `gpt.py`.
*   **Normalization**: RMSNorm with no learnable parameters (`torch.nn.functional.rms_norm`).
*   **Attention**: Grouped Query Attention (GQA) to reduce KV cache size. Supports Sliding Window Attention (e.g., "SSSL" pattern) to handle long contexts efficiently.
*   **Value Embeddings**: A "ResFormer"-inspired technique where value embeddings are mixed into the attention mechanism.

### The Optimizer (`optim.py`)
NanoChat uses a split optimization strategy:
1.  **Muon**: Applied to all 2D internal matrices (Attention projections, MLP weights). Optimizes the *topology* of the loss landscape by orthogonalizing updates.
2.  **AdamW**: Applied to 1D tensors (LayerNorms, biases if any) and Embeddings.
3.  **Learning Rate Schedule**: Auto-calculated based on batch size and `depth`.

### Tokenizer (`tokenizer.py`)
Uses the GPT-4 regex split pattern but slightly modified (`\p{N}{1,2}` instead of `{1,3}`).
*   **Special Tokens**:
    *   `<|bos|>`: Beginning of Sequence.
    *   `<|user_start|>`, `<|user_end|>`: User message delimiters.
    *   `<|assistant_start|>`, `<|assistant_end|>`: Assistant message delimiters.
    *   `<|python_start|>`, `<|python_end|>`: Model-generated code blocks.
    *   `<|output_start|>`, `<|output_end|>`: Execution output blocks.

---

## 4. End-to-End Workflow

### Step 1: Data Preparation
```bash
# Downloads FineWeb-Edu shards to data/base_data/
python -m nanochat.dataset
```
This downloads parquet files. The `dataloader.py` reads these directly, tokenizes on the fly, and packs them into batches.

### Step 2: Pretraining
```bash
# Train a model with depth=12 (approx GPT-2 Small size)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=12 --model-tag="d12"
```
*   Configures itself automatically.
*   Saves checkpoints to `checkpoints/base_checkpoints/d12/`.

### Step 3: Supervised Fine-Tuning (SFT)
```bash
# Load the pretrained d12 model and fine-tune
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag="d12"
```
*   Uses `TaskMixture` to blend General Chat, Math, and Logic tasks.
*   Saves to `checkpoints/chatsft_checkpoints/d12/`.

### Step 4: Reinforcement Learning (RL)
```bash
# Further optimize the SFT model on GSM8K
torchrun --nproc_per_node=8 -m scripts.chat_rl --model-tag="d12"
```
*   Generates multiple solutions for math problems.
*   Verifies correctness (Reward = 1 if correct, 0 otherwise).
*   Updates model to increase probability of correct reasoning paths.

### Step 5: Inference / Serving
```bash
# Serve the model
python -m scripts.chat_web
```
*   Loads the latest checkpoint.
*   Opens a web server at `http://localhost:8000`.
*   The `Engine` handles the chat loop and will automatically execute Python code if the model generates `<|python_start|>`.

---

## 5. Tool Use & Reasoning
NanoChat features integrated Python tool use.
1.  **Generation**: The model generates `<|python_start|> print("Hello") <|python_end|>`.
2.  **Detection**: `Engine.generate` detects the closing tag.
3.  **Execution**: The code is extracted and passed to `execution.execute_code` (sandboxed).
4.  **Feedback**: The output is formatted as `<|output_start|> Hello <|output_end|>` and appended to the context.
5.  **Continuation**: The model continues generating based on the tool output.
