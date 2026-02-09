# NanoChat Component: Tokenizer

**File:** `nanochat/tokenizer.py`
**Training Script:** `scripts/tok_train.py`

## 1. The Gist
The Tokenizer is the gateway to the LLM. It translates human-readable text strings into **Integers** (Token IDs), not Embeddings.

> **Crucial Distinction**:
> *   **Tokenizer**: Text $\to$ Integers (e.g., "Hello" $\to$ `15496`)
> *   **Model**: Integers $\to$ Embeddings (Vectors) (e.g., `15496` $\to$ `[0.12, -0.4, ...]`)

NanoChat uses a **Byte-Pair Encoding (BPE)** tokenizer with a vocabulary size of **32,768**.

### Why this design?
*   **Dual-Backend**:
    *   **Training**: Uses `rustbpe`, a custom Rust implementation that is extremely fast at learning BPE merges from raw text.
    *   **Inference**: Uses OpenAI's `tiktoken`. Tiktoken is highly optimized, releases the Python GIL (Global Interpreter Lock), and allows for massive throughput during data loading.
*   **Vocab Size (32,768)**: Standard BPE size (2^15). Fits easily into `uint16`, keeping dataset storage compact.
*   **GPT-4 Style**: Uses regex-based splitting to ensure high-quality tokens (separating numbers, punctuation, etc.).

---

## 2. Key Concepts

### Regex Splitting
Before BPE merges occur, text is split into chunks to prevent "bleeding" across semantic boundaries. NanoChat uses a pattern very similar to GPT-4:

```python
# nanochat/tokenizer.py
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

**Key Difference**: `\p{N}{1,2}` (Numbers are grouped in chunks of 1 or 2).
*   *GPT-4* uses `{1,3}` (up to 3 digits).
*   *NanoChat* uses `{1,2}`. This reduces the number of unique number tokens, saving vocabulary space for more words, which is crucial for smaller vocabularies like 32k.

### Special Tokens
These tokens structure the conversation. They are **not** BPE-merged but added explicitly.

| Token | ID (approx) | Purpose |
| :--- | :--- | :--- |
| `<|bos|>` | ~32768 | **Beginning of Sequence**. Starts every example. |
| `<|user_start|>` | ~32769 | Starts a User message. |
| `<|user_end|>` | ~32770 | Ends a User message. |
| `<|assistant_start|>` | ~32771 | Starts an Assistant response. |
| `<|assistant_end|>` | ~32772 | Ends an Assistant response. |
| `<|python_start|>` | ~32773 | Model opens a code block. |
| `<|python_end|>` | ~32774 | Model closes a code block. |
| `<|output_start|>` | ~32775 | System provides code execution output. |
| `<|output_end|>` | ~32776 | End of execution output. |

| `<|output_end|>` | ~32776 | End of execution output. |

---

## 3. Why Train a Tokenizer?

You might ask: *Why not just use ASCII/Unicode bytes? Why train this BPE thing?*

**1. Compression & Efficiency**
*   **Raw Text**: "The quick brown fox" is 19 bytes (integers).
*   **Tokenized**: Using BPE, common words like "The", "quick", "brown" become **single integers**. The sequence length drops from 19 to 4.
*   **Benefit**: The model has a fixed context window (e.g., 2048 items). If we use bytes, it sees 2000 *characters* (~400 words). If we use tokens, it sees 2000 *tokens* (~1500 words). **Training allows the model to "see" 4x more context.**

**2. Semantics**
*   Training learns that "ing", "ed", "tion" are common sub-units. This gives the model a head start on understanding language structure compared to raw character-level inputs.

**3. What do we get?**
When "training" is done, we get two things:
1.  **The Vocabulary**: A list of 32,768 unique byte sequences.
    *   ID 65 -> "A"
    *   ID 145 -> "ing"
    *   ID 5042 -> " Apple"
2.  **The Merges**: The rules to convert text into these IDs.
    *   "See `t` and `h` adjacent? Merge them into `th`."
    *   "See `th` and `e` adjacent? Merge them into `the`."

---

## 4. The Training Process (`scripts/tok_train.py`)

You don't just "get" a tokenizer; you train it on your data.

1.  **Data Ingestion**: The script iterates over the FineWeb-Edu parquet files (`nanochat.dataset`).
2.  **Sampling**: It caps documents at 10,000 chars and samples up to 2 billion chars total to keep training fast.
3.  **RustBPE Training**:
    ```python
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    ```
    This learns the merge rules (e.g., `t` + `h` -> `th`).
4.  **Conversion to Tiktoken**: The valid BPE ranks are extracted and loaded into a `tiktoken.Encoding` object for efficient runtime usage.
5.  **Artifacts**:
    *   `tokenizer.pkl`: The pickled tiktoken object.
    *   `token_bytes.pt`: A mapping of `Token ID -> Byte Length`. Used for calculating **Bits Per Byte (BPB)**, a metric that allows comparing models with different tokenizers.

---

## 4. Chat Templating (`render_conversation`)

This is the most critical function for **Supervised Fine-Tuning (SFT)**. It turns a JSON conversation into a sequence of IDs and safeguards the training process via **Masking**.

### Code Walkthrough
**Function**: `RustBPETokenizer.render_conversation(conversation)`

**Input**:
```json
[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

**Step 1: Tokenization & Structure**
The function wraps messages in special tokens:
```text
<|bos|>
<|user_start|> Hello <|user_end|>
<|assistant_start|> Hi there! <|assistant_end|>
```

**Step 2: Mask Creation**
We only want to train the model to *generate the assistant's response*. We do NOT want to train it to generate user queries (that's predicting the user, not being a helpful assistant).
*   **Mask = 0 (Ignore)**: `<|bos|>`, User tokens, `<|user_start|>`, `<|user_end|>`.
*   **Mask = 1 (Train)**: Assistant text tokens (`Hi there!`), `<|assistant_end|>`.

**Visual Representation**:
```
Tokens: <|bos|> <|user_start|> Hello <|user_end|> <|assistant_start|> Hi there! <|assistant_end|>
Mask:      0           0          0         0               0            1    1            1
```

### Handling Tool Use
When the model generates code, we train on the code. When the system returns output, we mask the output (because the model doesn't generate the output, the Python interpreter does).

```
Tokens: ... <|python_start|> print(2+2) <|python_end|> <|output_start|> 4 <|output_end|>
Mask:              1              1             1               0         0        0
```

---

## 5. Usage in Code

### Loading
```python
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer() # loads from data/tokenizer/
```

### Encoding / Decoding
```python
# Basic
ids = tokenizer.encode("Hello world")
text = tokenizer.decode(ids)

# With delimiters (useful for manual prompt construction)
ids = tokenizer.encode("Hello", prepend="<|bos|>")
```

### For SFT (Training)
```python
ids, mask = tokenizer.render_conversation(conversation)
# ids: [32768, 32769, 15496, 32770, ...]
# mask: [0, 0, 0, 0, ...]
```

### For Inference (RL / Generation)
```python
# render_for_completion() strips the last assistant message and primes the model
ids = tokenizer.render_for_completion(conversation)
# Ends with <|assistant_start|>, ready for the model to complete.
```

---

## 6. Deep Dive: Inside `rustbpe`

You asked: *How is the Rust tokenizer actually coded?*

Since `rustbpe` is a compiled extension, we can't "see" it in Python, but we can infer its architecture. It is built using **PyO3** (to bind Rust to Python) and **Rayon** (for parallelism).

### 1. The Data Structures (Rust)
Conceptually, the tokenizer in Rust looks like this:

```rust
// pseudo-code
struct Tokenizer {
    // The Vocabulary: Maps bytes -> Token ID
    vocab: HashMap<Vec<u8>, u32>,
    // The Merge Rules: Maps pair of bytes -> Rank (priority)
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
}
```

### 2. The Training Algorithm (BPE)
The training process (`train_from_iterator`) implements the standard BPE algorithm but optimized for speed.

**The Loop:**
1.  **Count Pairs**: Iterate over all data. Count how often every pair of adjacent tokens appears.
    *   *python*: `('hu', 'man')`, `('hu', 'g')`...
    *   *optimization*: This is **parallelized**. The text is split into chunks. Multiple threads (via `Rayon`) count pairs in their chunk. Then, the counts are aggregated (Map-Reduce).
2.  **Find Best Pair**: Identify the pair with the highest count (e.g., `('e', 'r')`).
3.  **Merge**: Create a new token `er`. Replace all instances of `e` followed by `r` with `er`.
4.  **Repeat**: Do this until `vocab_size` is reached.

**Why Rust?**
*   **Memory Overhead**: Python integers are 28 bytes. Rust `u32` is 4 bytes. For billions of tokens, this is the difference between fitting in RAM or crashing.
*   **GIL-Free**: Rust releases the Global Interpreter Lock, allowing true multi-core processing.

### 3. The `tiktoken` Handoff
After training, `RustBPETokenizer` exports the learned **ranks** to `tiktoken`.

**Why?** `rustbpe` is great for training (which `tiktoken` doesn't do), but `tiktoken` is the state-of-the-art for **inference**.
*   **Inference Logic**: When encoding "hello", you don't iteratively merge. You use the ranks to perform a specialized search (often a Min-Heap or exact lookup) to find the segmentation with the *least number of tokens* that aligns with the BPE merges.
*   `tiktoken`'s core loop runs in pure Rust at gigabytes per second.

### 4. PyO3 Binding
Using `PyO3`, the Rust code is exposed as a Python class.

```rust
#[pyclass]
struct Tokenizer { inner: BPEModel }

#[pymethods]
impl Tokenizer {
    #[new]
    fn new() -> Self { ... }

    fn train(&mut self, text: Vec<String>) -> PyResult<()> {
        // ... heavy lifting in Rust ...
    }
}
```
This is why you see `import rustbpe` but no `.py` files. It's a `.so` (Linux) or `.pyd` (Windows) binary.

