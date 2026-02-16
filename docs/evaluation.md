# NanoChat Component: Evaluation

**Files:** `nanochat/core_eval.py`, `nanochat/loss_eval.py`, `tasks/`

## 1. Overview

NanoChat uses two categories of evaluation:

| Category | Metric | When | What it Measures |
|----------|--------|------|------------------|
| **Base model** | Bits Per Byte (BPB) | During pretraining | Raw language modeling quality |
| **Base model** | CORE score | During/after pretraining | GPT-2 equivalent capability |
| **Chat model** | Pass@k on GSM8K | During RL training | Math reasoning accuracy |
| **Chat model** | Task-specific accuracy | After SFT/RL | Diverse capabilities |

---

## 2. Bits Per Byte (BPB) - `nanochat/loss_eval.py`

### `evaluate_bpb(model, batches, steps, token_bytes)` (line 9)

**What:** A tokenizer-invariant loss metric. Unlike raw cross-entropy loss (which depends on vocab size), BPB normalizes by the number of **bytes** each token represents.

**Why it matters:** If you change the tokenizer (e.g., from 32K to 64K vocab), the raw loss changes even if the model is equally good. BPB stays comparable across different tokenizers.

**Algorithm:**
```
BPB = total_nats / (ln(2) * total_bytes)

where:
  total_nats  = sum of cross-entropy losses for all valid tokens
  total_bytes = sum of byte lengths of all valid target tokens
```

**Implementation details:**
1. Compute per-token losses with `loss_reduction='none'`
2. For each target token, look up its byte length from `token_bytes` tensor
3. **Mask out:**
   - Special tokens (byte length = 0 in `token_bytes`)
   - Ignored tokens (target = -1, used for padding in SFT)
4. Sum losses and bytes separately
5. If distributed: `all_reduce` both sums across ranks
6. Divide: `total_nats / (log(2) * total_bytes)`

The `log(2)` conversion factor converts from nats (natural log) to bits.

---

## 3. CORE Metric - `nanochat/core_eval.py`

### What is CORE?

CORE (Comprehensive Open-source Reproducible Evaluation) is from the DCLM paper. It evaluates a base model (not chat model) on a suite of NLP tasks to produce a single aggregate score. The score is calibrated against GPT-2's performance.

### Task Types

The CORE evaluation handles three types of tasks:

#### Multiple Choice (`task_type='multiple_choice'`)
- Given a question + N answer choices, pick the most likely one
- Score: Mean cross-entropy loss over each choice's tokens
- Lowest loss = predicted answer
- Examples: ARC, MMLU, HellaSwag

#### Schema (`task_type='schema'`)
- Given N different contexts + same continuation, pick which context fits best
- Score: Mean cross-entropy loss over the continuation given each context
- Lowest loss = predicted context
- Examples: WinoGrande, COPA

#### Language Modeling (`task_type='language_modeling'`)
- Given a context, predict the exact continuation token-by-token
- Score: 1 if all tokens predicted correctly (argmax), 0 otherwise
- Examples: LAMBADA

### Key Functions

#### `render_prompts_mc(item, continuation_delimiter, fewshot_examples)` (line 17)
Renders N prompts for a multiple choice question, one per answer option. Each prompt has the same few-shot prefix but a different answer continuation.

```
Few-shot example 1: Q1? Answer: A1
Few-shot example 2: Q2? Answer: A2
Current question: Q? Answer: <option>
```

#### `render_prompts_schema(item, continuation_delimiter, fewshot_examples)` (line 36)
Renders N prompts for a schema question, one per context option. Each prompt has a different context but the same continuation.

#### `render_prompts_lm(item, continuation_delimiter, fewshot_examples)` (line 56)
Renders two prompts: one without the continuation (just the context) and one with it. The difference in token IDs tells us which tokens are the "continuation" to predict.

#### `find_common_length(token_sequences, direction)` (line 86)
Finds how many tokens are shared across sequences from the left (prefix) or right (suffix). Used to identify where the answer/continuation starts in tokenized prompts.

#### `batch_sequences_mc(tokenizer, prompts)` (line 113)
Tokenizes MC prompts and finds the common prefix (all prompts share the same question text). Returns `(tokens, start_indices, end_indices)` where `start:end` marks each answer's tokens.

#### `batch_sequences_schema(tokenizer, prompts)` (line 123)
Tokenizes schema prompts and finds the common suffix (all prompts share the same continuation). Returns indices marking the continuation.

#### `batch_sequences_lm(tokenizer, prompts)` (line 133)
Tokenizes LM prompts. The continuation starts where the "without" prompt ends.

#### `stack_sequences(tokens, pad_token_id)` (line 104)
Pads token sequences to equal length and stacks into a `(B, T)` tensor. Uses BOS as the pad token.

#### `forward_model(model, input_ids)` (line 144)
Runs the model on `input_ids` and returns:
- `losses`: Cross-entropy at each position (target is the next token)
- `predictions`: Argmax token at each position
Last column's loss is NaN (no target for the last token).

#### `evaluate_example(idx, model, tokenizer, data, device, task_meta)` (line 167)
Evaluates a single example:
1. Sample few-shot examples (deterministic RNG per example)
2. Render and tokenize prompts based on task type
3. Truncate to model's max sequence length if needed
4. Forward through model
5. Score:
   - **MC/Schema**: Option with lowest mean loss wins
   - **LM**: Check if all predicted tokens match actual tokens

#### `evaluate_task(model, tokenizer, data, device, task_meta)` (line 244)
Evaluates all examples in a task, distributed across ranks:
- Each rank processes examples `rank, rank+world_size, rank+2*world_size, ...`
- Results aggregated via `all_reduce(SUM)`
- Returns mean accuracy

---

## 4. Task Datasets (`tasks/`)

### `tasks/common.py` - Base Classes

#### `TaskMixture` (line ~1)
Combines multiple datasets by concatenation. Accessing index `i` finds which sub-dataset it belongs to and returns the corresponding example.

Used in SFT to blend SmolTalk + MMLU + GSM8K + SpellingBee.

#### `TaskSequence`
Sequential access to a dataset with DDP-aware indexing.

### `tasks/gsm8k.py` - Grade School Math 8K

- 8K training / 1.3K test math problems
- Each problem has a step-by-step solution ending with `#### <answer>`
- **Reward function**: Extract the number after `####` from both ground truth and generation, compare
- **Tool use**: Solutions include calculator expressions in `<|python_start|>...<|python_end|>` format
- The task provides both SFT training data (question + solution pairs) and RL rewards

### `tasks/mmlu.py` - Massive Multitask Language Understanding

- Multiple choice questions across 57 subjects
- Format: Question + 4 options (A/B/C/D)
- Used in SFT training and evaluation

### `tasks/smoltalk.py` - SmolTalk

- 460K general conversation examples from HuggingFace
- Multi-turn dialogues covering diverse topics
- Primary source of conversational ability during SFT

### `tasks/spellingbee.py` - Spelling & Letter Counting

Two variants:
- **SimpleSpelling**: "Spell the word 'apple'" -> "a-p-p-l-e"
- **SpellingBee**: "How many 'r' are in 'strawberry'?" -> Uses calculator tool

These teach the model character-level awareness (normally weak in BPE-tokenized models).

### `tasks/arc.py` - AI2 Reasoning Challenge

- Science questions from grade school exams
- Multiple choice format
- Tests basic scientific reasoning

### `tasks/humaneval.py` - HumanEval

- Python coding problems from OpenAI
- Model generates function bodies
- Evaluated by running test cases in sandbox (`execution.py`)

### `tasks/customjson.py` - Custom JSONL

Loads conversations from a JSONL file. Used for identity conversations (teaching the model its name/personality).

---

## 5. Evaluation Scripts

### `scripts/base_eval.py`
Evaluates a pretrained base model:
- CORE metric (aggregate of multiple NLP benchmarks)
- BPB on validation set
- Text samples from the model

### `scripts/chat_eval.py`
Evaluates a chat model (SFT or RL):
- GSM8K accuracy (math reasoning)
- MMLU accuracy (general knowledge)
- ARC accuracy (science)
- HumanEval Pass@1 (coding)
- SpellingBee accuracy (letter counting)

### `scripts/tok_eval.py`
Evaluates the tokenizer:
- Compression rates on various text types
- Comparison with GPT-2/GPT-4 tokenizers
- Token frequency analysis
