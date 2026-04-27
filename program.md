# autochat — Autonomous Research Program

You are an AI research agent optimizing a small GPT language model.
The model is a word-level Transformer trained on Chinese QA pairs, built on
top of [`numpy-grad`](https://github.com/ryansoq/numpy-grad) — our own
array-level autograd engine in pure NumPy.

## Your Goal

Lower **val_bpb** (validation bits per byte). Lower is better.

## Rules

1. **Modify `train.py`** for model/training changes. The autograd framework
   (`numpy-grad`) is also fair game if a change there unlocks a real bpb win
   (e.g. dtype, fused op, init scheme); keep its tests passing if you do.
2. **Run training** with `python3 train.py --auto` (20-minute time budget).
3. **Check the result**: look at the final `bpb=` value printed at the end.
4. **Compare** with the previous best bpb in `experiments.jsonl`.
   - If bpb improved → **KEEP** the change, commit with a descriptive message.
   - If bpb got worse → **REVERT** (`git checkout train.py` and any
     numpy-grad changes), log why it failed.
5. **Repeat** with a new hypothesis. **One change per experiment** so you
   know what moved the needle.

## What You Can Change

| Layer | Where | Examples |
|-------|-------|----------|
| Architecture | `train.py` GPTMini | num_layers, d_model, d_ff, num_heads |
| Activation | `train.py` block | GELU → SiLU/Swish; SwiGLU vs GELU FFN |
| Normalization | `train.py` block | LN pre vs post; RMSNorm |
| Optimizer | `train.py` train() | Adam (β1, β2, ε); lr schedule shape |
| Training loop | `train.py` train() | warmup, cosine vs step LR, grad-clip threshold |
| Attention | `train.py` MultiHeadAttention (or numpy-grad) | window patterns, sparse attention |
| Regularization | `train.py` | dropout, weight decay, label smoothing |
| Init | `train.py` `__init__` (or numpy-grad Linear) | He/Xavier scale, embedding init std |
| Framework | `numpy-grad/numpy_grad/` | dtype (float32 vs float64), fused ops, grad accum |

## Strategy

- **Start with high-impact changes** (architecture, optimizer, dtype) before fine-tuning.
- **One change at a time** so you know what worked.
- **Log your reasoning** in commit messages.
- **Don't over-fit** — we care about generalization (val_bpb), not just training loss.
- **If a numpy-grad change is on the table**, run `pytest tests/` first; if any
  test breaks, fix it before kicking off training.

## Constraints

- Pure NumPy only (no PyTorch, no JAX, no external ML libraries).
- Must run on CPU (no GPU).
- Time budget: 20 minutes per experiment.
- The `TRAINING_DATA` list is fixed — don't modify it.
- Keep the `compute_bpb()` function and `experiments.jsonl` logging intact.

## Experiment Log Format

Each run appends to `experiments.jsonl`. Review past experiments before proposing changes.
