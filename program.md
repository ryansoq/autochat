# autochat — Autonomous Research Program

You are an AI research agent optimizing a small GPT language model (`train.py`).
The model is a char-level Transformer trained on Chinese QA pairs, built with pure NumPy (no PyTorch).

## Your Goal

Lower **val_bpb** (validation bits per byte). Lower is better.

## Rules

1. **Only modify `train.py`.** Do not touch `program.md` or any other files.
2. **Run training** with `python3 train.py --auto` (10-minute time budget).
3. **Check the result**: look at the final `bpb=` value printed at the end.
4. **Compare** with the previous best bpb in `experiments.jsonl`.
   - If bpb improved → **KEEP** the change, commit with a descriptive message.
   - If bpb got worse → **REVERT** (`git checkout train.py`), log why it failed.
5. **Repeat** with a new hypothesis.

## What You Can Change

Everything in `train.py` is fair game:

| Category | Examples |
|----------|----------|
| Architecture | num_layers, d_model, d_ff, num_heads |
| Activation | GELU → Swish, ReLU, SiLU |
| Normalization | LayerNorm position (pre vs post), RMSNorm |
| Optimizer | Adam params (β1, β2, ε), learning rate schedule |
| Training | batch ordering, sequence length, warmup strategy |
| Attention | window patterns, causal mask variants |
| Regularization | dropout, weight decay |
| Initialization | Xavier, Kaiming, scaled init |

## Strategy

- **Start with high-impact changes** (architecture, optimizer) before fine-tuning.
- **One change at a time** so you know what worked.
- **Log your reasoning** in commit messages.
- **Don't over-fit** — we care about generalization (val_bpb), not just training loss.

## Constraints

- Pure NumPy only (no PyTorch, no JAX, no external ML libraries).
- Must run on CPU (no GPU).
- Time budget: 10 minutes per experiment.
- The `TRAINING_DATA` list is fixed — don't modify it.
- Keep the `compute_bpb()` function and `experiments.jsonl` logging intact.

## Experiment Log Format

Each run appends to `experiments.jsonl`. Review past experiments before proposing changes.
