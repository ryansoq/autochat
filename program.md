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

## Loop mechanics — for any agent picking this up

This file is the contract. Any agent (Claude Code, Aqua, GPT-4-tools, a
self-hosted runner) can drive the loop as long as it follows the cycle
below. The state file `autoresearch-state.json` is the single source of
truth — read it first, write it last, hold the lock in between.

### Per-tick algorithm

```
1. read program.md (this file) — confirm rules + budget haven't changed
2. read autoresearch-state.json — see best_bpb, current_pid, last_result
3. branch:
   3a. current_pid is set AND alive
       → tail current_log; do nothing else; exit
   3b. current_pid is set AND finished (process gone, log has "Total:" line)
       → harvest the result:
         - extract final bpb from log
         - if bpb < best_bpb_numpy_grad → KEEP path (3b-K)
         - else → REVERT path (3b-R)
       → clear current_pid / current_started / current_hypothesis / current_log
   3c. current_pid is null
       → propose next hypothesis (HYP{N+1}), apply edit to train.py
         and/or numpy-grad, run pytest if numpy-grad touched, then:
           PYTHONPATH=<numpy-grad path> nohup python3 -u train.py --auto \
             > /tmp/autochat-hyp{N}-$(date +%s).log 2>&1 &
       → write new current_pid / current_started / current_hypothesis /
         current_log into state file
```

### KEEP path (3b-K)

```
- update experiments.jsonl entry: {"kept": true, "notes": "HYP{N} ..."}
- update autoresearch-state.json: best_bpb, best_bpb_commit (placeholder),
  push hypothesis to tried_hypotheses
- regenerate progress.png:  python3 plot_progress.py
- git add train.py progress.png autoresearch-state.json
- git commit with descriptive message: HYP{N} ... — bpb X (-Y% vs prior best)
- git push
```

### REVERT path (3b-R)

```
- update experiments.jsonl entry: {"kept": false, "notes": "HYP{N} ... — REVERT"}
- git checkout train.py  (and any numpy-grad files modified)
- update autoresearch-state.json: leave best_bpb unchanged, push hypothesis
  to tried_hypotheses with kept:false
- regenerate progress.png:  python3 plot_progress.py
- git add progress.png autoresearch-state.json   (NOT train.py — it's reverted)
- git commit with message: HYP{N} ... — REVERT (bpb X vs best Y, +Z%)
- git push  (only the chart + state — main branch stays at the best train.py)
```

### Conventions

- **HYP numbering** monotonically increases. Skipped numbers are fine
  (HYP6 then HYP9 is OK if you abandoned HYP7-8 mid-design).
- **One change per experiment**. If you mix two ideas you can't tell
  which one moved the needle.
- **state file `current_pid` is the lock**. Never start a new experiment
  while one is running. If you crash mid-experiment, manually clear
  current_pid so the next tick can proceed.
- **`experiments.jsonl` is gitignored** — it's a runtime log. Failed
  entries stay there with `kept:false` so `progress.png` can show them
  as small grey dots. Don't delete failed entries.
- **Quiet hours (23:00-08:00 local)**: commit OK, push OK, but avoid
  starting new experiments — let any in-flight one finish, leave fresh
  proposals to the next morning.

### Hooking into a heartbeat

Any periodic mechanism that injects "read program.md and run one tick"
into your agent will work. Examples:

- ClawX `config.json` schedule:
  ```json
  "autochat-loop": {
    "enabled": true,
    "cron": "*/30 * * * *",
    "prompt": "Read /path/to/autochat/program.md and run one tick of the autoresearch loop."
  }
  ```
- A standalone shell loop with sleep:
  `while true; do <invoke-agent-with-program.md>; sleep 1800; done`
- An external scheduler (systemd timer, GitHub Actions cron, etc) that
  drives the agent through `claude --inject "..."` or similar.

The agent's job is just to follow the per-tick algorithm above. The
heartbeat mechanism is interchangeable.
