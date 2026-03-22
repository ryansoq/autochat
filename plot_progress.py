#!/usr/bin/env python3
"""Generate progress.png from experiments.jsonl"""

import json
import os

def plot():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.jsonl')
    if not os.path.exists(log_path):
        print("No experiments.jsonl found")
        return

    experiments = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))

    if not experiments:
        print("No experiments logged yet")
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, generating text report instead")
        print(f"\n{'#':>4} | {'bpb':>8} | {'loss':>8} | {'epochs':>6} | {'time':>6} | timestamp")
        print("-" * 70)
        for i, exp in enumerate(experiments):
            print(f"{i+1:4d} | {exp.get('final_bpb', 0):8.4f} | {exp.get('final_loss', 0):8.4f} | "
                  f"{exp.get('epochs', 0):6d} | {exp.get('elapsed_s', 0):5.0f}s | {exp.get('timestamp', '')}")
        return

    bpbs = [e.get('final_bpb', 0) for e in experiments]
    losses = [e.get('final_loss', 0) for e in experiments]
    xs = list(range(1, len(experiments) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('autochat — Autonomous Research Progress', fontsize=14, fontweight='bold')

    # BPB plot
    ax1.plot(xs, bpbs, 'b-o', markersize=4, label='val_bpb')
    ax1.set_ylabel('Bits Per Byte (lower is better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if len(bpbs) > 1:
        best_idx = bpbs.index(min(bpbs))
        ax1.annotate(f'best: {bpbs[best_idx]:.4f}',
                     xy=(best_idx + 1, bpbs[best_idx]),
                     fontsize=9, color='green', fontweight='bold')

    # Loss plot
    ax2.plot(xs, losses, 'r-o', markersize=4, label='loss')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_xlabel('Experiment #')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.png')
    plt.savefig(out_path, dpi=150)
    print(f"📊 Saved progress.png ({len(experiments)} experiments)")


if __name__ == '__main__':
    plot()
