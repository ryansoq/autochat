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
        for i, exp in enumerate(experiments):
            status = "KEPT" if exp.get('kept', True) else "DISC"
            print(f"{i+1:4d} | {exp.get('final_bpb', 0):8.4f} | {status} | {exp.get('timestamp', '')}")
        return

    bpbs = [e.get('final_bpb', 0) for e in experiments]
    xs = list(range(1, len(experiments) + 1))

    # Determine kept vs discarded
    kept_flags = []
    for e in experiments:
        kept_flags.append(e.get('kept', True))  # default: kept

    kept_xs = [x for x, k in zip(xs, kept_flags) if k]
    kept_bpbs = [b for b, k in zip(bpbs, kept_flags) if k]
    disc_xs = [x for x, k in zip(xs, kept_flags) if not k]
    disc_bpbs = [b for b, k in zip(bpbs, kept_flags) if not k]

    # Running best line
    running_best = []
    best = float('inf')
    for b, k in zip(bpbs, kept_flags):
        if k and b < best:
            best = b
        running_best.append(best)

    num_kept = sum(kept_flags)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'Autochat Progress: {len(experiments)} Experiments, {num_kept} Kept Improvements',
                 fontsize=11, loc='left')

    # Discarded — subtle background dots
    if disc_xs:
        ax.scatter(disc_xs, disc_bpbs, color='#BBBBBB', s=10, zorder=2,
                   alpha=0.5, label='Discarded', edgecolors='none')

    # Kept — prominent green
    if kept_xs:
        ax.scatter(kept_xs, kept_bpbs, color='#4CAF50', s=55, zorder=4,
                   label='Kept', edgecolors='#2E7D32', linewidths=0.5)

    # Running best — stepped so improvements stair-step down
    ax.plot(xs, running_best, '-', color='#4CAF50', linewidth=1.3,
            alpha=0.75, label='Running best', drawstyle='steps-post', zorder=3)

    # Annotations on every Kept point — diagonal up-right
    for i, exp in enumerate(experiments):
        if kept_flags[i]:
            label = exp.get('notes') or exp.get('description', '')
            if label:
                if len(label) > 38:
                    label = label[:35] + '...'
                ax.annotate(label, xy=(xs[i], bpbs[i]),
                            xytext=(6, 6), textcoords='offset points',
                            fontsize=7, color='#2E7D32', rotation=20, alpha=0.85)

    ax.set_ylabel('Validation BPB (lower is better)')
    ax.set_xlabel('Experiment #')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Zoom y-axis so kept points + running best fill the frame, like
    # karpathy/autoresearch — dropping early outliers off-screen is OK
    # because they're not informative once the search is past them.
    if kept_bpbs:
        y_lo = min(running_best) * 0.95
        y_hi = max(kept_bpbs) * 1.18
        ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.png')
    plt.savefig(out_path, dpi=150)
    print(f"📊 Saved progress.png ({len(experiments)} experiments, {num_kept} kept)")


if __name__ == '__main__':
    plot()
