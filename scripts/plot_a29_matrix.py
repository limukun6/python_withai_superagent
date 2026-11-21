"""Plot recorded A29 development outcomes and actual runtime; no new experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ARMS = ('L0', 'L1', 'R0', 'E0', 'R1', 'E1')
COLORS = ['#64748b', '#0f766e', '#94a3b8', '#c084fc', '#38bdf8', '#7c3aed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    tasks = ('coupon', 'preference')
    x = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axes, tasks):
        rows = summary['tasks'][name]['arms']
        values = [(rows[arm]['score'] - rows['L0']['score']) * 100 for arm in ARMS]
        ax.bar(x, values, color=COLORS, width=.7)
        ax.axhline(0, color='#334155', linewidth=.8)
        ax.set_xticks(x, ARMS)
        for index, value in enumerate(values):
            ax.annotate(f'{value:+.3f}', (index, value), xytext=(0, 5 if value >= 0 else -13),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(f'{name}: L0 accuracy = {rows["L0"]["score"]:.4f}')
        ax.set_ylabel('Accuracy difference from L0 (percentage points)')
        ax.set_ylim(min(values) - .6, max(values) + .6)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('A29: six complete agent paths per development task', fontsize=13)
    fig.text(.5, .01, 'One development split per task; one run per arm. No significance or unseen-task claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / ('accuracy_differences.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for column, name in enumerate(tasks):
        rows = summary['tasks'][name]['arms']
        explore = np.array([rows[arm]['action_counts'].get('explore', 0) for arm in ARMS])
        confirm = np.array([rows[arm]['action_counts'].get('confirm', 0) for arm in ARMS])
        fits = np.array([rows[arm]['actual_classifier_fits'] for arm in ARMS])
        refit = fits - explore - confirm
        if not np.all(refit == 1):
            raise ValueError('Recorded fits do not equal exploration + confirmation + one refit')
        ax = axes[0, column]
        ax.bar(x, explore, label='Explore', color='#38bdf8')
        ax.bar(x, confirm, bottom=explore, label='Confirm', color='#7c3aed')
        ax.bar(x, refit, bottom=explore + confirm, label='Refit', color='#334155')
        ax.set_xticks(x, ARMS)
        ax.set_title(name)
        ax.set_ylabel('Actual classifier fits')
        ax.set_ylim(0, 12)
        ax.legend(fontsize=8, ncol=3, loc='upper left')
        planning = np.array([sum(rows[arm]['call_latencies_seconds']) for arm in ARMS])
        total = np.array([rows[arm]['path_seconds'] for arm in ARMS])
        ax = axes[1, column]
        ax.bar(x, total - planning, label='Training and other', color='#0f766e')
        ax.bar(x, planning, bottom=total - planning, label='LLM request time', color='#c084fc')
        ax.set_xticks(x, ARMS)
        ax.set_ylabel('Observed path seconds')
        ax.legend(fontsize=8)
    fig.suptitle('A29: actual budget use and observed runtime', fontsize=13)
    fig.text(.5, .01, 'Equal limits do not imply equal cost. Runtime is a single observation, not a stable speed estimate.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / ('fits_and_runtime.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='plotted', classifier_fits=0, api_requests=0, figures=4)))
