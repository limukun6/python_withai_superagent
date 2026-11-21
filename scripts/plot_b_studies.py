"""Render the two B-line study figures from the audited, portable receipts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


TASKS = {'coupon': 'Task 1 · Coupons', 'headlines': 'Task 2 · Headlines', 'bean': 'Dry Bean', 'sms': 'SMS Spam'}
SEEDS = ('45102','45103')
INK, BLUE, GOLD, GRAY = '#253946', '#126a76', '#ad5321', '#62747c'
ROOT = Path(__file__).resolve().parents[1] / 'research'


def read(file):
    return json.loads(file.read_text(encoding='utf-8'))


def style(axis):
    axis.spines[['top','right','left']].set_visible(False)
    axis.spines['bottom'].set_color('#b9c8ca')
    axis.tick_params(length=0, pad=7, colors=INK)
    axis.grid(axis='x', color='#e1e9e8', zorder=0)
    axis.set_axisbelow(True)
    axis.set_ylim(3.6, -0.6)


def catalogue(out):
    result = read(ROOT/'bq_01_portfolios/results.json')['tasks']
    fig, ax = plt.subplots(figsize=(8.8,4.7))
    fig.subplots_adjust(left=.25, right=.96, top=.77, bottom=.19)
    fig.suptitle('What changed when more methods were available?', x=.04, ha='left',
                 fontsize=15, fontweight='bold', color=INK)
    ax.set_title('Each pair uses the same training holdout. SMS uses macro-F1. Other tasks use accuracy.',
                 loc='left', fontsize=9, color=GRAY, pad=10)
    for i, name in enumerate(TASKS):
        row = result[name]
        delta = row['primary_difference_percentage_points']
        ax.barh(i, delta, height=.42, color=BLUE)
        if delta == 0:
            ax.scatter([0],[i],color=BLUE,s=34, zorder=4)
        ax.text(delta+.055, i, f"{delta:+.2f} pp" if delta else '0.00 pp', va='center', color=INK, fontsize=10)
    ax.set_yticks(range(4), [
        f"{title}\n{100*result[k]['original_score']:.2f}% → {100*result[k]['expanded_score']:.2f}%"
        for k,title in TASKS.items()])
    ax.set_xlim(-.05,2.95)
    ax.set_xlabel('Expanded catalogue minus original catalogue (percentage points)', color=INK, labelpad=9)
    style(ax)
    fig.savefig(out/'catalogue.png',dpi=190,facecolor='white')
    plt.close(fig)


def allocation(out):
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.7),gridspec_kw={'width_ratios': [1,1.1]})
    fig.subplots_adjust(left=.17, right=.97, wspace=.19, top=.75, bottom=.23)
    fig.suptitle('When did feedback-guided scheduling help?',x=.035,ha='left',fontsize=16,
                 fontweight='bold',color=INK)
    for ax, title in zip(axes,['Change in primary score', 'Training and selection time saved'],strict=True):
        style(ax)
        ax.axvline(0,color='#a0b1b4',lw=1.1)
        ax.set_title(title,loc='left',fontsize=11,color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for i,name in enumerate(TASKS):
        folder = ROOT/'b1_07_allocation' if name=='headlines' else ROOT/'b1_08_allocation'/name
        receipt = read(folder/'results.json')
        paths = {p['name']:p for p in receipt['paths']}
        for idx,seed in enumerate(SEEDS):
            f,s = (paths[seed+'_'+p] for p in ('fixed','adaptive'))
            assert f['delivery']['primary_metric'] == s['delivery']['primary_metric']
            gain = 100*(s['delivery']['score']-f['delivery']['score'])
            saved = f['seconds']-s['seconds']
            for ax,x in zip(axes,[gain,saved],strict=True):
                ax.scatter(x,i+(-.13 if idx==0 else .13),marker=('o','D')[idx],
                           color=(BLUE,GOLD)[idx],s=50,edgecolors='white',linewidth=.55,zorder=3)
    axes[0].set_yticks(range(4),[title+'\n'+('Macro-F1' if name=='sms' else 'Accuracy') for name,title in TASKS.items()])
    axes[1].set_yticks(range(4),['']*4)
    axes[0].set_xlim(-.24,3.05)
    axes[1].set_xlim(-4,46)
    axes[0].set_xlabel('Adaptive minus fixed (percentage points)',color=INK,labelpad=9)
    axes[1].set_xlabel('Fixed minus adaptive (seconds)',color=INK,labelpad=9)
    symbols = [Line2D([0],[0],ls='',marker=m,color=c,label=f'Holdout {s}') for m,c,s in zip(('o','D'),(BLUE,GOLD),SEEDS)]
    fig.legend(handles=symbols,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
    fig.savefig(out/'allocation.png',dpi=190,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    catalogue(args.output)
    allocation(args.output)
