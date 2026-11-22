"""Scientific plots of one-run development factor results and actual elapsed time."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--local-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--update-layout',action='store_true',help='Regenerate only derived figures; no experiment or score is rerun')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.update_layout);data=read(args.summary);local=read(args.local_summary)
    order=['00','10','01','11'];colors=['#777777','#3975b6','#d39132','#2b927c'];labels=['00: neither','10: facts','01: calibrated','11: both']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),constrained_layout=True)
    for ax,name in zip(axes,('coupon','preference'),strict=True):
        arms=data['tasks'][name]['arms'];scores=[100*arms[a]['score'] for a in order]
        ax.scatter(range(4),scores,c=colors,s=95,zorder=3)
        base=local['tasks'][name]['arms'].get('full',local['tasks'][name]['arms'].get('shared'))
        ax.axhline(100*base['score'],ls='--',color='#333333',lw=1.2,label='Same-source local full')
        ax.set_xticks(range(4),['00','10','01','11']);ax.set_title(name.capitalize());ax.set_ylabel('Development accuracy (%)')
        ax.set_ylim(max(0,min(scores+[100*base['score']])-1.4),min(100,max(scores+[100*base['score']])+1.4))
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for i,score in enumerate(scores):ax.text(i,score+.07,f'{score:.2f}',ha='center',va='bottom',fontsize=9)
        ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Facts and calibrated proposals: one run per cell\n00 neither | 10 facts | 01 calibrated | 11 both',fontsize=11)
    for extension in ('png','svg'):
        path=args.output/('accuracy_factors.'+extension);fig.savefig(path,dpi=180)
        if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),constrained_layout=True)
    for ax,name in zip(axes,('coupon','preference'),strict=True):
        for a,color,label in zip(order,colors,labels,strict=True):
            value=data['tasks'][name]['arms'][a];ax.scatter(value['seconds'],100*value['score'],s=70,color=color,label=label)
            offset={'00':(5,8),'10':(5,8),'01':(-18,8),'11':(-18,-12)}[a]
            ax.annotate(a,(value['seconds'],100*value['score']),xytext=offset,textcoords='offset points',fontsize=8)
        base=local['tasks'][name]['arms'].get('full',local['tasks'][name]['arms'].get('shared'))
        ax.scatter(base['seconds'],100*base['score'],marker='^',s=75,color='#252525',label='Local full')
        ax.set_title(name.capitalize());ax.set_xlabel('Full CLI path elapsed seconds');ax.set_ylabel('Development accuracy (%)');ax.grid(alpha=.18)
    handles,texts=axes[0].get_legend_handles_labels();fig.legend(handles,texts,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Measured accuracy and runtime; no replicate error bars',fontsize=11)
    for extension in ('png','svg'):
        path=args.output/('accuracy_runtime.'+extension);fig.savefig(path,dpi=180)
        if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    save(args.output/'provenance.json',dict(summary_sha256=sha(args.summary),local_summary_sha256=sha(args.local_summary),
        classifier_fits=0,API_requests=0,scope='Known development tasks, one cell each, no uncertainty or generalization claim'))
