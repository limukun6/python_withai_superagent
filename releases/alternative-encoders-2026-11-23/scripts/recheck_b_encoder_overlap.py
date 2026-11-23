"""Count unique training rows, disagreement, and clustered uncertainty across matched replications."""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import matched_runs,predictions
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.data import load_development_task
from superagent.splits import sample_groups


def run(inputs,control,batch,followup,out):
    groups=sample_groups(load_development_task(inputs/'anonymous.zip'),'shared_text')
    protocol=read(batch/'protocol.json')
    check(ab_run.sha(inputs/'anonymous.zip')==protocol['identities']['packets']['anonymous.zip'],'Wrong training packet')
    signed=matched_runs(control,batch)
    def path(split,name):
        if (split,name) in signed:
            folder,trial,_=signed[split,name]
            return folder/f'trial_{trial}.joblib'
        check(split=='confirmation_1' and name=='bge','Unknown missing replication')
        file=followup/'deep-bge/worker.joblib'
        check(ab_run.sha(file)==read(followup/'fits.json')['deep-bge']['model'],'Wrong retrospective checkpoint')
        return file
    observed={name:[] for name in ('e5','bge')}
    ids=[]
    for split in ('primary','confirmation_1'):
        run_dir=batch/'deep-e5'
        index=np.asarray(read(run_dir/(split+'_split.json'))['search_indices'])
        data=joblib.load(run_dir/(split+'.joblib'))
        check(ab_run.sha(run_dir/(split+'.joblib'))==protocol['splits'][split]['evaluation_payload_sha256'],'Split changed')
        hard={}
        prob={}
        for name in ('deep','e5','bge'):
            artifact=joblib.load(path(split,name))
            hard[name],prob[name]=predictions(artifact['model'],data['X_search'])
            check(np.array_equal(hard[name],artifact['search_predictions']),'Scores do not reproduce')
        raw=data['X_search']
        first=np.asarray([l<r for l,r in raw])
        for name in observed:
            p=(prob['deep']+prob[name])/2
            vote=(p>.5)|((p==.5)&first)
            baseline=hard['deep']==data['y_search']
            gain=(vote==data['y_search']).astype(int)-baseline
            observed[name].append(gain)
        ids.append(index)
    stacked=np.concatenate(ids)
    unique,inv,counts=np.unique(stacked,return_inverse=True,return_counts=True)
    result={}
    expected={'e5':8,'bge':read(followup/'results.json')['bge_replication']['pooled_gain']}
    for name,x in observed.items():
        all_diffs=np.concatenate(x)
        net=np.bincount(inv,weights=all_diffs)/counts
        check(all_diffs.sum()==expected[name],'Unaligned paired counts')
        result[name]=dict(pooled_row_observations=len(stacked),distinct_rows=len(unique),
            rows_seen_twice=int((counts==2).sum()),unique_groups=len(set(groups[unique])),
            net_count_all_observations=int(all_diffs.sum()),
            net_exclusive_observations=int(all_diffs[counts[inv]==1].sum()),
            net_repeated_observations=int(all_diffs[counts[inv]>1].sum()),
            unique_row_weighted_net_equivalent=float(net.sum()),unique_row_weighted=paired(net,groups[unique]))
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(results=result,plan=ab_run.sha(ROOT/'docs/batches/B_encoder_overlap_1123_zh.md'),new_fits=0,model_api=0))
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','control','batch','followup','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('inputs','control','batch','followup','output')))
