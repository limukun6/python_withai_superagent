"""Verify A/B reversal and tied outputs of frozen deployments on feature-only inputs."""
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_encoder_vote_followup import classes
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.external_assets import model_resources,verify_resources
from superagent.models import probability_with_model,predict_with_model


def run(features,following,bge,native,out):
    with zipfile.ZipFile(features) as z,z.open('rows.csv') as f:
        reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf8'))
        check(reader.fieldnames==['option_a','option_b'],'Feature-only input required')
        pairs=[(r['option_a'],r['option_b']) for r in reader]
    titles=sorted(set(t for p in pairs for t in p),key=lambda t:(hashlib.sha256(t.encode()).digest(),t))[:64]
    check(len(titles)==64 and len(pairs)==1693,'Wrong input')
    rows=pairs+[(b,a) for a,b in pairs]+[(t,t) for t in titles]
    files=[('e5_vote',following,'soft_vote','soft_vote'),('bge_vote',bge,'bge_vote','bge_vote'),
           ('native',native,'autonomous','new')]
    results={}
    for name,base,model_name,key in files:
        folder=base/model_name
        modelpath=folder/'selected_model.joblib'
        selection=read(folder/'final_selection.json')
        check(ab_run.sha(modelpath)==selection['model_sha256'],'Frozen model changed')
        model=joblib.load(modelpath)['model']
        assets,_=model_resources(model)
        verify_resources(assets)
        expected=classes(base/(model_name+'_export_0')/'rows.csv',read(base/'exports_frozen.json')[key],len(pairs))
        hard=np.asarray(predict_with_model('pair_text',model,rows))
        labels,prob=probability_with_model('pair_text',model,rows)
        positive=prob[:,labels.tolist().index(1)]
        cut=len(pairs)
        distinct=np.array([a!=b for a,b in pairs])
        mismatches=int((hard[:cut]!=expected).sum())
        flipped_errors=int((hard[:cut][distinct]+hard[cut:2*cut][distinct]!=1).sum())
        swap_error=float(np.max(np.abs(positive[:cut]+positive[cut:2*cut]-1)))
        tie_error=float(np.max(np.abs(positive[2*cut:]-.5)))
        results[name]=dict(model_sha256=selection['model_sha256'],compared_rows=cut,distinct_pairs=int(distinct.sum()),
            exported_prediction_mismatches=mismatches,hard_swap_errors=flipped_errors,max_probability_swap_error=swap_error,
            equal_inputs=len(titles),max_equal_probability_error=tie_error,resources=[r['name'] for r in assets],
            passed=mismatches==flipped_errors==0 and max(swap_error,tie_error)<1e-12)
    save(out,dict(results=results,features_sha256=ab_run.sha(features),new_fits=0,api_requests=0,labels_opened=False))
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','features','following','bge','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('features','following','bge','native','output')))
