"""Resolve an exact primary tie with at most one previously unobserved confirmation fit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import compare
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes,verified_export
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.ab_complement import MIN_GAIN,PreferenceProbabilityVote
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import checked_score_record
from superagent.metrics import MetricSpec


def columns(artifact,payload,digest,name):
    record=artifact['prediction_record']
    check(record['evaluation_payload_sha256']==digest and record['label_sha256']==fingerprint(payload['y_search']),
          'Refit or confirmation data changed')
    m,ids=checked_score_record(record['score_record'],record,MetricSpec.from_payload(payload),name,payload['y_search'])
    return np.asarray(record['predictions']),m[:,ids.index(1)]


def replicate(name,source,folder,env,menu):
    file=folder/'extra_confirmation.joblib'
    admission=folder/'admission.json'
    ledger=folder/'worker_fits.json'
    physical=folder/'physical.jsonl'
    if not admission.exists():
        save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
        with (folder/'confirm.log').open('x') as log:
            rc=subprocess.run([sys.executable,'-m','superagent.experiment_worker',str(source),name,str(file),
                '--fit-admission',str(admission)],cwd=ROOT,env=dict(env,**{PREFERENCE_CATALOG_ENV:menu},
                AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),stdout=log,stderr=subprocess.STDOUT,timeout=300)
        check(rc.returncode==0,'Tied-partner confirmation failed')
    rec=read(ledger)
    check(len(physical.read_text().splitlines())==rec['attempted_classifier_fits']==rec['completed_classifier_fits']==1
          and rec['candidate']==name,'Wrong incremental cost or candidate')
    return file


def save_model(folder,name,model,spec):
    dest=folder/name
    dest.mkdir(exist_ok=True)
    file=dest/'selected_model.joblib'
    if not file.exists():
        joblib.dump(dict(kind='pair_text',candidate=name,model=model),file)
        shutil.copyfile(spec,dest/'task_spec.json')
    sealed=dict(selected_experiment=name,family=name,model_file=file.name,model_format='candidate_bundle_v1',
        model_sha256=ab_run.sha(file),task_spec_sha256=ab_run.sha(spec),test_accessed=False,refit=True,training_rows=6608)
    frozen_save(dest/'final_selection.json',sealed)
    return dest


def run(inputs,native,old,out,env):
    out.mkdir(parents=True,exist_ok=True)
    folder=native/'autonomous'
    orig=read(folder/'confirmation.json')['b_scheduler_review']
    anchor=orig['anchor']
    combos=orig['opening_combinations']
    tied=[name for name,score in combos.items() if score==max(combos.values())]
    check(len(tied)==2 and orig['promoted'] and orig['partner'] in tied,'Requires the registered two-way tie')
    source=folder/'confirmation_1.joblib'
    digest=ab_run.sha(source)
    history=read(folder/'history.json')
    first={r['candidate']:r for r in history if r['split_id']=='primary'}
    measured={r['candidate']:folder/f"trial_{r['trial']}.joblib" for r in history if r['split_id']=='confirmation_1'}
    missing=[n for n in [anchor,*tied] if n not in measured]
    check(len(missing)==1,'At most one additional replication is admitted')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_native_tied_partners_1123_zh.md'),
        script=ab_run.sha(Path(__file__)),native_decision=ab_run.sha(folder/'confirmation.json'),tie=tied,
        missing=missing[0],confirmation_payload=digest,additional_physical_fit_limit=1,model_api_limit=0))
    measured[missing[0]]=replicate(missing[0],source,out,env,read(folder/'run_manifest.json')['preference_catalog'])
    payload=joblib.load(source)
    y=payload['y_search']
    first_pos=np.array([a<b for a,b in payload['X_search']])
    observed={n:columns(joblib.load(measured[n]),payload,digest,n) for n in [anchor,*tied]}
    n1,n2=orig['primary_rows'],len(y)
    stats={}
    for name in tied:
        p=(observed[anchor][1]+observed[name][1])/2
        vote=(p>.5)|((p==.5)&first_pos)
        hard=[int((observed[n][0]==y).sum()) for n in (anchor,name)]
        best_primary=max(orig['opening_correct'][n] for n in (anchor,name))
        gain1=combos[name]-best_primary
        gain2=int((vote==y).sum())-max(hard)
        pooled=combos[name]+int((vote==y).sum())-max(orig['opening_correct'][n]+k for n,k in zip((anchor,name),hard))
        stats[name]=dict(primary_combo=combos[name],confirmation_singles=hard,confirmation_combo=int((vote==y).sum()),
            weakest_gain=min(gain1/n1,gain2/n2),pooled_gain=pooled,
            qualified=gain1>0 and gain2>0 and pooled/(n1+n2)>=MIN_GAIN)
    eligible=[n for n in tied if stats[n]['qualified']]
    chosen=max(eligible,key=lambda n:(stats[n]['weakest_gain'],stats[n]['pooled_gain'],orig['opening_correct'][n]),default=None)
    frozen_save(out/'decision_before_export.json',dict(anchor=anchor,original=orig['partner'],chosen=chosen,tied=tied,evidence=stats))
    full={}
    model_ids={}
    payload_digest=ab_run.sha(folder/'refit_data.joblib')
    for path in (folder,old/'deep-e5'):
        check(ab_run.sha(path/'refit_data.joblib')==payload_digest,'Full refit inputs differ')
        for file in sorted(path.glob('refit_component_*.joblib')):
            art=joblib.load(file)
            facts=art['metrics']['resolved_configuration']
            name=facts['id']
            if name in [anchor,*tied]:
                check(art['metrics']['training_rows']==6608 and facts['params']==first[name]['params'],'Different fitted configuration')
                full[name]=art['model']
                model_ids[name]=ab_run.sha(file)
    check(set(full)==set([anchor,*tied]),'A full member is unavailable')
    packages={'anchor':save_model(out,'anchor',full[anchor],folder/'task_spec.json')}
    keys={n:'pair_'+str(j) for j,n in enumerate(tied)}
    for n,k in keys.items():
        packages[k]=save_model(out,k,PreferenceProbabilityVote('pair_text',[full[anchor],full[n]],np.array([0,1])),
            folder/'task_spec.json')
    frozen_save(out/'models_frozen.json',dict(components=model_ids,pairs=keys,
        package_hashes={k:ab_run.sha(v/'selected_model.joblib') for k,v in packages.items()}))
    seals={k:verified_export(v,inputs,env,out) for k,v in packages.items()}
    check(seals[keys[orig['partner']]]==read(native/'exports_frozen.json')['new'],'Original deployed vote changed')
    frozen_save(out/'exports_frozen.json',seals)
    signed=read(native/'protocol.json')['packets']
    check(all(ab_run.sha(inputs/k)==signed[k] for k in ('features.zip','scoring_only.json','groups_only.json')),'Wrong reserve')
    truth=np.array(read(inputs/'scoring_only.json')['labels'])
    groups=np.array(read(inputs/'groups_only.json'))
    preds={k:classes(out/(k+'_export_0')/'rows.csv',seals[k],len(truth)) for k in packages}
    reference=classes(old/'deep-e5_export_0/rows.csv',read(native/'exports_frozen.json')['original'],len(truth))
    results={k:dict(versus_anchor=compare(v,preds['anchor'],truth),versus_original_twelve=compare(v,reference,truth),
        delta_vs_anchor=paired((v==truth).astype(int)-(preds['anchor']==truth),groups)) for k,v in preds.items()}
    selected_key=keys[chosen] if chosen else 'anchor'
    result=dict(results=results,chosen=chosen,anchor=anchor,selected_key=selected_key,
        original_key=keys[orig['partner']],actual_new_fits=1,required_fit_equivalents=4+3+(2 if chosen else 1),
        model_api=0,course_validation=False,formal_test=False,reserve_rows=len(truth),retrospective=True)
    frozen_save(out/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','native','old','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,n).resolve() for n in ('inputs','native','old','output')),offline_environment(a.workspace.resolve()))
