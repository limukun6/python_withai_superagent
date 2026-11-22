"""Run the registered original/enumerated/feedback comparisons with a shared freeze before grading."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import export_frozen
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run, ab_text_recipes
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_training_task, load_prediction_features
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import fingerprint
from tools.ab_registered_rows import content_digest
from tools.ab_text_composition_run import POLICIES


ORDER = (('headlines','original'),('headlines','expanded'),('headlines','composed'),
         ('sms','composed'),('sms','expanded'),('sms','original'))
SOURCE_FILES = {'headlines':'f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95',
                'sms':'630a11762c8154429d5b82d67ae97f1b7fd53736c66c29bfd600e5129f754378'}


def registration(root,name):
    if ab_run.sha(root/'preflight.json')!=SOURCE_FILES[name]:
        raise ValueError('Registration predates this experiment and must match its published identity')
    prep = rec.read(root/'preflight.json')
    if name=='headlines':
        entry = prep['entries']['45102_original']
        packet,old = root/'45102',root/'45102_original'
        labels = prep['tasks']['45102']
    else:
        entry = prep['sms']
        packet = old = root/'sms'
        labels = entry
    dev = load_development_task(packet/'train.zip')
    contract = TaskSpec.from_task(dev).record()
    caps = ab_run.registered_capabilities(contract,len(dev.y_train),'full')
    original = rec.read(old/'run/run_manifest.json')['capabilities']
    restored = {k:v for k,v in caps.items() if k not in ab_text_recipes.ROUTES}
    expected_new = [c.family for c in ab_text_recipes.candidates(dev.kind)]
    if (list(restored.items())!=list(original.items()) or list(caps)!=[*original,*expected_new]
            or len(expected_new)!=8 or rec.digest(original)!=entry['catalog_sha256'] or list(original)!=entry['candidate_ids']
            or contract!=entry['contract'] or dev.X_val is not None or dev.y_val is not None
            or ab_run.sha(packet/'train.zip')!=entry['files']['train.zip'] or ab_run.sha(packet/'inner.json')!=entry['inner_sha256']):
        raise ValueError('Task, original executable menu, added space or optimization data differs')
    return dict(packet=packet,old=old,entry=entry,labels=labels,original=original,expanded=caps)


def run_path(out,name,policy,data,source):
    packet = data['packet']
    case = out/(name+'_'+policy)
    case.mkdir()
    for f in ('train.zip','inner.json'):
        os.link(packet/f,case/f)
    shutil.copytree(packet/'registered',case/'registered',copy_function=os.link)
    menu = data['original'] if policy=='original' else data['expanded']
    entry = dict(data['entry'],name=case.name,catalog_sha256=rec.digest(menu),candidate_ids=list(menu))
    command = ('-m','tools.ab_text_composition_run','--policy',policy,'--split-manifest',case/'inner.json',
               '--split-sha256',entry['inner_sha256'],'--','run','--catalog-scope','full','--history-policy','strict','--planner','local')
    before = len(rec.trace(out))
    job = pilot.train_one(case,entry,source,launcher=command)
    actual = rec.trace(out)[before:]
    expected = [w['candidate'] for w in job['workers'] for _ in range(w['attempted_classifier_fits'])]
    run = case/'run'
    manifest = rec.read(run/'run_manifest.json')
    if (calls:=len(actual))!=job['fits'] or calls>11 or [r['candidate'] for r in actual]!=expected:
        raise ValueError('Independent supervised fits do not match the fully executed work')
    if list(manifest['capabilities'].items())!=list(menu.items()) or manifest['task_contract']!=entry['contract']:
        raise ValueError('Real task or ordered catalogue did not match the registered arm')
    proposals = []
    for step,action in enumerate(job['actions'],1):
        state = rec.read(run/f'state_{step}.json')
        receipt = rec.read(run/'bounded_recipes'/f'step_{step}.json')
        if receipt['state_sha256']!=fingerprint(state) or receipt['action']!=action or receipt['policy']!=policy:
            raise ValueError('Proposer receipt differs from the official decision')
        for proof in receipt['proposals']:
            if (policy!='composed' or action['action']!='explore' or proof['chosen_id']!=action['candidate_id']
                    or proof['chosen_id'] not in proof['feasible_recipe_ids']
                    or proof['target_eligible']!=(proof['target_id'] in proof['feasible_recipe_ids'])
                    or not set(proof['evidence_trials'])<=set(action['evidence_trials'])):
                raise ValueError('Recorded composition is not bound to executable evidence')
            proposals.append(dict(step=step,**proof))
    job['compositions'] = proposals
    job['recipe_observations'] = [dict(trial=r['trial'],candidate=r['candidate'],configuration=r['params'],
                                         split=r['split_id'],primary=r['search_score'],worker_seconds=r['worker_wall_seconds'])
                                  for r in rec.read(run/'history.json') if r['candidate'] in ab_text_recipes.ROUTES]
    return job


def comparisons(out,name,data,staged,jobs):
    packet,entry = data['packet'],data['entry']
    dev = load_development_task(packet/'train.zip')
    features = packet/'features.zip'
    if ab_run.sha(features)!=(entry['features_sha256'] if name=='headlines' else entry['files']['features.zip']):
        raise ValueError('Unlabeled prediction input changed')
    rows = load_prediction_features(features,dev.kind,dev.spec)
    if name=='headlines':
        raw = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
        if (rec.sha(raw)!=data['labels']['outer_ids_sha256'] or ab_run.sha(staged)!=
                rec.read(rec.ROOT/'research/ba19_released_delivery/preflight.json')['headlines']['source_staged_sha256']):
            raise ValueError('Training-only headline identities changed')
        truth = load_training_task(staged).y_train[json.loads(raw)['held']]
        held_hash = data['labels']['held_content_sha256']
    else:
        if ab_run.sha(packet/'holdout.joblib')!=entry['files']['holdout.joblib']:
            raise ValueError('SMS held-out truth changed')
        held = joblib.load(packet/'holdout.joblib')
        truth,held_hash = held['y'],content_digest(held['X'],held['y'])
    if content_digest(rows,truth)!=held_hash or len(rows)!=entry['rows']:
        raise ValueError('Prediction order and scoring data do not match')
    correct = {}
    for policy in POLICIES:
        key = name+'_'+policy
        case = out/key
        digest = export_frozen(case,features,len(truth))
        pred = np.asarray(read_export(case/'export_0/rows.csv',len(truth)),dtype=truth.dtype)
        correct[policy] = pred==truth
        jobs[key]['grade'] = dict(correct=int(correct[policy].sum()),rows=len(truth),scores=MetricSpec.for_task(dev).evaluate(truth,pred),
                                  csv_sha256=digest)
    deltas = {}
    for a,b in zip(POLICIES,POLICIES[1:]):
        left,right = correct[a],correct[b]
        gain = jobs[name+'_'+b]['grade']['scores']['score']-jobs[name+'_'+a]['grade']['scores']['score']
        deltas[b+'_minus_'+a] = dict(delta_primary_pp=100*gain,gained=int((right & ~left).sum()),lost=int((left & ~right).sum()))
    return deltas


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    source = rec.digest(ab_run.sources())
    if verified_directory() is None:
        raise ValueError('Required registered MiniLM weights and dependencies')
    rec.write(out/'protocol.json',dict(batch='B2-01',plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/B2_01_bounded_text_composition_plan.md'),
              runner_sha256=ab_run.sha(Path(__file__)),adapter_sha256=ab_run.sha(rec.ROOT/'tools/ab_text_composition_run.py'),
              source=source,previous_registrations=SOURCE_FILES,order=ORDER,fit_limit=11,total_fit_limit=66,api_limit=0,
              wall_seconds=1200,worker_seconds=300,environment=pilot.environment()))
    jobs = {}
    try:
        data = {name:registration(getattr(args,name).resolve(),name) for name in ('headlines','sms')}
        rec.write(out/'preflight.json',{name:dict(train_sha256=d['entry']['files']['train.zip'],inner_sha256=d['entry']['inner_sha256'],
                  splits=d['entry']['splits'],contract=d['entry']['contract'],base_ids=list(d['original']),expanded_ids=list(d['expanded']),
                  base_sha256=rec.digest(d['original']),expanded_sha256=rec.digest(d['expanded'])) for name,d in data.items()})
        for name,policy in ORDER:
            jobs[name+'_'+policy] = run_path(out,name,policy,data[name],source)
            rec.write(out/'checkpoint.json',jobs)
        rec.write(out/'all_six_models_frozen.json',{key:j['model_sha256'] for key,j in jobs.items()})
        deltas = {name:comparisons(out,name,d,args.staged,jobs) for name,d in data.items()}
        fits = len(rec.trace(out))
        if fits!=sum(j['fits'] for j in jobs.values()) or fits>66 or rec.digest(ab_run.sources())!=source:
            raise ValueError('Final source, physical fit count or registration differs')
        rec.write(out/'results.json',dict(paths=jobs,paired=deltas))
        rec.write(out/'completion.json',dict(status='completed',new_fits=fits,new_cli_exports=12,new_model_api_requests=0,
                  new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),new_fits=len(rec.trace(out)),completed_paths=list(jobs)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('headlines','sms','weights','staged','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
