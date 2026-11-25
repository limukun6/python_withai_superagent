"""Cold-start replication and fresh-seed check through the protected unified CLI."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from scripts.run_planner_self_review import packet, source_identity
from scripts.run_review_budget_study import action_key
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.metrics import MetricSpec


PLAN = ROOT / 'research/rsi_budget_1125/entry_protocol.json'
RUNS = (('n83','protected'),('n97','active'),('n97','protected'))


def prepare(root, prior):
    root.mkdir()
    (root / 'evaluation').mkdir()
    for case in ('n83','n97'):
        (root / case).mkdir()
    for name in ('p.zip','q.zip'):
        shutil.copyfile(prior / 'n83' / name, root / 'n83' / name)
    shutil.copyfile(prior / 'evaluation/n83.npy', root / 'evaluation/n83.npy')
    x,y = make_classification(n_samples=1000, n_features=16, n_informative=8, n_redundant=3, n_classes=3,
                              n_clusters_per_class=2, weights=[.57,.29,.14], class_sep=1.1,
                              flip_y=.09, random_state=941228)
    require(len(np.unique(x,axis=0)) == len(y), 'Duplicate replication rows')
    a,b = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=730358)
    packet(root/'n97/p.zip', 'tabular', 'balanced_accuracy', x[a], y[a])
    packet(root/'n97/q.zip', 'tabular', 'balanced_accuracy', x[b])
    np.save(root/'evaluation/n97.npy', y[b], allow_pickle=False)
    cases = {c:{'training':sha(root/c/'p.zip'), 'features':sha(root/c/'q.zip'),
                'truth':sha(root/'evaluation'/(c+'.npy'))} for c in ('n83','n97')}
    require(cases['n83']['training'] == read(prior/'protocol.json')['cases']['n83']['train_sha256'] and
            cases['n83']['truth'] == read(prior/'protocol.json')['cases']['n83']['truth_sha256'], 'Prior data changed')
    save(root/'protocol.json', dict(source=source_identity(ROOT), plan=sha(PLAN), runner=sha(Path(__file__)),
         prior=sha(prior/'completion.json'), cases=cases))


def run_one(root, case, mode, weights):
    work = root / case / mode
    work.mkdir()
    settings = env(weights, work / 'fits.jsonl')
    settings.pop('AB_B1_PER_WORKER_FIT_LIMIT',None)
    for key in ('SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_TASK_BUDGET_ID'):
        settings.pop(key,None)
    def call(command, args):
        with (work / (command+'.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'unified_agent', command, *map(str,args)], cwd=ROOT, env=settings,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=690, check=True)
    run = work / 'run'
    delivery = work / 'delivery/predictions.csv'
    call('solve', [root/case/'p.zip','--prediction-input',root/case/'q.zip','--run-dir',run,'--output',delivery,
                   '--fit-limit',16,'--time-limit',600, '--'+mode+'-self-review'])
    replay = work / 'replay/predictions.csv'
    call('predict',[root/case/'q.zip','--run-dir',run,'--output',replay])
    agent = run/'agent'
    result, final = read(agent/'run_outcome.json'), read(agent/'final_selection.json')
    hist, decisions = read(agent/'history.json'), read(agent/'decisions.json')
    manifest = read(agent/'run_manifest.json')
    count = len((work/'fits.jsonl').read_text().splitlines())
    require(result['status']=='completed' and result['new_api_requests']==0 and not result['formal_test_accessed']
            and result['unresolved_api_requests']==result['unknown_failed_attempts']==0 and
            0 < count == result['attempted_classifier_fits'] == result['completed_classifier_fits'] <= 16
            and sha(delivery)==sha(replay) and sha(agent/final['model_file'])==final['model_sha256']
            and manifest['self_review']==mode and manifest['fit_limit']==16 and final['refit'], 'Unverified delivery')
    require(all(d['observed_trials']==[h['trial'] for h in hist[:i]] and d['source']=='local evidence policy'
                for i,d in enumerate(decisions)), 'Trace lacks independent prior-only decisions')
    receipt = dict(case=case, mode=mode, fits=count, prediction=sha(delivery), model=final['model_sha256'],
                   prefix=[action_key(d['choice']) for d in decisions], winner=result['members'],
                   primary_split=sha(agent/'primary_split.json'), api_calls=result['new_api_requests'])
    if mode=='protected':
        review = read(agent/'stop_review.json')
        old = read(agent/'incumbent_selection.json')
        require(review['incumbent_model']==sha(agent/old['model_file'])==old['model_sha256'] and
                review['delivered_model']==final['model_sha256'] and review['final_members']==result['members'] and
                decisions[-1]['after_refit']=='stop_review.json' and decisions[-1]['choice']['members']==review['incumbent']
                and review['fits_at_stop']+review['additional_fits']==count, 'Broken incumbent or decision linkage')
        gains=[item['delta'] for item in review['comparisons']]
        passes=len(gains)==2 and min(gains)>0 and np.mean(gains)>=.003
        require((review['status']=='accepted')==passes and review['status'] in
                ('accepted','retained','insufficient_room_or_evidence'), 'Unverified promotion or unexpected failure')
        receipt['review']=review
    save(work/'sealed.json',receipt)
    print(case,mode,count,'fits',receipt.get('review',{}).get('status','sealed'),flush=True)


def evaluate(root, prior):
    config=read(root/'protocol.json')
    require(config['plan']==sha(PLAN) and config['runner']==sha(Path(__file__)) and
            config['source']==source_identity(ROOT) and config['prior']==sha(prior/'completion.json'), 'Frozen source changed')
    records={(c,m):read(root/c/m/'sealed.json') for c,m in RUNS}
    sealed={c+'/'+m:r['prediction'] for (c,m),r in records.items()}
    for (c,m),r in records.items():
        require(r['prediction']==sha(root/c/m/'delivery/predictions.csv') and
                config['cases'][c]['training']==sha(root/c/'p.zip') and config['cases'][c]['features']==sha(root/c/'q.zip'),
                'Packet or delivery changed')
    r=records['n83','protected']
    baseline=read(prior/'n83/active16/sealed.json')
    challenge=read(prior/'stop-challenge/n83/sealed.json')
    require(r['prediction']==challenge['prediction'] and r['fits']==challenge['combined_fits'] and
            r['prefix']==baseline['actions'] and r['review']['incumbent']==baseline['winner'] and
            r['review']['status']==challenge['status'] and r['review']['candidate']==challenge['candidate'] and
            r['review']['comparisons']==challenge['comparisons'], 'Full cold-start reproduction differs')
    left,right=records['n97','active'],records['n97','protected']
    require(left['prefix']==right['prefix'] and left['primary_split']==right['primary_split'] and
            left['winner']==right['review']['incumbent'], 'New arms differ before stopping')
    save(root/'sealed-predictions.json',sealed)
    results = {}
    for c in ('n83','n97'):
        truth=root/'evaluation'/(c+'.npy')
        require(sha(truth)==config['cases'][c]['truth'],'Held-out labels changed')
        y=np.load(truth,allow_pickle=False)
        scorer=MetricSpec('balanced_accuracy',tuple(np.unique(y).tolist()))
        result,answers={},{}
        for (case,m),rec in records.items():
            if case!=c:continue
            with (root/c/m/'delivery/predictions.csv').open() as file:
                rows=list(csv.DictReader(file))
            require([row['row_id'] for row in rows]==list(map(str,range(len(y)))),'Unaligned prediction rows')
            pred=np.asarray([row['prediction'] for row in rows],dtype=y.dtype)
            answers[m]=pred
            result[m]=dict(score=scorer.evaluate(y,pred)['score'], correct=int(np.sum(pred==y)), rows=len(y),
                           fits=rec['fits'], winner=rec['winner'], prediction=rec['prediction'], review=rec.get('review'))
        if c=='n97':
            first,second=answers['active']==y,answers['protected']==y
            result['comparison']=dict(delta=result['protected']['score']-result['active']['score'],
                gained=int(np.sum(~first & second)), lost=int(np.sum(first & ~second)), identical_prefix=True)
        results[c]=result
    total=sum(r['fits'] for r in records.values())
    previous=read(prior/'completion.json')['total_including_continuation']
    require(total <= 48 and total+previous <= 197,'Wider study budget exceeded')
    save(root/'completion.json',dict(cases=results,new_fits=total,combined_new_fits=total+previous,
         api_requests=sum(r['api_calls'] for r in records.values()), source=config['source'], plan=config['plan'],
         runner=config['runner'], n83_cold_reproduction=True, formal_test_accessed=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run','score'))
    parser.add_argument('root',type=Path)
    parser.add_argument('prior',type=Path)
    parser.add_argument('--weights',type=Path)
    args=parser.parse_args()
    root,prior=args.root.resolve(),args.prior.resolve()
    if args.stage=='prepare':
        prepare(root,prior)
    elif args.stage=='run':
        config=read(root/'protocol.json')
        require(config['source']==source_identity(ROOT) and config['runner']==sha(Path(__file__)) and
                config['plan']==sha(PLAN),'Changed before execution')
        for c,m in RUNS:
            require(sha(root/c/'p.zip')==config['cases'][c]['training'] and
                    sha(root/c/'q.zip')==config['cases'][c]['features'], 'Inputs changed before training')
            run_one(root,c,m,args.weights.resolve())
    else:
        evaluate(root,prior)
