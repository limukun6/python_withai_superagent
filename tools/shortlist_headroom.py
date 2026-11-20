"""Test a diverse shortlist and bound any three-finalist policy that protects the first-five leader."""
import argparse
from itertools import combinations
import os
from pathlib import Path
import shutil
import statistics

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.data import TrainingTask, load_training_task
from superagent.evidence_agent import route_of
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import sample_groups, take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def diverse_shortlist(source, history, current, anchor):
    payload = joblib.load(source/'inner_data.joblib')
    y = payload['y_search']
    groups = sample_groups(TrainingTask(payload['kind'],payload['X_search'],y),'shared_text')
    _, within = np.unique(groups,return_inverse=True)
    counts = np.bincount(within)
    if len(counts)<2:
        return current
    correct = {r['candidate']:joblib.load(source/f"trial_{r['trial']}.joblib")['search_predictions']==y for r in history}
    removable = next(n for n in reversed(current) if n!=anchor)
    protected = [n for n in current if n!=removable]
    missed = ~(correct[protected[0]] | correct[protected[1]])
    families = {route_of(n) for n in protected}
    eligible = []
    for name, mask in correct.items():
        if name in current or route_of(name) in families:
            continue
        paired = mask.astype(int) - correct[removable]
        mean = float(paired.mean())
        sums = np.bincount(within,weights=paired) - mean*counts
        se = float(np.sqrt(len(counts)/(len(counts)-1)*np.sum(sums*sums))/len(y))
        recovered = int((mask & missed).sum())
        extra = recovered - int((correct[removable]&missed).sum())
        if mean>=-se and extra>=max(1,np.ceil(.002*len(y))):
            eligible.append((-recovered,-int(mask.sum()),name))
    return [*protected,min(eligible)[2]] if eligible else current


def complete_scores(source, dest, record, history, confirmation):
    means = {r['candidate']:r['mean_score'] for r in confirmation['candidates']}
    lookup = {r['candidate']:r['search_score'] for r in history}
    worker = dest/'worker'
    worker.mkdir(exist_ok=True)
    for path in [*source.glob('method_*.json'),*source.glob('*vectors.sqlite'),
                 source/'pretrained_runtime.json',source/'pretrained_decision.json']:
        if path.is_file() and not (worker/path.name).exists():shutil.copyfile(path,worker/path.name)
    for name in record['extra']:
        number = record['trials'][name]
        scores = [lookup[name]]
        for i in (1,2):
            payload = joblib.load(source/f'confirmation_data_{i}.joblib')
            payload.pop('feature_cache_dir',None)
            data_path = worker/f'confirm_{i}.joblib'
            if not data_path.exists():joblib.dump(payload,data_path)
            artifact = worker/f'c_{i}_{number}.joblib'
            proof = artifact.with_suffix('.json')
            if not proof.exists():
                _, metrics = train_candidate(data_path,name,artifact,480)
                expect(metrics['status']=='completed' and metrics['classifier_fits']==1,str(metrics))
                freeze_file(proof,dict(sha256=digest(artifact)))
            expect(digest(artifact)==read(proof)['sha256'],'Confirmation classifier changed')
            trained = joblib.load(artifact)
            correct = float(np.mean(trained['search_predictions']==payload['y_search']))
            expect(abs(correct-trained['metrics']['search_score'])<1e-12,'Inner accuracy does not reproduce')
            scores.append(correct)
        means[name] = statistics.mean(scores)
    expect(not (worker/'downstream_usage.jsonl').exists() and not (worker/'llm_usage.jsonl').exists(),
           'Extra model forward in classification-only confirmation')
    return means


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('archive','reference','finalists','seven','output'):
        parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--arm',default='paired')
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE']='1'
    os.environ.pop('SUPERAGENT_PRETRAINED_RUNTIME',None)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    ref=read(args.reference/'protocol.json')
    seven=read(args.seven/'protocol.json')
    expect(ref['archive_sha256']==digest(args.archive) and seven['reference_sha256']==digest(args.reference/'protocol.json')
           and seven['finalist_protocol_sha256']==digest(args.finalists/'protocol.json') and seven['arm']==args.arm,'Different frozen runs')
    freeze_file(root/'protocol.json',dict(reference_sha256=digest(args.reference/'protocol.json'),arm=args.arm,
        seven_sha256=digest(args.seven/'protocol.json'),source_sha256=digest(Path(__file__)),
        rule='Keep two original finalists including the first-five leader. Replace only the worst unprotected finalist, '
             'only by an unrepresented family within one clustered standard error of its primary-search accuracy, gaining '
             'at least 0.2% additional rows both protected models miss. Select by the original three-split mean. '
             'Separately evaluate all seven with all three splits and enumerate the outer-label upper bound over all triples '
             'containing the first-five leader. All inner decisions and rankings freeze before inspecting outer predictions.'))
    if args.freeze:return
    opened=[]
    with train_only(args.archive,opened):task=load_training_task(args.archive)
    expect(task.kind=='categorical_tabular','This replay uses self-contained tabular prediction bundles')
    for i,block in enumerate(ref['locations']):
        source=Path(block[args.arm])
        folder=root/f'fold_{i}'
        folder.mkdir(exist_ok=True)
        rec=receipt(source)
        history=read(source/'trials.json')
        confirm=read(source/'confirmation.json')
        expect(confirm['primary_metric']=='accuracy','This shortlist gate requires accuracy evidence')
        candidate_list=diverse_shortlist(source,history,confirm['shortlist'],rec['incumbent'])
        scores=complete_scores(source,folder,seven['records'][i],history,confirm)
        def winner(names):return min(names,key=lambda n:(-scores[n],n))
        expect(winner(confirm['shortlist'])==rec['winner'],'Original winner changed')
        legal={winner(combo) for combo in combinations(scores,3) if rec['incumbent'] in combo}
        freeze_file(folder/'decision.json',dict(diverse=candidate_list,original=confirm['shortlist'],rank=scores,
                    feasible=sorted(legal),selected=winner(candidate_list),all_confirmed=winner(scores),incumbent=rec['winner']))
    folds=[]
    for i,block in enumerate(ref['locations']):
        source=Path(block[args.arm])
        info=seven['records'][i]
        held=ref['indices'][i]['outer']
        x,y=take(task.X_train,held),task.y_train[held]
        folder=root/f'fold_{i}'
        decision=read(folder/'decision.json')
        models={decision['incumbent']:(source/'selected_model.joblib',info['original_model'])}
        for j,n in enumerate(info['shortlist'][1:],1):
            models[n]=(args.finalists/f'fold_{i}'/f'full_{j}.joblib',info['previous_full'][n])
        for n in info['extra']:
            path=args.seven/f'fold_{i}'/f"extra_{info['trials'][n]}.joblib"
            models[n]=(path,read(path.with_suffix('.json'))['sha256'])
        predictions={}
        for n,(path,sha) in models.items():
            expect(digest(path)==sha,'Frozen whole-data model changed')
            predictions[n]=predict_with_model(task.kind,joblib.load(path)['model'],x)
        frozen=read(args.reference/f"fold_{ref['fold_numbers'][i]-1}"/'outer_predictions.json')
        np.testing.assert_array_equal(predictions[decision['incumbent']],frozen[args.arm])
        expect(frozen['rows']==held,'Wrong outer fold order')
        best=min(decision['feasible'],key=lambda n:(-int((predictions[n]==y).sum()),n))
        choices=dict(incumbent=decision['incumbent'],diverse=decision['selected'],all_confirmed=decision['all_confirmed'],
                     best_allowable_shortlist=best)
        vectors={k:predictions[n] for k,n in choices.items()}
        write(folder/'outer_predictions.json',dict(rows=held,truth=y.tolist(),**{k:v.tolist() for k,v in vectors.items()}))
        folds.append(dict(rows=len(y),changed_shortlist=set(decision['diverse'])!=set(decision['original']),
                          choices=choices,correct={k:int((v==y).sum()) for k,v in vectors.items()}))
    totals={k:sum(f['correct'][k] for f in folds) for k in folds[0]['correct']}
    write(root/'results.json',dict(folds=folds,correct=totals,rows=sum(f['rows'] for f in folds),
        changed_lists=sum(f['changed_shortlist'] for f in folds),additional_diagnostic_fits=8*len(folds),deployment_fits_diverse=14,
        deployment_fits_all_confirmed=22,additional_encoder_forwards=0,new_api_calls=0,training_members=opened))
    print(totals,flush=True)


if __name__=='__main__':main()
