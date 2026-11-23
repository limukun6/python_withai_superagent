"""Verify the released fit counts, decisions and paired differences without training or model weights."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
E = ROOT/'evidence'


def require(value, message):
    if not value:
        raise ValueError(message)


def raw(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())


def record(path):
    return json.loads(raw(path))


def digest(path):
    return hashlib.sha256(raw(path)).hexdigest()


def compare(groups, result, old='baseline', new='new'):
    cells = np.array([np.bincount(rows,minlength=4) for rows in groups])
    total = cells.sum(axis=0)
    n, gains, losses = int(total.sum()),int(total[2]),int(total[1])
    require((n,total[1]+total[3],total[2]+total[3],gains,losses)==
            (result['rows'],result[old],result[new],result['gained'],result['lost']), 'Paired totals differ')
    change = result.get('paired',result.get('delta'))
    require(change['groups']==len(cells) and change['net_correct']==gains-losses,'Group count or direction differs')
    counts, delta = cells.sum(axis=1),cells[:,2]-cells[:,1]
    rng = np.random.default_rng(60423)
    draws = []
    for _ in range(100):
        ix = rng.integers(0,len(cells),size=(50,len(cells)))
        draws.extend(100*delta[ix].sum(axis=1)/counts[ix].sum(axis=1))
    np.testing.assert_allclose(np.quantile(draws,[.025,.975]),change['group_bootstrap_95_pp'],rtol=0,atol=1e-10)
    return gains-losses


def seal(folder, key, path):
    require(record(folder/'completion.json')[key]==digest(folder/path),f'{folder.name} {key} integrity')


def physical(folder, name, trace):
    fit = record(folder/name/'classifier_fits.json')
    logs = [json.loads(row) for row in raw(folder/trace).splitlines()]
    require([p['candidate'] for p in logs]==[p['candidate'] for p in fit['workers']] and
            len(logs)==fit['attempted_classifier_fits']==fit['completed_classifier_fits']
            and fit['unknown_failed_attempts']==0, 'Physical ledger mismatch')
    return len(logs)


def main():
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'File differs: '+name)
    integration = record(E/'integration.json')
    with zipfile.ZipFile(ROOT/'runtime-source.zip') as archive:
        require(archive.testzip() is None,'Broken code archive')
        for path, checksum in integration['native_source_files'].items():
            require(hashlib.sha256(archive.read(path)).hexdigest()==checksum,'Different native source '+path)
        for study in (ROOT/'studies').glob('*.py'):
            require(study.read_bytes()==archive.read('scripts/'+study.name),'Study source differs')
        collector = hashlib.sha256(archive.read('historical/initial_numeric_collector.py')).hexdigest()
        require(collector==record(E/'adaptive/protocol.json')['runner'],'Initial collector hash')
    result = {}
    for folder in ('geometry','kernel','adaptive','fixed','transfer','generations','transitive'):
        case = E/folder
        for k in ('protocol','results'):
            seal(case,k,k+'.json')
        result[folder]=record(case/'results.json')
        require(result[folder]['model_api_calls']==result[folder]['new_rmb']==0,'Incorrect API receipt')
    geometry = E/'geometry'
    for part, methods in result['geometry']['panels'].items():
        groups=record(geometry/part/'grouped_correctness.json')
        for k, values in methods.items():
            compare(groups[k],values)
        require(digest(geometry/part/'grouped_correctness.json')==record(geometry/'completion.json')['groups'][part],
                'Geometry correctness evidence changed')
    require(not result['geometry']['eligible_for_outer'] and result['geometry']['physical_fits']==4,'Geometry gate')
    kernel=E/'kernel'
    ds=[]
    for part, values in result['kernel']['panels'].items():
        ds.append(compare(record(kernel/part/'grouped_correctness.json'),values))
        require(digest(kernel/part/'grouped_correctness.json')==record(kernel/'completion.json')['groups'][part],
                'Kernel correctness evidence changed')
    require(ds==[14,7,5] and result['kernel']['eligible_for_outer'],'Kernel confirmation gate')
    for name, number in [('geometry',4),('kernel',3),('transitive',1),('generations',2)]:
        fits=list((E/name).rglob('fit_completed.json'))
        require(len(fits)==number,'Research fit count')
        for end in fits:
            fit=record(end)
            require(fit.get('start',fit.get('started'))==digest(end.with_name('fit_started.json')) and
                    fit['fits']==1,'Incomplete research fit')
    for runner, n in [('adaptive',7),('fixed',11)]:
        case=E/runner
        for name in ('exports','grouped_correctness','models_frozen','scoring_protocol'):
            seal(case,name,name+'.json')
        vals=result[runner]
        require(compare(record(case/'grouped_correctness.json'),vals)==16,'Unexpected Bean delta')
        require(physical(case,'agent','physical.jsonl')==vals['physical_fits']==n,'Native fit count')
        actions=record(case/'agent/actions.json')
        require(actions[-1]['action']=='stop' and vals['keep_optional'] and
                vals['selected']=='numeric_tabular_fourier_logistic','Native choice')
        require(sum(a['action']=='confirm' for a in actions)==(6 if runner=='fixed' else 0),'Scheduler allocation')
        exp=record(case/'exports.json')['exports']
        require(exp['new_first']==exp['new_second']==integration['a_csv_sha256']==integration['b_csv_sha256'],
                'A and B prediction hashes differ')
    compare(record(E/'generations/grouped_correctness.json')['bn01'],result['generations']['panels']['bn01'],
            'old_kernel','new_kernel')
    compare(record(E/'generations/grouped_correctness.json')['a94'],result['generations']['panels']['a94'],
            'old_kernel','new_kernel')
    require(result['generations']['improved_both_by_three'] and result['generations']['overlapping_feature_groups']==402,
            'Historical cross-comparison')
    trans=E/'transitive'
    seal(trans,'grouped_correctness','grouped_correctness.json')
    deltas={k:compare(v,result['transitive']['panels'][k]) for k,v in record(trans/'grouped_correctness.json').items()}
    require(deltas=={'A':3,'B':0,'combined':3} and not result['transitive']['eligible_for_confirmation']
            and record(trans/'training_graph.json')['added_weak_pairs']==411,'Indirect preference gate')
    transfer=E/'transfer'
    seal(transfer,'models_frozen','models_frozen.json')
    seal(transfer,'exports','exports.json')
    ranks=record(transfer/'rank_counts.json')
    wins,previous=0.,0
    for n,p in ranks['ordered_negative_positive']:
        wins+=p*(previous+.5*n)
        previous+=n
    class_counts=np.sum(ranks['ordered_negative_positive'],axis=0)
    auc=wins/np.prod(class_counts)
    require(set(record(transfer/'exports.json')['csv'].values())=={ranks['predictions_sha256']},'AUC exports differ')
    for arm, count in [('standard',7),('expanded',8)]:
        observed=result['transfer']['arms'][arm]
        require(physical(transfer,arm,arm+'.jsonl')==observed['fits']==count and
                int(np.trace(ranks['confusion_true_by_pred']))==observed['correct']==111 and
                np.isclose(auc,observed['auc']),'Transfer score or physical fits differ')
    print('Verified 7 studies. Bean 2005/2178. Matched kernels +9/+14. Transfer 111/114. Task2 +3/1591.')


if __name__=='__main__':
    main()
