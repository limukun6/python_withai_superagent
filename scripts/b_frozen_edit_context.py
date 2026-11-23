"""Run the predeclared edit-context experiment with frozen downstream models."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from b_frozen_head_frontier import REPO, SEEDS, load_head, paired, read, require, save, sha
sys.path.insert(0,str(REPO/'superagent'))
from ab_edit_views import edit_view


def vectors_for(study, cache_dir, data, ids):
    from superagent.minilm_features import DeeperMiniLMEncoder
    cache_dir.mkdir()
    config = cache_dir/'runtime.json'
    save(config,dict(deep_minilm_weights=str(study.WEIGHTS/'all-MiniLM-L12-v2')))
    bank = DeeperMiniLMEncoder(config)
    require(bank.folder.resolve()==cache_dir.resolve(),'Unexpected external runtime override')
    views = {}
    for i in ids:
        i = int(i)
        result, changed = edit_view(*data[i])
        reversed_pair, flag = edit_view(*data[i][::-1])
        require(flag==changed and result==reversed_pair[::-1],'Edit view violates swap equivariance')
        if changed:
            views[i] = result
    texts = sorted({s for pair in views.values() for s in pair})
    if texts:
        bank.encode(texts)
    return views, bank, dict(eligible_rows=len(views),distinct_adapter_input_titles=len(texts))


def margins(head, full_bank, edit_bank, rows, ids, views, old_z):
    base = head.classifier.decision_function(head.features(rows))
    np.testing.assert_allclose(base,old_z,atol=1e-6,rtol=0)
    adjusted = base.copy()
    positions = np.array([j for j,i in enumerate(ids) if int(i) in views],dtype=int)
    if len(positions):
        original = [title for pos in positions for title in rows[pos]]
        edited = [title for pos in positions for title in views[int(ids[pos])]]
        delta = edit_bank.encode(edited)-full_bank.encode(original)
        correction = (delta[::2]-delta[1::2])@head.classifier.coef_[0,-384:]
        adjusted[positions] += .5*correction
    return base, adjusted


def local_usage(folder):
    p = folder/'downstream_usage.jsonl'
    events = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    require(all(e['remote_calls']==0 and e['api_rmb']==0 for e in events),'Unexpected remote use')
    return dict(local_feature_tokens=sum(v['input_tokens'] for v in events),
                local_feature_sequences=sum(v['sequences'] for v in events),remote_requests=0)


def score_dev(study, archive, groups, path, inventory):
    sets, stats, fold_changes = [], {}, {}
    for seed in SEEDS:
        parts = []
        fold_changes[str(seed)] = []
        for fold in range(5):
            key=f'{seed}_fold_{fold}'
            name=path/(key+'.npz')
            require(sha(name)==inventory[key]['sha256'],'Unsealed prediction file')
            with np.load(name) as a:
                parts.append([a[k] for k in ('ids','base','edited','active')])
        ids,base,edit,active = [np.concatenate(v) for v in zip(*parts,strict=True)]
        order=np.argsort(ids)
        ids,base,edit,active = [v[order] for v in (ids,base,edit,active)]
        require(len(ids)==len(np.unique(ids))==6608,'Incomplete OOF coverage')
        labels = study.opened_rows(archive,('label_pairwise',),set(ids))
        y = np.array([int(labels[i][0]) for i in ids])
        left,right = (base>0)==y,(edit>0)==y
        gain = right.astype(int)-left
        require(not gain[~active].any(),'Unrouted rows changed')
        by_id = {int(i):g for i,g in zip(ids,gain,strict=True)}
        fold_changes[str(seed)] = [int(sum(by_id[int(i)] for i in block[0])) for block in parts]
        stats[str(seed)] = dict(correct_original=int(left.sum()),correct_edit=int(right.sum()),
            gained=int((gain==1).sum()),lost=int((gain==-1).sum()),active=int(active.sum()),
            **paired(gain,groups[ids]))
        sets.append((ids,left,right,active))
    a,b = sets
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[3],b[3])
    both_wrong = ~a[1]&~b[1]
    both_right = a[1]&b[1]
    avg = ((a[2].astype(int)-a[1])+(b[2].astype(int)-b[1]))/2
    overall = paired(avg,groups[a[0]])
    passed = all(x['net_correct']>0 for x in stats.values()) and overall['net_pp']>=.2
    return dict(rows=len(a[0]),per_seed=stats,mean_paired=overall,fold_net_correct=fold_changes,
        any_edit_eligible=int(a[3].sum()),original_joint_errors=int(both_wrong.sum()),
        eligible_joint_errors=int((both_wrong&a[3]).sum()),both_corrected=int((both_wrong&a[2]&b[2]).sum()),
        either_corrected=int((both_wrong&(a[2]|b[2])).sum()),joint_new_errors=int((both_right&~a[2]&~b[2]).sum()),
        gate_to_reserve=passed)


def run(args):
    old = args.workspace/'outputs/b-preference-replay-1123/study'
    sys.path.insert(0,str(old))
    import study
    from superagent.minilm_features import DEEP_FILES
    require(study.ROOT==args.workspace.resolve(),'Wrong workspace')
    old_run = args.head_run
    old_seal = read(old_run/'dev_seal.json')
    study.verified(args.archive,study.ARCHIVE_SHA)
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    helper = REPO/'superagent/ab_edit_views.py'
    save(out/'protocol.json',dict(plan=sha(REPO/'docs/batches/B_edit_context_protocol_1123_zh.md'),
        source={p.name:sha(p) for p in (Path(__file__),helper,Path(__file__).with_name('b_frozen_head_frontier.py'),old/'study.py')},
        archive=study.ARCHIVE_SHA,member=study.MEMBER,old_dev_seal=sha(old_run/'dev_seal.json'),
        model_files={k:sha(study.verified(study.WEIGHTS/'all-MiniLM-L12-v2'/k,v[1])) for k,v in DEEP_FILES.items()},
        fits=0,model_api=0))
    data = list(study.opened_rows(args.archive,('headline_A','headline_B')).values())
    groups = study.sample_groups(SimpleNamespace(kind='pair_text',X_train=data),'shared_text')
    first = read(study.ref_paths(SEEDS[0])[-1]/'protocol.json')['indices']
    dev = {i for s in first for i in s['outer']}
    require(len(dev)==6608,'Wrong development pool')
    views, cache, meta = vectors_for(study,out/'dev_vectors',data,sorted(dev))
    files = {}
    for seed in SEEDS:
        wide = study.ref_paths(seed)[-1]
        for fold, s in enumerate(read(wide/'protocol.json')['indices']):
            key = f'{seed}_fold_{fold}'
            old_file = study.verified(old_run/(key+'.npz'),old_seal['files'][key]['sha256'])
            with np.load(old_file) as arr:
                ids, old_z = arr['ids'], arr['base']
            np.testing.assert_array_equal(ids,s['outer'])
            require(set(s['train'])==dev-set(ids) and set(groups[ids]).isdisjoint(groups[s['train']]),'Split overlap')
            root=wide/f'fold_{fold}'
            full=study.ReadOnlyVectors(root/'encoder-mixture-wide_prediction_features/minilm_l12_vectors.sqlite')
            head, source=load_head(root/'encoder-mixture-wide',old_seal['files'][key]['source_sha256'],full,len(s['train']))
            base, new=margins(head,full,cache,[data[i] for i in ids],ids,views,old_z)
            file = out/(key+'.npz')
            with file.open('xb') as f:
                np.savez_compressed(f,ids=ids,base=base,edited=new,active=np.array([i in views for i in ids]))
            files[key]=dict(model=source['source_sha256'],sha256=sha(file))
    usage = dict(development=local_usage(out/'dev_vectors'))
    save(out/'dev_seal.json',dict(protocol=sha(out/'protocol.json'),files=files,**meta))
    result = score_dev(study,args.archive,groups,out,files)
    result['dev_seal']=sha(out/'dev_seal.json')
    save(out/'development.json',result)
    promote = False
    if result['gate_to_reserve']:
        ids = np.array(sorted(set(range(len(data)))-dev))
        require(len(ids)==1693 and set(groups[ids]).isdisjoint(groups[list(dev)]),'Contaminated reserve')
        views,cache,meta=vectors_for(study,out/'reserve_vectors',data,ids)
        root=study.OUT/'mainline-release-models/task2-wide'
        record=read(old/'run2/predictions.json')['blocks']['reserve']
        full=study.ReadOnlyVectors(root/'original.csv.features/minilm_l12_vectors.sqlite')
        head,source=load_head(root/'run',record['source_digests'][1],full,len(dev))
        rows=[data[i] for i in ids]
        base=head.classifier.decision_function(head.features(rows))
        with np.load(study.verified(old/'run2/reserve.npz',record['sha256'])) as saved:
            np.testing.assert_array_equal(ids,saved['row_ids'])
            np.testing.assert_array_equal(base>0,saved['deep'])
        base,new=margins(head,full,cache,rows,ids,views,base)
        with (out/'reserve.npz').open('xb') as f:
            np.savez_compressed(f,ids=ids,original=base>0,edited=new>0)
        save(out/'reserve_seal.json',dict(predict=sha(out/'reserve.npz'),dev=sha(out/'development.json'),**meta))
        labels=study.opened_rows(args.archive,('label_pairwise',),set(ids))
        y=np.array([int(labels[i][0]) for i in ids])
        a,b=(base>0)==y,(new>0)==y
        promote=int(b.sum())>=1235 and b.sum()>a.sum()
        save(out/'reserve.json',dict(correct_original=int(a.sum()),correct_edit=int(b.sum()),rows=len(ids),
             paired=paired(b.astype(int)-a,groups[ids]),course_validation_gate=bool(promote),
             prediction_seal=sha(out/'reserve_seal.json')))
        usage['reserve']=local_usage(out/'reserve_vectors')
    save(out/'completion.json',dict(development=sha(out/'development.json'),reserve=sha(out/'reserve.json')
        if result['gate_to_reserve'] else None,course_validation_gate=bool(promote),actual_new_fits=0,model_api_calls=0,
        api_rmb=0,local_usage=usage,course_validation_scored=False,official_test_scored=False,H_scored=False))
    print('completed',result['mean_paired'],'reserve',result['gate_to_reserve'],flush=True)


if __name__=='__main__':
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','archive','head-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
