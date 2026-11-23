"""Compare fixed coefficient shrinkage and cross-partition averaging without fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.special import expit


REPO = Path(__file__).resolve().parents[1]
SEEDS = (47147,47381)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path, data):
    with path.open('x') as f:
        json.dump(data,f,indent=2)
        f.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_head(folder, expected_sha, vectors, outer_train):
    from superagent.minilm_features import DeeperMiniLMTextClassifier
    release = read(folder/'final_selection.json')
    model_path = folder/release['model_file']
    require(sha(model_path)==release['model_sha256']==expected_sha,'Model fingerprint differs')
    require(release['training_rows']==outer_train and release['refit'] is True,'Full refit size differs')
    mixed = joblib.load(model_path)['model']
    heads = [m for m in (mixed.anchor,mixed.other) if isinstance(m,DeeperMiniLMTextClassifier)]
    require(len(heads)==1,'Expected one twelve-layer component')
    head = heads[0]
    role = 'full_refit' if head is mixed.anchor else 'retained_primary'
    n = outer_train if role=='full_refit' else len(read(folder/'split.json')['train_indices'])
    head.encoder = vectors
    return head, dict(role=role,training_pairs=n,source_sha256=expected_sha)


def score_head(head, pairs, n):
    x = head.features(pairs)
    model = head.classifier
    np.testing.assert_array_equal(model.classes_, [0,1])
    base = model.decision_function(x)
    coefs = model.coef_[0].copy()
    start = 0
    for vectorizer in (head.word,head.char):
        require(vectorizer.smooth_idf and vectorizer.min_df==2,'Unexpected TF-IDF settings')
        dfs = (2*n+1)*np.exp(1-vectorizer.idf_)-1
        np.testing.assert_allclose(dfs,np.round(dfs),atol=1e-6,rtol=0)
        dfs = np.round(dfs)
        require(np.all((dfs>=2)&(dfs<=2*n)), 'Document frequencies out of range')
        end = start+len(dfs)
        coefs[start:end] *= dfs/(dfs+4)
        start = end
    require(x.shape[1]-start==head.scaler.n_features_in_+384, 'Unexpected feature block order')
    shrink = np.asarray(x@coefs).ravel()+model.intercept_[0]
    require(np.all(np.isfinite(base)) and np.all(np.isfinite(shrink)), 'Non-finite margin')
    return base, shrink


def paired(delta, groups):
    unique, ids = np.unique(groups,return_inverse=True)
    counts = np.bincount(ids)
    net = np.bincount(ids,weights=delta)
    rng = np.random.default_rng(60423)
    draws = []
    for _ in range(100):
        sampled = rng.integers(0,len(unique),size=(50,len(unique)))
        draws.extend(100*net[sampled].sum(axis=1)/counts[sampled].sum(axis=1))
    return dict(groups=len(unique),net_correct=float(delta.sum()),net_pp=float(delta.mean()*100),
                group_bootstrap_95_pp=np.quantile(draws,[.025,.975]).tolist())


def infer_dev(study, data, groups, prior, out):
    inventory = read(prior/'predictions.json')['blocks']
    artifacts = {}
    for seed in SEEDS:
        _, _, _, wide = study.ref_paths(seed)
        splits = read(wide/'protocol.json')['indices']
        for fold, split in enumerate(splits):
            stem = f'{seed}_fold_{fold}'
            old = inventory[stem]
            train, ids = np.array(split['train']), np.array(split['outer'])
            require(not set(groups[train])&set(groups[ids]),'Outer title group reached fitting')
            with np.load(study.verified(prior/(stem+'.npz'),old['sha256'])) as saved:
                np.testing.assert_array_equal(saved['row_ids'],ids)
                historical = saved['deep']
            fold_dir = wide/f'fold_{fold}'
            vectors = study.ReadOnlyVectors(fold_dir/'encoder-mixture-wide_prediction_features/minilm_l12_vectors.sqlite')
            head, meta = load_head(fold_dir/'encoder-mixture-wide',old['source_digests'][1],vectors,len(train))
            require(meta['role']==old['l12_role'],'Training role differs')
            base, shrink = score_head(head,[data[i] for i in ids],meta['training_pairs'])
            np.testing.assert_array_equal(base>0,historical)
            file = out/(stem+'.npz')
            with file.open('xb') as f:
                np.savez_compressed(f,ids=ids,base=base,shrink=shrink)
            artifacts[stem] = dict(**meta,rows=len(ids),sha256=sha(file))
            print(stem,meta['role'],len(ids),flush=True)
    save(out/'dev_seal.json',dict(protocol=sha(out/'protocol.json'),files=artifacts))


def development(study, archive, groups, out):
    seal = read(out/'dev_seal.json')
    require(seal['protocol']==sha(out/'protocol.json'),'Development seal differs')
    margins = {}
    common = None
    for seed in SEEDS:
        folds = []
        for fold in range(5):
            stem = f'{seed}_fold_{fold}'
            with np.load(study.verified(out/(stem+'.npz'),seal['files'][stem]['sha256'])) as a:
                folds.append([a[k] for k in ('ids','base','shrink')])
        ids,base,shrink = [np.concatenate(items) for items in zip(*folds,strict=True)]
        order = np.argsort(ids)
        ids = ids[order]
        require(len(ids)==len(np.unique(ids))==6608,'Incomplete unique OOF rows')
        if common is None:
            common = ids
        np.testing.assert_array_equal(common,ids)
        margins[seed] = (base[order],shrink[order])
    truth = study.opened_rows(archive,('label_pairwise',),set(common))
    y = np.array([int(truth[i][0]) for i in common])
    g = groups[common]
    base_hits, shrunk_hits = [], []
    by_seed = {}
    for seed,(base,shrink) in margins.items():
        left, right = (base>0)==y, (shrink>0)==y
        base_hits.append(left)
        shrunk_hits.append(right)
        delta = right.astype(int)-left
        by_seed[str(seed)] = dict(base=int(left.sum()),shrink=int(right.sum()),
            gained=int((delta==1).sum()),lost=int((delta==-1).sum()),**paired(delta,g))
    base_hits,shrunk_hits = np.array(base_hits),np.array(shrunk_hits)
    avg_prob = np.mean([expit(margins[s][0]) for s in SEEDS],axis=0)
    bag = (avg_prob>.5)==y
    reference = int(np.argmax(base_hits.sum(axis=1)))
    delta = bag.astype(int)-base_hits[reference]
    bag_stats = dict(correct=int(bag.sum()),reference_seed=SEEDS[reference],gained=int((delta==1).sum()),
        lost=int((delta==-1).sum()),**paired(delta,g))
    avg_delta = np.mean(shrunk_hits.astype(int)-base_hits.astype(int),axis=0)
    shrink_stats = paired(avg_delta,g)
    passed = {'shrink':all(v['net_correct']>0 for v in by_seed.values()) and shrink_stats['net_pp']>=.2,
              'bag':bag_stats['net_pp']>=.2}
    gains = {'shrink':shrink_stats['net_pp'],'bag':bag_stats['net_pp']}
    winner = max((name for name in passed if passed[name]),key=gains.get,default=None)
    disagree = base_hits[0]!=base_hits[1]
    outcome = dict(rows=len(y),per_seed=by_seed,shrink_paired_two_seed_mean=shrink_stats,bag=bag_stats,
        disagreement=dict(rows=int(disagree.sum()),bag_correct=int(bag[disagree].sum()),
            both_wrong=int((~base_hits.any(axis=0)).sum()),both_right=int(base_hits.all(axis=0).sum())),
        gates=passed,selected=winner,prediction_seal=sha(out/'dev_seal.json'))
    save(out/'development.json',outcome)
    return winner, set(common)


def reserve(study, archive, data, groups, prior, out, winner, dev):
    ids = np.array(sorted(set(range(len(data)))-dev))
    require(len(ids)==1693 and set(groups[ids]).isdisjoint(groups[list(dev)]),'Reserve partition differs')
    release = study.OUT/'mainline-release-models/task2-wide'
    vectors = study.ReadOnlyVectors(release/'original.csv.features/minilm_l12_vectors.sqlite')
    old = read(prior/'predictions.json')['blocks']['reserve']
    pairs = [data[i] for i in ids]
    head, meta = load_head(release/'run',old['source_digests'][1],vectors,len(dev))
    require(meta['role']=='full_refit','Reference is not the full development fit')
    base, shrink = score_head(head,pairs,len(dev))
    with np.load(study.verified(prior/'reserve.npz',old['sha256'])) as saved:
        np.testing.assert_array_equal(ids,saved['row_ids'])
        np.testing.assert_array_equal(base>0,saved['deep'])
    scores = {'full_deep':base>0}
    sources = [meta]
    if winner=='shrink':
        scores[winner] = shrink>0
    else:
        files = read(out/'dev_seal.json')['files']
        fold = next(i for i in range(5) if all(files[f'{s}_fold_{i}']['role']=='full_refit' for s in SEEDS))
        probabilities = []
        for seed in SEEDS:
            wide = study.ref_paths(seed)[-1]
            source = files[f'{seed}_fold_{fold}']
            train = read(wide/'protocol.json')['indices'][fold]['train']
            require(set(train)<=dev, 'Bag member reaches reserve')
            head, member = load_head(wide/f'fold_{fold}/encoder-mixture-wide',source['source_sha256'],vectors,len(train))
            z = head.classifier.decision_function(head.features(pairs))
            scores[str(seed)] = z>0
            probabilities.append(expit(z))
            sources.append(dict(seed=seed,fold=fold,**member))
        scores[winner] = np.mean(probabilities,axis=0)>.5
    predictions = out/'reserve_predictions.npz'
    with predictions.open('xb') as f:
        np.savez_compressed(f,ids=ids,**scores)
    save(out/'reserve_seal.json',dict(development=sha(out/'development.json'),sha256=sha(predictions),sources=sources))
    labels = study.opened_rows(archive,('label_pairwise',),set(ids))
    y = np.array([int(labels[i][0]) for i in ids])
    hits = {k:v==y for k,v in scores.items()}
    correct = {k:int(v.sum()) for k,v in hits.items()}
    others = {k:paired(hits[winner].astype(int)-v,groups[ids]) for k,v in hits.items() if k!=winner}
    promote = all(result['net_correct']>0 for result in others.values()) and correct[winner]>=1235
    save(out/'reserve.json',dict(rows=len(ids),correct=correct,paired=others,gate_to_course_validation=promote,
        prediction_seal=sha(out/'reserve_seal.json')))
    return promote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    old_study = args.workspace/'outputs/b-preference-replay-1123/study'
    sys.path.insert(0,str(old_study))
    import study
    require(study.ROOT==args.workspace.resolve(),'Wrong experiment source directory')
    prior, out = old_study/'run2', args.output
    out.mkdir(exist_ok=False,parents=True)
    study.verified(args.archive,study.ARCHIVE_SHA)
    save(out/'protocol.json',dict(plan_sha256=sha(REPO/'docs/batches/B_accuracy_frontier_protocol_1123_zh.md'),
        script_sha256=sha(Path(__file__)),helper_sha256=sha(old_study/'study.py'),archive_sha256=study.ARCHIVE_SHA,
        only_member=study.MEMBER,prior_inventory_sha256=sha(prior/'predictions.json'),
        prior_inventory=read(prior/'predictions.json')['protocol'],seeds=list(SEEDS),
        fit_cap=0,api_cap=0,official_test=False))
    features = list(study.opened_rows(args.archive,('headline_A','headline_B')).values())
    groups = study.sample_groups(SimpleNamespace(kind='pair_text',X_train=features),'shared_text')
    infer_dev(study,features,groups,prior,out)
    winner, dev = development(study,args.archive,groups,out)
    validation_gate = reserve(study,args.archive,features,groups,prior,out,winner,dev) if winner else False
    save(out/'completion.json',dict(selected=winner,development=sha(out/'development.json'),
        reserve=sha(out/'reserve.json') if winner else None,course_validation_gate=validation_gate,
        actual_new_supervised_fits=0,model_api_requests=0,new_api_rmb=0,official_validation_scored=False,
        official_test_scored=False,H_scored=False))
    print('completed',winner,'course validation gate',validation_gate,flush=True)


if __name__=='__main__':
    main()
