"""Give a genuinely promoted rule its own feedback and challenge its descendant on new tasks."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from scripts.replay_planner_review_archive import episode, propose
from scripts.run_b_kernel_domain_transfer import read_original
from scripts.run_planner_self_review import distinct, packet
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_planner_lineage import freeze, original_traces, provenance, run_panel, trace_digest, verdict


PLAN = ROOT/'research/rsi_lineage_1125/descendant_protocol.json'


def packets(root, source):
    sentences, labels = zip(*read_original(source)['yelp'])
    rng = np.random.default_rng(655408)
    x = rng.normal(size=(1000, 12))
    y = np.where(x[:,0] < .15, 1.3*x[:,1]+x[:,2] > .45, x[:,3]*x[:,4] > .22)
    y = np.logical_xor(y, rng.random(len(y)) < .09).astype(int)
    iris = load_iris()
    rng = np.random.default_rng(655409)
    codes = rng.integers(0, 7, size=(1000, 7))
    order = rng.permutation(7)
    target = np.where(codes[:,2] % 3 == 0, codes[:,3] == codes[:,4], order[codes[:,0]] > order[codes[:,1]])
    target = np.logical_xor(target, rng.random(len(target)) < .075).astype(int)
    specs = [('x11','text','macro_f1',np.array(sentences),np.array(labels)),
             ('x12','tabular','balanced_accuracy',x,y),
             ('x21','tabular','macro_f1',iris.data,iris.target),
             ('x22','categorical','balanced_accuracy',np.char.add('v',codes.astype(str)),target)]
    (root/'evaluation').mkdir()
    result = {}
    for key, modality, metric, x, y in specs:
        folder = root/key
        folder.mkdir()
        ids = distinct(x,y)
        fit, held = train_test_split(ids, test_size=.25, stratify=y[ids], random_state=655407)
        packet(folder/'p.zip',modality,metric,x[fit],y[fit])
        packet(folder/'q.zip',modality,metric,x[held])
        truth = root/'evaluation'/(key+'.npy')
        np.save(truth,y[held],allow_pickle=False)
        result[key] = dict(metric=metric, train_rows=len(fit), held_rows=len(held),
                           train=sha(folder/'p.zip'), features=sha(folder/'q.zip'), truth=sha(truth))
    return result


def prepare(root, main, prior, source):
    root.mkdir(parents=True, exist_ok=False)
    (root/'rules').mkdir()
    waves = [read(main/f'wave_{i}_result.json') for i in range(1,4)]
    require([w['verdict']['promote'] for w in waves] == [False, False, True],
            'Extension requires an actual first promotion on the final wave')
    parent = waves[-1]['next_parent']
    require(parent == waves[-1]['roles']['child'] != waves[-1]['roles']['parent'], 'Not a promoted child')
    history = original_traces(prior)
    require({k:trace_digest(v) for k,v in history.items()} == read(main/'protocol.json')['traces'],
            'Historical inputs changed')
    for i, wave in enumerate(waves):
        rule = parent if i == len(waves)-1 else wave['roles']['parent']
        for key in wave['cases']:
            path = main/key/rule/'run/agent'
            require(trace_digest(path) == read(main/key/rule/'sealed.json')['training_trace'],
                    'Feedback does not match this actual solver')
            history[key] = path
    require(len(history) == 12, 'Each previous task should contribute exactly once')
    old = read(main/'rules'/(parent+'.json'))['weights']
    require(freeze(root, old) == parent, 'Incorrect parent weights')
    step = min(1., 1.25 * read(main/'wave_3_proposal.json')['step'])
    mutated = propose(np.array(old), {k: episode(v) for k,v in history.items()}, step)
    child = freeze(root, mutated)
    require(child != parent, 'No new planning rule was produced')
    proposal = dict(roles=dict(parent=parent,child=child), parent_weights=old, child_weights=mutated.tolist(),
        step=step, fit_traces={k:trace_digest(v) for k,v in history.items()},
        feedback_from_promoted_run=list(waves[-1]['cases']), parent_admission=sha(main/'wave_3_result.json'))
    save(root/'proposal.json',proposal)
    save(root/'protocol.json', dict(source=provenance(), extra_plan=sha(PLAN), extra_runner=sha(Path(__file__)),
         source_reviews=sha(source), proposal=sha(root/'proposal.json'), cases=packets(root, source)))


def run(root, weights):
    config = read(root/'protocol.json')
    require(config['extra_plan'] == sha(PLAN) and config['extra_runner'] == sha(Path(__file__)) and
            config['proposal'] == sha(root/'proposal.json'), 'Frozen extension changed')
    proposal = read(root/'proposal.json')
    roles = proposal['roles']
    challenge = run_panel(root, ['x11','x12'], roles, weights, config, 'challenge')
    decision = verdict(challenge)
    save(root/'challenge_result.json',dict(cases=challenge, verdict=decision))
    held = run_panel(root, ['x21','x22'], roles, weights, config, 'transfer') if decision['promote'] else {}
    receipts = [read(p) for p in root.glob('*/*/sealed.json')]
    physical = sum(len(f.read_text().splitlines()) for f in root.glob('*/*/fits.jsonl'))
    require(0 < physical == sum(r['fits'] for r in receipts) <= read(PLAN)['maximum_new_classifier_fits']
            and all(r['api'] == 0 for r in receipts), 'Extension accounting mismatch')
    save(root/'completion.json',dict(config=sha(root/'protocol.json'), proposal=proposal, challenge=challenge,
        verdict=decision, transfer=held, fits=physical, runs=len(receipts), api_calls=0, course_test_accessed=False))
    print('Descendant verdict', decision, 'verified fits', physical, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--main-panel', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--source-reviews', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root.resolve(), args.main_panel.resolve(), args.prior.resolve(), args.source_reviews.resolve())
    else:
        run(args.root.resolve(), args.weights.resolve())
