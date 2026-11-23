"""Give the production Agent an anonymous movie training packet and evaluate on unseen films."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, packets, read, save
from scripts.run_b_quote_transfer import MEMBERS, movie_pairs, normalized
from superagent import ab_run
from superagent.ab_methods import JOINT_PREFERENCE
from superagent.splits import sample_groups


PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_quote_adaptation_1123_zh.md'
sha = ab_run.sha
SEED = 18643
FIT_CAP = 11


def movie_groups(rows, films):
    edges = [('film:' + film, 'text:' + normalized(text))
             for row, film in zip(rows, films) for text in row[:2]]
    roots = sample_groups(SimpleNamespace(kind='pair_text', X_train=edges), 'shared_text')
    check(np.array_equal(roots[::2], roots[1::2]), 'Two sides of a film belong to different components')
    by_root = defaultdict(set)
    for film, root in zip(films, roots[::2]):
        by_root[root].add(film)
    key = {}
    for root, names in by_root.items():
        slug = json.dumps(['quote-adaptation-1123-v1', *sorted(names)], ensure_ascii=False,
                          separators=(',', ':')).encode('utf-8')
        key[root] = hashlib.sha256(slug).hexdigest()
    return [key[root] for root in roots[::2]]


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    rows, films, omitted = movie_pairs(a.archive)
    groups = movie_groups(rows, films)
    held = [i for i, group in enumerate(groups) if int(group, 16) % 5 == 0]
    hold = set(held)
    train = [i for i in range(len(rows)) if i not in hold]
    def members(ids):
        return ({groups[i] for i in ids}, {films[i] for i in ids},
                {normalized(t) for i in ids for t in rows[i][:2]})
    training, heldout = members(train), members(held)
    check(all(x.isdisjoint(y) for x, y in zip(training, heldout)), 'A movie, text or component crosses the outer split')
    check(train and held, 'Empty outer partition')
    packets(rows, train, True, a.out / 'anonymous.zip')
    packets(rows, held, False, a.out / 'features.zip')
    score = a.out / 'scoring_only.json'
    save(score, dict(ids=held, labels=[rows[i][2] for i in held], components=[groups[i] for i in held]))
    save(a.out / 'protocol.json', dict(seed=SEED, fit_limit=FIT_CAP, api_cap=0, membership=MEMBERS,
        source_rows=len(rows), omitted=omitted, train_rows=len(train), held_rows=len(held),
        train_components=len(training[0]), held_components=len(heldout[0]), train_films=len(training[1]),
        held_films=len(heldout[1]), shared_outer_components=0, shared_outer_texts=0, shared_outer_films=0,
        division='SHA256 of salted sorted movie names mod 5; 0 is held', method_menu=JOINT_PREFERENCE,
        train_packet=sha(a.out/'anonymous.zip'), features=sha(a.out/'features.zip'), scoring=sha(score),
        plan=sha(PLAN), runner=sha(Path(__file__)), production=ab_run.sources()))


def train_once(a, plan, env):
    d = a.out / 'autonomous'
    physical = a.out / 'physical.jsonl'
    if not d.exists() and not physical.exists() and not (a.out/'run.log').exists():
        args = ['run', str(a.out/'anonymous.zip'), '--portfolio', 'ab-v1', '--planner', 'local',
                '--scheduler', 'ab-adaptive-complement', '--preference-catalog', JOINT_PREFERENCE,
                '--cover-untried-encoder', '--seed', str(SEED), '--fit-limit', str(FIT_CAP),
                '--trial-timeout', '300', '--time-limit', '1200', '--run-dir', str(d)]
        code, _ = command(args, a.out/'run.log',
                          dict(env, AB_B1_FIT_LOG=str(physical), AB_B1_PER_WORKER_FIT_LIMIT='1'), 1260)
        check(code == 0, 'Training process stopped. Inspect its existing fit and outcome records.')
    outcome = read(d/'run_outcome.json')
    selection = read(d/'final_selection.json')
    history = read(d/'history.json')
    ledger = read(d/'classifier_fits.json')
    actions = read(d/'actions.json')
    manifest = read(d/'run_manifest.json')
    logs = [json.loads(line) for line in physical.read_text().splitlines()]
    usage = [json.loads(line) for line in (d/'llm_usage.jsonl').read_text().splitlines()]
    members = selection['ensemble_components'] or [selection['family']]
    check(manifest['cover_untried_encoder'] and manifest['preference_catalog'] == JOINT_PREFERENCE and
          outcome['requested_planner_completed'] and outcome['source_unchanged'] and
          outcome['full_refit_completed'] and not outcome['formal_test_accessed'] and
          outcome['new_api_requests'] == 0 and len(usage) == 1 and usage[0]['event'] == 'ledger_initialized'
          and usage[0]['cost_rmb'] == 0, 'Native outcome or budget differs from protocol')
    check(selection['training_rows'] == plan['train_rows'] and
          sha(d/'selected_model.joblib') == selection['model_sha256'] and
          len(logs) == ledger['attempted_classifier_fits'] == ledger['completed_classifier_fits'] <= FIT_CAP,
          'Model record or physical fit accounting differs')
    frozen_save(a.out/'model_frozen.json', dict(model=selection['model_sha256'], selected=members,
        history=[{'candidate': h['candidate'], 'split': h['split_id'], 'score': h['search_score']} for h in history],
        actions=[{k: act.get(k) for k in ('action','candidate_id','split_id','reason','evidence_trials')} for act in actions],
        decision_digest=sha(d/'actions.json'), fit_log=sha(physical), fits=len(logs),
        model_api_calls=0, training_rows=selection['training_rows']))


def score(a, plan, env):
    d = a.out/'autonomous'
    choice = read(a.out/'model_frozen.json')
    check(sha(d/'selected_model.joblib') == choice['model'] and sha(d/'actions.json') == choice['decision_digest'],
          'Frozen decision changed')
    export = verified_export(d, a.out, env, a.out)
    frozen_save(a.out/'outer_export.json', dict(model=choice['model'], predictions=export, processes=2))
    check(sha(a.out/'scoring_only.json') == plan['scoring'], 'Outer evaluation rows changed')
    frozen = read(a.transfer/'protocol.json')
    seals = read(a.transfer/'exports.json')
    completed = read(a.transfer/'completion.json')
    check(frozen['members'] == MEMBERS and frozen['rows'] == plan['source_rows'] and
          completed['exports'] == sha(a.transfer/'exports.json') and
          frozen['evaluation_record'] == sha(a.transfer/'scoring_only.json'), 'Frozen comparison changed')
    labels = read(a.out/'scoring_only.json')
    y, ids, groups = np.array(labels['labels']), np.array(labels['ids']), np.array(labels['components'])
    np.testing.assert_array_equal(y, np.array(read(a.transfer/'scoring_only.json')['winner_left'])[ids])
    hits = {'adapted': classes(a.out/'autonomous_export_0/rows.csv', export, len(y)) == y}
    for name, subdir in [('joint', 'autonomous_export_0'), ('control', 'soft_vote_export_0')]:
        hits[name] = classes(a.transfer/subdir/'rows.csv', seals[name], frozen['rows'])[ids] == y
    delta = hits['adapted'].astype(int) - hits['joint'].astype(int)
    summary = {name: {'correct': int(h.sum()), 'accuracy': float(h.mean()),
                      'versus_chance': paired(h.astype(float) - .5, groups)} for name, h in hits.items()}
    result = dict(rows=len(y), films=plan['held_films'], components=plan['held_components'], scores=summary,
        fixed_left_correct=int(y.sum()), fixed_right_correct=int(len(y)-y.sum()), adapted_only=int((delta>0).sum()),
        joint_only=int((delta<0).sum()), adapted_minus_frozen=paired(delta, groups), selected=choice['selected'],
        new_fits=choice['fits'], model_api_calls=0, new_rmb=0, prediction=export, formal_test_accessed=False)
    frozen_save(a.out/'results.json', result)
    frozen_save(a.out/'completion.json', dict(results=sha(a.out/'results.json'), protocol=sha(a.out/'protocol.json'),
        decision=sha(a.out/'model_frozen.json'), exports=sha(a.out/'outer_export.json'), fits=choice['fits'], model_api_calls=0))
    print(json.dumps(result))


def run(a):
    if not a.out.exists():
        prepare(a)
    plan = read(a.out/'protocol.json')
    check(plan['plan'] == sha(PLAN) and plan['runner'] == sha(Path(__file__)) and
          plan['production'] == ab_run.sources() and plan['train_packet'] == sha(a.out/'anonymous.zip') and
          plan['features'] == sha(a.out/'features.zip'), 'Registered code or packets changed')
    env = offline_environment(a.workspace)
    if a.phase in ('train', 'all'):
        train_once(a, plan, env)
    if a.phase in ('score', 'all'):
        score(a, plan, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace', 'archive', 'transfer', 'out'):
        parser.add_argument('--'+key, type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--phase', choices=('prepare','train','score','all'), default='all')
    run(parser.parse_args())
