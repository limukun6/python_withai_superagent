"""Verify released digests, charges, decisions and paired counts. Optionally check both patches against the base checkout."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(value, message):
    if not value:
        raise ValueError(message)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def folders():
    cold, warm, repeat = (E / key for key in ('cold','resumed','repeatability'))
    for folder in (cold,warm,repeat):
        check(all(sha(folder/(k+'.json')) == v for k,v in read(folder/'completion.json').items()), folder.name+' receipt')
    protocol, frozen, export, outcome = [read(cold/(x+'.json')) for x in ('protocol','models_frozen','exports','results')]
    check(frozen['protocol'] == sha(cold/'protocol.json') and export['frozen'] == sha(cold/'models_frozen.json')
          and export['features'] == protocol['features'], 'Cold freeze to export chain')
    check(len(frozen['catalog']) == 35 and (protocol['training_rows'], protocol['held_rows'],protocol['held_groups']) == (7364,937,702)
          and protocol['training_shared_groups'] == protocol['prior_shared_groups'] == 0, 'Cold data registration')
    predictions, actions = {}, {}
    for arm, run in frozen['arms'].items():
        path = cold/arm
        check([r['candidate'] for r in lines(path/'physical.jsonl')] == run['physical']
              and len(run['physical']) == run['actual_fits'] == 9 <= protocol['fit_cap_each'], 'Cold worker receipts')
        a, b = [path/f'export_{i}'/'rows.csv' for i in (0,1)]
        check(sha(a) == sha(b) == export['predictions'][arm], 'Prediction files or line endings differ')
        with a.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        check([r['row_id'] for r in rows] == list(map(str,range(937))) and {r['prediction'] for r in rows} == {'0','1'}, 'Export order')
        predictions[arm] = [r['prediction'] for r in rows]
        actions[arm] = [(a['action'],a['candidate_id'],a['split_id']) for a in run['actions']]
    check(actions['control'] == actions['proposal'] and not outcome['actions_differ'] and not outcome['gate'], 'Policy result')
    cells = read(cold/'groups.json')
    check(all(len(r) == 4 and all(type(x) is int and x >= 0 for x in r) for r in cells), 'Group counts')
    both_wrong, lost, gained, both_right = map(sum,zip(*cells,strict=True))
    total = both_wrong + lost + gained + both_right
    check((len(cells), total, gained, lost) == (702,937,2,0) and (gained,lost)==(outcome['gained'],outcome['lost'])
          and {'control':lost+both_right,'proposal':gained+both_right} == outcome['correct']
          and sum(a != b for a,b in zip(*predictions.values(),strict=True)) == gained+lost, 'Pairing and correct totals')
    audit = read(cold/'refit_audit.json')
    check(audit['search_and_confirmation_records_identical'] == 7 and audit['new_fits'] == 0 and
          audit['predicted_mismatches_vs_control_then_proposal'] == {'00':[0,2],'01':[2,0],'10':[0,2],'11':[2,0]}, 'Head substitution')
    prior, after = [read(E/'replay'/(k+'.json')) for k in ('before_guard','after_guard')]
    check(prior['runs'] == after['runs'] and prior['runs']['full_titles']['diverged'] and
          not prior['runs']['movie_dialogue']['diverged'], 'Replays before or after the guard')
    paths = read(E/'replay/observed_paths.json')
    check(sorted(v['matching_actions'] for v in paths['paths'].values()) == [8,8,8] and paths['production'] == after['replay_source'],
          'Native observed action reproduction')
    w, wf, wp = [read(warm/(k+'.json')) for k in ('results','model_frozen','protocol')]
    check(w['frozen'] == sha(warm/'model_frozen.json') and wf['protocol'] == sha(warm/'protocol.json')
          and wp['source_files'] == paths['production'] and wp['replay'] == sha(E/'replay/before_guard.json')
          and len(wf['reused_primary']) == w['physical_fits_reused'] == 5 and len(wf['new_workers']) == w['physical_fits_new'] == 3
          and [x['candidate'] for x in lines(warm/'physical.jsonl')] == [x['candidate'] for x in wf['new_workers']]
          and w['logical_path_fits'] == 8 and len(set(w['predictions_sha256'])) == 1, 'Resumed accounting or freeze')
    c = w['review']
    check(c['fresh_rows'] == 1358 and c['fresh_correct'][c['challenger']] - c['fresh_correct'][c['anchor']] == c['fresh_net'] == -15
          and w['delivered'] == w['old_winner'] == c['anchor'] and c['status'] == 'retain', 'Warm decision')
    rep, rp = read(repeat/'results.json'), read(repeat/'protocol.json')
    check(rep['protocol'] == sha(repeat/'protocol.json') and rp['features'] == protocol['features'] and
          rp['training'] == protocol['training'] and rp['frozen_model'] == frozen['arms']['control']['model'], 'Stability inputs')
    for i, expected in enumerate(rep['worker_sha256']):
        folder = repeat / str(i)
        worker = read(folder/'worker.json')
        check(expected == sha(folder/'worker.json') and worker['physical_sha256'] == sha(folder/'physical.jsonl')
              and [a['tolerance'] for a in lines(folder/'physical.jsonl')] == [1e-4,1e-6], 'Diagnostic fit meter')
    check(rep['physical_fits']==4 and not rep['accuracy_measured'] and all(rep['paired_fingerprints_match'].values())
          and all(x['vote_label_differences']==0 for x in rep['between_processes'].values()), 'Stability result')
    check(outcome['school_api_calls']==w['model_api_calls']==rep['model_api_calls']==0
          and not outcome['designated_validation_opened'] and not w['designated_validation_opened']
          and not outcome['formal_test_opened'] and not w['formal_test_opened'], 'Evaluation scope')
    overlay = ROOT / 'overlay'
    for folder, doc, runner in ((cold,'B_autonomous_close_singles_plan_20261124_zh','run_b_autonomous_close_singles'),
                                 (warm,'B_resume_close_singles_plan_20261124_zh','run_b_close_singles_continuation'),
                                 (repeat,'B_minilm_refit_repeatability_20261124_zh','check_b_minilm_refit_repeatability')):
        info = read(folder/'protocol.json')
        check(info['plan'] == sha(overlay/'docs/batches'/(doc+'.md')) and
              info['runner'] == sha(overlay/'scripts'/(runner+'.py')), 'Runner or plan changed')
    check(audit['runner_sha256'] == sha(overlay/'scripts/audit_b_close_singles_refits.py') and
          prior['runner_sha256'] == after['runner_sha256'] == sha(overlay/'scripts/replay_b_close_singles.py') and
          paths['runner'] == sha(overlay/'scripts/replay_b_close_singles_paths.py'), 'Audit source')
    return cells, outcome, protocol, wp['source_files']


def rank_routing():
    root = E/'rank-routing'
    run, frozen, protocol = [read(root/(k+'.json')) for k in ('results','predictions_frozen','protocol')]
    check(run['frozen'] == sha(root/'predictions_frozen.json') and frozen['protocol'] == sha(root/'protocol.json') and
          protocol['plan'] == sha(ROOT/'overlay/docs/batches/B_disagreement_ranks_plan_20261124_zh.md') and
          protocol['runner'] == sha(ROOT/'overlay/scripts/check_b_disagreement_ranks.py') and
          protocol['cold_receipt'] == sha(E/'cold/models_frozen.json') and protocol['full_receipt'] == sha(E/'resumed/model_frozen.json'),
          'Confidence rule and input chain')
    check(run['new_fits'] == run['model_api_calls'] == 0 and not run['course_validation_read'] and not run['formal_test_read']
          and run['shared_group_count_between_title_panels'] == 133, 'Confidence evaluation scope')
    review = []
    for name, expected in read(root/'completion.json').items():
        if not name.endswith('_groups'):
            check(expected == sha(root/(name+'.json')), name+' receipt')
            continue
        raw = gzip.decompress((root/(name+'.json.gz')).read_bytes())
        check(hashlib.sha256(raw).hexdigest() == expected, name+' groups digest')
        cells = json.loads(raw)
        panel = run['panels'][name.removesuffix('_groups')]
        neither, lost, gained, both = map(sum,zip(*cells,strict=True))
        check(len(cells) == panel['groups'] and sum((neither,lost,gained,both)) == panel['rows'] and
              (lost,gained,both+lost,both+gained,gained-lost) ==
              (panel['lost'],panel['gained'],panel['correct']['baseline'],panel['correct']['proposed'],panel['net']),
              name+' paired counts')
        review.append((cells, panel['paired_group_95_pp']))
    cold = run['panels']['cold_titles']
    nets = [p['net'] for p in run['panels'].values()]
    gate = sum(n > 0 for n in nets)>=2 and all(n>=0 for n in nets) and cold['delta_pp']>=.5 and cold['paired_group_95_pp'][0]>0
    check(gate == run['passes_gate'] and not gate, 'Confidence promotion test')
    return review


def interval(cells, expected, seed):
    import numpy as np
    groups = np.asarray(cells)
    delta, sizes = groups[:,2]-groups[:,1], groups.sum(axis=1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(100):
        ix = rng.integers(0,len(groups),size=(50,len(groups)))
        values.extend(100*delta[ix].sum(axis=1)/sizes[ix].sum(axis=1))
    check(np.allclose(np.quantile(values,[.025,.975]),expected,rtol=0,atol=1e-12), 'Paired confidence interval')


def patches(base, protocol, hardened):
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        source = subprocess.check_output(['git','-C',str(base),'archive',protocol['base'],'--','superagent'])
        with tarfile.open(fileobj=io.BytesIO(source)) as archive:
            for member in archive.getmembers():
                check(not member.name.startswith('/') and '..' not in Path(member.name).parts and
                      (member.isdir() or member.isfile()), 'Unexpected archived pathname')
            archive.extractall(repo)
        def sources():
            root = repo/'superagent'
            return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*.py')}
        check(sources()==protocol['sources']['control'], 'Base source digest')
        for name, expected in (('01-measured.patch',protocol['sources']['proposal']),('02-budget.patch',hardened)):
            subprocess.run(['git','apply','--check',str(ROOT/name)],cwd=repo,check=True)
            subprocess.run(['git','apply',str(ROOT/name)],cwd=repo,check=True)
            check(sources()==expected, name+' source digest')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',action='store_true')
    parser.add_argument('--base-repo',type=Path)
    args=parser.parse_args()
    cells, result, protocol, hardened = folders()
    ranks = rank_routing()
    if args.interval:
        interval(cells,result['group_bootstrap_95_pp'],62114)
        for rank_cells, expected in ranks:
            interval(rank_cells,expected,29013)
    if args.base_repo is not None:
        patches(args.base_repo.resolve(),protocol,hardened)
    print('Verified cold 676/937 and 678/937, retrospective routing 680/937, 5 reused and 25 new fits, 0 model API calls.')
