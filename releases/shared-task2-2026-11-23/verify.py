"""Recount the published evidence. Supply the official archive to additionally grade both prediction CSVs."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
SHADOW = ROOT / 'borderline'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def predictions(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = csv.DictReader(stream)
        require(rows.fieldnames == ['row_id', 'prediction'], 'Unexpected CSV schema')
        values = list(rows)
    require([r['row_id'] for r in values] == list(map(str, range(940))), 'Validation row IDs changed')
    require({r['prediction'] for r in values} == {'0', '1'}, 'Invalid predicted class')
    return [int(r['prediction']) for r in values]


def completed(folder):
    for name, digest in read(folder/'completion.json').items():
        require(sha(folder/(name+'.json')) == digest, 'Experiment seal differs: '+name)


def check_calibration():
    folder = ROOT/'calibration'
    completed(folder)
    protocol = read(folder/'protocol.json')
    require(protocol['runner_sha256'] == sha(ROOT/'overlay/scripts/study_b_grouped_blend.py') and
            protocol['plan_sha256'] == sha(ROOT/'overlay/docs/batches/B_grouped_blend_plan_20261124_zh.md') and
            protocol['shadow_completion_sha256'] == sha(SHADOW/'completion.json') and
            protocol['model_digests'] == read(SHADOW/'frozen.json')['models'], 'Calibration sources differ')
    results = read(folder/'results.json')
    net = []
    for n, (groups,panel) in enumerate(zip(read(folder/'grouped.json'), results['panels'], strict=True)):
        cells = list(map(sum, zip(*groups, strict=True)))
        counts = [sum(x for k,x in enumerate(cells) if k & bit) for bit in (1,2,4)]
        expected = [panel[key] for key in ('single','equal','learned')]
        receipt = read(folder/f'fit_{1-n}_completed.json')
        require(sum(cells) == panel['rows'] and len(groups) == panel['groups'] and counts == expected and
                receipt['started'] == sha(folder/f'fit_{1-n}_started.json') and receipt['source_panel'] == 1-n
                and receipt['target_panel'] == n and receipt['supervised_calibration_fits'] == 1 and
                0 <= receipt['weight_on_anchor'] == panel['weight_from_other_panel'] <= 1,
                'Cross-panel training or grouped counts differ')
        net.append(counts[2]-max(counts[:2]))
    require(sum(net) == results['net_vs_stronger'] and
            (min(net) >= 0 and sum(net) >= 8) == results['pass_gate'] and
            results['new_base_fits'] == results['model_api_calls'] == 0 and results['supervised_calibration_fits'] == 2,
            'Calibration decision or cost differs')
    return sum(net)


def check_e5():
    folder = ROOT/'matched-e5'
    completed(folder)
    p = read(folder/'protocol.json')
    frozen = read(folder/'models_frozen.json')
    workers = [json.loads(line)['candidate'] for line in (folder/'physical.jsonl').read_text().splitlines()]
    receipt = read(folder/'worker_fit.json')
    require(p['plan'] == sha(ROOT/'overlay/docs/batches/B_matched_e5_plan_20261124_zh.md') and
            p['runner'] == sha(ROOT/'overlay/scripts/study_b_matched_e5.py') and
            p['shadow_completion'] == sha(SHADOW/'completion.json') and p['rows'] == 4971 and
            p['training_ids'] == read(SHADOW/'protocol.json')['training_ids_sha256'] and
            p['base_model'] == read(SHADOW/'frozen.json')['models'][0] and
            frozen['physical'] == sha(folder/'physical.jsonl') and frozen['worker'] == sha(folder/'worker_fit.json')
            and workers == [p['candidate']] and receipt['completed_classifier_fits'] == receipt['attempted_classifier_fits'] == 1,
            'E5 comparison sources or supervised counts differ')
    results = read(folder/'results.json')
    differences = []
    for n, (groups,panel) in enumerate(zip(read(folder/'grouped.json'),results['panels'],strict=True)):
        cells = list(map(sum,zip(*groups,strict=True)))
        counts = [sum(x for i,x in enumerate(cells) if i & bit) for bit in (1,2,4)]
        require(counts == [panel[k] for k in ('single','e5','equal')] and
                counts[0] == read(SHADOW/'results.json')['panels'][n]['anchor'] and
                sum(cells) == panel['rows'] and len(groups) == panel['groups'] and
                cells[4]+cells[6] == panel['gained_vs_anchor'] and cells[1]+cells[3] == panel['lost_vs_anchor'],
                'Matched E5 panel differs')
        differences.append(counts[2]-max(counts[:2]))
    require(sum(differences) == results['net_vs_stronger'] and
            results['pass_gate'] == (min(differences) >= 0 and sum(differences) >= 8) and
            results['classifier_fits'] == 1 and results['model_api_calls'] == 0,
            'Matched E5 decision differs')
    return sum(differences)


def check_fresh(interval):
    folder = ROOT/'fresh-singles'
    completed(folder)
    p = read(folder/'protocol.json')
    f = read(folder/'models_frozen.json')
    r = read(folder/'results.json')
    require(p['previous_experiment'] == sha(ROOT/'matched-e5/completion.json') and
            p['runner'] == sha(ROOT/'overlay/scripts/study_b_fresh_single_encoders.py') and
            p['plan'] == sha(ROOT/'overlay/docs/batches/B_fresh_single_encoders_plan_20261124_zh.md') and
            p['train_rows'] + sum(x['rows'] for x in p['panels']) == 8301 and
            p['excluded_evaluation_rows'] == 3330 and
            all(x['prior_shared_groups'] == x['training_shared_groups'] == 0 for x in p['panels']),
            'Fresh panel protocol differs')
    names = [json.loads(line)['candidate'] for line in (folder/'physical.jsonl').read_text().splitlines()]
    require(f['physical'] == sha(folder/'physical.jsonl') and names == p['names'] and
            f['receipts'] == [sha(folder/f'fit_{n}.json') for n in range(2)] and
            len(names) == r['classifier_fits'] == 2 and r['model_api_calls'] == 0, 'Fresh panel fitting differs')
    flat = []
    grouped = read(folder/'grouped.json')
    for cells, panel, registered in zip(grouped,r['panels'],p['panels'],strict=True):
        neither,lost,gained,both = map(sum,zip(*cells,strict=True))
        require(neither+lost+gained+both == panel['rows'] == registered['rows'] and
                len(cells) == panel['groups'] == registered['groups'] and both+lost == panel['l12']
                and both+gained == panel['e5'] and gained-lost == panel['net'], 'Fresh grouped counts differ')
        flat.extend(cells)
    net = sum(x[2]-x[1] for x in flat)
    pp = 100*net/sum(map(sum,flat))
    require(r['net'] == net and abs(r['delta_pp']-pp)<1e-12 and
            r['pass_gate'] == (all(x['net']>0 for x in r['panels']) and pp>=.5 and r['group_bootstrap_95_pp'][0]>0),
            'Fresh decision rule differs')
    if interval:
        import numpy as np

        values = np.array(flat)
        sizes, delta = values.sum(axis=1), values[:,2]-values[:,1]
        rng = np.random.default_rng(60424)
        reps = []
        for _ in range(100):
            picks = rng.integers(0,len(flat),size=(50,len(flat)))
            reps.extend(100*delta[picks].sum(axis=1)/sizes[picks].sum(axis=1))
        require(np.allclose(np.quantile(reps,[.025,.975]),r['group_bootstrap_95_pp'],rtol=0,atol=1e-12),
                'Fresh panel interval differs')
    return net


def main(args):
    for row in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest, name = row.split('  ', 1)
        require(sha(ROOT/name) == digest, 'File hash differs: '+name)
    completed(EVIDENCE)
    q = read(EVIDENCE/'protocol.json')
    frozen = read(EVIDENCE/'models_frozen.json')
    r = read(EVIDENCE/'results.json')
    ex = read(EVIDENCE/'exports.json')
    groups = read(EVIDENCE/'grouped_correctness.json')
    require(q['runner'] == sha(ROOT/'overlay/scripts/run_b_shared_full_task2.py') and
            q['plan'] == sha(ROOT/'overlay/docs/batches/B_shared_full_task2_plan_20261123_zh.md')
            and len(q['production']) == 91 and q['model_api_cap'] == 0, 'Production protocol mismatch')
    require(frozen['protocol'] == sha(EVIDENCE/'protocol.json') and ex['frozen'] == sha(EVIDENCE/'models_frozen.json'),
            'The freeze and export chain differs')
    require(q['rows'] == frozen['training_rows'] == r['training_rows'] == 8301, 'Training rows differ')
    previous = predictions(EVIDENCE/'previous_predictions.csv')
    new = predictions(EVIDENCE/'validation_predictions.csv')
    require(sha(EVIDENCE/'previous_predictions.csv') == q['previous_csv'] == ex['old'] and
            [sha(EVIDENCE/'validation_predictions.csv')]*2 == ex['new'], 'CSV seals differ')
    cells = Counter(value for group in groups for value in group)
    require(set(cells) == {0,1,2,3} and sum(cells.values()) == ex['rows'] == r['rows'] == 940
            and len(groups) == r['validation_groups'], 'Grouped sample differs')
    require(cells[1]+cells[3] == r['previous'] == 689 and cells[2]+cells[3] == r['correct'] == 684
            and cells[1] == r['lost'] and cells[2] == r['gained'] and r['accuracy'] == 684/940,
            'Paired counts differ')
    require(sum(a != b for a,b in zip(previous,new,strict=True)) == r['lost']+r['gained'],
            'Changed prediction count differs')
    workers = read(EVIDENCE/'classifier_fits.json')
    physical = [json.loads(line)['candidate'] for line in (EVIDENCE/'physical.jsonl').read_text().splitlines()]
    require(sha(EVIDENCE/'physical.jsonl') == frozen['physical'] and
            physical == [v['candidate'] for v in workers['workers']] and
            len(physical) == workers['attempted_classifier_fits'] == workers['completed_classifier_fits'] ==
            frozen['completed_fits'] == r['physical_fits'] == 6 and workers['unknown_failed_attempts'] == 0,
            'Physical fits differ')
    receipt = read(EVIDENCE/'portable_receipt.json')
    require(receipt['predictions_sha256'] == ex['new'][0] and receipt['model_sha256'] == frozen['model']
            and receipt['rows'] == 940 and receipt['archive_replay']['network_forbidden']
            and not receipt['archive_replay']['training_performed'], 'Portable model evidence differs')
    h = read(EVIDENCE/'h2_training_readiness.json')
    require(h['script_sha256'] == sha(ROOT/'overlay/scripts/check_b_shared_h2_inputs.py') and
            h['plan_sha256'] == sha(ROOT/'overlay/docs/batches/B_h2_receiver_preflight_20261123_zh.md')
            and set(h['inputs']) == {'45101','45102'} and h['official_H2_score'] is None and
            not h['outer_features_read'] and not h['outer_labels_read'] and h['classifier_fits'] == 0,
            'H2 readiness record differs')
    for pack in h['inputs'].values():
        require(pack['rows'] == 2500 and len(pack['candidates']) == 9 and len(pack['splits']) == 3 and
                all(v['training_rows'] == 2000 and v['internal_rows'] == 500 for v in pack['splits'].values()),
                'H2 training registration differs')
    completed(SHADOW)
    study = read(SHADOW/'results.json')
    sp = read(SHADOW/'protocol.json')
    require(sp['runner'] == sha(ROOT/'overlay/scripts/study_b_borderline_stop.py') and
            sp['plan'] == sha(ROOT/'overlay/docs/batches/B_near_threshold_plan_20261123_zh.md') and
            sp['selection_review'] == sha(EVIDENCE/'selection.json') and
            sp['source_frozen'] == sha(EVIDENCE/'models_frozen.json'), 'Diagnostic protocol differs')
    for records, panel in zip(read(SHADOW/'grouped.json'),study['panels'],strict=True):
        n0, lost, gained, both = map(sum, zip(*records,strict=True))
        require(n0+lost+gained+both == panel['rows'] and lost+both == panel['anchor']
                and gained+both == panel['pair'] and gained-lost == panel['net']
                and len(records) == panel['groups'], 'Diagnostic paired panel differs')
    shadow_fits = [json.loads(line)['candidate'] for line in (SHADOW/'physical.jsonl').read_text().splitlines()]
    net = sum(p['net'] for p in study['panels'])
    count = sum(p['rows'] for p in study['panels'])
    require(study['combined_rows'] == count and study['combined_gain'] == net and
            study['meets_gate'] == (all(p['net']>0 for p in study['panels']) and net/count >= sp['gain_gate'])
            and shadow_fits == [sp['anchor'], sp['partner']] and study['physical_fits'] == len(shadow_fits) == 2,
            'Diagnostic result or physical count differs')
    if args.course_archive:
        require(sha(args.course_archive) == q['released_source'], 'Unexpected course archive')
        with zipfile.ZipFile(args.course_archive) as archive, archive.open('Released_Task2-Headlines_CTR/val.csv') as f:
            labels = [int(v['label_pairwise']) for v in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))]
        matrix = Counter((truth, pred) for truth,pred in zip(labels,new,strict=True))
        paired = Counter(int(a==t)+2*int(b==t) for a,b,t in zip(previous,new,labels,strict=True))
        require(paired == cells and [[matrix[i,j] for j in (0,1)] for i in (0,1)] == r['confusion'],
                'Independent grading with course validation labels differs')
    if args.interval:
        import numpy as np

        rng = np.random.default_rng(60423)
        sizes = np.array(list(map(len,groups)))
        gains = np.array([v.count(2)-v.count(1) for v in groups])
        values = []
        for _ in range(100):
            indices = rng.integers(0,len(groups),size=(50,len(groups)))
            values.extend(100*gains[indices].sum(axis=1)/sizes[indices].sum(axis=1))
        require(np.allclose(np.quantile(values,[.025,.975]),r['paired_group_bootstrap_95_pp'],atol=1e-12,rtol=0),
                'Group bootstrap result differs')
    calibrated = check_calibration()
    matched_e5 = check_e5()
    fresh = check_fresh(args.interval)
    print(f"Task2 {r['correct']}/{r['rows']}, previous {r['previous']}, +{r['gained']}/-{r['lost']}; "
          f"two training panels net {net:+d}/{count}; calibration net {calibrated:+d} versus stronger control per panel; "
          f"matched E5 vote net {matched_e5:+d}; fresh E5 minus L12 {fresh:+d}; "
          "H2 train-only 2 x 3 splits; 11 base fits, 2 supervised calibration fits, 0 model API.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-archive', type=Path)
    parser.add_argument('--interval', action='store_true', help='Repeat the grouped bootstrap using NumPy')
    main(parser.parse_args())
