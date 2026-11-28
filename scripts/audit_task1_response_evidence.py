"""Check Task1 development receipts and emit an aggregate without accessing any dataset."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(research):
    names = ('exposure', 'aligned_fields', 'personal_offsets', 'response_classes')
    directories = {n: research / f'task1_{n}_01' for n in names}
    first = directories['exposure']
    control = read(first / 'results.json')
    spec = read(first / 'protocol.json')
    held_hashes = [s['held_sha'] for s in spec['splits']]
    totals = {}
    for name, path in directories.items():
        protocol, end, rows = [read(path / f'{key}.json') for key in ('protocol', 'completion', 'results')]
        assert end['status'] == 'complete' and end['api_calls'] == 0 and end['course_yuan'] == 0
        assert end.get('designated_validation_opened', end.get('validation_opened')) is False
        assert end.get('formal_test_opened', end.get('test_opened')) is False
        assert protocol['seed'] == spec['seed']
        assert protocol.get('members', protocol.get('train_members')) == spec['train_members']
        if name == 'aligned_fields':
            data = (first / 'results.json').read_bytes().replace(b'\r\n', b'\n')
            assert protocol['control_sha'] in {hashlib.sha256(version).hexdigest()
                                               for version in (data, data.replace(b'\n', b'\r\n'))}
        if name in ('personal_offsets', 'response_classes'):
            assert protocol.get('held_sha', protocol.get('folds')) == held_hashes
        evidence = {}
        for i, r in enumerate(rows):
            assert i == r['fold']
            base = r['methods']['baseline']
            if i < len(control):
                assert base['prediction_sha'] == control[i]['methods']['baseline']['prediction_sha']
                assert base['correct'] == control[i]['methods']['baseline']['correct'] and r['held'] == control[i]['held']
            for method, s in r['methods'].items():
                net = s['fixed'] - s['broken']
                assert net == s['correct'] - base['correct']
                assert abs(s['accuracy'] - s['correct'] / r['held']) < 1e-10
                agg = evidence.setdefault(method, dict(rows=0, correct=0, fixed=0, broken=0, net_by_fold=[]))
                agg['rows'] += r['held']
                for key in ('correct', 'fixed', 'broken'):
                    agg[key] += s[key]
                agg['net_by_fold'].append(net)
        for v in evidence.values():
            v['delta_pooled_pp'] = round(100 * (v['fixed'] - v['broken']) / v['rows'], 5)
        totals[name] = dict(methods=evidence, resources=end)
    contexts = {}
    for r in control[:3]:
        for key, values in r['baseline_by_context'].items():
            bucket = contexts.setdefault(key, dict(rows=0, correct=0))
            for field in bucket:
                bucket[field] += values[field]
    return dict(batches=totals, held_hashes=held_hashes, common_first_three_rows=sum(r['held'] for r in control[:3]),
                low_confidence_rows=sum(r['uncertain_rows'] for r in control[:3]),
                low_confidence_correct=sum(r['methods']['baseline']['low_confidence_correct'] for r in control[:3]),
                baseline_by_context=contexts, prior_designated_score=read(research / 'task1_context_agent_01/completion.json')['scores']['agent'],
                wall_seconds_sum=round(sum(v['resources']['seconds'] for v in totals.values()), 3))


def audit_gate(research, earlier):
    path = research / 'task1_uncertain_gate_01'
    spec, end, rows = [read(path / f'{key}.json') for key in ('protocol', 'completion', 'results')]
    assert end['status'] == 'complete' and end['classifier_fits'] == 4 and end['api_calls'] == 0
    assert not end['designated_validation_opened'] and not end['formal_test_opened'] and end['course_yuan'] == 0
    assert spec['held_sha'] == earlier['held_hashes'] and [r['fold'] for r in rows] == [3, 4]
    first = read(research / 'task1_exposure_01/results.json')[:3]
    explore_net = sum(r['methods']['blend']['low_confidence_correct'] -
                      r['methods']['baseline']['low_confidence_correct'] for r in first)
    confirm_net = 0
    for r in rows:
        gate, base = (r['methods'][name] for name in ('gate', 'baseline'))
        net = gate['fixed'] - gate['broken']
        assert net == gate['correct'] - base['correct'] == gate['low_confidence_correct'] - base['low_confidence_correct']
        confirm_net += net
    assert explore_net == end['prior_three_net'] and confirm_net == end['confirm_net']
    all_rows = first + rows
    total = sum(r['held'] for r in all_rows)
    assert abs(100 * (explore_net + confirm_net) / total - end['five_fold_delta_pp']) < 1e-10
    return dict(design_net=explore_net, confirm_net=confirm_net, total_rows=total,
                base_correct=sum(r['methods']['baseline']['correct'] for r in all_rows),
                uncertain_rows=earlier['low_confidence_rows'] + sum(r['gate_rows'] for r in rows),
                uncertain_correct=sum(r['methods']['baseline']['low_confidence_correct'] for r in all_rows),
                five_batches_seconds=round(earlier['wall_seconds_sum'] + end['seconds'], 3),
                resources=end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, default=Path('research'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--with-gate', action='store_true', help='Also verify the subsequent two-fold confirmation.')
    args = parser.parse_args()
    verified = audit(args.research)
    if args.with_gate:
        verified['gate_confirmation'] = audit_gate(args.research, verified)
    result = json.dumps(verified, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as file:
            file.write(result)
    else:
        print(result)
