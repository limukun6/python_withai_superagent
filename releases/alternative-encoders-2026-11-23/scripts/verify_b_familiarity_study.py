"""Verify published aggregates, checksums and decisions without models, data or fitting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'research/b_learned_familiarity_1123'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    names = ('learned_familiarity_01', 'native_familiarity_01', 'e5_familiarity_01', 'cross_seed_portfolio_01')
    plans = ('learned_familiarity', 'native_familiarity', 'e5_familiarity', 'cross_seed_portfolio')
    runners = ('run_b_learned_familiarity', 'run_b_native_familiarity', 'run_b_e5_familiarity', 'check_b_cross_seed_portfolio')
    results = []
    for name, plan, runner in zip(names, plans, runners):
        folder = EVIDENCE / name
        complete = read(folder / 'completion.json')
        protocol = read(folder / 'protocol.json')
        require(complete['status'] == 'complete' and complete.get('results', complete.get('result')) == sha(folder / 'results.json'),
                'Wrong result fingerprint in ' + name)
        require(protocol['plan'] == sha(ROOT / 'docs/batches' / ('B_' + plan + '_1123_zh.md'))
                and protocol.get('runner', protocol.get('code')) == sha(ROOT / 'scripts' / (runner + '.py')),
                'Study plan or runner differs in ' + name)
        results.append(read(folder / 'results.json'))
    feature, native, e5, replay = results
    require(feature['development_reserve']['familiarity_correct'] == 1232 and feature['actual_new_supervised_fits'] == 5
            and feature['qualified_for_full_refit'], 'Wrong ablation or gate')
    selection = read(EVIDENCE / names[1] / 'selection_frozen.json')
    relocation = read(EVIDENCE / names[1] / 'relocation.json')
    signature = read(EVIDENCE / names[1] / 'exports_frozen.json')
    require(sha(EVIDENCE / names[1] / 'selection_frozen.json') == signature['selection'] and
            native['selected_new'] and native['members'] == selection['decision']['members'] and
            native['correct'] == {'new':1224, 'old':1208, 'deep':1224, 'feature_only':1232} and
            selection['new_fits'] == native['new_fits'] == 8 and relocation['rows'] == native['rows'] == 1693 and
            relocation['sha256'] == signature['methods']['new'] and relocation['model_sha256'] == selection['model'],
            'Wrong independent native selection or export')
    require(e5['passed'] and e5['physical_new_fits'] == 2 and e5['group_disjoint_pool']['rows'] == 2382
            and e5['development_reserve']['correct']['new_vote'] == 1237 and e5['development_reserve']['correct']['original_vote'] == 1234,
            'Wrong E5 comparison')
    require(not replay['qualifies_for_native_run'] and replay['actual_new_fits'] == 0 and
            replay['within_fold']['total_rows'] == replay['cross_seed']['total_rows'] == 2069, 'Wrong scheduler replay')
    require(feature['model_api_calls'] == native['model_api_requests'] == e5['paid_model_calls'] == replay['model_api_requests'] == 0,
            'Unexpected model API usage')
    print('Verified 4 completed experiments, 15 new physical fits, 0 model API calls, 1693-row autonomous export and 2382-row paired comparison.')


if __name__ == '__main__':
    main()
