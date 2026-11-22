"""Check already-frozen Released exports, C1 configurations, and H1 contract-only eligibility."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

from scripts import run_b1_ab_acceptance as rec, run_b_c1_matrix as matrix
from scripts.check_b_a25_frozen_delivery import predict
from superagent import ab_run


def check(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),SUPERAGENT_OFFLINE='1')
    source=rec.digest(ab_run.sources())
    plan=rec.ROOT/'docs/batches/BA27_02_delivery_readiness_plan.md'
    rec.write(out/'protocol.json',dict(batch='BA27-02',source=source,plan_sha256=ab_run.sha(plan),
        runner_sha256=ab_run.sha(Path(__file__)),frozen_predictions=4,formal_arms_to_prepare=12,fit_cap=0,model_api_cap=0))
    contract=rec.read(rec.ROOT/'research/a12_a16_day/a13_holdout_packages/registration.json')['H1']
    original=rec.read(rec.ROOT/'research/a26_a27_methods_and_flash/H_capability_update.json')['H1']
    if contract['task_archive_sha256']!=original['task_archive_sha256']:
        raise ValueError('H1 registration differs')
    spec,n=contract['task_spec'],contract['rows']
    full=ab_run.registered_capabilities(spec,n,'full')
    with patch.dict(os.environ,{'SUPERAGENT_MINILM_WEIGHTS':''}):
        light=ab_run.registered_capabilities(spec,n,'full')
    if (ab_run.registered_capabilities(spec,n,'baseline') or list(full)!=[*light,'unordered_minilm_lexical'] or
        list(light)!=[*original['capabilities'],'unordered_word','unordered_char','unordered_word_char'] or
        any(full[k]!=v for k,v in original['capabilities'].items()) or
        any(v['swap_label_rule']!='preserves' or v['classifier_fits']!=1 or v['refit_classifier_fits']!=1 for v in full.values())):
        raise ValueError('Registered H1 methods, fit accounting or weight availability differs')
    formal=matrix.freeze_inputs(args.portable.resolve())
    old=rec.read(rec.ROOT/'research/ba25_joint/b_c1_protocol.json')
    if formal['source']!=source or any(formal[k]!=v for k,v in old.items() if k!='source'):
        raise ValueError('C1 data, menus, versioned runner, budget or configurations changed beyond production sources')
    rec.write(out/'c1_protocol.json',formal)
    ref=rec.ROOT/'research/ba19_released_delivery'
    initial=rec.read(ref/'preflight.json')
    selected=rec.read(ref/'results.json')['paths']
    frozen=rec.read(ref/'all_models_frozen.json')
    exports={}
    for key,model in frozen.items():
        task=key.split('_')[0]
        folder=args.released.resolve()
        packet=folder/task/'features.zip'
        if ab_run.sha(packet)!=initial[task]['features_sha256']:
            raise ValueError('Frozen query changed')
        exports[key]=predict(folder/key/'run',packet,out/key,model,selected[key]['csv_sha256'],initial[task]['held_rows'])
    if rec.trace(out) or rec.digest(ab_run.sources())!=source or any(v['status']!='identical' for v in exports.values()):
        raise ValueError('Source, frozen exports or training count changed')
    rec.write(out/'results.json',dict(released=exports,H1_contract_only=dict(source_rows=n,
        signed_archive=contract['task_archive_sha256'],methods_with_weights=list(full),methods_without_weights=list(light)),
        H2_registration='Rice and EuroSAT retained; mutual freeze pending',formal_task_ids=matrix.TASK_BUDGET_IDS,
        formal_arms_prepared=sum(len(v['order']) for v in formal['tasks'].values())))
    rec.write(out/'completion.json',dict(status='passed',independent_cli_exports=4,formal_arms_started=0,model_fits=0,
        model_api_requests=0,added_course_rmb=0,outer_scores=0,H_samples_opened=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('released','portable','weights','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    check(parser.parse_args())
