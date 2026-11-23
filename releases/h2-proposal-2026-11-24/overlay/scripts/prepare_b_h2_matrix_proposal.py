"""Prepare a source-bound H2 proposal and validate both real local entries up to the fit boundary."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from superagent import ab_run, ab_vision
from superagent.cli import build_parser
from tools.ab_registered_run import run as registered_run


ROOT=Path(__file__).resolve().parents[1]
BASE='449b766eb4a29af2f4020092ab6b3eab425d25be'
PLAN=ROOT/'docs/batches/B_h2_matrix_proposal_plan_20261124_zh.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)


def require(ok,message):
    if not ok:
        raise ValueError(message)


class BeforeTraining(BaseException):
    pass


def prepare(args):
    # The same offline/fit guard used for production frozen-model checks, before any path can run.
    from scripts import a38_predict_guard

    args.output.mkdir(parents=True,exist_ok=False)
    source=ab_run.sources()
    receipt=ROOT/'research/a137_numeric_acceptance/run/run_manifest.json'
    require(source==read(receipt)['source_files'] and len(source)==93,'Expected the registered A137 production source')
    os.environ[ab_vision.WEIGHT_ENV]=str(args.vision)
    require(ab_vision.weights_file()==args.vision,'Frozen MobileNet asset does not match')
    reg=read(ROOT/'research/a109_H2_canonical_inputs/training_results.json')
    previous=read(ROOT/'research/bh_03_training_inputs/results.json')
    commands=read(ROOT/'research/a115_H_entry/results.json')['command_templates']
    funding=read(ROOT/'research/a130_a132_H1_summary/current_ledger_owner.json')
    h1=read(ROOT/'research/a128_H1_snapshot/experiment_snapshot.json')['policy']
    require(h1['arms']==['L0','R0'] and h1['Fstar']=='L0','Revisit the source of the proposed comparison')
    write(args.output/'protocol.json',dict(source=BASE,files=source,plan=ab_run.sha(PLAN),runner=ab_run.sha(Path(__file__)),
          A_source_manifest=ab_run.sha(receipt),A_training_registry=ab_run.sha(ROOT/'research/a109_H2_canonical_inputs/training_results.json'),
          B_training_registry=ab_run.sha(ROOT/'research/bh_03_training_inputs/results.json'),
          model_api_cap=0,classifier_fit_cap=0,outer_data_allowed=False))
    entries,proof={},{}
    for seed in ('45101','45102'):
        meta=reg[seed]
        require(meta==previous[seed],'A and B training registries differ')
        src=args.inputs/seed
        archive,inner=src/'train.zip',src/'inner_splits.json'
        require(ab_run.sha(archive)==meta['train_archive_sha256'] and ab_run.sha(inner)==meta['inner_manifest_sha256'],
                'Training input or partition manifest differs')
        common=['--scheduler','ab-fixed','--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11',
                '--seed','707','--time-limit','1200','--trial-timeout','300']
        local=['--planner','local',*common]
        comparison=['--planner','raw-history',*common,'--model','DeepSeek-V4.1-Flash','--api-key-file','LOCAL_KEY_PATH',
                    '--budget-ledger','EXISTING_SHARED_LEDGER_PATH','--task-budget-id','h2_transfer','--llm-budget-rmb','.30',
                    '--history-policy','incremental-research','--proposal-contract','calibrated-v2']
        old=commands['H2_'+seed]
        for arm,flags in (('L0',local),('R0',comparison)):
            prior=old[arm][old[arm].index('--planner'):]
            require(flags==['h2_transfer' if x=='H2_transfer' else x for x in prior], 'Review historical flag changes')
        # The real paid handler requires an exclusively owned nonempty ledger. Parse the command only.
        parsed=build_parser().parse_args(['run',str(archive),*comparison])
        ab_run.validate_options(parsed)
        require(parsed.task_budget_id=='h2_transfer' and not parsed.numeric_kernel and not parsed.confidence_cascade
                and parsed.fit_limit==11 and parsed.scheduler=='ab-fixed','Unexpected paid path defaults')
        entry=read(ROOT/f'research/a115_H_entry/H2_{seed}_L0_manifest.json')
        run_folder=args.output/seed
        seen=[]
        def boundary(directory,caps,metric,*unused):
            require(Path(directory)==run_folder and caps==entry['capabilities'] and metric.name=='accuracy',
                    'H2 catalogue or metric changed')
            manifest=read(run_folder/'run_manifest.json')
            require(manifest['source_files']==source and manifest['task_contract']==entry['task_contract']
                    and manifest['task_archive_sha256']==meta['train_archive_sha256']
                    and manifest['planner_mode']=='local' and manifest['fit_limit']==11,'CLI manifest mismatch')
            for name,registered in meta['splits'].items():
                ss=read(run_folder/(name+'_split.json'))
                require(all(ss[k]==registered[k] for k in ('seed','row_ids_sha256','label_sha256','evaluation_payload_sha256'))
                        and ss['train_rows']==registered['train_rows'] and ss['search_rows']==registered['internal_eval_rows']
                        and ab_run.sha(run_folder/(name+'.joblib'))==registered['evaluation_payload_sha256'],
                        'Official worker bytes or registered IDs differ: '+seed+'/'+name)
                seen.append(name)
            raise BeforeTraining()
        with patch.object(ab_run,'Executor',boundary):
            try:
                registered_run(inner,meta['inner_manifest_sha256'],['run',str(archive),'--run-dir',str(run_folder),*local])
            except BeforeTraining:
                pass
            else:
                raise ValueError('The real entry did not reach its training boundary')
        require(set(seen)==set(meta['splits']) and not list(run_folder.glob('trial_*.joblib')),'Unexpected training outputs')
        entries[seed]=dict(training_sha256=meta['train_archive_sha256'],registered_splits_sha256=meta['inner_manifest_sha256'],
                          menu=list(entry['capabilities']),worker_sha256={k:v['evaluation_payload_sha256'] for k,v in meta['splits'].items()},
                          flags=dict(L0=local,R0=comparison))
        proof[seed]=dict(actual_train_rows=meta['train_rows'],candidate_count=len(entry['capabilities']),verified_workers=seen,
                         byte_checks_passed=True,verified_at='official registered CLI before Executor')
    matrix=dict(status='B_proposal_for_A_registration',source=BASE,source_files=len(source),Fstar_proposed='L0',
                arms=['L0','R0'],seeds=list(entries),entries=entries,maximum_fits_per_path=11,
                maximum_flash_per_paid_path=3,maximum_fits_if_adopted=44,maximum_flash_if_adopted=6,
                paid_threshold_rmb_per_path=.30,task_scope='h2_transfer',pool_all_models_before_outer=True,
                historical_ledger_owner=funding['owner'],historical_requests=funding['project_summary']['requests'],
                historical_unknown=funding['project_summary']['unresolved_requests'],
                formal_matrix_adopted=False,h2_performance_observed=False)
    write(args.output/'proposed_matrix.json',matrix)
    write(args.output/'results.json',dict(entries=proof,paid_configuration='parsed_without_ledger_or_network',
          executed_training_paths=0,supervised_fits=0,model_api_calls=0,ledger_opened=False,
          h2_outer_features_opened=False,h2_outer_labels_opened=False))
    write(args.output/'completion.json',{k:ab_run.sha(args.output/(k+'.json')) for k in ('protocol','proposed_matrix','results')})
    print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','vision','output'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    prepare(parser.parse_args())
