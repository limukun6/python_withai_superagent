"""Replay observed B decisions and verify prediction compatibility on the exact A135 project sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from superagent import ab_adaptive_opening, ab_run, ab_vision
from superagent.ab_catalog import untried_candidates
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import validate_action
from tools.ab_registered_rows import registered_partitioner

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_a135_receiver_plan_20261124_zh.md'
CURRENT='6b6ace830ad9eefe92cf6236cb88c7ff706a116b'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False)


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def run(a):
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Replay is offline')
    sys.addaudithook(offline)
    a.out.mkdir(parents=True,exist_ok=False)
    old=a.original/'agent'
    current=ab_run.sources()
    reference=ROOT/'research/a133_a135_summary/a135/run/run_manifest.json'
    protocol=read(a.original/'protocol.json')
    selection=read(a.original/'models_frozen.json')
    old_manifest=read(old/'run_manifest.json')
    require(len(current)==92 and current==read(reference)['source_files'] and
            old_manifest['source_files']==protocol['production'] and len(protocol['production'])==91,
            'At least one version differs from its published production source')
    require(ab_run.sha(a.original/'anonymous.zip')==protocol['training'] and ab_run.sha(a.features)==protocol['feature_packet']
            and ab_run.sha(old/'selected_model.joblib')==selection['model'],'Training-only input, features or model differs')
    save(a.out/'protocol.json',dict(plan=ab_run.sha(PLAN),runner=ab_run.sha(Path(__file__)),shared_source=CURRENT,
         source_files=current,source_manifest=ab_run.sha(reference),original_protocol=ab_run.sha(a.original/'protocol.json'),
         original_source=protocol['shared_commit'],original_model=selection['model'],options=protocol['options'],
         fit_cap=0,model_api_cap=0))
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L6-v2'),
                      SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),
                      SUPERAGENT_E5_SMALL_WEIGHTS=str(a.weights/'e5-small-v2'),TOKENIZERS_PARALLELISM='false',
                      TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
    args=build_parser().parse_args(['run',str(a.original/'anonymous.zip'),*protocol['options']])
    ab_run.validate_options(args)
    require(not args.confidence_cascade and not args.resolve_single_conflict and args.cover_untried_encoder
            and args.planner=='local','Shared flags differ')
    os.environ[PREFERENCE_CATALOG_ENV]=args.preference_catalog
    task=load_development_task(a.original/'anonymous.zip')
    spec=TaskSpec.from_task(task).record()
    caps=ab_run.registered_capabilities(spec,len(task.y_train),args.catalog_scope)
    require(spec==old_manifest['task_contract'] and caps==old_manifest['capabilities'],'Task2 contract or menu differs')
    split_dir=a.out/'partitions'
    split_dir.mkdir()
    trusted,allowed=ab_run.freeze_partitions(task,split_dir,args.seed,MetricSpec.for_task(task),preference_vote=True)
    for split in allowed:
        require(read(split_dir/(split+'_split.json'))==read(old/(split+'_split.json')),'Trusted split differs')
    actions=read(old/'actions.json')
    require(actions==selection['actions'],'Prior actions no longer match the frozen observations')
    for i,expected in enumerate(actions,1):
        state=read(old/f'state_{i}.json')
        require(state['task']==spec and state['available_candidates']==untried_candidates(caps,state['history']),
                'A historical state did not match the current menu')
        action=ab_adaptive_opening.action(state,caps,trusted,cover_untried_encoder=args.cover_untried_encoder)
        require(action==expected,'Default action differs at state '+str(i))
        validate_action(state,action,caps,allowed)
    review=ab_adaptive_opening.review(state,trusted)
    require(actions[-1]['action']=='stop' and review==read(old/'selection.json')['confirmation']['b_scheduler_review']
            and review['members']==[selection['selection']], 'The final model choice differs')
    # The receiving process rejects network and fitting before importing the production CLI.
    child="""import os, pathlib, sys
def offline(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'): raise PermissionError('Offline prediction')
sys.addaudithook(offline)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
def reject(*a, **kw): raise RuntimeError('Supervised fitting forbidden')
for cls in (LogisticRegression,SVC,LinearSVC,ExtraTreesClassifier): cls.fit=reject
sys.path.insert(0,sys.argv.pop(1))
from superagent.cli import main
main()
"""
    output=a.out/'receiver/rows.csv'
    childenv=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    with (a.out/'predict.log').open('x') as f:
        subprocess.run([sys.executable,'-I','-c',child,str(ROOT),'predict',str(a.features),'--run-dir',str(old),
                        '--output',str(output)],env=childenv,cwd=a.out,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
    expected=read(a.original/'exports.json')['new'][0]
    proof=read(output.parent/'prediction_manifest.json')
    require(ab_run.sha(output)==expected and proof['prediction_count']==940 and not proof['test_labels_read'],
            'Receiver did not reproduce the unchanged model predictions')
    os.environ[ab_vision.WEIGHT_ENV]=str(a.vision)
    os.environ[PREFERENCE_CATALOG_ENV]='standard'
    require(ab_vision.weights_file()==a.vision,'Registered vision asset differs')
    registered=read(ROOT/'research/a109_H2_canonical_inputs/training_results.json')
    h2={}
    for seed in ('45101','45102'):
        folder=a.h2/seed
        row=registered[seed]
        require(ab_run.sha(folder/'train.zip')==row['train_archive_sha256'],'H2 train archive changed')
        image_task=load_development_task(folder/'train.zip')
        contract=TaskSpec.from_task(image_task).record()
        menu=ab_run.registered_capabilities(contract,len(image_task.y_train))
        previous=read(ROOT/f'research/a115_H_entry/H2_{seed}_L0_manifest.json')
        require(image_task.X_val is None and image_task.y_val is None and contract==previous['task_contract']
                and menu==previous['capabilities'] and ab_vision.NAME in menu,'H2 training contract or menu changed')
        split=registered_partitioner(image_task,folder/'inner_splits.json',row['inner_manifest_sha256'])
        checks={}
        for name,item in row['splits'].items():
            fit,held,stats=split(image_task,item['seed'])
            require(len(fit)==item['train_rows'] and len(held)==item['internal_eval_rows']
                    and stats['class_domain_preserved'],'H2 partition changed')
            checks[name]=dict(seed=item['seed'],fit_rows=len(fit),eval_rows=len(held))
        h2[seed]=dict(training_rows=len(image_task.y_train),candidates=len(menu),splits=checks)
    result=dict(old_source=protocol['shared_commit'],current_source=CURRENT,source_files=len(current),
         changed_production_files=sorted(k for k in current if current[k]!=protocol['production'].get(k)),
         historical_model=selection['model'],current_training_performed=False,source_model=selection['selection'],
         matching_actions=len(actions),action_types=[v['action'] for v in actions],matching_task2_menu=len(caps),
         matching_partitions=len(allowed),fresh_prediction_reader=True,matching_predictions=940,
         prediction_sha256=expected,confidence_cascade=False,course_validation_labels_read=False,
         course_test_read=False,h2_train=h2,h2_outer_access=False,classifier_fits=0,model_api_calls=0)
    save(a.out/'results.json',result)
    save(a.out/'completion.json',{k:ab_run.sha(a.out/(k+'.json')) for k in ('protocol','results')})
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','features','weights','h2','vision','out'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
