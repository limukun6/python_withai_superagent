"""Exercise order, duplication, batching and declared pair semantics on real frozen predictors."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from superagent.ab_run import sha as file_hash
from superagent.data import load_prediction_features
from superagent.models import predict_with_model, probability_with_model
from superagent.splits import take


TOLERANCE=2e-5


def audit_one(config):
    base,case,features,out=(Path(config[k]) for k in ('base','case','features','out'))
    import superagent
    if Path(superagent.__file__).resolve()!=base/'superagent/__init__.py':
        raise ValueError('Wrong receiving checkout')
    model_dir=case/'relocated'
    task=json.loads((model_dir/'task_spec.json').read_text())
    selection=json.loads((model_dir/'final_selection.json').read_text())
    path=model_dir/selection['model_file']
    if file_hash(path)!=config['model_sha256'] or file_hash(features)!=config['features_sha256'] or selection['model_sha256']!=config['model_sha256']:
        raise ValueError('Input or frozen model changed')
    bundle=joblib.load(path)
    if bundle['candidate']!=selection['family']:
        raise ValueError('Model bundle does not implement the frozen candidate')
    kind,model=bundle['kind'],bundle['model']
    inputs=load_prediction_features(features,kind,task)
    original_csv=Path(config['accepted_export'])
    if file_hash(original_csv)!=config['csv_sha256']:
        raise ValueError('Native prior delivery is not the accepted export')
    with original_csv.open(encoding='utf-8',newline='') as f:
        reference=np.asarray([r['prediction'] for r in csv.DictReader(f)])
    if len(reference)!=config['rows'] or len(inputs)!=len(reference):
        raise ValueError('Accepted row count or declared unlabeled query changed')
    rng=np.random.default_rng(112208)
    ids=np.sort(rng.choice(len(inputs),size=min(97,len(inputs)),replace=False))
    observed=take(inputs,ids)
    primary=np.asarray(predict_with_model(kind,model,observed)).reshape(-1)
    manifest=json.loads((case/'run/run_manifest.json').read_text())
    cap=manifest['capabilities'][selection['family']]
    probability=cap['supports_probabilities']
    if probability:
        classes,probs=probability_with_model(kind,model,observed)
        classes=np.asarray(classes)
        probs=np.asarray(probs)
        if (set(classes.tolist())!=set(task['metric_spec']['labels']) or probs.shape!=(len(ids),len(classes))
                or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(axis=1),1.,atol=TOLERANCE,rtol=0)):
            raise ValueError('Invalid base probabilities')
    mode=task.get('input_schema',{}).get('pair_semantics')
    perturb=np.concatenate([np.arange(len(ids)),np.linspace(0,len(ids)-1,9,dtype=int)])
    rng.shuffle(perturb)
    measured={}
    sizes=(1,47,48,49)
    offset=0
    parts=[]
    while offset<len(perturb):
        size=sizes[len(parts)%len(sizes)]
        parts.append(perturb[offset:offset+size])
        offset+=size
    all_classes_match=True
    for name,blocks in [('permuted_once',[perturb]),('permuted_in_chunks',parts)]:
        labels=[]
        maxima=[]
        for block in blocks:
            x=take(observed,block)
            labels.extend(np.asarray(predict_with_model(kind,model,x)).reshape(-1).tolist())
            if probability:
                other,p=probability_with_model(kind,model,x)
                all_classes_match=all_classes_match and np.array_equal(other,classes)
                difference=np.abs(np.asarray(p)-probs[block])
                if not np.isfinite(difference).all():
                    raise ValueError('Nonfinite reordered probability')
                maxima.append(float(np.max(difference)))
        measured[name]=dict(rows=len(labels),sizes=[len(b) for b in blocks],label_mismatch=int(np.sum(np.asarray(labels)!=primary[perturb])),
                            max_probability_difference=max(maxima) if maxima else None)
    swap=None
    if mode in ('preference','symmetric'):
        swapped=[(b,a) for a,b in observed]
        reverse=np.asarray(predict_with_model(kind,model,swapped)).reshape(-1)
        other,p=probability_with_model(kind,model,swapped)
        if not np.array_equal(other,classes):
            raise ValueError('Swap changed class identities')
        if mode=='symmetric':
            diff=np.abs(probs-p)
            mismatch=int(np.sum(primary!=reverse))
            ties=0
        else:
            pos=np.flatnonzero(classes==1)[0]
            diff=np.abs(probs[:,pos]+p[:,pos]-1.)
            tied=(np.abs(probs[:,pos]-.5)<=TOLERANCE)|(np.abs(p[:,pos]-.5)<=TOLERANCE)
            ties=int(tied.sum())
            mismatch=int(np.sum((primary==reverse)&~tied))
        swap=dict(semantics=mode,rows=len(reverse),tolerance_ties=ties,label_mismatch=mismatch,max_probability_difference=float(np.max(diff)))
    baseline_mismatch=int(np.sum(primary.astype(str)!=reference[ids]))
    passed=(not baseline_mismatch and all_classes_match and all(not r['label_mismatch'] and (r['max_probability_difference'] or 0)<=TOLERANCE
            for r in measured.values()) and (not swap or (not swap['label_mismatch'] and swap['max_probability_difference']<=TOLERANCE)))
    report=dict(status='passed' if passed else 'observed_difference',candidate=selection['family'],kind=kind,registered_classes=len(task['metric_spec']['labels']),
        source_model_sha256=config['model_sha256'],chosen_rows=len(ids),source_rows=len(reference),source_csv_label_mismatches=baseline_mismatch,
        class_order_matches=bool(all_classes_match),probabilities_probed=probability,perturbations=measured,pair_swap=swap)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def study(args):
    from scripts.verify_b_a31_bundle import cases as new_cases, call
    work,out=args.workspace.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    record_new=json.loads(args.new_record.read_text())
    record_old=json.loads(args.old_record.read_text())
    new,old=out/'new_repo',out/'old_repo'
    for root,clone in ((args.new_receiver,new),(args.old_receiver,old)):
        call(['git','clone','--quiet','--no-checkout','--no-local','--no-tags',root.resolve(),clone],out)
    inputs=[]
    for branch,name,case,features,_,_,_,_ in new_cases(work):
        inputs.append((new,name,record_new['branch_tips'][branch],record_new['predictions'][name],case,features,case/'export_0/rows.csv'))
    flower=work/'ab_coverage_scheduler/runs/bs03_images_01/color_coverage'
    bean=work/'ab_numeric_kernel/runs/bn01_bean_01/full'
    phrase=work/'ab_time_resource/runs/time_coverage_01/mrpc_combined'
    extra={'flower':(flower,flower/'features.zip',flower/'pred_0/rows.csv'),
           'bean':(bean,work/'ab_portable_inputs/runs/bp02_environment_1_01/bean/delivery/features.zip',bean/'export_0/rows.csv'),
           'phrase':(phrase,work/'ab_coverage_scheduler/runs/bs01_two_splits_01/45102/features.zip',phrase/'export_0/rows.csv')}
    for entry in record_old['branches'].values():
        if entry['case'] in extra:
            inputs.append((old,entry['case'],entry['commit'],entry,*extra[entry['case']]))
    save('protocol',dict(batch='B-RECEIVER-ROWS-01',driver_sha256=file_hash(Path(__file__)),
        plan_sha256=file_hash(Path(__file__).resolve().parents[1]/'docs/batches/B_RECEIVER_ROWS_01_plan.md'),
        accepted_receipts=[file_hash(args.new_record),file_hash(args.old_record)],test_cases=[c[1] for c in inputs],seed=112208,
        source_rows_max=97,additional_duplicates=9,chunk_sizes=[1,47,48,49],probability_tolerance=TOLERANCE,fit_limit=0,model_api_limit=0))
    try:
        results={}
        for clone,name,commit,entry,case,features,export in inputs:
            call(['git','switch','-q','--detach',commit],clone)
            config=dict(base=str(clone),case=str(case),features=str(features),accepted_export=str(export),out=str(out/(name+'.json')),
                        model_sha256=entry['model_sha256'],features_sha256=entry['features_sha256'],csv_sha256=entry['csv_sha256'],rows=entry['rows'])
            config_file=out/(name+'_input.json')
            config_file.write_text(json.dumps(config))
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(clone/'scripts/ab_acceptance_guard',clone))),
                AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(out/'physical_fits.jsonl'),SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(work/'pretrained_weights/all-MiniLM-L6-v2'),
                SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(work/'ab_joint_a11/data/b_natural_photo/mobilenet_v3_small-047dcff4.pth'),
                OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            env.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
            with (out/(name+'.log')).open('x') as log:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--config',str(config_file)],
                                    cwd=clone,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if proc.returncode:
                raise ValueError('Frozen observation worker failed: '+name)
            results[name]=json.loads((out/(name+'.json')).read_text())
            save('checkpoint',results)
        if len(results)!=9 or (out/'physical_fits.jsonl').exists():
            raise ValueError('Unexpected count or training')
        save('results',results)
        save('completion',dict(status='completed',cases=len(results),passed=sum(r['status']=='passed' for r in results.values()),
             new_fits=0,new_model_api_requests=0,new_course_rmb=0,true_labels_read=False))
    except Exception as error:
        save('completion',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    parent=modes.add_parser('study')
    for name in ('workspace','output','new-receiver','old-receiver','new-record','old-record'):
        parent.add_argument('--'+name,type=Path,required=True)
    modes.add_parser('worker').add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='study':
        study(args)
    else:
        audit_one(json.loads(args.config.read_text()))
