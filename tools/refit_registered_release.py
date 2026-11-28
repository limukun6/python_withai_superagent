"""Refit a solver-selected single model or fixed vote on additional labelled development rows."""
import argparse
import os
from pathlib import Path
import shutil
import zipfile

import joblib
import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, make_zip, read, require, save
from superagent.ab_run import sources
from superagent.autonomous_run import installed_methods
from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.external_assets import activate_resource_directory
from superagent.first_delivery import digest
from superagent.models import SoftVoteModel
from superagent.preference_analogy import WeightedProbabilityVote
from superagent.splits import take


def add_validation_packet(out, packages, case):
    tag={'a':'table','b':'titles'}[case]
    names=({'C_val.npy':'C_train.npy','y_val.npy':'y_train.npy'} if tag=='table' else {'val.csv':'train.csv'})
    with zipfile.ZipFile(out/case/'training.zip') as z:
        (doc,)=[n for n in z.namelist() if Path(n).name.lower()=='readme.md']
        entries=[(doc,z.read(doc))]
    opened=[]
    with zipfile.ZipFile(archive(packages,tag)) as z:
        for validation_name,training_name in names.items():
            (member,)=[n for n in z.namelist() if '__MACOSX' not in Path(n).parts and Path(n).name==validation_name]
            opened.append(member)
            entries.append(('input/'+training_name,z.read(member)))
    packet=out/'augmented'/case/'additional_development.zip'
    make_zip(packet,entries)
    return packet,opened


def vote_parts(model, count):
    if count==1:
        return [model],None
    require(type(model) in (SoftVoteModel,WeightedProbabilityVote) and len(model.models)==count,'Expected the recorded fixed probability vote')
    weights=getattr(model,'weights',None)
    return model.models, list(map(float,weights)) if weights is not None else None


def fixed_vote(original,kind,models,classes,weights):
    if len(models)==1:
        return models[0]
    require(np.array_equal(original.classes_,classes),'Frozen class order differs')
    return type(original)(kind,models,classes,tuple(weights) if weights is not None else None)


def fitted_filename(base,index,seed,member):
    if seed is not None:
        return f'seed_refit_{index}_{seed}.joblib'
    # The regular worker and the optional adviser use distinct final-fit filenames.
    for name in (f'refit_component_{index}.joblib','analogy_full.joblib'):
        ledger=(base/name).with_suffix('.fits.json')
        if ledger.exists() and read(ledger)['candidate']==member:
            return name
    raise ValueError('Missing elected component fit for '+member)


def refit(args,case):
    out=args.out
    base=(args.election or out/case)/'run/agent'
    target=out/'augmented'/case
    target.mkdir(parents=True,exist_ok=False)
    native=target/'native'
    native.mkdir()
    seal=read(args.election_seal or out/case/'sealed.json')
    require(source_identity(ROOT)==seal['solver'] and not seal['course_test_opened'] and seal['independent_reload'],
            'Use the verified ordinary solve')
    selected=read(base/'final_selection.json')
    require(selected['refit'] and digest(base/selected['model_file'])==seal['model'],'Frozen base model required')
    packet,opened=add_validation_packet(out,args.packages,case)
    original=load_training_task(out/case/'training.zip')
    extra=load_training_task(packet)
    baseline=joblib.load(base/'refit_data.joblib')
    require(extra.kind==original.kind==baseline['kind'] and
            np.array_equal(np.asarray(original.X_train,dtype=str),np.asarray(baseline['X_train'],dtype=str)) and
            np.array_equal(original.y_train,baseline['y_train']) and len(original.y_train)==selected['training_rows'],
            'Original training payload or kind differs')
    require(set(np.unique(extra.y_train))<=set(np.unique(original.y_train)) and len(extra.y_train)==seal['rows'],
            'Added rows or class domain differ')
    rows=baseline['X_train']
    full=rows+extra.X_train if isinstance(rows,list) else np.concatenate([rows,extra.X_train])
    labels=np.concatenate([baseline['y_train'],extra.y_train])
    require(len(full)==len(labels),'Input/label count mismatch')
    payload=dict(baseline,X_train=full,y_train=labels,X_search=take(full,[0]))
    classes=np.asarray(baseline['metric_spec']['labels'])
    registry=base/'candidate_registry.json'
    if registry.exists():
        shutil.copyfile(registry,native/registry.name)
    shutil.copyfile(base/'task_spec.json',native/'task_spec.json')
    reference=joblib.load(base/selected['model_file'])['model']
    members=selected.get('ensemble_components') or [selected['single_candidate']]
    outer,outer_weights=vote_parts(reference,len(members))
    require('weights' not in selected or selected['weights']==outer_weights,'Vote weights differ from the election')
    plan=selected.get('seed_replication') or {}
    repeated=set(plan.get('members_repeated',[]))
    require(repeated<=set(members),'Unknown member in the recorded seed recipe')
    observations=[]
    groups=[]
    physical=target/'physical.jsonl'
    os.environ['AB_B1_FIT_LOG']=str(physical)
    for i,(member,parent_model) in enumerate(zip(members,outer)):
        seeds=[None,*(plan.get('seeds',[]) if member in repeated else [])]
        leaves,shares=vote_parts(parent_model,len(seeds))
        fitted=[]
        for seed,source_leaf in zip(seeds,leaves):
            filename=fitted_filename(base,i,seed,member)
            before=joblib.load(base/filename)
            require(type(before['model']) is type(source_leaf),'Source member order or family differs')
            config=before['metrics']['resolved_configuration']
            require(config['id']==member and config.get('training_random_seed')==seed,'Recorded fit identity differs')
            expected=read((base/filename).with_suffix('.fits.json'))['completed_classifier_fits']
            request=native/(Path(filename).stem+'_input.joblib')
            joblib.dump(dict(payload,**({'supervised_random_seed':seed} if seed is not None else {})),request)
            dest=native/filename
            ledger=dest.with_suffix('.fits.json')
            admission=dest.with_suffix('.admission.json')
            save(admission,dict(maximum_classifier_fits=expected,ledger_path=str(ledger)))
            model,facts=train_candidate(request,member,dest,600,fit_admission_path=admission)
            require(model is not None and facts.get('status')=='completed',str(facts.get('error','Refit failed')))
            require(facts['resolved_configuration']==config and read(ledger)['completed_classifier_fits']==expected,
                    'The refit changed the original hyperparameters, seed or fit count')
            fitted.append(model)
            observations.append(dict(candidate=member,seed=seed,parameters=config['params'],fits=expected,
                                     runtime_configuration_equal=True,source_artifact=digest(base/filename)))
        groups.append(fixed_vote(parent_model,original.kind,fitted,classes,shares))
    final=fixed_vote(reference,original.kind,groups,classes,outer_weights)
    filename='augmented_model.joblib'
    joblib.dump(dict(kind=original.kind,candidate=selected['family'],model=final),native/filename)
    observed=sum(1 for _ in physical.open())
    require(observed==sum(item['fits'] for item in observations),'Physical fit accounting differs')
    record=dict(task_kind=original.kind,solver=seal['solver'],base_model=seal['model'],new_model=digest(native/filename),
                original_training_rows=len(original.y_train),added_development_rows=len(extra.y_train),
                total_training_rows=len(labels),same_selected_family=selected['family'],members=members,
                outer_weights=outer_weights,seeds=plan.get('seeds',[]),final_fits=observations,
                actual_classifier_fits=observed,model_api=0,additional_packet=digest(packet),opened_development_members=opened,
                selection_repeated=False,independent_designated_validation_score=None,official_test_opened=False)
    save(target/'refit.json',record)
    save(native/'run_manifest.json',dict(source_files=sources(),source_model=seal['model'],mode='fixed_final_refit'))
    save(native/'final_selection.json',dict(selected_experiment=selected['selected_experiment'],family=selected['family'],ensemble_components=selected['ensemble_components'],
         single_candidate=selected['single_candidate'],refit=True,training_rows=len(labels),primary_metric=selected['primary_metric'],
         selection_scope='Inherited from the training-only autonomous run',base_model_sha256=seal['model'],
         model_file=filename,model_sha256=record['new_model'],task_spec_sha256=digest(native/'task_spec.json'),
         model_format='candidate_bundle_v1',delivery_status='complete',test_accessed=False,
         **{key:selected[key] for key in ('weights','seed_replication') if key in selected}))
    print(case,len(labels),observed,record['new_model'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','packages','runtime'):
        parser.add_argument('--'+name,required=True,type=lambda x:Path(x).resolve())
    parser.add_argument('--case',choices=['a','b'],required=True)
    for name in ('election','election-seal','feature-bank'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve())
    cmd=parser.parse_args()
    if cmd.feature_bank:
        os.environ['SUPERAGENT_PAIRED_FEATURE_CACHE']=str(cmd.feature_bank)
    os.environ.update(AB_B1_OFFLINE='1',SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT,cmd.runtime))))
    activate_resource_directory(cmd.out/'weights')
    with installed_methods():
        refit(cmd,cmd.case)
