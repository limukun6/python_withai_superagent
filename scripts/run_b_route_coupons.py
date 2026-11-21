"""Compare both schedulers on two training-only coupon holdouts with original-format inputs."""
import argparse
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot, run_b_route_titles as trial
from scripts.run_b_a19_released_delivery import write_packet
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, grouped_partition, sample_groups, take
from tools.ab_registered_rows import content_digest, registered_partitioner


ORDER = (('45101','original'),('45101','representative'),('45102','representative'),('45102','original'))


def prepare(seed, packet, previous, original, readme, reference):
    packet.mkdir()
    if seed == '45101':
        raw = gzip.decompress((rec.ROOT/'research/bq_01_portfolios/coupon/row_ids.json.gz').read_bytes())
        if rec.sha(raw)!=reference['outer_ids_sha256']:
            raise ValueError('Previously signed original row set changed')
        outer = json.loads(raw)['outer']
        fit,held = (np.asarray(outer[key],dtype=int) for key in ('train','held'))
        for name,field in [('train.zip','train_sha256'),('features.zip','features_sha256'),('inner.json','inner_sha256')]:
            if ab_run.sha(previous/name)!=reference[field]:
                raise ValueError('Previously registered coupon packet changed')
            os.link(previous/name,packet/name)
    else:
        fit,held,_ = grouped_partition(original,int(seed),'shared_text')
        write_packet(packet/'train.zip','coupon',take(original.X_train,fit),original.y_train[fit],readme)
        write_packet(packet/'features.zip','coupon',take(original.X_train,held))
    groups = sample_groups(original,'shared_text')
    if (len(set(fit)|set(held))!=len(original.y_train) or len(fit)+len(held)!=len(original.y_train)
            or set(groups[fit]) & set(groups[held])):
        raise ValueError('An optimization or held row crosses its assigned group')
    task = load_development_task(packet/'train.zip')
    task_contract = TaskSpec.from_task(task).record()
    train_hash = content_digest(take(original.X_train,fit),original.y_train[fit])
    held_hash = content_digest(take(original.X_train,held),original.y_train[held])
    features = load_prediction_features(packet/'features.zip',task.kind,task.spec)
    if (task.X_val is not None or task.y_val is not None or task_contract!=reference['task_contract']
            or train_hash!=content_digest(task.X_train,task.y_train) or held_hash!=content_digest(features,original.y_train[held])):
        raise ValueError('Task meaning, training values, or unlabeled delivery rows changed')
    ids_hash = pilot.identities_file(packet/'outer_ids.json.gz',dict(fit=fit.tolist(),held=held.tolist()))
    if seed=='45102':
        grouping = effective_grouping(task.kind,'shared_text')
        items = []
        for label,s in (('primary',707),('confirmation_1',808),('confirmation_2',909)):
            train,search,_ = grouped_partition(task,s,grouping)
            items.append(dict(id=label,seed=s,train_indices=train.tolist(),search_indices=search.tolist(),
                              row_ids_sha256=fingerprint(search),label_sha256=fingerprint(task.y_train[search])))
        rec.write(packet/'inner.json',dict(version=1,kind=task.kind,grouping=grouping,training_content_sha256=train_hash,splits=items))
    inner_sha = ab_run.sha(packet/'inner.json')
    (packet/'registered').mkdir()
    with patch.object(ab_run,'grouped_partition',registered_partitioner(task,packet/'inner.json',inner_sha)):
        _,splits = ab_run.freeze_partitions(task,packet/'registered',707,MetricSpec.for_task(task))
    if seed=='45101' and (splits!=reference['splits'] or held_hash!=reference['held_content']):
        raise ValueError('Original coupon partition fingerprints changed')
    caps = ab_run.registered_capabilities(task_contract,len(task.y_train),'full')
    if list(caps)!=reference['catalogs']['full']['ids'] or rec.digest(caps)!=reference['catalogs']['full']['sha256']:
        raise ValueError('Execution no longer offers the complete registered coupon catalogue')
    return dict(files={'train.zip':ab_run.sha(packet/'train.zip')},features_sha256=ab_run.sha(packet/'features.zip'),inner_sha256=inner_sha,
                outer_ids_sha256=ids_hash,held_content_sha256=held_hash,training_content_sha256=train_hash,train_rows=len(fit),rows=len(held),
                candidate_ids=list(caps),catalog_sha256=rec.digest(caps),splits=splits,contract=task_contract),set(held.tolist())


def main(args):
    out = args.output.resolve()
    trial.setup(out,args.weights)
    roots = {'original':args.control.resolve(),'representative':rec.ROOT}
    source = trial.sources(roots['original'])
    reference = rec.read(rec.ROOT/'research/ba19_released_delivery/preflight.json')['coupon']
    if ab_run.sha(args.staged)!=reference['source_staged_sha256']:
        raise ValueError('Incorrect training-only package')
    environment = pilot.environment()
    rec.write(out/'protocol.json',dict(batch='B-ROUTE-03',plan_sha256=ab_run.sha(rec.ROOT/trial.PLAN),sources=source,
              runner_sha256=ab_run.sha(Path(__file__)),worker_sha256=ab_run.sha(Path(trial.__file__)),
              staged_training_sha256=reference['source_staged_sha256'],prior_preflight_sha256=ab_run.sha(rec.ROOT/'research/ba19_released_delivery/preflight.json'),
              order=ORDER,outer_seeds=trial.SEEDS,inner_seeds=[707,808,909],environment=environment,fit_limit=11,
              max_new_fits=44,wall_seconds=1200,worker_seconds=300,model_api_limit=0))
    jobs = {}
    try:
        original = load_training_task(args.staged)
        member = original.spec['task_description_source']['member']
        with zipfile.ZipFile(args.staged) as archive:
            readme = (member,archive.read(member))
        if rec.sha(readme[1])!=reference['readme_sha256']:
            raise ValueError('Original task README changed')
        entries,held_sets = {},{}
        for seed in trial.SEEDS:
            entries[seed],held_sets[seed] = prepare(seed,out/seed,args.previous.resolve(),original,readme,reference)
        del original
        overlap = len(held_sets['45101'] & held_sets['45102'])
        rec.write(out/'preflight.json',dict(entries=entries,shared_outer_rows=overlap))
        for seed,arm in ORDER:
            key = seed+'_'+arm
            packet = out/seed
            case = out/key
            trial.stage_case(case,packet,entries[seed])
            jobs[key] = trial.train(case,roots[arm],source[arm],packet/'features.zip',out)
            rec.write(out/'checkpoint.json',jobs)
        rec.write(out/'all_models_frozen.json',{k:j['model_sha256'] for k,j in jobs.items()})
        truth_source = load_training_task(args.staged)
        pairs = {}
        for seed,entry in entries.items():
            packet = out/seed
            raw = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
            if rec.sha(raw)!=entry['outer_ids_sha256']:
                raise ValueError('Scoring row IDs changed after training')
            truth = truth_source.y_train[json.loads(raw)['held']]
            dev = load_development_task(packet/'train.zip')
            if content_digest(load_prediction_features(packet/'features.zip',dev.kind,dev.spec),truth)!=entry['held_content_sha256']:
                raise ValueError('Scoring labels or delivery order changed')
            row_files = {}
            for arm,root in roots.items():
                case = out/(seed+'_'+arm)
                trial.invoke(case,root,source[arm],'deliver',packet/'features.zip')
                row_files[arm] = case/'export_0/rows.csv'
            grade,delta = trial.paired_score(truth,MetricSpec.for_task(dev),row_files)
            pairs[seed] = dict(grade=grade,paired=delta,selected={arm:jobs[seed+'_'+arm]['selected'] for arm in roots})
        count = len(rec.trace(out))
        if count!=sum(j['fits'] for j in jobs.values()) or count>44 or trial.sources(roots['original'])!=source:
            raise ValueError('Final source or independent fit count changed')
        rec.write(out/'results.json',dict(paths=jobs,pairs=pairs,shared_outer_rows=overlap))
        rec.write(out/'completion.json',dict(status='completed',new_fits=count,new_cli_exports=8,new_model_api_requests=0,
                  new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),verified_paths=list(jobs),new_fits=len(rec.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control','previous','weights','staged','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
