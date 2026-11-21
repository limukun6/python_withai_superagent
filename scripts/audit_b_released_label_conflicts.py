"""Count unavoidable contradictions for feature-only predictions on registered development rows."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts import run_b1_ab_acceptance as record
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_prediction_features, load_training_task
from tools.ab_registered_rows import content_digest


def categorical_key(row):
    result=[]
    for value in row:
        if isinstance(value,np.generic):
            value=value.item()
        if value is None or type(value) in (str,int,bool):
            result.append((type(value).__name__,value))
        elif type(value) is float:
            result.append(('float',value.hex()))
        else:
            raise ValueError(f'Unsupported feature type {type(value).__name__}')
    return tuple(result)


def groups(rows, labels, *, pair=False, swap=False):
    observed=defaultdict(Counter)
    for row,label in zip(rows,labels,strict=True):
        if pair:
            a,b=row
            if not isinstance(a,str) or not isinstance(b,str) or int(label) not in (0,1):
                raise ValueError('Expected two original headline strings and a binary label')
            key=(a,b)
            label=int(label)
            if swap and b<a:
                key=(b,a)
                label=1-label
        else:
            key=categorical_key(row)
        observed[key][label]+=1
    return observed


def summary(observed):
    sizes=[sum(v.values()) for v in observed.values()]
    count=sum(sizes)
    unavoidable=sum(sum(v.values())-max(v.values()) for v in observed.values())
    return dict(rows=count,unique_inputs=len(sizes),groups_with_repetition=sum(n>1 for n in sizes),
        rows_in_repeated_groups=sum(n for n in sizes if n>1),surplus_duplicate_rows=count-len(sizes),
        conflicting_groups=sum(len(v)>1 for v in observed.values()),minimum_wrong=unavoidable,
        finite_sample_accuracy_ceiling=1-unavoidable/count)


def preference_graph(observed):
    # Label 1 means the first title is preferred. Ignore exact ties of opposing
    # votes after accounting for their unavoidable errors in the pairwise count.
    graph=defaultdict(set)
    edges={}
    tied=identical=0
    for (a,b),count in observed.items():
        if a==b:
            identical+=sum(count.values())
            continue
        gap=count[1]-count[0]
        if gap==0:
            tied+=1
            continue
        if a>b:
            raise ValueError('Graph needs canonical unordered title pairs')
        graph[a].add(b)
        graph[b].add(a)
        edges[a,b]=gap
    triangles=cycles=packed=weight=0
    used=set()
    for a in sorted(graph):
        for b in sorted(n for n in graph[a] if n>a):
            for c in sorted(n for n in graph[a]&graph[b] if n>b):
                triangles+=1
                ab,bc,ac=edges[a,b],edges[b,c],edges[a,c]
                if (ab>0)==(bc>0) and (ab>0)!=(ac>0):
                    cycles+=1
                    keys={(a,b),(b,c),(a,c)}
                    if used.isdisjoint(keys):
                        used.update(keys)
                        packed+=1
                        weight+=min(abs(ab),abs(bc),abs(ac))
    return dict(titles_in_residual_graph=len(graph),residual_comparisons=len(edges),same_string_rows=identical,
        distinct_pairs_tied_after_cancellation=tied,closed_triangles=triangles,directed_cycles=cycles,
        edge_disjoint_cycles=packed,additional_mandatory_errors_for_one_global_order=weight)


def characterize(rows,labels,pair):
    exact=groups(rows,labels,pair=pair)
    findings=dict(ordered=summary(exact))
    canonical=exact
    if pair:
        canonical=groups(rows,labels,pair=True,swap=True)
        merged=summary(canonical)
        graph=preference_graph(canonical)
        floor=merged['minimum_wrong']+graph['additional_mandatory_errors_for_one_global_order']
        findings.update(unordered=merged,preference_graph=graph,
            global_order_minimum_wrong=floor,global_order_finite_sample_accuracy_ceiling=1-floor/len(labels))
    return findings,exact,canonical


def inputs(work,name):
    if name=='coupon':
        repo=work/'ab_route_representative'
        root=repo/'runs/route_coupons_01'
        signed=repo/'research/b_route_coupons'
        preflight=record.read(signed/'preflight.json')
        for seed in ('45101','45102'):
            case=root/(seed+'_representative')
            entry=preflight['entries'][seed]
            grade=record.read(signed/'results.json')['pairs'][seed]['grade']['representative']
            yield seed,root/seed,case,entry,grade
    else:
        ctx=work/'ab_contextual_preference/runs/context_headlines_02'
        signed=work/'ab_nonlinear_text/research/b_text_kernel_titles'
        if ab_run.sha(ctx/'preflight.json')!='f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95':
            raise ValueError('Registered headline split changed')
        preflight=record.read(ctx/'preflight.json')
        for seed in ('45101','45102'):
            entry=dict(preflight['tasks'][seed],features_sha256=preflight['entries'][seed+'_original']['features_sha256'])
            case=work/'ab_nonlinear_text/runs/text_kernel_titles_01'/seed
            grade=record.read(signed/'results.json')['pairs'][seed]['grades']['extended']
            yield seed,ctx/seed,case,entry,grade


def overlap(learned,held):
    shared=learned.keys()&held.keys()
    return dict(shared_inputs=len(shared),held_rows_matching_an_optimization_input=sum(sum(held[k].values()) for k in shared))


def audit_task(work,name,expected):
    is_pair=name=='headlines'
    label='preference' if is_pair else 'coupon'
    path=work/'ab_joint_a19/runs/ba19_released_inputs_02'/label/'train_only.zip'
    if ab_run.sha(path)!=expected:
        raise ValueError('Training-only source archive differs')
    with zipfile.ZipFile(path) as archive:
        names={Path(s).name.lower() for s in archive.namelist()}
        if names!=({'readme.md','train.csv'} if is_pair else {'readme.md','c_train.npy','y_train.npy'}):
            raise ValueError('Source must contain exactly the registered training members')
    task=load_training_task(path)
    if is_pair and (task.spec['input_schema']['pair_label_one']!='left' or task.spec['input_schema']['pair_semantics']!='preference'):
        raise ValueError('Expected the registered left-positive preference meaning')
    x,y=task.X_train,task.y_train
    full,_,_=characterize(x,y,is_pair)
    result=dict(train_only_archive_sha256=expected,training_source=full,development_partitions={})
    for seed,packet,model,entry,grade in inputs(work,name):
        raw=gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
        if record.sha(raw)!=entry['outer_ids_sha256'] or ab_run.sha(packet/'features.zip')!=entry['features_sha256']:
            raise ValueError('Development row membership or submitted model inputs differ')
        ids=json.loads(raw)
        learn,held=ids['fit'],ids['held']
        if sorted(learn+held)!=list(range(len(y))):
            raise ValueError('Development rows do not partition the training-only source')
        features=load_prediction_features(packet/'features.zip',task.kind,task.spec)
        truth=y[held]
        if (content_digest(features,truth)!=entry['held_content_sha256']
                or content_digest([x[i] for i in held],truth)!=entry['held_content_sha256']):
            raise ValueError('Evaluation feature values, labels or row order changed')
        findings,ordered,canon=characterize(features,truth,is_pair)
        training_x=[x[i] for i in learn]
        train_groups=groups(training_x,y[learn],pair=is_pair)
        findings['ordered_optimization_overlap']=overlap(train_groups,ordered)
        if is_pair:
            findings['unordered_optimization_overlap']=overlap(groups(training_x,y[learn],pair=True,swap=True),canon)
        csv=model/'export_0/rows.csv'
        preds=read_export(csv,len(held))
        if (ab_run.sha(csv)!=grade['csv_sha256'] or ab_run.sha(model/'export_1/rows.csv')!=grade['csv_sha256']
                or sum(p==str(t) for p,t in zip(preds,truth,strict=True))!=grade['correct'] or grade['rows']!=len(held)):
            raise ValueError('Existing frozen model export or published correct count differs')
        model_errors=len(held)-grade['correct']
        mandatory=findings['ordered']['minimum_wrong']
        findings['existing_frozen_export']=dict(rows=len(held),correct=grade['correct'],wrong=model_errors,
            csv_sha256=grade['csv_sha256'],ordered_conflict_floor_over_existing_errors=mandatory/model_errors if model_errors else None)
        result['development_partitions'][seed]=findings
    return result


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    sources=record.read(record.ROOT/'research/ba19_released_delivery/preflight.json')
    record.write(out/'protocol.json',dict(batch='B-RELEASED-LABEL-CONFLICTS-01',
        driver_sha256=ab_run.sha(Path(__file__)),plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_RELEASED_LABEL_CONFLICTS_01_plan.md'),
        source_receipt_sha256=ab_run.sha(record.ROOT/'research/ba19_released_delivery/preflight.json'),fit_limit=0,api_limit=0))
    try:
        result={n:audit_task(args.workspace.resolve(),n,sources[n]['source_staged_sha256']) for n in ('coupon','headlines')}
        record.write(out/'results.json',result)
        record.write(out/'completion.json',dict(status='completed',tasks=2,prior_frozen_exports_checked=4,
            new_fits=0,new_model_predictions=0,new_model_api_requests=0,new_course_rmb=0,formal_validation_test_or_h_opened=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
