"""Freeze unlabeled row strata, then unpack errors in four existing development exports."""
import argparse
from collections import Counter
import gzip
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scripts import run_b1_ab_acceptance as record
from scripts.audit_b_released_label_conflicts import categorical_key, groups, inputs
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_prediction_features,load_training_task
from superagent.splits import normalized_text
from tools.ab_registered_rows import content_digest


BINS={'unseen_field_value':('none','one_or_more'),'unseen_field_combinations':('0','1','2_to_5','6_plus'),
      'word_edits':('0','1','2_to_4','5_plus'),'component_comparisons':('one','multiple'),
      'titles_in_optimization':('0','1','2')}


def categorical_strata(training,held):
    optimized=[categorical_key(row) for row in training]
    fields=list(combinations(range(len(optimized[0])),2))
    if len(fields)!=210 or any(len(row)!=21 for row in optimized):
        raise ValueError('Expected 21 registered coupon fields')
    seen_values=[set(column) for column in zip(*optimized,strict=True)]
    seen_pairs=[{(row[i],row[j]) for row in optimized} for i,j in fields]
    result={'unseen_field_value':[],'unseen_field_combinations':[]}
    for row in held:
        key=categorical_key(row)
        if len(key)!=len(seen_values):
            raise ValueError('Evaluation row has a different column count')
        novel=sum(v not in options for v,options in zip(key,seen_values,strict=True))
        missing=sum((key[i],key[j]) not in known for (i,j),known in zip(fields,seen_pairs,strict=True))
        result['unseen_field_value'].append('one_or_more' if novel else 'none')
        result['unseen_field_combinations'].append(str(missing) if missing<2 else '2_to_5' if missing<=5 else '6_plus')
    return result


def components(nodes,edges,*,strong=False):
    ids={name:i for i,name in enumerate(sorted(nodes))}
    rows=[ids[a] for a,b in edges]
    cols=[ids[b] for a,b in edges]
    matrix=csr_matrix((np.ones(len(rows),dtype=bool),(rows,cols)),shape=(len(ids),len(ids)))
    _,labels=connected_components(matrix,directed=True,connection='strong' if strong else 'weak')
    return {name:int(labels[i]) for name,i in ids.items()}


def word_distance(a,b):
    left,right=(re.findall(r'[^\W_]+',s.casefold(),flags=re.UNICODE) for s in (a,b))
    previous=list(range(len(right)+1))
    for i,x in enumerate(left,1):
        current=[i]
        for j,y in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def title_strata(training,held):
    known={normalized_text(title) for pair in training for title in pair}
    nodes={title for pair in held for title in pair}
    memberships=components(nodes,held)
    counts=Counter(memberships[a] for a,b in held)
    result={'word_edits':[],'component_comparisons':[],'titles_in_optimization':[]}
    for a,b in held:
        distance=word_distance(a,b)
        result['word_edits'].append(str(distance) if distance<2 else '2_to_4' if distance<5 else '5_plus')
        result['component_comparisons'].append('one' if counts[memberships[a]]==1 else 'multiple')
        result['titles_in_optimization'].append(str(sum(normalized_text(t) in known for t in (a,b))))
    return result


def graph_counts(x,y):
    counts=groups(x,y,pair=True,swap=True)
    nodes={title for pair in x for title in pair}
    edges=[]
    for (a,b),votes in counts.items():
        delta=votes[1]-votes[0]
        if a!=b and delta:
            edges.append((a,b) if delta>0 else (b,a))
    weak,strong=components(nodes,edges),components(nodes,edges,strong=True)
    sizes=Counter(weak.values())
    edge_counts=Counter(weak[a] for a,b in edges)
    strong_sizes=Counter(strong.values())
    cyclic={k for k,size in strong_sizes.items() if size>1}
    return dict(titles=len(nodes),majority_edges=len(edges),weak_components=len(sizes),
        components_with_one_comparison=sum(n==1 for n in edge_counts.values()),largest_component_titles=max(sizes.values()),
        largest_component_edges=max(edge_counts.values(),default=0),independent_undirected_cycles=len(edges)-len(nodes)+len(sizes),
        strong_components=len(strong_sizes),cyclic_strong_components=len(cyclic),titles_on_directed_cycles=sum(strong_sizes[k] for k in cyclic),
        edges_inside_cyclic_components=sum(strong[a]==strong[b] and strong[a] in cyclic for a,b in edges),acyclic=not cyclic)


def grade_strata(membership,correct):
    result={}
    for axis,values in membership.items():
        rows={key:dict(rows=0,correct=0) for key in BINS[axis]}
        for key,ok in zip(values,correct,strict=True):
            rows[key]['rows']+=1
            rows[key]['correct']+=int(ok)
        for key,v in rows.items():
            v['errors']=v['rows']-v['correct']
            v['accuracy']=v['correct']/v['rows'] if v['rows'] else None
        if sum(v['correct'] for v in rows.values())!=sum(correct):
            raise ValueError('Strata fail to recover original correct count')
        result[axis]=rows
    return result


def run(args):
    work,out=args.workspace.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    previous=record.ROOT/'research/b_released_label_conflicts/results.json'
    signed=record.read(previous)
    record.write(out/'protocol.json',dict(batch='B-RELEASED-INPUT-DIFFICULTY-02',prior_sha256=ab_run.sha(previous),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_RELEASED_INPUT_DIFFICULTY_02_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),bins=BINS,fit_limit=0,model_api_limit=0))
    try:
        tasks,labels,rows,registrations={},{},{},{}
        for name in ('coupon','headlines'):
            is_pair=name=='headlines'
            stage=work/'ab_joint_a19/runs/ba19_released_inputs_02'/('preference' if is_pair else 'coupon')/'train_only.zip'
            if ab_run.sha(stage)!=signed[name]['train_only_archive_sha256']:
                raise ValueError('Training-only archive changed')
            task=load_training_task(stage)
            tasks[name]=task
            for seed,packet,model,entry,expected in inputs(work,name):
                raw=gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
                if record.sha(raw)!=entry['outer_ids_sha256'] or ab_run.sha(packet/'features.zip')!=entry['features_sha256']:
                    raise ValueError('Registered development split or order changed')
                ids=json.loads(raw)
                if sorted(ids['fit']+ids['held'])!=list(range(len(task.X_train))):
                    raise ValueError('Unlabeled assignment does not partition training input')
                training_x=[task.X_train[i] for i in ids['fit']]
                features=load_prediction_features(packet/'features.zip',task.kind,task.spec)
                membership=title_strata(training_x,features) if is_pair else categorical_strata(training_x,features)
                key=name+'_'+seed
                rows[key]=(membership,features,model,expected,entry['held_content_sha256'])
                labels[key]=(task,ids['held'])
                registrations[key]={axis:dict(row_count=len(values),ordered_strata_sha256=record.digest(values),
                    counts={group:values.count(group) for group in BINS[axis]}) for axis,values in membership.items()}
        record.write(out/'all_strata_frozen.json',registrations)
        findings={}
        for key,(membership,features,model,expected,content) in rows.items():
            task,held=labels[key]
            truth=task.y_train[held]
            if content_digest(features,truth)!=content:
                raise ValueError('Development features or labels changed')
            if any(record.digest(v)!=registrations[key][a]['ordered_strata_sha256'] for a,v in membership.items()):
                raise ValueError('Unlabeled strata changed before grading')
            csv=model/'export_0/rows.csv'
            if ab_run.sha(csv)!=expected['csv_sha256'] or ab_run.sha(model/'export_1/rows.csv')!=expected['csv_sha256']:
                raise ValueError('Original or relocated frozen predictions changed')
            predictions=read_export(csv,len(truth))
            correct=[p==str(label) for p,label in zip(predictions,truth,strict=True)]
            if sum(correct)!=expected['correct'] or len(correct)!=expected['rows']:
                raise ValueError('Failed to recover the original overall score')
            findings[key]=dict(rows=len(correct),correct=sum(correct),csv_sha256=expected['csv_sha256'],strata=grade_strata(membership,correct))
            if key.startswith('headlines'):
                findings[key]['preference_graph']=graph_counts(features,truth)
        titles=tasks['headlines']
        record.write(out/'results.json',dict(development=findings,whole_training_title_graph=graph_counts(titles.X_train,titles.y_train)))
        record.write(out/'completion.json',dict(status='completed',real_exports=4,new_fits=0,new_model_predictions=0,
            new_model_api_requests=0,new_course_rmb=0,formal_validation_test_or_h_opened=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
