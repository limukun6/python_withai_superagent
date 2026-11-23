"""Recompute confidence selection, paired outcome changes, token savings and physical-fit counts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_b_grouped_titles import compare, digest, read, require

DATA=ROOT/'research/b_encoder_cascade_1123'


def verify_policy(d):
    policy=d['policy']
    ordered=sorted(policy['first_correct'])
    require(max(ordered,key=policy['first_correct'].get)==policy['first'], 'First candidate selection')
    first,cf=policy['primary'],policy['confirmation']
    require([x['threshold'] for x in first]==[0.,.1,.2,.3,.4,1.] and
            all(x['rows']==d['primary_source']['used_rows'] for x in first),'Primary threshold grid')
    best=max((x for x in first if x['correct']>=x['full_correct']),key=lambda x:(x['skips'],x['threshold']))
    require(best['threshold']==policy['proposed']==cf['threshold'] and cf['rows']==d['fresh_confirmation_source']['used_rows'],
            'Threshold selected or confirmation differs')
    approved=cf['correct']>=cf['full_correct'] and cf['skips']>=.1*cf['rows']
    require(bool(policy['adopted'])==approved and policy['threshold']==(cf['threshold'] if approved else 1.),
            'Confirmation decision')


def batch(name, runner, plan, fits):
    p=DATA/name
    end=read(p/'completion.json')
    r=read(p/'results.json')
    protocol=read(p/'protocol.json')
    require(all(end[k]==digest(p/(k+'.json')) for k in ('decisions','results','exports','protocol')),name+' seals')
    require(protocol['plan']==digest(ROOT/'docs/batches'/plan) and protocol['runner']==digest(ROOT/'scripts'/runner) and
            protocol['policy']==digest(ROOT/'scripts/b_encoder_cascade.py'),name+' source')
    require(fits==end['fits']==r['physical_fits']==protocol['fit_cap'] and
            r['model_api_calls']==end['model_api_calls']==protocol['model_api_cap']==r['new_rmb']==0 and
            not r['course_validation_or_test'],name+' budget')
    sources=list(p.rglob('fit_completed.json'))
    require(len(sources)==fits==len(list(p.rglob('fit_started.json'))), 'Physical fit count')
    for f in sources:
        require(end['fit']==digest(f) and read(f)['start']==digest(f.with_name('fit_started.json')), 'Fit receipt')
    decisions,ex=read(p/'decisions.json'),read(p/'exports.json')
    result={}
    for key,task in r['tasks'].items():
        verify_policy(decisions[key])
        gz=p/key/'grouped_correctness.json'
        require(end['groups'][key]==digest(gz),'Outer group seal')
        tab=read(gz)
        for arm,v in task['comparisons'].items():
            n=compare(tab,None,arm,'selected',v,(task['correct'][arm],task['correct']['selected']))
            require(n==task['rows'],'Rows used')
        control,actual=ex[key]['control_usage'],ex[key]['selected_usage']
        total=task['rows']
        require(actual['first']==control['first'] and actual['first']['rows']==control['second']['rows']==total and
                actual['second']['rows']+actual['skips']==total and control['skips']==0, 'Physical routes')
        skipped=actual['skips']
        require(skipped==task['second_rows_skipped'] and control['second']['input_tokens']-actual['second']['input_tokens']
                ==task['second_tokens_saved'],'Measured row and token savings')
        require(task['supports_followup']==(skipped>=.1*total and (task['correct']['selected']-task['correct']['full'])
                /total>=-.005),'Follow-up gate')
        result[key]=dict(correct=task['correct'],skipped=skipped)
    require(r['eligible']==all(t['supports_followup'] for t in r['tasks'].values()),'Batch gate')
    return result


def main():
    first=batch('initial','run_b_encoder_cascade.py','B_encoder_cascade_1123_zh.md',0)
    second=batch('replication','run_b_encoder_cascade_replication.py','B_encoder_cascade_replication_1123_zh.md',1)
    extra=read(DATA/'replication/protocol.json')
    require(extra['initial']==digest(DATA/'initial/completion.json') and
            extra['order_audit']==digest(ROOT/'docs/batches/B_encoder_cascade_probability_audit_1123_zh.md'),
            'Follow-up or input-order note')
    exp=read(DATA/'initial/exports.json')
    for key,m in read(DATA/'initial/native_controls.json').items():
        require(m['vector']==exp[key]['always_'+('dual' if m['column']=='full' else 'first')]
                and m['rows']==exp[key]['rows'] and m['repeated_native_exports']==2,'Cross-process native controls')
    print(json.dumps(dict(initial=first,replication=second,fits=1,model_api_calls=0)))


if __name__=='__main__':
    main()
