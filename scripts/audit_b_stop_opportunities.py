"""Measure whether visible challengers and untried routes warrant more search before stopping."""
import argparse
from pathlib import Path
import sys

from audit_b_stopping_prefixes import (
    HERE, RULES, archived_grade, inputs, inspect_path, read, same_files, save, sha,
)


PAIRED, CHALLENGERS, ROUTES = RULES[1], 'resolve_leading_challengers', 'also_cover_affordable_routes'


def review_inputs(records, arm):
    root = records/'runs/text_kernel_domains_01'
    signed = records/'research/b_text_kernel_domains'
    same_files(root,signed,('protocol.json','preflight.json','results.json','all_six_models_frozen.json','completion.json'))
    source_hash = read(signed/'protocol.json')['sources'][arm]
    entries = read(signed/'preflight.json')
    history = read(signed/'results.json')
    models = read(signed/'all_six_models_frozen.json')
    for domain in ('amazon','imdb','yelp'):
        name = domain+'_'+arm
        case = root/name
        pre = entries[domain]
        entry = dict(pre,name=name,**pre['catalogs'][arm])
        if entry != read(case/'entry.json'):
            raise ValueError('Review-domain registration differs')
        expected = dict(history['paths'][name],grade=history['pairs'][domain]['grades'][arm])
        yield domain,'held_source',arm,case,entry,source_hash,expected,models[name]


def opportunities(state, caps, incumbent):
    from superagent.ab_comparison import MIN_GAIN
    from superagent.metrics import MetricSpec, score_of
    verified = {p['trial'] for p in state['prediction_evidence']}
    observed = {}
    for row in state['history']:
        if row['trial'] in verified:
            observed.setdefault(row['split_id'],{})[row['candidate']]=score_of(row)
    primary = observed['primary']
    needs_confirmation = []
    for name,score in primary.items():
        if name != incumbent and score-primary[incumbent]>=MIN_GAIN:
            compared = any({name,incumbent} <= observed.get(s,{}).keys() for s in state['registered_confirmation_splits'])
            if not compared:
                needs_confirmation.append(name)
    covered = {caps[n]['route'] for n in primary}
    remaining = state['budget']['remaining_fit_allowance']
    reserve = state['budget']['reserved_final_refit']
    metric = MetricSpec.from_payload({'metric_spec':state['task']['metric']})
    untried = set()
    for cap in state['available_candidates']:
        if cap != caps[cap['id']]:
            raise ValueError('State and registered capability facts differ')
        if (cap['refit_classifier_fits']<=reserve and 2*cap['classifier_fits']+reserve<=remaining
                and (not metric.requires_scores or cap['supports_probabilities']) and cap['route'] not in covered):
            untried.add(cap['route'])
    return sorted(needs_confirmation),sorted(untried)


def decision_trace(case, old, caps):
    from superagent.scheduler_bridge import fingerprint
    trace, chosen = [],{}
    previous = []
    for prefix in old['trajectory']:
        count = prefix['observations']
        state = read(case/'run'/f'state_{count+1}.json')
        if fingerprint(state)!=prefix['state_sha256']:
            raise ValueError('State differs from the reproduced decision history')
        winner = prefix['incumbent']
        previous.append(winner)
        paired = (prefix['primary_count']>=3 and len(previous)>=3 and len(set(previous[-3:]))==1
                  and prefix['has_paired_confirmation'])
        terminal = count==old['full_search_observations']
        pending, uncovered = opportunities(state,caps,winner)
        live_names = {r['candidate'] for r in state['history'] if r['split_id']=='primary'}
        why = dict(observations=count,state_sha256=prefix['state_sha256'],incumbent=winner,
                   unpaired_leading_candidates=pending,affordable_untried_routes=uncovered)
        trace.append(why)
        options = (paired, paired and not pending, paired and not pending and not uncovered)
        for policy,eligible in zip((PAIRED,CHALLENGERS,ROUTES),options,strict=True):
            if policy in chosen or (not eligible and not terminal):
                continue
            saved=old['original_fits']-prefix['physical_search_fits']-caps[winner]['refit_classifier_fits']
            if not 0<=saved<old['original_fits']:
                raise ValueError('Estimated saving differs from observed cost and refit allowance')
            chosen[policy]=dict(**why,rule_met=eligible,earlier=not terminal,potential_fits_saved=saved,
                same_winner=winner==old['original'],eventual_winner_had_primary=old['original'] in live_names,
                eventual_winner_was_unpaired_leading=old['original'] in pending,
                eventual_winner_route_was_untried=caps[old['original']]['route'] in uncovered)
    baseline=chosen[PAIRED]
    if any(baseline[k]!=old['rules'][PAIRED][k] for k in ('observations','incumbent','rule_met','earlier','potential_fits_saved','same_winner')):
        raise ValueError('Original paired stopping rule was not reproduced')
    return trace,chosen


def run(args):
    source=args.source.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source))
    from scripts import run_b1_ab_acceptance as record
    from superagent.ab_run import sources
    from superagent.ab_comparison import MIN_GAIN

    if record.ROOT!=source or MIN_GAIN!=.002:
        raise ValueError('Unexpected source checkout or original promotion margin')
    record.offline(out)
    current=record.digest(sources())
    review=args.cohort.startswith('reviews_')
    archive=review_inputs(args.review_records.resolve(),args.cohort.split('_')[1]) if review else inputs(source,args.cohort)
    save(out/'protocol.json',dict(batch='B-STOPPING-PREFIX-02',cohort=args.cohort,registered_source_sha256=current,
        driver_sha256=sha(Path(__file__)),prior_driver_sha256=sha(HERE/'scripts/audit_b_stopping_prefixes.py'),
        plan_sha256=sha(HERE/'docs/batches/B_STOPPING_PREFIX_02_plan.md'),confirmed_margin=MIN_GAIN,new_fits_allowed=0,api_limit=0))
    try:
        sources_held, menus, records = {},{},{}
        for task,seed,arm,case,entry,source_hash,expected,model in archive:
            key='/'.join((task,seed,arm))
            old,caps=inspect_path(args.cohort,task,seed,arm,case,entry,source_hash,expected,model,current)
            if not review:
                prior=read(HERE/'research/b_stopping_prefixes'/args.cohort/'all_prefix_decisions_frozen.json')[key]
                if prior!=old:
                    raise ValueError('Previous study decisions failed to reproduce')
            trace,decisions=decision_trace(case,old,caps)
            records[key]=dict(task=task,split=seed,arm=arm,original=old['original'],original_fits=old['original_fits'],
                states=old['states_verified'],identity=old['identity'],decisions=decisions,opportunities=trace)
            menus[key]=caps
            sources_held[key]=(case,expected)
        save(out/'all_decisions_frozen.json',records)
        grades={n:archived_grade(case,args.cohort,expected) for n,(case,expected) in sources_held.items()}
        for key,r in records.items():
            actual=grades[key]
            for effect in r['decisions'].values():
                candidate=effect['incumbent']
                matches=[name for name,other in records.items() if other['identity']==r['identity']
                         and other['original']==candidate and menus[name][candidate]==menus[key][candidate]
                         and grades[name]['features_sha256']==actual['features_sha256']]
                if len({grades[n]['csv_sha256'] for n in matches})>1:
                    raise ValueError('Two matching full runs disagree')
                effect['matching_full_runs']=matches
                effect['grade_difference_pp']=None
                if matches:
                    alt=grades[matches[0]]
                    if (actual['metric'],actual['rows'])!=(alt['metric'],alt['rows']):
                        raise ValueError('Comparable outputs have different score definitions')
                    effect['grade_difference_pp']=100*(alt['score']-actual['score'])
        summary={}
        for rule in (PAIRED,CHALLENGERS,ROUTES):
            rows=[r['decisions'][rule] for r in records.values()]
            summary[rule]=dict(paths=len(rows),earlier=sum(r['earlier'] for r in rows),
                different_winner=sum(not r['same_winner'] for r in rows),
                potential_fits_saved=sum(r['potential_fits_saved'] for r in rows),
                matched_full_runs=sum(bool(r['matching_full_runs']) for r in rows))
        save(out/'results.json',dict(summary=summary,paths=records))
        save(out/'completion.json',dict(status='completed',cohort=args.cohort,paths=len(records),states=sum(r['states'] for r in records.values()),
            new_fits=0,new_model_predictions=0,new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_h_opened=False))
    except Exception as error:
        save(out/'completion.json',dict(status='failed',error=str(error),new_fits=0,new_model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True,choices=('b1','b2','reviews_original','reviews_extended'))
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--review-records',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.cohort.startswith('reviews_') and args.review_records is None:
        parser.error('--review-records is required for review-domain cohorts')
    run(args)
