"""Evaluate sealed paired actions on two common training-internal partitions."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import time
import joblib
from scripts.run_a2_state_study import read,sha,save,sources,fingerprint


def validate_inputs(prepared,calls,fixture=False):
    from superagent.contracts import DecisionState,ExperimentProposal
    from superagent.checked_history_policy import decision_context
    prepared,calls=Path(prepared),Path(calls)
    protocol=read(prepared/'protocol.json');done=read(prepared/'completion.json')
    if done.get('status')!='prepared' or protocol['source_sha256']!=sources():raise ValueError('Prepared source identity mismatch')
    if protocol.get('evaluation_seeds')!=[2707,3707]:raise ValueError('Unregistered evaluation partitions')
    for package,version in protocol['dependencies'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Prepared dependency version mismatch')
    binding=read(calls/'protocol.json');completion=read(calls/'completion.json')
    if binding.get('fixture',False)!=fixture:raise ValueError('Fixture identity mismatch')
    if binding.get('prepared_protocol_sha256')!=sha(prepared/'protocol.json') or binding.get('source_sha256')!=sources():
        raise ValueError('Decision batch binding mismatch')
    if completion.get('status')!='complete':raise ValueError('Cannot evaluate incomplete decision batch')
    entries={(e['task'],e['prefix']):e for e in read(prepared/'states.json')}
    decisions=read(calls/'results.json');pairs={};states={}
    for decision in decisions:
        key=(decision['task'],decision['prefix']);checked=decision['checked']
        if key not in entries or type(checked) is not bool:raise ValueError('Unknown task/prefix/arm')
        if checked in pairs.setdefault(key,{}):raise ValueError('Duplicate paired decision')
        entry=entries[key];path=prepared/entry['file']
        if sha(path)!=entry['state_sha256']:raise ValueError('Frozen state bytes changed')
        state=DecisionState(read(path));record=state.record();states[key]=record
        for arm in (False,True):
            expected=entry['checked_context_sha256' if arm else 'raw_context_sha256']
            if fingerprint(decision_context(state,arm))!=expected:raise ValueError('Frozen context identity mismatch')
        proposal=decision['proposal'];candidate=proposal['candidate']
        configuration=next((c for c in record['available_candidates'] if c['id']==candidate),None)
        if configuration is None or proposal['params']!=configuration['params']:raise ValueError('Unavailable or modified candidate')
        validated=ExperimentProposal.from_decision(record['next_trial'],configuration,proposal['reason'],proposal['raw_proposal'],record['history'])
        if validated.record()!=proposal:raise ValueError('Proposal record disagrees with frozen decision')
        if not fixture and validated.raw_proposal is None:raise ValueError('Real decision missing original model proposal')
        pairs[key][checked]=deepcopy(configuration)
    if any(set(v)!={False,True} for v in pairs.values()) or not pairs:raise ValueError('Incomplete paired arms')
    if not fixture and (set(pairs)!=set(entries) or len(decisions)!=24 or completion.get('valid_proposals')!=24):
        raise ValueError('Formal evaluation requires every predeclared paired state')
    if fixture and len(pairs)>2:raise ValueError('Fixture batch limited to two states')
    jobs={(task,c['id'],seed) for (task,_),arms in pairs.items() for c in arms.values() for seed in protocol['evaluation_seeds']}
    if len(jobs)>48:raise ValueError('Evaluation exceeds 48-fit budget')
    return protocol,pairs,states,sorted(jobs)


def evaluate(prepared,calls,output,fixture=False,validate_only=False):
    from superagent.benchmark import train_candidate
    from superagent.contracts import TaskSpec
    from superagent.models import search_candidates
    from superagent.splits import grouped_partition,take,overlap_audit
    from superagent.prediction_evidence import fingerprint as vector_hash
    from tools.search_budget_study import outer_partition
    prepared=Path(prepared);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();attempted=0;completed=0;measurements={};results=[]
    try:
        protocol,pairs,states,jobs=validate_inputs(prepared,calls,fixture)
        save(out/'protocol.json',dict(fixture=fixture,scope='Controlled evaluator validation' if fixture else 'Sealed action quality diagnostic',
            evaluator_sha256=sha(__file__),dependencies=protocol['dependencies'],
            prepared_protocol_sha256=sha(prepared/'protocol.json'),calls_results_sha256=sha(Path(calls)/'results.json'),source_sha256=sources(),
            evaluation_seeds=protocol['evaluation_seeds'],unique_classifier_fits_planned=len(jobs),logical_evaluations=len(pairs)*4,
            automatic_retries=False,outer_evaluated=False,validate_only=validate_only))
        # Validate every data/candidate identity before the first fit.
        tasks={};configs={}
        for name in sorted({k[0] for k in pairs}):
            archive=Path(protocol['tasks'][name]['archive'])
            if sha(archive)!=protocol['tasks'][name]['sha256']:raise ValueError('Task archive identity changed')
            task,_,_,indices=outer_partition(archive,protocol['outer_seed'])
            if indices!=read(prepared/(name+'_partition.json')):raise ValueError('Training/outer partition changed')
            tasks[name]=task;catalog={c.family:c.params for c in search_candidates(task.kind,'expanded')}
            for key,arms in pairs.items():
                if key[0]!=name:continue
                if TaskSpec.from_task(task).record()['metric']!=states[key]['task_spec']['metric']:raise ValueError('Task objective changed')
                for config in arms.values():
                    if catalog.get(config['id'])!=config['params']:raise ValueError('Static executable catalog changed')
                    configs[name,config['id']]=config
        if validate_only:
            save(out/'completion.json',dict(status='validated',attempted_classifier_fits=0,completed_classifier_fits=0,api_requests=0));return
        for name,candidate,seed in jobs:
            if sources()!=protocol['source_sha256']:raise ValueError('Production source changed during evaluation')
            task=tasks[name];metric=TaskSpec.from_task(task).metric;train,held,split=grouped_partition(task,seed,'shared_text')
            folder=out/f'job_{len(measurements)+1:02d}';folder.mkdir()
            save(folder/'split.json',dict(task=name,candidate=candidate,seed=seed,train_rows=train.tolist(),search_rows=held.tolist(),
                resource=split,audit=overlap_audit(task,train,held)))
            payload=folder/'input.joblib'
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,train),y_train=task.y_train[train],
                X_search=take(task.X_train,held),y_search=task.y_train[held],metric_spec=metric.record(),
                pair_semantics=task.spec.get('input_schema',{}).get('pair_semantics'),retain_model=False),payload)
            attempted+=1
            _,measured=train_candidate(payload,candidate,folder/'observed.joblib',90)
            save(folder/'measurement.json',measured)
            if measured['status']!='completed':raise RuntimeError('Evaluation worker failed; partial component count unknown')
            completed+=measured.get('classifier_fits',1)
            if measured.get('classifier_fits')!=1:raise RuntimeError('Unexpected component cost; stop evaluation')
            metric.check_identity(measured.get('metric_spec'))
            artifact=joblib.load(folder/'observed.joblib');prediction=artifact['prediction_record']
            if (prediction['evaluation_payload_sha256']!=sha(payload) or prediction['label_sha256']!=vector_hash(task.y_train[held])
                    or prediction['prediction_sha256']!=vector_hash(prediction['predictions'])):raise ValueError('Prediction identity mismatch')
            # Four A2 tasks do not use AUC; reject accidental protocol expansion instead of substituting accuracy.
            if metric.requires_scores:raise ValueError('AUC action study needs a separately frozen score-aware protocol')
            recomputed=metric.evaluate(task.y_train[held],prediction['predictions'])
            if abs(recomputed['score']-measured['search_score'])>1e-12:raise ValueError('Reported score disagrees with observed prediction')
            measurements[name,candidate,seed]=dict(task=name,candidate=candidate,seed=seed,score=recomputed['score'],
                accuracy=recomputed['accuracy'],metric=metric.name,prediction_sha256=prediction['prediction_sha256'],
                evaluation_payload_sha256=prediction['evaluation_payload_sha256'])
            save(out/'checkpoint.json',dict(measurements=list(measurements.values()),attempted_fits=attempted,completed_fits=completed))
        for (name,prefix),arms in sorted(pairs.items()):
            raw=[measurements[name,arms[False]['id'],s]['score'] for s in protocol['evaluation_seeds']]
            checked=[measurements[name,arms[True]['id'],s]['score'] for s in protocol['evaluation_seeds']]
            deltas=[b-a for a,b in zip(raw,checked)];mean=sum(deltas)/len(deltas)
            results.append(dict(task=name,prefix=prefix,metric=states[name,prefix]['task_spec']['metric']['name'],
                raw_candidate=arms[False]['id'],checked_candidate=arms[True]['id'],same_action=arms[False]==arms[True],
                raw_scores=raw,checked_scores=checked,paired_differences=deltas,mean_difference=mean,
                outcome='tie' if abs(mean)<=1e-12 else 'checked_higher' if mean>0 else 'raw_higher'))
        save(out/'results.json',results)
        save(out/'completion.json',dict(status='complete',fixture=fixture,paired_states=len(results),
            logical_evaluations=4*len(results),reused_evaluations=4*len(results)-len(jobs),
            attempted_classifier_fits=attempted,completed_classifier_fits=completed,api_requests=0,
            elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==protocol['source_sha256'],outer_evaluated=False))
    except Exception as error:
        save(out/'completion.json',dict(status='stopped',error=f'{type(error).__name__}: {error}',
            attempted_worker_calls=attempted,completed_classifier_fits=completed,failed_attempt_component_count_unknown=attempted>len(measurements),
            api_requests=0,elapsed_seconds=time.monotonic()-start));raise
