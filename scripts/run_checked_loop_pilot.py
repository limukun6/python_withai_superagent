"""Two task kinds, three equal-fit-budget arms, twelve Flash decisions maximum."""
from pathlib import Path
import time
from unittest.mock import patch

from run_composition_study import REPO, INTEGRATION, validate
from run_flash_error_feedback import StudySession
from superagent.checked_history_policy import CheckedHistoryPolicy
from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.cli import CAMPUS_CHAT_ENDPOINT,CAMPUS_FLASH_MODEL
from superagent.models import search_candidates
from superagent.targeted_feedback import candidate_facts
from superagent.evidence_schedule import duel
from tools.expanded_feedback_pilot import RUN_OPTIONS,SEARCH_SEED
from tools.search_budget_study import outer_partition,read,predict


def main():
    root=REPO/'runs/checked_loop_pilot_v1'
    root.mkdir(parents=True,exist_ok=False)
    logger=RunLogger(root)
    logger.initialize_usage_ledger()
    client=ExperimentLLM(CAMPUS_CHAT_ENDPOINT,str(REPO/'api_key.txt'),CAMPUS_FLASH_MODEL,
        batch_budget=1.2,max_requests=12,streaming=True,
        allowed_served_models=['DeepSeek-V4.1-Flash','DeepSeek-V4.1-Flash-1017','DSF'])
    options=RUN_OPTIONS|dict(time_limit=1200,error_feedback=True)
    seed=43911
    save(root/'protocol.json',dict(outer_seed=seed,search_seed=SEARCH_SEED,options=options,
        arms=['local','raw','checked'],maximum_requests=12,timeout=60,
        source_sha256={p.name:sha(p) for p in sorted((INTEGRATION/'superagent').glob('*.py'))},
        runner_sha256=sha(Path(__file__)),
        task_sha256={name:sha(REPO/'data'/f'released_{name}.zip') for name in ('task1','task2')},
        intervention='Extra computed hypothesis checks and memory; same output schema and menu',
        planned_order={'task1':['local','raw','checked'],'task2':['local','checked','raw']},
        scope='One new development partition per released task, three autonomous decisions after shared anchor'))
    started=time.monotonic()
    results=[]
    try:
        for name,order in [('task1',['local','raw','checked']),('task2',['local','checked','raw'])]:
            case=root/name
            case.mkdir()
            task,features,labels,partition=outer_partition(REPO/'data'/f'released_{name}.zip',seed)
            save(case/'partition.json',partition)
            facts={c.family:candidate_facts(task.kind,dict(id=c.family,params=c.params))
                   for c in search_candidates(task.kind,'expanded')}
            if any(f is None for f in facts.values()):
                raise ValueError(f'Missing candidate facts for {name}')
            save(case/'candidate_facts.json',facts)
            arms={}
            for arm in order:
                folder=case/arm
                policy=None if arm=='local' else CheckedHistoryPolicy(client,logger,arm=='checked')
                print(f'{name} {arm} started',flush=True)
                with patch('requests.Session',StudySession):
                    run_one(task,'local_interleaved_routes',SEARCH_SEED,folder,None,
                        **options,decision_policy=policy)
                arms[arm]=validate(folder)
                history=read(folder/'trials.json')
                arms[arm]['checks']=[dict(trial=r['trial'],check=r['hypothesis_check']) for r in history]
                print(f'{name} {arm} complete: {arms[arm]["candidate"]}',flush=True)
            save(case/'frozen_arms.json',arms)
            predicted={arm:predict(record,features,labels) for arm,record in arms.items()}
            correct={arm:values==labels for arm,values in predicted.items()}
            for arm,record in arms.items():
                record.update(accuracy=float(correct[arm].mean()),correct=int(correct[arm].sum()))
            comparisons={f'{b}_minus_{a}':duel(correct[a],correct[b])
                         for a,b in [('local','raw'),('local','checked'),('raw','checked')]}
            result=dict(task=name,kind=task.kind,seed=seed,arms=arms,comparisons=comparisons)
            results.append(result)
            save(root/'results.json',results)
            save(case/'outer_predictions.json',{a:v.tolist() for a,v in predicted.items()})
            print(str(dict(task=name,comparisons=comparisons)),flush=True)
        save(root/'completion.json',dict(status='complete',new_fits=66,requests=client.request_count,
            course_cost_rmb=client.spent,elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        details=f'{type(error).__name__}: {error}'.replace(client.key,'[redacted]')
        save(root/'completion.json',dict(status='stopped',error=details,requests_attempted=client.request_count,
            known_course_cost_rmb=client.spent,elapsed_seconds=time.monotonic()-started))
        print(details,flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
