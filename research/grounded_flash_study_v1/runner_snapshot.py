"""Two structured Flash decisions with candidate facts and measured checks."""
from pathlib import Path
import time
from unittest.mock import patch

from run_flash_error_feedback import StudySession, context_for, validate_answer
from run_composition_study import REPO, INTEGRATION, SEEDS, validate, check_prefix
from run_targeted_edit_study import EditPolicy
from superagent.adaptive_local import save, sha
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.cli import CAMPUS_CHAT_ENDPOINT, CAMPUS_FLASH_MODEL
from superagent.models import search_candidates
from superagent.contracts import ExperimentProposal
from superagent.targeted_feedback import candidate_facts
from superagent.hypothesis_agent import check_expectation
from superagent.benchmark import run_one
from superagent.evidence_schedule import duel
from tools.expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED
from tools.search_budget_study import read, outer_partition, predict


class FrozenPolicy(EditPolicy):
    def __init__(self, path, answer):
        super().__init__(path)
        self.answer = answer

    def propose(self, state):
        record = state.record()
        trial = record['next_trial']
        chosen = self.answer['candidate'] if trial == 3 else self.path[trial - 1]
        config = next(c for c in record['available_candidates'] if c['id'] == chosen)
        return ExperimentProposal.from_decision(trial, config, 'Frozen Flash action',
            self.answer if trial == 3 else None, record['history'])


def main():
    root = REPO / 'runs/grounded_flash_study_v1'
    root.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(root)
    logger.initialize_usage_ledger()
    started = time.monotonic()
    client = ExperimentLLM(CAMPUS_CHAT_ENDPOINT, str(REPO/'api_key.txt'), CAMPUS_FLASH_MODEL,
        batch_budget=.2, max_requests=2, streaming=True,
        allowed_served_models=['DeepSeek-V4.1-Flash', 'DeepSeek-V4.1-Flash-1017', 'DSF'])
    save(root/'protocol.json', dict(seeds=SEEDS, requests=2, timeout=60, portfolio='challenger',
         source_sha256={p.name:sha(p) for p in sorted((INTEGRATION/'superagent').glob('*.py'))},
         runner_sha256=sha(Path(__file__)),
         scope='Exploratory structured-interface pilot with expanded edit action menu',
         minimum_improvement=.001, all_decisions_before_new_training=True))
    answers = {}
    try:
        for seed in SEEDS:
            folder = root/str(seed)
            folder.mkdir()
            baseline = REPO/'runs/composition_study_v1'/str(seed)/'baseline'
            context = context_for(baseline, True)
            excluded = set(read(baseline/'trials.json')[i]['candidate'] for i in (0,1,3))
            context['available_candidates'] = [dict(id=c.family, params=c.params)
                for c in search_candidates('pair_text','challenger') if c.family not in excluded]
            context['candidate_facts'] = {c['id']:candidate_facts('pair_text',c)
                                           for c in context['available_candidates']}
            context['instruction'] += (
                ' Also return task_kind="pair_text", configuration_claim={classifier: actual classifier from candidate_facts}, '
                'expected_change={metric:"search_accuracy",reference_trial:2,direction:"increase",min_delta:0.001}, '
                'and slice_expectation={slice_id:target_slice,reference_trial:2,min_delta:0.001}. '
                'Choose a slice actually observed in trial 2. Both numerical expectations will be checked after training. '
                'Use registered candidate facts. Extra edit features are a hypothesis and have unknown accuracy.')
            save(folder/'prompt.json', context)
            print(f'{seed} structured Flash request started', flush=True)
            with patch('requests.Session', StudySession):
                answer = client.ask(context,logger,f'{seed}_grounded')
            save(folder/'answer.json',answer)
            validate_answer(answer,context)
            if answer.get('task_kind') != 'pair_text' or answer.get('configuration_claim') is None:
                raise ValueError('Missing task or classifier claim')
            if answer.get('slice_expectation') is None or answer.get('expected_change') is None:
                raise ValueError('Missing prospective numerical checks')
            if answer['slice_expectation']['slice_id'] != answer['target_slice']:
                raise ValueError('Conflicting slice declarations')
            config = next(c for c in context['available_candidates'] if c['id']==answer['candidate'])
            ExperimentProposal.from_decision(3,config,answer['reason'],answer,read(baseline/'trials.json')[:2])
            answers[seed]=answer
            print(f'{seed} accepted {answer["candidate"]}',flush=True)
        results=[]
        new_fits=0
        for seed in SEEDS:
            folder=root/str(seed)
            baseline=REPO/'runs/composition_study_v1'/str(seed)/'baseline'
            control=validate(baseline)
            path=list(control['path'])
            path[2]=answers[seed]['candidate']
            candidates=[baseline, REPO/'runs/flash_error_feedback_v2/43111/detailed',
                        REPO/'runs/flash_error_feedback_v2/43117/scalar',
                        REPO/'runs/targeted_edit_study_v1'/str(seed)/'edit']
            reused=None
            for candidate in candidates:
                if candidate!=baseline and str(seed) not in candidate.parts:
                    continue
                if candidate.exists() and read(candidate/'summary.json') and validate(candidate)['path']==path:
                    reused=candidate
                    break
            task,features,labels,partition=outer_partition(REPO/'data/released_task2.zip',seed)
            if partition!=read(baseline.parent/'partition.json'):
                raise ValueError('Partition mismatch')
            if reused is None:
                reused=folder/'trained'
                run_one(task,'local_interleaved_routes',SEARCH_SEED,reused,None,
                    **(RUN_OPTIONS|dict(portfolio='challenger',error_feedback=True)),
                    decision_policy=FrozenPolicy(control['path'],answers[seed]))
                new_fits+=11
                check_prefix(baseline,reused)
            arm=validate(reused)
            trials=read(reused/'trials.json')
            check=check_expectation(answers[seed],trials[2],trials[:2])
            save(folder/'hypothesis_check.json',check)
            # For reused paths this is a new decision checked against existing measurements.
            save(folder/'execution.json',dict(model_file=arm['model_file'],source_path=str(reused),
                 new_fit_performed=reused==folder/'trained',predeclared_answer=answers[seed]))
            a=predict(control,features,labels)==labels
            b=predict(arm,features,labels)==labels
            result=dict(seed=seed,candidate=answers[seed]['candidate'],check=check,
                 baseline_accuracy=float(a.mean()),accuracy=float(b.mean()),comparison=duel(a,b))
            results.append(result)
            save(root/'results.json',results)
            print(str(result),flush=True)
        save(root/'completion.json',dict(status='complete',requests=client.request_count,
            course_cost_rmb=client.spent,new_fits=new_fits,elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        details=f'{type(error).__name__}: {error}'.replace(client.key,'[redacted]')
        save(root/'completion.json',dict(status='stopped',error=details,requests_attempted=client.request_count,
             known_course_cost_rmb=client.spent))
        print(details,flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
