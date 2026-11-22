"""No fitting/HTTP: exact old defaults and common profiled/calibrated context reconstruction."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_flash import context_for
from superagent.ab_run import sources
from superagent.data import load_training_task
from superagent.task_profile import training_facts


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);frozen=sources();results=[]
    entries={r['task']:r for r in read(args.prepared/'readiness.json')}
    save(args.output/'protocol.json',dict(batch='A44-context-audit',source_files=frozen,runner_sha256=sha(Path(__file__)),
        scope='Existing histories only; no new LLM answer or performance evaluated',real_api_requests=0,classifier_fits=0))
    for task in ('coupon','preference'):
        entry=entries[task];archive=Path(entry['training_archive'])
        if sha(archive)!=entry['training_archive_sha256']:raise ValueError('Input changed')
        facts=training_facts(load_training_task(archive))
        for arm in ('R0','E0','R1','E1'):
            directory=args.matrix/(task+'_'+arm);caps=read(directory/'run_manifest.json')['capabilities']
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            for event in (r for r in events if r['kind']=='llm_request'):
                original=json.loads(event['messages'][1]['content']);trial=len(original['history'])+1
                state=read(directory/f'state_{trial}.json');ids=tuple(c['id'] for c in original['available_candidates'])
                rebuilt=context_for(state,caps,ids,arm.startswith('E'))
                if rebuilt!=original:raise ValueError('Default contract changed an old payload')
                profiled={**deepcopy(state),'training_profile':facts}
                raw=context_for(profiled,caps,ids,False,'calibrated-v2')
                checked=context_for(profiled,caps,ids,True,'calibrated-v2')
                if raw!={k:v for k,v in checked.items() if k not in ('checks','memory')}:raise ValueError('Unequal common context')
                results.append(dict(task=task,arm=arm,trial=trial,old_default_exact=True,
                    common_profile_calibrated_equal=True,old_payload_sha256=digest(original),new_raw_payload_sha256=digest(raw)))
    save(args.output/'results.json',results)
    save(args.output/'completion.json',dict(status='passed',old_payloads_rebuilt=len(results),new_common_context_pairs=len(results),
        actual_classifier_fits=0,real_api_requests=0,source_unchanged=sources()==frozen))
    print(json.dumps(read(args.output/'completion.json')))
