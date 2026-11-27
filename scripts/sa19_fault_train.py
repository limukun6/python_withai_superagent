"""Research-only single pre-launch fault. All successful workers are unmodified."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--query',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    from superagent.ab_run import Executor
    from unified_agent import train
    original=Executor.worker;injected=[]
    def once(self,payload,name,destination,timeout,reserve=0):
        if not injected:
            injected.append(name)
            self.logger.event('research_prelaunch_fault',candidate=name,actual_fits_for_fault=0,
                           cause='controlled unavailable dependency before worker startup')
            raise RuntimeError('SA19 controlled unavailable dependency before worker startup')
        return original(self,payload,name,destination,timeout,reserve=reserve)
    with patch.object(Executor,'worker',once):
        code=train(a.package,a.run,fit_limit=4,time_limit=360,task_interpretation='auto',prediction_input=a.query)
    (a.run/'fault_receipt.json').write_text(json.dumps(dict(candidates=injected,prelaunch=True,
        fault_classifier_fits=0,method_selection_injected=False)),encoding='utf-8')
    return code


if __name__=='__main__':raise SystemExit(main())
