"""Isolated acceptance-only faults. The production launcher never imports this file."""
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[2]
runpy.run_path(str(root/'scripts/ab_acceptance_guard/sitecustomize.py'))
case=os.environ.get('AIAA_RECOVERY_FIXTURE')
argv=getattr(sys,'orig_argv',[])
if case and 'superagent' in argv and 'unified' in argv:
    from superagent.ab_run import Executor
    from superagent.events import RunLogger
    if case in ('before_first_fit','refit_failure'):
        original=Executor.worker
        def worker(self,payload,name,destination,timeout,**kwargs):
            if case=='before_first_fit' or Path(payload).name=='refit_data.joblib':
                raise TimeoutError('Acceptance fixture interrupted the admitted training stage before dispatch')
            return original(self,payload,name,destination,timeout,**kwargs)
        Executor.worker=worker
    elif case=='hard_exit':
        write=RunLogger.write_json
        def publish(self,name,value):
            write(self,name,value)
            if name=='final_selection.json' and value.get('model_format')=='verified_trial_v1':
                os._exit(87)
        RunLogger.write_json=publish
    elif case=='stream_cut':
        import json
        from unittest.mock import MagicMock
        import requests
        import superagent.unified
        from superagent.ab_flash import ENDPOINT,MODEL
        from superagent.experiment_llm import ExperimentLLM
        def configured(directory):
            key=Path(os.environ['AIAA_DUMMY_KEY'])
            return ExperimentLLM(ENDPOINT,str(key),MODEL,.10,max_requests=1,allowed_served_models={MODEL},streaming=True)
        def fake_post(*args,**kwargs):
            response=MagicMock(status_code=200)
            response.__enter__.return_value=response
            def parts():
                yield b'data: '+json.dumps(dict(model=MODEL,choices=[dict(index=0,delta=dict(content='{'))])).encode()
                raise requests.exceptions.ChunkedEncodingError('Controlled SSE interruption before token usage')
            response.iter_lines.side_effect=parts
            return response
        superagent.unified.configured_flash=configured
        requests.Session.post=fake_post
