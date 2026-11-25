"""Cold course-entry acceptance for unfamiliar packages, with a separate mock-transport case."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

from superagent.ab_flash import ENDPOINT, MODEL
from superagent.ab_run import sources
from superagent.autonomous_run import train
from superagent.experiment_llm import ExperimentLLM
from tests.test_adaptive_formats import FIXED_DOC, FIXED_PLAN, QUERY, TRAIN, nested_package, tar_package, workbook

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def cli(folder, tag, arguments, env):
    with (folder/(tag+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-m','unified_agent',*map(str,arguments)],cwd=ROOT,env=env,
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1280)


def fixed(folder):
    (folder/'README.md').write_text(FIXED_DOC)
    training, query = folder/'train.dat',folder/'test.dat'
    training.write_text(''.join(f'{i:03}{2*i:03}{("L" if y=="low" else "H"):>2}\n' for i,y in TRAIN))
    query.write_text(''.join(f'{i:03}{2*i:03}\n' for i in QUERY))
    return training, query


def simulated_client(folder):
    token = folder/'dummy.fixture'
    token.write_text('synthetic-test-fixture')
    client = ExperimentLLM(ENDPOINT,str(token),MODEL,.30,max_requests=1,allowed_served_models={MODEL},streaming=True)
    message = json.dumps(FIXED_PLAN)
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    chunks = [dict(model=MODEL,choices=[dict(index=0,delta=dict(content=message),finish_reason='stop')]),
              dict(model=MODEL,choices=[],usage=dict(prompt_tokens=400,completion_tokens=128))]
    response.iter_lines.return_value = [b'data: '+json.dumps(v).encode() for v in chunks]+[b'data: [DONE]']
    return client, response


def bitmap_package(folder):
    import io
    from PIL import Image
    formats = [('BMP','bmp'),('TIFF','tif'),('WEBP','webp'),('GIF','gif')]
    def image(i, fmt):
        data=io.BytesIO()
        Image.new('RGB',(8,8),(i*4,i*3,i*2)).save(data,format=fmt,**({'lossless':True} if fmt=='WEBP' else {}))
        return data.getvalue()
    training, query = {}, {}
    for i,label in TRAIN:
        fmt, ext = formats[i % len(formats)]
        training[f'train/{label}/{i:03}.{ext}'] = image(i,fmt)
    for n,i in enumerate(QUERY):
        fmt, ext = formats[n]
        query[f'test/{n:03}.{ext}'] = image(i,fmt)
    return tar_package(folder/'p.tar.gz',training),tar_package(folder/'q.tar.gz',query)


def run(root, final_check=False):
    root.mkdir(parents=True,exist_ok=False)
    protocol = read(ROOT/'research/adaptive_formats_1125/protocol.json')
    save(root/'protocol.json',protocol)
    before = sources()
    result = {}
    cases = ['compressed_nested','workbook','documented_layout'] + (['static_bitmaps'] if final_check else [])
    for case in cases:
        folder = root/case
        folder.mkdir()
        run_dir = folder/'run'
        dest = folder/'delivery'/'predictions.csv'
        replay = folder/'reloaded'/'predictions.csv'
        env = dict(os.environ, SUPERAGENT_OFFLINE='1', AB_B1_OFFLINE='1', HF_HUB_OFFLINE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                   AB_B1_FIT_LOG=str(folder/'physical.jsonl'), AB_B1_PER_WORKER_FIT_LIMIT='1',
                   PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))))
        for key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID'):
            env.pop(key,None)
        if case == 'documented_layout':
            training, query = fixed(folder)
            run_dir.mkdir()
            client, response = simulated_client(folder)
            with patch.dict(os.environ,env), patch('requests.Session.post',return_value=response) as transport:
                rc = train(training,run_dir/'agent',fit_limit=3,flash=client)
            assert rc == 0 and transport.call_count == 1
            messages = transport.call_args.kwargs['json']['messages']
            prompt = json.loads(messages[-1]['content'])
            assert set(prompt) == {'instruction','layout_sentences','goal_sentences','record_width'}
            cli(folder,'predict',['predict',query,'--run-dir',run_dir,'--output',dest],env)
        else:
            if case == 'workbook':
                training = query = workbook(folder/'anonymous.blob')
            elif case == 'static_bitmaps':
                training,query = bitmap_package(folder)
            else:
                training,query = nested_package(folder)
            if final_check:
                run_dir.mkdir()
                with (folder/'train.log').open('x') as log:
                    subprocess.run([sys.executable,'-m','superagent','unified',str(training),'--run-dir',
                        str(run_dir/'agent'),'--fit-limit','3'],env=env,cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,timeout=1250,check=True)
                cli(folder,'predict',['predict',query,'--run-dir',run_dir,'--output',dest],env)
            else:
                cli(folder,'solve',['solve',training,'--prediction-input',query,'--run-dir',run_dir,'--output',dest],env)
        cli(folder,'reload',['predict',query,'--run-dir',run_dir,'--output',replay],env)
        native = run_dir/'agent'
        outcome = read(native/'run_outcome.json')
        receipt = read(dest.parent/'prediction_manifest.json')
        physical = (folder/'physical.jsonl').read_text().splitlines()
        with dest.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert outcome['status']=='completed' and outcome['training_rows']==len(TRAIN)
        assert len(physical) == outcome['attempted_classifier_fits'] <= (3 if final_check or case=='documented_layout' else 10)
        assert outcome['new_api_requests'] == int(case=='documented_layout')
        assert [row['row_id'] for row in rows] == list(map(str,range(len(QUERY))))
        assert dest.read_bytes()==replay.read_bytes() and receipt['prediction_count']==len(QUERY)
        result[case] = dict(status='completed', codec=read(native/'task_spec.json')['input_format'],
                            train_rows=len(TRAIN), prediction_rows=len(rows), fits=len(physical),
                            live_api_requests=0, simulated_transport_calls=int(case=='documented_layout'),
                            fresh_process_bytes_identical=True, predictions_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        save(root/'results.json',result)
    if before != sources():
        raise ValueError('Production source changed during the acceptance run')
    save(root/'completion.json',dict(status='complete',source_sha256=hashlib.sha256(
         json.dumps(before,sort_keys=True).encode()).hexdigest(),physical_fits=sum(r['fits'] for r in result.values()),
         live_api_requests=0,simulation_used_for='documented_layout'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--final-check',action='store_true')
    args=parser.parse_args()
    run(args.output.resolve(),args.final_check)
