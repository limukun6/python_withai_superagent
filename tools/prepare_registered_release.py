"""Verify an independently elected predecessor artifact with the integrated prediction runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, read, require, save
from superagent.first_delivery import digest

PUBLIC = ROOT / 'research/paired_encoder_search_1127'


def verify(out, historical, archive):
    published = read(PUBLIC / 'corrected_delivery_portable.json')
    historical_score = read(PUBLIC / 'corrected_delivery_comparison.json')
    outcome = read(historical / 'run/agent/run_outcome.json')
    selection = read(historical / 'run/agent/final_selection.json')
    model = historical / 'run/agent' / selection['model_file']
    features = out / 'b/queries.zip'
    require(published['archive_sha256'] == digest(archive)
            and published['model_sha256'] == digest(model) == selection['model_sha256']
            and published['predictions_sha256'] == digest(historical / 'reloaded.csv')
            and digest(features) == read(historical / 'protocol.json')['features_sha256']
            and historical_score['identical_predictions'] and outcome['training_only']
            and outcome['status'] == 'completed' and not outcome['formal_test_accessed'], 'Historical election changed')
    with zipfile.ZipFile(archive) as package:
        source = json.loads(package.read('solution/bundle_manifest.json'))
    source_manifest = historical / 'run/agent/run_manifest.json'
    require(source['source_files'] == read(source_manifest)['source_files'] and
            source['model_sha256'] == selection['model_sha256'], 'Archived training source differs')
    target = out / 'registered/b'
    target.mkdir(parents=True,exist_ok=True)
    result = target / 'reload.csv'
    if not result.exists():
        # Start a new interpreter with the integrated code and no path to the old implementation or feature bank.
        environment = {k: os.environ[k] for k in ('PATH','TMPDIR','TMP','TEMP','SYSTEMROOT','WINDIR') if k in os.environ}
        environment.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))),
                           AB_B1_OFFLINE='1',SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',
                           OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        program = ('from superagent.external_assets import activate_resource_directory; '
                   'from superagent.predict import create_predictions; import sys; '
                   'activate_resource_directory(sys.argv[1]); create_predictions(*sys.argv[2:])')
        with (target/'replay.log').open('w') as log:
            subprocess.run([sys.executable,'-c',program,str(out/'weights'),str(features),
                            str(historical/'run/agent'),str(result)],cwd=ROOT,env=environment,stdout=log,
                           stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True,timeout=600)
    require(digest(result)==published['predictions_sha256'] and len(result.read_bytes().splitlines())-1==published['rows'],
            'Current source did not reproduce the original validation predictions')
    seal = dict(solver=source_identity(ROOT), training_solver=published['runtime_sha256'], model=digest(model),
                spec=selection['task_spec_sha256'], predictions=digest(result),rows=published['rows'],
                training_rows=outcome['training_rows'], family=selection['family'],members=outcome['members'],
                weights=selection['weights'], fits=outcome['attempted_classifier_fits'],seconds=outcome['wall_seconds'],
                original_training_source_manifest=digest(source_manifest), original_archive=digest(archive),
                original_scored_predictions=published['predictions_sha256'],designated_correct=historical_score['carried_correct'],
                new_model_api=0,independent_reload=True,course_test_opened=False)
    save(target/'sealed.json',seal)
    print(json.dumps(seal),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for option in ('out','historical','archive'):
        cli.add_argument('--'+option,type=lambda v:Path(v).resolve(),required=True)
    args=cli.parse_args()
    verify(args.out,args.historical,args.archive)
