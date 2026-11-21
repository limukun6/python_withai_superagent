"""Copy text/JSON study evidence, omitting task data, images, worker models and keys."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inventory=[]
    def copy(run,name):
        src=a.runs/run/name
        if not src.is_file():return
        dst=a.output/run/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        inventory.append(dict(file=dst.relative_to(a.output).as_posix(),sha256=sha(dst),bytes=dst.stat().st_size))
    for run in ('a17_catalog_checks','a19_readme_checks','a19_final_readme_checks','a19_conservative_final_checks'):
        for file in ('protocol.json','completion.json','tests.log'):copy(run,file)
    for file in ('protocol.json','completion.json','results.json','all_models_frozen.json','physical_fits.jsonl'):
        copy('a17_photo_catalog_Q',file)
    for scope in ('baseline','full'):
        for file in ('run_manifest.json','run_outcome.json','actions.json','history.json','classifier_fits.json',
                     'final_selection.json','verification.json'):
            copy('a17_photo_catalog_Q',scope+'/'+file)
    for run in ('a18_c1_readiness','a18_c1_readiness_declared','a18_c1_readiness_final','a18_c1_readiness_audited'):
        # Superseded commands are intentionally not exported as executable current matrix.
        for file in ('protocol.json','completion.json'):copy(run,file)
    run='a18_c1_readiness_current'
    for file in ('protocol.json','completion.json','readiness.json','environment.txt'):copy(run,file)
    save(a.output/'inventory.json',dict(files=inventory,
        note='Raw local byte digests; Git CRLF normalization may change clone bytes. Binary data/models excluded.'))
    print(dict(files=len(inventory),bytes=sum(x['bytes'] for x in inventory)))
