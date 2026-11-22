"""Copy a frozen model and its exact project source into a standalone prediction bundle."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


ROOT=Path(__file__).resolve().parents[1]
ENTRY='''"""Offline frozen prediction; source is loaded from this bundle only."""
import argparse, hashlib, importlib.metadata, json, pathlib, socket, sys
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Prediction bundle integrity check failed')
for name,version in manifest['dependencies'].items():
    if importlib.metadata.version(name)!=version:raise ValueError('Dependency version differs: '+name)
def offline(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network forbidden')
sys.addaudithook(offline)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC,LinearSVC
def no_fit(*a,**kw):raise RuntimeError('Fitting forbidden during frozen prediction')
for cls in (LogisticRegression,ExtraTreesClassifier,SVC,LinearSVC):cls.fit=no_fit
try:
    from catboost import CatBoostClassifier
    CatBoostClassifier.fit=no_fit
except ImportError:pass
sys.path.insert(0,str(root/'src'))
import superagent
if not pathlib.Path(superagent.__file__).resolve().is_relative_to(root/'src'):
    raise ValueError('Original repository source leaked into the isolated process')
from superagent.predict import create_predictions
p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
create_predictions(a.features,root,a.output)
print(json.dumps(dict(status='predicted',project_source_from_bundle=True,training_performed=False,network_forbidden=True)))
'''


def export(run_dir,output):
    run_dir=Path(run_dir).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    selected=read(run_dir/'final_selection.json');manifest=read(run_dir/'run_manifest.json')
    if manifest['source_files']!=sources():raise ValueError('Use the exact model-training project source for export')
    if Path(selected['model_file']).name!=selected['model_file']:raise ValueError('Expected a model filename inside the run')
    if sha(run_dir/selected['model_file'])!=selected['model_sha256']:raise ValueError('Model digest differs')
    bundle=output/'bundle';bundle.mkdir();(bundle/'src/superagent').mkdir(parents=True)
    for path in sorted((ROOT/'superagent').rglob('*.py')):
        target=bundle/'src/superagent'/path.relative_to(ROOT/'superagent');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in ('task_spec.json','final_selection.json',selected['model_file']):shutil.copy2(run_dir/name,bundle/name)
    (bundle/'predict.py').write_text(ENTRY,encoding='utf-8')
    dependencies={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','joblib','Pillow','requests')}
    # Some selected models pickle CatBoost objects; include it when installed, not a downloaded encoder.
    try:dependencies['catboost']=importlib.metadata.version('catboost')
    except importlib.metadata.PackageNotFoundError:pass
    (bundle/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in dependencies.items()),encoding='utf-8')
    (bundle/'README.md').write_text('Frozen prediction package. Install requirements.txt, then run:\n\n'
        'python -I predict.py --features TASK_FEATURES.zip --output predictions.csv\n\n'
        'No API key, training input, ledger, or network is required for prediction. '
        'The recorded partial/full model status is preserved in final_selection.json.\n',encoding='utf-8')
    files={p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    save(bundle/'bundle_manifest.json',dict(bundle_version=1,source_files=manifest['source_files'],files=files,dependencies=dependencies,
        model_sha256=selected['model_sha256'],delivery_status=selected.get('delivery_status','complete' if selected['refit'] else 'partial'),
        source_run_manifest_sha256=sha(run_dir/'run_manifest.json'),classifier_fits=0,API_requests=0))
    archive=Path(shutil.make_archive(str(output/'model_bundle'),'zip',root_dir=bundle))
    save(output/'completion.json',dict(status='exported',archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        delivery_status=read(bundle/'bundle_manifest.json')['delivery_status'],training_data_copied=False,credential_copied=False,
        project_source_included=True,actual_classifier_fits=0,real_api_requests=0))
    return bundle


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    export(a.run_dir,a.output)
