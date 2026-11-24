"""Write a distinct preference-prediction folder from an already frozen native model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import joblib
from sklearn.pipeline import Pipeline

from superagent.ab_preference_output import CanonicalPreferencePipeline
from superagent.ab_run import sha


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output folder for the adapted model')
    selection = json.loads((source/'final_selection.json').read_text())
    schema_file = source/'task_spec.json'
    spec = json.loads(schema_file.read_text())
    schema = spec.get('input_schema', {})
    name = selection['model_file']
    if (selection.get('model_format') != 'candidate_bundle_v1' or selection.get('output_adapter') is not None
            or Path(name).name != name or spec.get('task_kind') not in ('pair_text_binary', 'pair_text_classification')
            or schema.get('pair_semantics') != 'preference' or schema.get('pair_label_one') not in ('left', 'right')):
        raise ValueError('Expected an unadapted native model with explicitly directional binary pair semantics')
    source_file = source/name
    if sha(schema_file) != selection.get('task_spec_sha256') or sha(source_file) != selection.get('model_sha256'):
        raise ValueError('Frozen source model or schema differs from its selection record')
    bundle = joblib.load(source_file)
    model = bundle.get('model')
    if (bundle.get('kind') != 'pair_text' or bundle.get('candidate') != selection['family']
            or type(model) is not Pipeline or set(model.classes_.tolist()) != {0, 1}):
        raise ValueError('Expected the recorded fitted binary pair pipeline')
    options = model.get_params(deep=False)
    wrapped = CanonicalPreferencePipeline(**options)
    if hasattr(model, 'training_metadata'):
        wrapped.training_metadata = model.training_metadata
    adapter = dict(name='canonical_preference_v1',source_model_sha256=selection['model_sha256'],
                   source_schema_sha256=selection['task_spec_sha256'],new_fits=0,
                   selection_score_scope='original trained model',exact_tie='lexicographic first; identical rows choose class 0')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp)/'model'
        staging.mkdir()
        shutil.copy2(schema_file,staging/'task_spec.json')
        joblib.dump(dict(bundle,model=wrapped),staging/name)
        final = dict(selection,model_sha256=sha(staging/name),output_adapter=adapter)
        (staging/'final_selection.json').write_text(json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        if output.exists():
            raise FileExistsError('Output folder appeared during export')
        os.rename(staging, output)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.run_dir,args.output)
