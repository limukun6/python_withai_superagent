"""SA24 controlled image-design conditions; no prediction answers are exported."""
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image,ImageOps
from sklearn.datasets import load_digits

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.sa8_real_delivery_cases import partition
from scripts.sa14_native_cases import output


def build(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    data=load_digits(n_class=3);fit,query=partition(data.images,data.target,90,30)
    pixels=np.rint(data.images/16*255).astype(np.uint8);classes=['amber','cobalt','jade']
    train={};future={};labels={};dimensions=[]
    def bitmap(index,position):
        size=[(8,8),(10,13),(16,12)][position%3]
        image=Image.fromarray(pixels[index]).convert('RGB').resize(size,Image.Resampling.BILINEAR)
        stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue(),list(size)
    for i,index in enumerate(fit):
        label=classes[int(data.target[index])];member=f'gallery/{label}/{i:04d}.png'
        train[member],size=bitmap(index,i);labels[member]=label;dimensions.append(size)
    for i,index in enumerate(query):future[f'incoming/{i:04d}.png']=bitmap(index,i)[0]
    common=('Classify the images. Only gallery/ supplies supervised training images; the immediate class '
        'subdirectory is the target, with unchanged class names amber, cobalt, jade. The separate prediction '
        'package supplies unlabeled images under incoming/. Image dimensions vary. Use macro F1 as the primary '
        'metric. Export UTF-8 comma-separated CSV with header Row,Class. Row starts at 1 in ascending full '
        'prediction image path order. Class contains the unchanged class directory name. No original record '
        'IDs, label sidecars or prediction answers are supplied. ')
    registry=[];recipe=output();recipe['columns'][1]['name']='Class'
    for name,extra,constraints in [
        ('F','The task imposes no image preprocessing dimensions or color conversion. Choose the internal '
             'representation autonomously within the execution budget, and freeze it for prediction.',{}),
        ('P','Required preprocessing: the final image width must be 12 pixels and the final color mode must '
             'be RGB. Height and other internal representation settings are not specified; choose them '
             'autonomously within the execution budget and freeze them for prediction.',dict(width=12,mode='RGB'))]:
        for path,members in [(root/(name+'.zip'),dict(train,**{'README.md':common+extra})),
                             (root/(name+'.query.zip'),future)]:
            with zipfile.ZipFile(path,'w') as z:
                for member,value in members.items():z.writestr(member,value)
        oracle=dict(kind='image',metric='macro_f1',classes=classes,training_rows=90,prediction_rows=30,
            training_members=sorted(train),training_labels=[labels[n] for n in sorted(train)],
            prediction_members=sorted(future),task_constraints=constraints,output=recipe,
            source='sklearn.load_digits(n_class=3), previously used public development source',
            source_indices=dict(training=fit.tolist(),query=query.tolist()),original_dimensions=sorted(set(map(tuple,dimensions))),
            prediction_answers_exported=False)
        save(root/(name+'.oracle.json'),oracle)
        registry.append(dict(id=name,package=name+'.zip',query=name+'.query.zip',
            package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),
            oracle_sha256=sha(root/(name+'.oracle.json')),training_rows=90,prediction_rows=30))
    save(root/'registry.json',registry);return registry


def independent_pixels(package,members,chosen):
    """Reconstruct from frozen choices, independently of production's image reader."""
    rows=[]
    with zipfile.ZipFile(package) as z:
        for member in members:
            with Image.open(io.BytesIO(z.read(member))) as original:
                rgba=ImageOps.exif_transpose(original).convert('RGBA')
                composed=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba)
                image=composed.convert(chosen['mode']).resize((chosen['width'],chosen['height']),Image.Resampling.BILINEAR)
                rows.append(np.array(image,dtype=np.uint8))
    return np.stack(rows)


def grade(package,query,proposal,spec,oracle):
    from superagent.format_sources import package_files
    from superagent.semantic_contract import evidence_workspace
    from superagent.native_semantics import compile_native
    from superagent.data import load_prediction_features
    from superagent.metrics import MetricSpec
    from scripts.sa7_multiturn_cases import array_fingerprint
    from tools.submission_equivalence import equivalent_csv_recipe
    with package_files(package) as files,package_files(query) as other:
        task=compile_native(files,evidence_workspace(files),proposal,prediction_files=other)
    chosen=spec['native_input_recipe']['image_preprocessing'];metric=MetricSpec.for_task(task)
    x=independent_pixels(package,oracle['training_members'],chosen)
    q=independent_pixels(query,oracle['prediction_members'],chosen)
    observed=load_prediction_features(query,task.kind,spec)
    decisions=spec.get('design_decisions',{}).get('image_preprocessing')
    checks=dict(kind=task.kind==oracle['kind'],metric=metric.name==oracle['metric'],
        classes=list(metric.labels)==oracle['classes'],training_rows=len(task.y_train)==oracle['training_rows'],
        training_pixels=np.array_equal(x,task.X_train),training_labels=task.y_train.tolist()==oracle['training_labels'],
        prediction_pixels=np.array_equal(q,observed),prediction_rows=len(q)==oracle['prediction_rows'],
        task_constraints=all(chosen[k]==v and type(chosen[k]) is type(v) for k,v in oracle['task_constraints'].items()),
        frozen_choices=task.spec['native_input_recipe']==spec['native_input_recipe'],
        output=equivalent_csv_recipe(spec.get('output_recipe'),oracle['output'],oracle['classes']))
    provenance=None
    if decisions:
        expected={k:'task_requirement' if k in oracle['task_constraints'] else 'agent_design_choice' for k in chosen}
        provenance=dict(extracted_constraints=decisions['task_constraints']==oracle['task_constraints'],
            origins=decisions['parameter_origins']==expected,chosen=decisions['chosen']==chosen,
            limits_explicit=decisions['semantic_requirement_extraction_certified'] is False)
    return dict(complete=all(checks.values()),checks=checks,chosen=chosen,design_record=decisions,
        provenance_checks=provenance,provenance_correct=provenance is not None and all(provenance.values()),
        training_features_sha256=array_fingerprint(x),prediction_features_sha256=array_fingerprint(q),
        interpretation_certification='oracle checks these two documents only, not arbitrary requirement extraction')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(build(p.parse_args().output))
