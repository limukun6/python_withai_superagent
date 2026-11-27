"""Describe external frozen encoders from actual pipeline objects, and verify their files."""
import hashlib
import os
from pathlib import Path


def activate_resource_directory(directory, environment=None):
    """Expose every verified supported local installation independently of the task and winner."""
    from .ab_sentence import ALTERNATIVES,DEEP_ENV,DEEP_FILES,DEEP_MODEL,FILES,MODEL,pinned_folder
    from .ab_vision import WEIGHT_BYTES,WEIGHT_ENV,WEIGHT_SHA256

    env = os.environ if environment is None else environment
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Encoder resource directory does not exist: {root}")
    text = [("minilm","SUPERAGENT_MINILM_WEIGHTS",MODEL,FILES),("minilm_l12",DEEP_ENV,DEEP_MODEL,DEEP_FILES)]
    text.extend((k,s["env"],s["model"],s["files"]) for k,s in ALTERNATIVES.items())
    for alias,variable,model,files in text:
        if env.get(variable):
            continue
        for relative in dict.fromkeys((model,model.split("/")[-1],alias)):
            location=root/relative
            if location.is_dir() and pinned_folder(location,files) is not None:
                env[variable]=str(location)
                break
    from .local_sentence import discover
    discover(root,env)
    if not env.get(WEIGHT_ENV):
        for location in (root/("mobilenet_v3_small-"+WEIGHT_SHA256[:8]+".pth"),root/"vision"/"weights"):
            if location.is_file() and location.stat().st_size==WEIGHT_BYTES:
                with location.open("rb") as stream:
                    if hashlib.file_digest(stream,"sha256").hexdigest()==WEIGHT_SHA256:
                        env[WEIGHT_ENV]=str(location)
                        break


def model_resources(model):
    from .ab_sentence import (ALTERNATIVES,AlternativeSentenceFeatures,DEEP_ENV, DEEP_FILES, DEEP_MODEL, DEEP_REVISION, DeepPreferenceFeatures,
                              FILES, MODEL, REVISION, SentenceFeatures)
    from .ab_symmetric import UnorderedSentenceFeatures
    from .ab_vision import FrozenVisionFeatures, WEIGHT_ID, WEIGHT_BYTES, WEIGHT_SHA256, WEIGHT_ENV
    from .confusion_specialist import ClassPairModel

    from .local_sentence import LocalSentenceFeatures
    modular={}
    found=set();visited=set()
    def walk(value):
        if id(value) in visited:return
        visited.add(id(value))
        if isinstance(value,LocalSentenceFeatures):modular[value.spec["signature"]]=value.resource()
        elif isinstance(value,AlternativeSentenceFeatures):found.add(value.variant)
        elif isinstance(value,DeepPreferenceFeatures):found.add('minilm_l12')
        elif isinstance(value,(SentenceFeatures,UnorderedSentenceFeatures)):found.add('minilm')
        if isinstance(value,FrozenVisionFeatures):found.add('vision')
        if isinstance(value, ClassPairModel):
            walk(value.base)
            walk(value.expert)
        elif isinstance(value,dict):
            for child in value.values():walk(child)
        elif isinstance(value,(list,tuple)):
            for child in value:walk(child)
        elif callable(getattr(value,'get_params',None)):
            for child in value.get_params(deep=False).values():walk(child)
        # Local vote wrappers store fitted pipelines rather than sklearn parameters.
        for name in ('models','components'):
            children=getattr(value,name,None)
            if isinstance(children,(list,tuple,dict)):walk(children)
    walk(model);resources=[];packages=set()
    if modular:
        resources.extend(modular[k] for k in sorted(modular))
        packages.update(('torch','transformers','tokenizers','safetensors','huggingface-hub'))
    if 'minilm' in found:
        resources.append(dict(name='minilm',environment_variable='SUPERAGENT_MINILM_WEIGHTS',path_kind='directory',
            model=MODEL,revision=REVISION,files={name:dict(bytes=size,sha256=digest) for name,(size,digest) in FILES.items()}))
        packages.update(('torch','transformers','tokenizers','safetensors','huggingface-hub'))
    if 'minilm_l12' in found:
        resources.append(dict(name='minilm_l12',environment_variable=DEEP_ENV,path_kind='directory',
            model=DEEP_MODEL,revision=DEEP_REVISION,
            files={name:dict(bytes=size,sha256=digest) for name,(size,digest) in DEEP_FILES.items()}))
        packages.update(('torch','transformers','tokenizers','safetensors','huggingface-hub'))
    for variant,spec in ALTERNATIVES.items():
        if variant in found:
            resources.append(dict(name=variant,environment_variable=spec['env'],path_kind='directory',
                model=spec['model'],revision=spec['revision'],
                files={n:dict(bytes=size,sha256=h) for n,(size,h) in spec['files'].items()}))
            packages.update(('torch','transformers','tokenizers','safetensors','huggingface-hub'))
    if 'vision' in found:
        resources.append(dict(name='vision',environment_variable=WEIGHT_ENV,path_kind='file',model=WEIGHT_ID,
                              file=dict(bytes=WEIGHT_BYTES,sha256=WEIGHT_SHA256)))
        packages.update(('torch','torchvision'))
    return resources,sorted(packages)


def verify_resources(resources, environment=None):
    environment=os.environ if environment is None else environment
    for resource in resources:
        env=resource['environment_variable'];provided=environment.get(env)
        if not provided:raise ValueError('Missing frozen encoder resource: set '+env)
        root=Path(provided).expanduser().resolve()
        entries=[(root/name,record) for name,record in resource['files'].items()] if resource['path_kind']=='directory' else [(root,resource['file'])]
        for path,record in entries:
            if not path.is_file() or path.stat().st_size!=record['bytes']:
                raise ValueError('Frozen encoder asset size/missing file: '+resource['name'])
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=record['sha256']:raise ValueError('Frozen encoder asset checksum mismatch: '+resource['name'])
