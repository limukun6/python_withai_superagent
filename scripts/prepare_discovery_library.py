"""Complete declarative resources from public model publishers and record the immutable bytes."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from scripts.unified_released_study import require
from tools.encoder_assets import matches


def read_public(url, path):
    subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
                    '--max-time','420','--output',str(path),url],check=True)


def pinned_resource(value):
    repo, separator, revision = value.partition('@')
    if not separator or not re.fullmatch(r'[\w-][\w.-]*/[\w-][\w.-]*',repo,re.ASCII) or not re.fullmatch('[0-9a-f]{40}',revision):
        raise argparse.ArgumentTypeError('Use an owner/model identifier followed by @ and the full immutable revision')
    return repo, revision


def prepare(weights, expanded, record, requested_resources=()):
    expanded.mkdir(parents=True,exist_ok=True)
    for source in weights.rglob('*'):
        if not source.is_file() or any(p.startswith('.') for p in source.relative_to(weights).parts):
            continue
        target = expanded/source.relative_to(weights)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            os.link(source,target)
    revisions = dict(requested_resources) if requested_resources else {'sentence-transformers/all-mpnet-base-v2':'e8c3b32edf5434bc2275fc9bab85f82640a19130',
                 'sentence-transformers/all-distilroberta-v1':'842eaed40bee4d61673a81c92d5689a8fed7a09f',
                 'sentence-transformers/paraphrase-MiniLM-L3-v2':'4ca70771034acceecb2e72475f72050fcdde4ddc'}
    result = json.loads(record.read_text()) if record.exists() else {}
    for repo,requested in revisions.items():
        use = result.get(repo,{}).get('revision',requested)
        require(use==requested,'Existing receipt belongs to a different revision')
        with NamedTemporaryFile() as file:
            read_public(f'https://huggingface.co/api/models/{repo}/revision/{use}?blobs=true',Path(file.name))
            meta = json.loads(Path(file.name).read_text())
        revision = meta['sha']
        require(meta['id']==repo and len(revision)==40 and (use=='main' or revision==use),'Invalid publisher revision')
        present = {item['rfilename']:item for item in meta['siblings']}
        required = {'modules.json','config.json','tokenizer.json','model.safetensors'}
        # Older publishers store contextual pooling settings in separate files. Static
        # embedding bags instead declare their single table in the first module.
        if '1_Pooling/config.json' in present or 'sentence_bert_config.json' in present:
            required.update(('1_Pooling/config.json','sentence_bert_config.json'))
        else:
            with NamedTemporaryFile() as file:
                path=Path(file.name)
                read_public(f'https://huggingface.co/{repo}/resolve/{revision}/modules.json',path)
                declaration=present.get('modules.json')
                require(declaration and matches(path,(declaration['size'],declaration['blobId'])),'Missing valid module declaration')
                stages=json.loads(path.read_text())
            require([r.get('type') for r in stages] in (
                ['sentence_transformers.models.StaticEmbedding'],
                ['sentence_transformers.models.StaticEmbedding','sentence_transformers.models.Normalize']),
                'Unsupported published module declaration')
            require(stages[0].get('path') in ('','.'),'This installer expects root-level weights')
        require(required<=present.keys(),'Incomplete model metadata')
        names = sorted(required | ({'config_sentence_transformers.json','tokenizer_config.json'} & present.keys()))
        folder = expanded / repo.split('/')[-1]
        entries = {}
        for name in names:
            info = present[name]
            dig = info.get('lfs',{}).get('sha256',info['blobId'])
            spec = (info['size'],dig)
            local = folder/name
            local.parent.mkdir(parents=True,exist_ok=True)
            if not matches(local,spec):
                require(not local.exists(),'Will not overwrite an earlier pinned installation')
                with NamedTemporaryFile(dir=local.parent,delete=False) as temporary:
                    temp = Path(temporary.name)
                try:
                    read_public(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',temp)
                    require(matches(temp,spec),'Publisher checksum mismatch')
                    os.link(temp,local)
                finally:
                    temp.unlink(missing_ok=True)
            with local.open('rb') as stream:
                entries[name] = {'bytes':info['size'],'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
        observation = dict(revision=revision,files=entries)
        require(repo not in result or result[repo]==observation,'Installation record changed')
        result[repo] = observation
        temp = record.with_suffix('.pending')
        temp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        temp.replace(record)
        print(repo,revision,len(entries),'files verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('weights','expanded','record'):
        parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--resource',action='append',type=pinned_resource,default=[],
                        help='Public Sentence Transformers publisher/model@commit to add to the shared library')
    args=parser.parse_args()
    prepare(args.weights,args.expanded,args.record,args.resource)
