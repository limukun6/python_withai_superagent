"""Copy a source-only, reproducible snapshot of the research worker's local Python imports."""
import argparse
import ast
import hashlib
import json
import tokenize
import zipfile
from collections import deque
from pathlib import Path


ANCHORS=('superagent.minilm_features','superagent.pair_attention','superagent.pretrained_choice',
         'tools.milestone_validation','tools.preference_adapter','tools.token_pooling_study','tools.token_value_head')


def locate(root,name):
    if not name or name.split('.')[0] not in ('superagent','tools'):
        return None
    p=root.joinpath(*name.split('.'))
    for f in (p.with_suffix('.py'),p/'__init__.py'):
        if f.is_file():
            return f
    return None


def module_files(root):
    pending=deque(ANCHORS)
    seen=set()
    files={}
    while pending:
        name=pending.popleft()
        if name in seen:
            continue
        seen.add(name)
        path=locate(root,name)
        if path is None:
            continue
        rel=path.relative_to(root).as_posix()
        files[rel]=path
        parts=name.split('.')
        if len(parts)>1:
            pending.append('.'.join(parts[:-1]))
        package=parts if path.name=='__init__.py' else parts[:-1]
        with tokenize.open(path) as fd:
            tree=ast.parse(fd.read())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                pending.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    prefix=package[:len(package)-node.level+1]
                    base='.'.join([*prefix,*((node.module or '').split('.') if node.module else [])])
                else:
                    base=node.module or ''
                pending.append(base)
                pending.extend(base+'.'+a.name for a in node.names if a.name!='*')
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('legacy_root',type=Path)
    p.add_argument('target_zip',type=Path)
    a=p.parse_args()
    files=module_files(a.legacy_root.resolve())
    if not all(locate(a.legacy_root,name) is not None for name in ANCHORS):
        raise ValueError('Missing required research module')
    hashes={}
    with zipfile.ZipFile(a.target_zip,'x') as archive:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
        info=zipfile.ZipInfo('SOURCE_SHA256.json',date_time=(2020,1,1,0,0,0))
        archive.writestr(info,json.dumps(hashes,sort_keys=True,indent=2))
    print('Python source files:',len(files))
