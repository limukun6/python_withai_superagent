"""Lossless references for checked evidence; common observations and unknowns stay unchanged."""
from copy import deepcopy
import json

REF='$evidence_ref'
EXTRA=('checks','memory')


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _escape(part):return str(part).replace('~','~0').replace('/','~1')


def _walk(value,path):
    yield value,path
    if isinstance(value,dict):
        for k,v in value.items():yield from _walk(v,path+'/'+_escape(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from _walk(v,path+'/'+str(i))


def compact_context(full):
    if any(k not in full for k in EXTRA) or 'evidence_projection' in full:
        raise ValueError('Expected an unprojected checked context')
    base={k:deepcopy(v) for k,v in full.items() if k not in EXTRA}
    index={}
    def register(value,path):
        for v,p in _walk(value,path):
            if isinstance(v,(dict,list,str)) and len(_json(v).encode())>=48:
                key=_json(v)
                if key not in index or len(p)<len(index[key]):index[key]=p
    for k,v in base.items():register(v,'/'+_escape(k))
    def encode(value):
        if isinstance(value,dict) and set(value)=={REF}:raise ValueError('Reserved evidence reference marker in original checks')
        key=_json(value);path=index.get(key)
        if path is not None and len(_json({REF:path}).encode())+12<len(key.encode()):return {REF:path}
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,list):return [encode(v) for v in value]
        return deepcopy(value)
    for name in EXTRA:
        # Refuse marker collisions even when an enclosing object could be referenced.
        if any(isinstance(v,dict) and set(v)=={REF} for v,_ in _walk(full[name],'/'+name)):
            raise ValueError('Reserved evidence reference marker in original checks')
        base[name]=encode(full[name])
        if name=='checks':register(full[name],'/checks')
    base['evidence_projection']=dict(version=1,format='lossless_references',
        instruction='Resolve $evidence_ref JSON pointers within this context. References reuse original evidence; '
                    'they do not certify hypotheses or unknown claims. Common fields are unchanged.')
    if expand_context(base)!=full:raise ValueError('Compact evidence failed exact round-trip validation')
    return base


def expand_context(compact):
    if compact.get('evidence_projection',{}).get('format')!='lossless_references':
        raise ValueError('Unsupported compact feedback format')
    common={k:deepcopy(v) for k,v in compact.items() if k not in (*EXTRA,'evidence_projection')}
    def expand(value,active=()):
        if isinstance(value,dict) and set(value)=={REF}:
            path=value[REF]
            if not isinstance(path,str) or not path.startswith('/') or path in active:
                raise ValueError('Malformed or cyclic evidence reference')
            parts=[p.replace('~1','/').replace('~0','~') for p in path[1:].split('/')]
            if not parts or parts[0] not in (*common,'checks'):raise ValueError('Evidence reference outside allowed context')
            current=compact
            for part in parts:
                if isinstance(current,list):
                    if not part.isdigit() or str(int(part))!=part:raise ValueError('Invalid evidence array index')
                    current=current[int(part)]
                elif isinstance(current,dict):current=current[part]
                else:raise ValueError('Evidence pointer traverses a scalar')
            return expand(current,(*active,path)) if parts[0]=='checks' else deepcopy(current)
        if isinstance(value,dict):return {k:expand(v,active) for k,v in value.items()}
        if isinstance(value,list):return [expand(v,active) for v in value]
        return deepcopy(value)
    return {**common,**{name:expand(compact[name]) for name in EXTRA}}
