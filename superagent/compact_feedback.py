"""Lossless references for checked evidence; common observations and unknowns stay unchanged."""
from copy import deepcopy
import json

REF='$evidence_ref'
TABLE='$evidence_table'
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
        if isinstance(value,dict) and (set(value)=={REF} or set(value)=={TABLE}):
            raise ValueError('Reserved evidence reference marker in original checks')
        key=_json(value);path=index.get(key)
        if path is not None and len(_json({REF:path}).encode())+12<len(key.encode()):return {REF:path}
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,list):
            plain=[encode(v) for v in value]
            if len(value)>=2 and all(isinstance(v,dict) and set(v)==set(value[0]) for v in value):
                columns=list(value[0]);table={TABLE:dict(columns=columns,rows=[[encode(v[k]) for k in columns] for v in value])}
                if len(_json(table).encode())+32<len(_json(plain).encode()):return table
            return plain
        return deepcopy(value)
    for name in EXTRA:
        # Refuse marker collisions even when an enclosing object could be referenced.
        if any(isinstance(v,dict) and (set(v)=={REF} or set(v)=={TABLE}) for v,_ in _walk(full[name],'/'+name)):
            raise ValueError('Reserved evidence reference marker in original checks')
        base[name]=encode(full[name])
        if name=='checks':register(full[name],'/checks')
    base['evidence_projection']=dict(version=1,format='lossless_references',
        instruction='Resolve $evidence_ref JSON pointers within this context. $evidence_table rows use their ordered columns. References reuse original evidence; '
                    'they do not certify hypotheses or unknown claims. Common fields are unchanged.')
    if expand_context(base)!=full:raise ValueError('Compact evidence failed exact round-trip validation')
    return base


def expand_context(compact):
    if compact.get('evidence_projection',{}).get('format')!='lossless_references':
        raise ValueError('Unsupported compact feedback format')
    common={k:deepcopy(v) for k,v in compact.items() if k not in (*EXTRA,'evidence_projection')}
    def expand(value,active=()):
        if isinstance(value,dict) and set(value)=={TABLE}:
            table=value[TABLE]
            if not isinstance(table,dict) or set(table)!={'columns','rows'}:raise ValueError('Malformed evidence table')
            columns,rows=table['columns'],table['rows']
            if (not isinstance(columns,list) or any(not isinstance(c,str) for c in columns) or
                len(set(columns))!=len(columns) or not isinstance(rows,list) or
                any(not isinstance(row,list) or len(row)!=len(columns) for row in rows)):
                raise ValueError('Evidence table shape or columns differ')
            return [{k:expand(v,active) for k,v in zip(columns,row)} for row in rows]
        if isinstance(value,dict) and set(value)=={REF}:
            path=value[REF]
            if not isinstance(path,str) or not path.startswith('/') or path in active:
                raise ValueError('Malformed or cyclic evidence reference')
            parts=[p.replace('~1','/').replace('~0','~') for p in path[1:].split('/')]
            if not parts or parts[0] not in (*common,'checks'):raise ValueError('Evidence reference outside allowed context')
            # The pointer addresses the original logical tree, even when its checks
            # list is physically encoded as a column table in the compact object.
            current=expand(compact['checks'],(*active,path)) if parts[0]=='checks' else compact[parts[0]]
            for part in parts[1:]:
                if isinstance(current,list):
                    if not part.isdigit() or str(int(part))!=part:raise ValueError('Invalid evidence array index')
                    current=current[int(part)]
                elif isinstance(current,dict):current=current[part]
                else:raise ValueError('Evidence pointer traverses a scalar')
            return deepcopy(current)
        if isinstance(value,dict):return {k:expand(v,active) for k,v in value.items()}
        if isinstance(value,list):return [expand(v,active) for v in value]
        return deepcopy(value)
    return {**common,**{name:expand(compact[name]) for name in EXTRA}}
