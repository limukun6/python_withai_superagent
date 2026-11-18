"""Bounded parameter interpolation from observed trials; no generated Python code."""
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from .metrics import score_of,metric_of
from pathlib import Path

from .models import Candidate,search_candidates
from .evidence_agent import route_of

MAX_GENERATED=4
PER_STEP=2
ALLOWED_STRATEGIES=('budget_routes','local_greedy_routes','local_protected_routes','local_interleaved_routes')


def _positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0
def _same(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=0.)
def _without_c(params):return {k:v for k,v in params.items() if k!='C'}


def make_spec(kind,portfolio,template_id,c):
    if portfolio not in ('original','expanded'):raise ValueError('Refinement supports original or expanded portfolios only')
    if not _positive(c):raise ValueError('Refined C must be a positive finite number, not a boolean')
    catalog=search_candidates(kind,portfolio);template=next((item for item in catalog if item.family==template_id),None)
    if template is None or not _positive(template.params.get('C')) or not isinstance(template.fit_predict,partial) or 'c' not in template.fit_predict.keywords:
        raise ValueError('Template must be an existing static C-parameterized candidate')
    route=route_of(template.family)
    compatible=[item for item in catalog if route_of(item.family)==route and _without_c(item.params)==_without_c(template.params)]
    values=[item.params['C'] for item in compatible]
    c=float(f'{c:.12g}')
    if not min(values)<c<max(values):raise ValueError('Refinement must remain strictly inside the original route bounds')
    if any(_same(c,value) for value in values):raise ValueError('Refinement duplicates an existing static configuration')
    canonical=min(compatible,key=lambda item:(item.params['C'],item.family))
    return {'kind':kind,'portfolio':portfolio,'template_id':canonical.family,'C':c}


def materialize(kind,spec):
    if not isinstance(spec,dict) or set(spec)!={'kind','portfolio','template_id','C'} or spec['kind']!=kind:
        raise ValueError('Invalid generated candidate specification')
    canonical=make_spec(kind,spec['portfolio'],spec['template_id'],spec['C'])
    if canonical!=spec:raise ValueError('Generated candidate specification is not canonical')
    template=next(c for c in search_candidates(kind,spec['portfolio']) if c.family==spec['template_id'])
    params={**template.params,'C':spec['C']};route=route_of(template.family)
    identity={'kind':kind,'template':spec['template_id'],'params':params}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    return Candidate(f'{route}_refined_{digest}',params,partial(template.fit_predict,c=spec['C']))


def resolve_registered(kind,candidate_id,document):
    if isinstance(document,dict) and document.get('schema_version')==3:
        from .generated_pipeline import resolve_generated_registry
        return resolve_generated_registry(kind,candidate_id,document)
    if isinstance(document,dict) and document.get("schema_version")==2:
        from .recipe_registry import resolve_recipe_registry
        return resolve_recipe_registry(kind,candidate_id,document)
    if not isinstance(document,dict) or set(document)!={'schema_version','kind','portfolio','entries'} or document['schema_version']!=1 or document['kind']!=kind:
        raise ValueError('Invalid candidate registry schema or data kind')
    if document['portfolio'] not in ('original','expanded') or not isinstance(document['entries'],dict) or len(document['entries'])>MAX_GENERATED:
        raise ValueError('Invalid candidate registry portfolio or size')
    generated={}
    for key,entry in document['entries'].items():
        if not isinstance(entry,dict) or set(entry)!={'spec','params','origin'} or not isinstance(entry['spec'],dict) or entry['spec'].get('portfolio')!=document['portfolio']:
            raise ValueError('Invalid candidate registry entry')
        candidate=materialize(kind,entry['spec'])
        if candidate.family!=key or candidate.params!=entry['params']:raise ValueError('Generated candidate ID or parameters changed')
        generated[key]=candidate
    if candidate_id in generated:return generated[candidate_id]
    static=next((c for c in search_candidates(kind,'all') if c.family==candidate_id),None)
    if static is None:raise ValueError('Candidate is absent from both static and generated catalogs')
    return static


class CandidateRegistry:
    def __init__(self,kind,portfolio,path):
        if portfolio not in ('original','expanded'):raise ValueError('Refinement supports original or expanded portfolios only')
        self.kind,self.portfolio,self.path=kind,portfolio,Path(path)
        if self.path.exists():raise FileExistsError('Do not overwrite an existing generated-candidate registry')
        self.catalog=search_candidates(kind,portfolio)
        self.document={'schema_version':1,'kind':kind,'portfolio':portfolio,'entries':{}}
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.document,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(self.path)

    def propose(self,history):
        if len(self.document['entries'])>=MAX_GENERATED:return []
        static={c.family:c for c in self.catalog}
        registered={key:materialize(self.kind,row['spec']) for key,row in self.document['entries'].items()}
        known={**static,**registered};groups={}
        for row in history:
            if row.get('status')!='completed' or not _positive(row.get('params',{}).get('C')):
                continue
            score=score_of(row)
            if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:continue
            candidate=known.get(row['candidate'])
            if candidate is None or candidate.params!=row['params']:raise ValueError('Observed configuration disagrees with its catalog')
            if not isinstance(candidate.fit_predict,partial) or 'c' not in candidate.fit_predict.keywords:continue
            groups.setdefault(route_of(row['candidate']),[]).append(row)
        routes=sorted(groups,key=lambda route:(-max(score_of(r) for r in groups[route]),route))
        added=[]
        for route in routes:
            observed=groups[route]
            if len({r['params']['C'] for r in observed})<2:continue
            best=min(observed,key=lambda r:(-score_of(r),r['trial']))
            center=best['params']['C'];neighbors=[]
            lower=[r for r in observed if r['params']['C']<center]
            upper=[r for r in observed if r['params']['C']>center]
            if lower:neighbors.append(max(lower,key=lambda r:r['params']['C']))
            if upper:neighbors.append(min(upper,key=lambda r:r['params']['C']))
            template=next(c for c in self.catalog if route_of(c.family)==route and _without_c(c.params)==_without_c(best['params']))
            for neighbor in neighbors:
                value=float(f'{math.sqrt(center*neighbor["params"]["C"]):.12g}')
                if any(route_of(c.family)==route and _positive(c.params.get('C')) and _same(value,c.params['C']) for c in known.values()):continue
                try:spec=make_spec(self.kind,self.portfolio,template.family,value)
                except ValueError:continue
                candidate=materialize(self.kind,spec)
                origin={'rule':'geometric_midpoint_of_observed_same_route_C','after_trial':history[-1]['trial'],
                    'reference_trial':best['trial'],'neighbor_trial':neighbor['trial'],'route':route,
                    'primary_metric':metric_of(history),'reference_score':score_of(best),'neighbor_score':score_of(neighbor),
                    'reference_accuracy':best['search_accuracy'],'neighbor_accuracy':neighbor['search_accuracy'],
                    'C_endpoints':[center,neighbor['params']['C']]}
                entry={'spec':spec,'params':candidate.params,'origin':origin}
                self.document['entries'][candidate.family]=deepcopy(entry);known[candidate.family]=candidate
                added.append({'id':candidate.family,**deepcopy(entry)})
                if len(added)>=PER_STEP or len(self.document['entries'])>=MAX_GENERATED:break
            if len(added)>=PER_STEP or len(self.document['entries'])>=MAX_GENERATED:break
        if added:self.save()
        return added

    def description(self):
        return {'enabled':True,'mode':'midpoints','parameter':'C','bounds':'strictly inside static route range',
            'requires_completed_same_route_configurations':2,'maximum_generated_total':MAX_GENERATED,'maximum_generated_per_step':PER_STEP,
            'generated_candidates':len(self.document['entries']),'adds_training_attempts':False,'adds_api_requests':False,
            'not_supported_as_blind_ablation':True}
