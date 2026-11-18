"""Persistent, protocol-bound evidence and an anytime search scheduler."""
from copy import deepcopy
import math
import statistics

from .local_search import LocalSearchPolicy,confirmation_shortlist


class ConfirmationMemory:
    def __init__(self,protocol_id,seeds):
        if not protocol_id or len(set(seeds))!=len(seeds) or len(seeds)<2:
            raise ValueError('A named evaluation protocol and distinct split seeds are required')
        self.protocol_id=protocol_id
        self.seeds=tuple(seeds)
        self.records={}

    def add(self,candidate,seed,status,accuracy=None):
        if seed not in self.seeds:raise ValueError('Split is not part of this protocol')
        key=(candidate,seed)
        if key in self.records:raise ValueError('An earlier attempt cannot be overwritten or silently retried')
        if status=='completed':
            if type(accuracy) not in (int,float) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
                raise ValueError('Completed accuracy must be finite and between zero and one')
        elif accuracy is not None:raise ValueError('A failed attempt cannot supply a score')
        self.records[key]={'status':status,'accuracy':accuracy}

    def means(self):
        result={}
        for candidate in sorted({key[0] for key in self.records}):
            rows=[self.records.get((candidate,seed)) for seed in self.seeds]
            if all(row and row['status']=='completed' for row in rows):
                result[candidate]=statistics.mean(row['accuracy'] for row in rows)
        return result

    def incumbent(self):
        means=self.means()
        if not means:return None,None
        candidate=min(means,key=lambda cid:(-means[cid],cid))
        return candidate,means[candidate]

    def snapshot(self):
        return {'protocol_id':self.protocol_id,'seeds':list(self.seeds),
                'records':[{'candidate':c,'seed':s,**deepcopy(r)} for (c,s),r in self.records.items()]}

    @classmethod
    def restore(cls,state,expected_protocol_id):
        if state['protocol_id']!=expected_protocol_id:raise ValueError('Cannot mix different evaluation protocols')
        memory=cls(state['protocol_id'],state['seeds'])
        for row in state['records']:memory.add(row['candidate'],row['seed'],row['status'],row['accuracy'])
        return memory


def _tuples(value):return tuple(_tuples(v) for v in value) if isinstance(value,list) else value


class PersistentSearch:
    """Scheduler has no access to unrequested results or any external labels."""
    def __init__(self,catalog,policy_name,rng_seed,seeds,primary_seed,protocol_id):
        if primary_seed not in seeds:raise ValueError('Primary seed must belong to evaluation protocol')
        self.catalog=deepcopy(catalog)
        self.policy_name,self.rng_seed=policy_name,rng_seed
        self.policy=LocalSearchPolicy(policy_name,catalog,rng_seed)
        self.primary_seed=primary_seed
        self.memory=ConfirmationMemory(protocol_id,seeds)
        self.history=[]
        self.pending=[]
        self.active=None
        self.query_count=0

    def next_query(self):
        if self.active is not None:return deepcopy(self.active)
        if self.pending:self.active=self.pending.pop(0)
        elif len(self.history)<len(self.catalog):
            chosen,reason=self.policy.choose(self.history)
            self.active={'stage':'primary','candidate':chosen,'seed':self.primary_seed,'reason':reason}
        else:return None
        return deepcopy(self.active)

    def observe(self,status,accuracy=None):
        if self.active is None:raise ValueError('No outstanding query')
        query=self.active
        self.memory.add(query['candidate'],query['seed'],status,accuracy)
        self.query_count+=1
        if query['stage']=='primary':
            route=next(r['route'] for r in self.catalog if r['id']==query['candidate'])
            self.history.append({'candidate':query['candidate'],'route':route,'status':status,
                'search_accuracy':accuracy,'reason':query['reason']})
            shortlist=confirmation_shortlist(self.history)
            confirm=shortlist if len(self.history)==5 else (
                [query['candidate']] if len(self.history)>5 and query['candidate'] in shortlist else [])
            self.pending.extend({'stage':'confirmation','candidate':candidate,'seed':seed,'reason':'complete_fixed_protocol'}
                for candidate in confirm for seed in self.memory.seeds if (candidate,seed) not in self.memory.records)
        self.active=None
        return self.memory.incumbent()

    def snapshot(self):
        return {'catalog':deepcopy(self.catalog),'policy_name':self.policy_name,'rng_seed':self.rng_seed,
            'rng_state':self.policy.rng.getstate(),'primary_seed':self.primary_seed,'memory':self.memory.snapshot(),
            'history':deepcopy(self.history),'pending':deepcopy(self.pending),'active':deepcopy(self.active),'query_count':self.query_count}

    @classmethod
    def restore(cls,state,expected_protocol_id):
        memory=ConfirmationMemory.restore(state['memory'],expected_protocol_id)
        result=cls(state['catalog'],state['policy_name'],state['rng_seed'],memory.seeds,state['primary_seed'],expected_protocol_id)
        result.policy.rng.setstate(_tuples(state['rng_state']))
        result.memory=memory
        result.history=deepcopy(state['history']);result.pending=deepcopy(state['pending']);result.active=deepcopy(state['active'])
        result.query_count=state['query_count']
        if result.query_count!=len(memory.records):raise ValueError('Checkpoint query count does not match evidence')
        return result
