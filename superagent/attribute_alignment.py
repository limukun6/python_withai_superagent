"""Learn which comparable categorical field is informative in each observed context."""
import hashlib
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import GroupKFold

from .field_interactions import choose_pairs, rank_pairs
from .joint_context import ContextTokenTree, blocks, tokens
from .models import _clean_categories, _tabular_catboost
from .splits import sample_groups


def column_identity(values):
    digest=hashlib.sha256()
    for value in values:
        encoded=str(value).encode("utf8")
        digest.update(len(encoded).to_bytes(4,"big"))
        digest.update(encoded)
    return digest.hexdigest()


def comparable_fields(rows,columns):
    levels={j:set(rows[:,j])-{"__MISSING__"} for j in columns}
    available=[j for j in columns if 3<=len(levels[j])<=40]
    available.sort(key=lambda j:column_identity(rows[:,j]))
    families=[]
    for j in available:
        for family in families:
            if all(len(levels[j]&levels[k])/len(levels[j]|levels[k])>=.75 for k in family):
                family.append(j)
                break
        else:
            families.append([j])
    return [family for family in families if len(family)>=2]


def information(values,labels,num_classes):
    """Bias-corrected mutual information with a small-sample support discount."""
    _,codes=np.unique(values,return_inverse=True)
    n=len(labels)
    count=np.bincount(codes*num_classes+labels,minlength=(int(codes.max())+1)*num_classes)
    counts=count.reshape(-1,num_classes)
    totals=counts.sum(0,keepdims=True)
    expected=counts.sum(1,keepdims=True)*totals/n
    keep=counts>0
    value=np.sum(counts[keep]*np.log(counts[keep]/expected[keep]))/n
    bias=(len(counts)-1)*(np.count_nonzero(totals)-1)/(2*n)
    return float(max(0,value-bias)*n/(n+32))


def fitted_routes(rows,y,families,contexts):
    labels,target=np.unique(y,return_inverse=True)
    result=[]
    context_order=sorted(contexts,key=lambda j:column_identity(rows[:,j]))
    for family in families:
        common=[information(rows[:,j],target,len(labels)) for j in family]
        default=family[int(np.argmax(common))]
        best_gain=0.
        chosen=(None,{},default)
        for gate in context_order:
            values,counts=np.unique(rows[:,gate],return_counts=True)
            if len(values)<2 or len(values)>32:
                continue
            estimates=[];support=[];routing={}
            for value,n in zip(values,counts,strict=True):
                if n<32:
                    continue
                mask=rows[:,gate]==value
                scores=[information(rows[mask,j],target[mask],len(labels)) for j in family]
                routing[value]=family[int(np.argmax(scores))]
                estimates.append(scores)
                support.append(int(n))
            if len(set(routing.values()))<2:
                continue
            estimates=np.asarray(estimates);weights=np.asarray(support)/len(rows)
            gain=float((weights*estimates.max(1)).sum()-(weights[:,None]*estimates).sum(0).max())
            if gain>max(.002,best_gain):
                chosen=(gate,routing,default)
                best_gain=gain
        result.append(chosen)
    return result


def aligned_columns(rows,routes):
    extras=[]
    for gate,route,default in routes:
        if gate is None:
            extra=rows[:,default]
            context=np.repeat("",len(rows))
        else:
            context=rows[:,gate]
            cols=np.array([route.get(value,default) for value in context],dtype=int)
            extra=rows[np.arange(len(rows)),cols]
        extras.extend([extra,tokens(np.column_stack((context,extra)))])
    return np.column_stack(extras)


class AttributeAlignmentTree(ContextTokenTree):
    def fit(self,rows,labels):
        x=_clean_categories(rows)
        y=np.asarray(labels)
        self.profile,self.context=blocks(x)
        families=comparable_fields(x,self.profile)
        self.pairs=choose_pairs(rank_pairs(x,y),self.params["pair_limit"])
        base=ContextTokenTree.transform(self,x)
        self.routes_=[]
        self.audit_=dict(family_sizes=[len(f) for f in families],supervised_route_estimations=0,
                         tree_fits=1,context_field_count=len(self.context))
        if families:
            groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
            folds=min(3,len(set(groups)))
            if folds<2:
                raise ValueError("Too few independent row groups for learning a conditional field")
            added=np.empty((len(x),2*len(families)),dtype=object)
            for fit,held in GroupKFold(folds).split(x,y,groups):
                routes=fitted_routes(x[fit],y[fit],families,self.context)
                added[held]=aligned_columns(x[held],routes)
            self.routes_=fitted_routes(x,y,families,self.context)
            self.audit_.update(supervised_route_estimations=folds+1,
                selected_gates=[g for g,_,_ in self.routes_],
                distinct_fields_per_gate=[len(set(r.values())) for _,r,_ in self.routes_])
            base=np.column_stack((base,added))
        self.classifier=_tabular_catboost(y,6,3.)
        self.classifier.set_params(thread_count=6,**self.params.get("tree_params",{}))
        self.classifier.fit(base,y,cat_features=list(range(base.shape[1])))
        self.classes_=self.classifier.classes_
        return self

    def transform(self,rows):
        x=_clean_categories(rows)
        base=ContextTokenTree.transform(self,x)
        return np.column_stack((base,aligned_columns(x,self.routes_))) if self.routes_ else base
