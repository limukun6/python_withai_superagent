"""Fold-local numeric ranges and clock features for repeated categorical values."""
from collections import Counter
from functools import lru_cache
import math
import re
import unicodedata

import numpy as np

from .models import _clean_categories, _tabular_catboost
from .tabular_encoder import MISSING


NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)'
QUANTITY=re.compile(rf'([$€£¥]?)\s*({NUMBER})\s*(%?)')
RANGE=re.compile(r'^(.+?)\s*(?:\.\.|~|–|—|-|\s+to\s+)\s*(.+)$')
CLOCK=re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?')
BOUND=re.compile(r'(?:(<=|>=|<|>|≤|≥|under|below|over|above|at least|at most|less than|more than)\s*)(.+)')
SUFFIX=re.compile(r'(.+?)\s*(\+|and above|or more|or less|and below|plus)')


def quantity(text):
    match=QUANTITY.fullmatch(text.strip())
    if not match:return None
    currency,number,percent=match.groups()
    digits=number.lstrip('+-')
    if len(digits)>1 and digits[0]=='0' and digits[1].isdigit():return None
    value=float(number.replace(',',''))
    if not math.isfinite(value) or (currency and percent):return None
    return currency or percent or 'number',value


@lru_cache(maxsize=8192)
def parse_value(value):
    text=unicodedata.normalize('NFKC',value).replace('−','-').strip().casefold()
    clock=CLOCK.fullmatch(text)
    if clock and (clock[2] or clock[3]):
        hour=int(clock[1]);minute=int(clock[2] or 0)
        if minute>59 or (clock[3] and not 1<=hour<=12) or (not clock[3] and hour>23):return None
        if clock[3]:hour=hour%12+12*(clock[3]=='pm')
        angle=2*math.pi*(60*hour+minute)/1440
        return 'clock',(math.sin(angle),math.cos(angle))
    bound=BOUND.fullmatch(text)
    suffix=SUFFIX.fullmatch(text)
    if bound or suffix:
        mark,raw=(bound[1],bound[2]) if bound else (suffix[2],suffix[1])
        parsed=quantity(raw)
        if not parsed:return None
        unit,x=parsed
        upper=mark in ('<','<=','≤','below','under','at most','less than','or less','and below')
        return unit,((math.nan,x,x) if upper else (x,math.nan,x))
    simple=quantity(text)
    if simple:
        unit,x=simple
        return unit,(x,x,x)
    pair=RANGE.fullmatch(text)
    if not pair:return None
    left,right=quantity(pair[1]),quantity(pair[2])
    if not left or not right:return None
    u,a=left;v,b=right
    if a>b or (u!=v and 'number' not in (u,v)):return None
    return (v if u=='number' else u),(a,b,a/2+b/2)


def column_types(rows):
    types={}
    for i,col in enumerate(_clean_categories(rows).T):
        counts=Counter(str(v).strip() for v in col if str(v).strip().casefold() not in MISSING)
        if not 3<=len(counts)<=128 or sum(counts.values())<2*len(counts):continue
        parsed={value:parse_value(value) for value in counts}
        by_type=Counter()
        for value,result in parsed.items():
            if result:by_type[result[0]]+=counts[value]
        if not by_type:continue
        unit,n=by_type.most_common(1)[0]
        distinct=sum(result is not None and result[0]==unit for result in parsed.values())
        if n>=.85*sum(counts.values()) and distinct>=max(3,.80*len(counts)):
            types[i]=unit
    return types


class TypedCategoryClassifier:
    def __init__(self,params):
        self.params=params

    def features(self,rows):
        raw=_clean_categories(rows)
        if raw.shape[1]!=self.width:raise ValueError('The number of input columns changed')
        parts=[raw.astype(object)]
        for index,unit in self.types.items():
            size=2 if unit=='clock' else 3
            values,inverse=np.unique(raw[:,index],return_inverse=True)
            encoded=np.full((len(values),size),np.nan)
            for k,value in enumerate(values):
                parsed=parse_value(str(value))
                if parsed and parsed[0]==unit:encoded[k]=parsed[1]
            parts.append(encoded[inverse])
        return np.column_stack(parts)

    def fit(self,rows,labels):
        self.width=np.shape(rows)[1]
        self.types=column_types(rows)
        base=self.params['backbone_params']
        self.classifier=_tabular_catboost(labels,base['depth'],base['l2_leaf_reg'])
        if self.params.get('one_hot_max_size'):self.classifier.set_params(one_hot_max_size=self.params['one_hot_max_size'])
        self.classifier.fit(self.features(rows),labels,cat_features=list(range(self.width)))
        self.classes_=self.classifier.classes_
        self.training_metadata=dict(typed_columns={str(i):u for i,u in self.types.items()})
        return self

    def predict(self,rows):
        return np.asarray(self.classifier.predict(self.features(rows))).reshape(-1)

    def predict_proba(self,rows):
        return self.classifier.predict_proba(self.features(rows))
