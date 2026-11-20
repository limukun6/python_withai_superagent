"""Read validation loss along an already fitted boosting trajectory."""
import numpy as np
from sklearn.metrics import log_loss

from .models import _clean_categories


def measured_curve(model,search,labels):
    raw=_clean_categories(search)
    size=int(model.tree_count_)
    steps=sorted({max(1,round(size*i/4)) for i in (1,2,3,4)})
    rows=[]
    for last in steps:
        p=model.predict_proba(raw,ntree_end=last,thread_count=2)
        pred=np.asarray(model.classes_)[p.argmax(axis=1)]
        rows.append(dict(trees=last,log_loss=float(log_loss(labels,p,labels=model.classes_)),accuracy=float(np.mean(pred==labels))))
    return rows


def suggested_trees(curve):
    current=curve[-1]
    earlier=min(curve[:-1],key=lambda r:(r['log_loss'],r['trees']))
    if current['log_loss']-earlier['log_loss']>=.001:
        return earlier['trees']
    if curve[-2]['log_loss']-current['log_loss']>=.001:
        return min(1800,round(current['trees']*1.5))
    return None
