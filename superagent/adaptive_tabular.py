"""Training-internal iteration selection, followed by full-partition refitting."""
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math

import numpy as np

from .compute_resources import cpu_threads
from .data import TrainingTask
from .models import Candidate,_clean_categories
from .splits import inner_split,overlap_audit


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class AdaptiveCatBoostModel:
    classifier: object
    training_metadata: dict
    classifier_fits: int=2

    @property
    def classes_(self):return np.asarray(self.classifier.classes_)
    def predict(self,x):return np.asarray(self.classifier.predict(_clean_categories(x))).reshape(-1)
    def predict_proba(self,x):return np.asarray(self.classifier.predict_proba(_clean_categories(x)))


def fit_adaptive_catboost(x_train,y_train,x_search,depth=8,l2_leaf_reg=3.,max_iterations=1500,
                         early_stopping_rounds=80,selection_seed=4242):
    from catboost import CatBoostClassifier
    if type(depth) is not int or depth<1 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError('Depth and maximum iterations must be positive integers')
    if type(early_stopping_rounds) is not int or early_stopping_rounds<1:
        raise ValueError('Early-stopping patience must be a positive integer')
    if not math.isfinite(l2_leaf_reg) or l2_leaf_reg<0:raise ValueError('Invalid L2 leaf penalty')
    x=_clean_categories(x_train);y=np.asarray(y_train)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y):raise ValueError('Expected matching table rows and scalar labels')
    classes=np.unique(y)
    if len(classes)<2:raise ValueError('At least two classes are required')
    training=TrainingTask('categorical_tabular',x,y)
    fit_indices,holdout_indices=inner_split(training,selection_seed,'exact_rows')
    for indices in (fit_indices,holdout_indices):
        if set(np.unique(y[indices]).tolist())!=set(classes.tolist()):
            raise ValueError('Every class must remain in both iteration-selection partitions')
    audit=overlap_audit(training,fit_indices,holdout_indices)
    if audit['search_duplicate_rows']!=0:raise ValueError('Normalized duplicate rows cross the iteration-selection boundary')
    loss='Logloss' if len(classes)==2 else 'MultiClass'
    common={'depth':depth,'learning_rate':.05,'l2_leaf_reg':l2_leaf_reg,'loss_function':loss,
        'eval_metric':loss,'random_seed':42,'thread_count':cpu_threads(2),'verbose':False,'allow_writing_files':False}
    columns=list(range(x.shape[1]))
    selector=CatBoostClassifier(iterations=max_iterations,**common)
    selector.fit(x[fit_indices],y[fit_indices],cat_features=columns,
        eval_set=(x[holdout_indices],y[holdout_indices]),use_best_model=True,
        early_stopping_rounds=early_stopping_rounds)
    best=selector.get_best_iteration()
    curves=selector.get_evals_result();curve=[float(v) for v in curves['validation'][loss]]
    if type(best) is not int or best<0 or not curve or not all(math.isfinite(v) for v in curve):
        raise ValueError('Iteration selector did not return a valid measured best iteration')
    iterations=best+1
    if not 1<=iterations<=len(curve)<=max_iterations:raise ValueError('Inconsistent selected iteration count')
    if not math.isclose(curve[best],min(curve),abs_tol=1e-12):raise ValueError('Selected iteration disagrees with the saved loss curve')
    # Discard the selector; the second learner sees the entire current training
    # partition and no eval_set. x_search is used only for subsequent prediction.
    del selector
    classifier=CatBoostClassifier(iterations=iterations,**common)
    classifier.fit(x,y,cat_features=columns)
    if classifier.tree_count_!=iterations:raise ValueError('Full-partition refit did not retain the selected number of trees')
    metadata={'method':'training_internal_iteration_selection_then_full_partition_refit',
        'selection_data':'current_X_train_and_y_train_only','training_rows':len(y),
        'training_feature_sha256':_fingerprint(x.tolist()),'training_label_sha256':_fingerprint(y.tolist()),
        'selection_seed':selection_seed,'selection_grouping':'exact_normalized_category_rows',
        'selector_fit_indices':fit_indices.tolist(),'selector_holdout_indices':holdout_indices.tolist(),
        'selector_overlap_audit':audit,'stopping_metric':loss,'stopping_metric_direction':'minimize',
        'max_iterations':max_iterations,'early_stopping_rounds':early_stopping_rounds,
        'selector_iterations_executed':len(curve),'selector_holdout_loss_curve':curve,
        'selected_iterations':iterations,'selected_holdout_loss':curve[best],
        'final_tree_count':int(classifier.tree_count_),'refit_eval_set_used':False,
        'outer_search_labels_accessed':False,'classifier_fits':2}
    model=AdaptiveCatBoostModel(classifier,metadata)
    return model,model.predict(x_search)


def adaptive_candidates():
    return [Candidate(f'adaptive_catboost_depth{depth}_reg{reg}',
        {'depth':depth,'l2_leaf_reg':reg,'learning_rate':.05,'max_iterations':1500,'early_stopping_rounds':80,
         'iteration_selection':'training_internal_grouped_loss','classifier_fits_per_evaluation':2},
        partial(fit_adaptive_catboost,depth=depth,l2_leaf_reg=reg))
        for depth in (6,8) for reg in (3.,10.)]
