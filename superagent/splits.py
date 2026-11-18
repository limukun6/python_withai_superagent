"""Label-stratified development splits with explicit dependency grouping."""
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def take(rows, indices):
    if hasattr(rows,'take') and not isinstance(rows,np.ndarray):return rows.take(indices)
    return rows[indices] if isinstance(rows, np.ndarray) else [rows[int(i)] for i in indices]


def normalized_text(text):
    return " ".join(str(text).split()).casefold()


def effective_grouping(kind, grouping):
    if grouping not in ("exact_rows", "shared_text"):
        raise ValueError("Unknown development grouping")
    return grouping if kind == "pair_text" else "exact_rows"


def feature_key(kind,row):
    if kind=='single_text':return normalized_text(row)
    if kind=='image':
        values=np.ascontiguousarray(row)
        return (values.shape,values.dtype.str,values.tobytes())
    if kind=='pair_text':return tuple(sorted(row))
    return tuple(map(str,row))


def sample_groups(task, grouping="exact_rows"):
    mode = effective_grouping(task.kind, grouping)
    if mode == "exact_rows":
        keys = [feature_key(task.kind,row) for row in task.X_train]
        return np.asarray([hashlib.sha256(repr(key).encode()).hexdigest() for key in keys])
    # A-B and B-C belong to the same connected component, not merely the
    # same exact pair. Canonical roots make grouping independent of row order.
    pairs = [tuple(map(normalized_text, row)) for row in task.X_train]
    parents = {text: text for pair in pairs for text in pair}
    def root(text):
        while parents[text] != text:
            parents[text] = parents[parents[text]]
            text = parents[text]
        return text
    for left, right in pairs:
        a, b = root(left), root(right)
        parents[max(a,b)] = min(a,b)
    return np.asarray([hashlib.sha256(root(left).encode()).hexdigest() for left, _ in pairs])


def grouped_partition(task, seed, grouping="exact_rows"):
    groups = sample_groups(task, grouping)
    count=len(set(groups));classes=set(np.asarray(task.y_train).tolist())
    if count<2:raise ValueError('Need at least two independent groups for development evaluation')
    declared=getattr(task,'spec',{}).get('evaluation_protocol',{})
    if not isinstance(declared,dict):raise ValueError('Evaluation protocol must be an object')
    if declared.get('kind','grouped_stratified')!='grouped_stratified':
        raise ValueError('Declared temporal/entity evaluation protocol needs a supported explicit splitter')
    # Choose by group counts and class coverage, never by any model score.
    for folds in range(min(5,count),1,-1):
        if min(np.sum(np.asarray(task.y_train)==label) for label in classes)<folds:continue
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold,(train,held) in enumerate(splitter.split(np.zeros(len(task.y_train)),task.y_train,groups)):
            if set(np.asarray(task.y_train)[train].tolist())==classes and set(np.asarray(task.y_train)[held].tolist())==classes:
                return train,held,{'folds':folds,'heldout_fold':fold,'total_groups':count,
                    'train_groups':len(set(groups[train])),'heldout_groups':len(set(groups[held])),
                    'class_domain_preserved':True,'selection_rule':'largest_2_to_5_grouped_fold_count_with_full_class_coverage; no model scores',
                    'small_data_reduction':folds<5}
    raise ValueError('No leakage-free grouped partition retains every class on both sides')


def inner_split(task, seed, grouping="exact_rows"):
    train,held,_=grouped_partition(task,seed,grouping)
    return train,held


def overlap_audit(task, train, search):
    record = {"train_rows": len(train), "search_rows": len(search)}
    if task.kind == "pair_text":
        seen = {normalized_text(text) for i in train for text in task.X_train[int(i)]}
        record["search_rows_sharing_text"] = sum(
            any(normalized_text(text) in seen for text in task.X_train[int(i)]) for i in search)
    else:
        seen = {feature_key(task.kind,task.X_train[int(i)]) for i in train}
        record["search_duplicate_rows"] = sum(feature_key(task.kind,task.X_train[int(i)]) in seen for i in search)
    return record
