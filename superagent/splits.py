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


def _declared_column(task, column):
    """Locate a declared split-key column inside the feature matrix."""
    if not isinstance(column, str) or not column.strip():
        raise ValueError('An explicit protocol must name one split-key column')
    columns = (task.spec.get('input_schema') or {}).get('columns') or []
    names = [entry.get('name') for entry in columns if isinstance(entry, dict)]
    if column not in names:
        raise ValueError(f'Protocol column {column!r} is not declared in input columns')
    matrix = np.asarray(task.X_train)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError('An explicit protocol needs a table whose declared columns match the features')
    return matrix[:, names.index(column)]


def _require_both_sides(y_train, train, held, protocol):
    labels = np.asarray(y_train)
    if not len(train) or not len(held):
        raise ValueError(f'The {protocol} split leaves an empty side')
    if set(labels[train].tolist()) != set(labels[held].tolist()):
        raise ValueError(f'The {protocol} split does not keep the same classes on both sides; '
                         'refusing to score an incomparable held-out set')


def temporal_partition(task, declaration):
    """Split by a declared time-like column: earlier rows train, later rows are held out."""
    column = declaration.get('column') or declaration.get('time_column')
    fraction = declaration.get('train_fraction', 0.8)
    if not isinstance(fraction, (int, float)) or not 0.5 <= fraction < 1:
        raise ValueError('train_fraction must be a number in [0.5, 1)')
    values = _declared_column(task, column)
    order = np.argsort(values, kind='stable')
    cut = int(round(len(order) * float(fraction)))
    train, held = order[:cut], order[cut:]
    _require_both_sides(task.y_train, train, held, 'temporal')
    return train, held, {'protocol': 'temporal', 'column': column,
                         'train_fraction': float(fraction), 'total_rows': int(len(order)),
                         'train_rows': int(len(train)), 'heldout_rows': int(len(held)),
                         'selection_rule': 'declared time order; earlier rows train; no model scores'}


def entity_partition(task, seed, declaration):
    """Split whole entities: no entity value may appear on both sides."""
    column = declaration.get('column') or declaration.get('entity_column')
    fraction = declaration.get('train_fraction', 0.8)
    if not isinstance(fraction, (int, float)) or not 0.5 <= fraction < 1:
        raise ValueError('train_fraction must be a number in [0.5, 1)')
    values = [str(value) for value in np.asarray(_declared_column(task, column)).tolist()]
    unique = sorted(set(values),
                    key=lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest())
    target = int(round(len(values) * float(fraction)))
    chosen, running = [], 0
    for value in unique:
        members = values.count(value)
        if chosen and running + members > target:
            break
        chosen.append(value)
        running += members
    train_set = set(chosen)
    train = np.asarray([i for i, value in enumerate(values) if value in train_set], dtype=int)
    held = np.asarray([i for i, value in enumerate(values) if value not in train_set], dtype=int)
    if not len(held):
        raise ValueError('The entity split keeps every entity in training; lower train_fraction')
    _require_both_sides(task.y_train, train, held, 'entity')
    return train, held, {'protocol': 'entity', 'column': column, 'train_fraction': float(fraction),
                         'entities': len(unique), 'train_entities': len(train_set),
                         'heldout_entities': len(unique) - len(train_set),
                         'selection_rule': 'whole entities assigned once by seeded key order; no model scores'}


def grouped_partition(task, seed, grouping="exact_rows"):
    declared=getattr(task,'spec',{}).get('evaluation_protocol',{})
    if not isinstance(declared,dict):raise ValueError('Evaluation protocol must be an object')
    kind = declared.get('kind', 'grouped_stratified')
    if kind == 'temporal':
        return temporal_partition(task, declared)
    if kind == 'entity':
        return entity_partition(task, seed, declared)
    if kind != 'grouped_stratified':
        raise ValueError(f'Unsupported declared evaluation protocol {kind!r}; '
                         'only grouped_stratified, temporal and entity are implemented')
    groups = sample_groups(task, grouping)
    count=len(set(groups));classes=set(np.asarray(task.y_train).tolist())
    if count<2:raise ValueError('Need at least two independent groups for development evaluation')
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
