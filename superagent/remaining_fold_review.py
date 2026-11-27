"""Spend spare fits on unused assessment folds after the ordinary stopping decision is frozen."""
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np

from .ab_run import sha
from .experiment_policy import identity
from .near_full_confirmation import shortlist
from .splits import sample_groups, take
from .task_semantics import declared_semantics

NAMES = ("primary", "confirmation_1", "confirmation_2", "confirmation_3", "confirmation_4")


def assessment_folds(task, seed, search, partition):
    if (task.spec.get("evaluation_protocol") or {}).get("kind", "grouped_stratified") != "grouped_stratified":
        return {}
    full = partition(task, seed, names=NAMES)
    if tuple(search) != NAMES[:3] or tuple(full) != NAMES:
        return {}
    if any(not np.array_equal(search[n][i], full[n][i]) for n in search for i in (0, 1)):
        return {}
    groups = sample_groups(task, "shared_text")
    assessed = set(np.concatenate([groups[search[n][1]] for n in search]))
    result = {}
    for name in NAMES[3:]:
        fit, held, resource = full[name]
        test_groups = set(groups[held])
        if test_groups.intersection(assessed) or test_groups.intersection(groups[fit]):
            return {}
        assessed.update(test_groups)
        result[name] = (fit, held, resource)
    return result


def output(option, bank, classes):
    members = option["members"]
    if len(members) == 1:
        obs = bank[members[0]]
        return obs["predictions"], obs["probabilities"]
    probabilities = [bank[member]["probabilities"] for member in members]
    if any(p is None for p in probabilities):
        raise ValueError("A proposed vote has unavailable class probabilities")
    mean = np.mean(probabilities, axis=0)
    return classes[mean.argmax(axis=1)], mean


def change(metric, classes, truth, previous, alternative, mask=None):
    mask = slice(None) if mask is None else mask
    def score(value):
        predicted, prob = value
        return metric.evaluate(truth[mask], predicted[mask], None if prob is None else prob[mask],
                               None if prob is None else classes)["score"]
    return float(score(alternative)-score(previous))


def concatenate(outputs):
    prediction = np.concatenate([out[0] for out in outputs])
    probabilities = [out[1] for out in outputs]
    return prediction, None if any(p is None for p in probabilities) else np.concatenate(probabilities)


def admission(metric, classes, truths, old, new, keys):
    gains = [change(metric, classes, y, a, b) for y, a, b in zip(truths, old, new)]
    all_y = np.concatenate(truths)
    a, b = concatenate(old), concatenate(new)
    pooled = change(metric, classes, all_y, a, b)
    groups = np.concatenate(keys)
    buckets = np.array([hashlib.blake2b(str(g).encode(), digest_size=2).digest()[0] % 16 for g in groups])
    sampled = []
    for k in np.unique(buckets):
        mask = buckets != k
        if set(np.unique(all_y[mask])) == set(classes):
            sampled.append(change(metric, classes, all_y, a, b, mask))
    error = None
    if len(sampled) >= 8:
        estimate = np.asarray(sampled)
        error = float(np.sqrt((len(estimate)-1)*np.mean((estimate-estimate.mean())**2)))
    passes = min(gains)>=0 and pooled>=.003 and error is not None and pooled>error
    return bool(passes),dict(fold_gains=gains,pooled_gain=pooled,group_jackknife_se=error,
                             rows=[len(y) for y in truths])


def review(task, seed, search, choice, menu, history, metric, executor, directory, log, timeout, partition, observe):
    extra = assessment_folds(task, seed, search, partition)
    caps = executor.caps
    left = executor.limit-executor.accounting()["attempted_classifier_fits"]
    costs = {m:dict(c, classifier_fits=2*c["classifier_fits"]) for m,c in caps.items()}
    options = shortlist(choice, menu, costs, left)
    needed = sorted({m for a in options for m in a["members"]})
    tasks = [(n,m) for n in extra for m in needed]
    reserve = max(sum(caps[m]["refit_classifier_fits"] for m in a["members"]) for a in options)
    total = sum(caps[m]["classifier_fits"] for _,m in tasks)
    times = [r["training_seconds"] for r in history if r["status"]=="completed"]
    remaining = executor.start+executor.wall-time.monotonic()
    def retain(reason):
        log.write_json("remaining_fold_result.json",dict(action="retain",members=choice["members"],reason=reason))
        return choice
    if len(extra)!=2 or len(options)<2 or reserve+total>left or remaining<(len(tasks)+reserve)*max(times,default=1.)+12:
        return retain("No affordable confirmed alternative on two unused assessment folds")
    groups = sample_groups(task,"shared_text")
    log.write_json("remaining_fold_nomination.json",dict(ordinary=choice["members"],
         choices=[a["members"] for a in options],folds=list(extra),remaining_fits=left,reserved_refits=reserve))
    directory = Path(directory)
    bank = {n:{} for n in extra}
    classes = np.asarray(metric.labels)
    try:
        for name,(fit,held,_) in extra.items():
            path = directory/(name+"_review.joblib")
            labels = task.y_train[held]
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,fit),y_train=task.y_train[fit],
                       X_search=take(task.X_train,held),y_search=labels,metric_spec=metric.record(),
                       pair_semantics=declared_semantics(task),capture_probabilities=True),path)
            payload = dict(path=path,labels=labels,sha256=sha(path))
            for j,member in enumerate(needed):
                total -= caps[member]["classifier_fits"]
                dest=directory/(name+"_review_"+str(j)+".joblib")
                _,facts=executor.worker(path,member,dest,timeout,reserve=reserve+total)
                bank[name][member]=observe(joblib.load(dest),facts,payload,metric)
        truth = [task.y_train[held] for _,held,_ in extra.values()]
        keys = [groups[held] for _,held,_ in extra.values()]
        previous = [output(choice,b,classes) for b in bank.values()]
        passing = []
        report = []
        for alt in options[1:]:
            candidate = [output(alt,b,classes) for b in bank.values()]
            passed, scores = admission(metric,classes,truth,previous,candidate,keys)
            report.append(dict(members=alt["members"],passed=passed,**scores))
            if passed:
                passing.append((alt,scores["pooled_gain"]))
        winner = min(passing,key=lambda p:(-p[1],len(p[0]["members"]),identity(p[0]["members"])))[0] if passing else choice
        log.write_json("remaining_fold_result.json",dict(action="replace" if passing else "retain",ordinary=choice["members"],
                       members=winner["members"],checks=report,completed_review_workers=len(tasks)))
        return winner
    except (RuntimeError,ValueError,OSError,TypeError,KeyError) as exc:
        if executor.accounting()["unknown_failed_attempts"]:
            raise
        return retain(str(exc)[:250])
