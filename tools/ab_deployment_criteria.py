"""Compute the project's preregistered criteria on audited, matched development observations."""
from collections import defaultdict
from decimal import Decimal
from statistics import median


def number(value, *, positive=False):
    decoded = Decimal(str(value))
    if not decoded.is_finite() or (positive and decoded<=0):
        raise ValueError('Expected a finite, positive elapsed time' if positive else 'Expected a finite measurement')
    return decoded


def assess(pairs):
    if not pairs:
        raise ValueError('At least one measured pair is required')
    by_task,metrics,used = defaultdict(list),{},set()
    base_times,method_times,risks = [],[],[]
    requests=0
    for row in pairs:
        task,split=row['task'],row['split']
        if (task,split) in used or metrics.setdefault(task,row['metric'])!=row['metric']:
            raise ValueError('Repeated task/split or different metrics within one task')
        used.add((task,split))
        scores=[number(row[k]) for k in ('baseline_score','score')]
        if not all(0<=s<=1 for s in scores) or type(row['requests']) is not int or row['requests']<0:
            raise ValueError('Invalid classification score or actual request count')
        delta=(scores[1]-scores[0])*100
        by_task[task].append(delta)
        if delta<Decimal('-2'):
            risks.append(dict(task=task,split=split,difference_pp=float(delta)))
        base_times.append(number(row['baseline_seconds'],positive=True))
        method_times.append(number(row['seconds'],positive=True))
        requests+=row['requests']
    means={t:sum(values)/len(values) for t,values in by_task.items()}
    improvements=sum(v>=Decimal('.2') for v in means.values())
    worst=min(means.values())
    baseline_time,method_time=median(base_times),median(method_times)
    quality=worst>=Decimal('-.5') and improvements>=2
    efficiency=worst>=Decimal('-.2') and method_time<=Decimal('.8')*baseline_time
    return dict(tasks=len(means),pairs=len(used),task_mean_difference_pp={t:float(v) for t,v in means.items()},
                mean_task_difference_pp=float(sum(means.values())/len(means)),worst_task_difference_pp=float(worst),
                tasks_gaining_at_least_0_2pp=improvements,quality_condition=quality,efficiency_condition=efficiency,
                meets_measured_criteria=quality or efficiency,baseline_median_seconds=float(baseline_time),
                median_seconds=float(method_time),median_time_reduction_percent=float((1-method_time/baseline_time)*100),
                measured_requests=requests,total_seconds=float(sum(method_times)),risk_splits=risks)


def ranking_key(report):
    return (-report['mean_task_difference_pp'],-report['worst_task_difference_pp'],
            report['measured_requests'],report['total_seconds'])
