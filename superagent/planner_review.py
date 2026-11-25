"""Prospective self-review of a planner's forecasts for previously untried methods."""
import math

import numpy as np


def forecast(weights, expected, novelty):
    correction = float(np.clip(weights[0] + weights[1] * (novelty - .5), -.10, .10))
    return float(np.clip(expected + correction, 0., 1.))


class PlannerReview:
    """Each task starts at the original policy. A successor is tested before it can act."""

    def __init__(self):
        self.generation = 0
        self.weights = [0., 0.]
        self.step_size = .6
        self.observations = []
        self.versions = []
        self.pending = None
        self.ticket = None

    def menu(self, original):
        if self.generation == 0:
            return original
        ranked = []
        for action in original:
            if action['action'] != 'explore' or action['split'] != 'primary':
                ranked.append(action)
                continue
            evidence = action['evidence']
            expected = evidence['expected']
            adjustment = forecast(self.weights, expected, evidence['novelty']) - expected
            ranked.append(dict(action, utility=action['utility'] + adjustment,
                               evidence=dict(evidence, policy_generation=self.generation,
                                             original_utility=action['utility'], forecast_adjustment=adjustment)))
        return sorted(ranked, key=lambda row: -row['utility'])

    def before_fit(self, trial, action):
        self.ticket = None
        if action['action'] != 'explore' or action['split'] != 'primary':
            return
        evidence = action['evidence']
        expected, novelty = evidence['expected'], evidence['novelty']
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (expected, novelty)):
            raise ValueError('Invalid exploration forecast')
        self.ticket = dict(trial=trial, parent=self.generation, expected=expected, novelty=novelty,
                           parent_forecast=forecast(self.weights, expected, novelty))
        if self.pending is not None:
            self.ticket.update(proposal=self.pending['version'],
                               child_forecast=forecast(self.pending['weights'], expected, novelty))

    def observed(self, trial, score):
        if self.ticket is None:
            return
        if self.ticket['trial'] != trial or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Forecast and observed trial disagree')
        row = dict(self.ticket, score=score)
        self.ticket = None
        self.observations.append(row)
        if self.pending is not None:
            if row.get('proposal') != self.pending['version'] or trial in self.pending['fit_trials']:
                raise ValueError('A prospective evaluation must follow the recorded proposal')
            self.pending['evaluation'].append(dict(trial=trial, parent_error=abs(score-row['parent_forecast']),
                                                   child_error=abs(score-row['child_forecast'])))
            if len(self.pending['evaluation']) == 2:
                self._decide()
        if self.pending is None and len(self.observations) >= 2 and len(self.versions) < 3:
            self._propose()

    def _propose(self):
        features = np.array([[1., row['novelty']-.5] for row in self.observations])
        residual = np.array([row['score']-forecast(self.weights,row['expected'],row['novelty'])
                             for row in self.observations])
        delta = np.linalg.solve(features.T @ features + np.eye(2), features.T @ residual)
        weights = np.clip(np.asarray(self.weights) + self.step_size * delta, -.12, .12)
        self.pending = dict(version=len(self.versions)+1, parent=self.generation, parent_weights=self.weights.copy(),
                            weights=weights.tolist(), proposal_step=self.step_size,
                            fit_trials=[r['trial'] for r in self.observations], evaluation=[], status='shadow')
        self.versions.append(self.pending)

    def _decide(self):
        gains = [row['parent_error']-row['child_error'] for row in self.pending['evaluation']]
        mean_gain = float(np.mean(gains))
        accepted = mean_gain >= .003 and min(gains) >= -.03
        if accepted:
            self.generation = self.pending['version']
            self.weights = self.pending['weights'].copy()
            self.step_size = min(1., self.step_size*1.25)
        else:
            self.step_size = max(.15, self.step_size*.5)
        self.pending.update(status='accepted' if accepted else 'rejected', mean_absolute_error_gain=mean_gain,
                            next_proposal_step=self.step_size)
        self.pending = None

    def record(self):
        return dict(version=1, generation=self.generation, active_weights=self.weights, proposal_step=self.step_size,
                    forecasts=self.observations, proposals=self.versions, forecast_before_fit=self.ticket)
