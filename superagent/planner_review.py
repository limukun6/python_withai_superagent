"""Prospective self-review of a planner's forecasts for previously untried methods."""
import math

import numpy as np


def forecast(weights, expected, novelty):
    correction = float(np.clip(weights[0] + weights[1] * (novelty - .5), -.10, .10))
    return float(np.clip(expected + correction, 0., 1.))


def add_review_arguments(parser):
    control=parser.add_mutually_exclusive_group()
    control.add_argument('--task-analysis',choices=('local','flash','feedback-control','feedback-local','feedback-flash','feedback-design','feedback-design-flash'),
                         help='Check a task interpretation at opening or after observing a real training experiment')
    control.add_argument('--feedback-reflection', choices=('control','local','flash','conditional-flash','scheduled-flash',
                        'evidence-control','evidence-local','proposal-flash','delivered-local','delivered-flash','opening-local','opening-flash','opening-feedback-control','opening-feedback-local','opening-feedback-flash','opening-feedback-design','opening-feedback-design-flash'),
                        help='Review an early action or a measured disagreement that could change the delivered model')
    parser.add_argument('--reflection-history', help='Verified anonymous prior outcome counts')
    parser.add_argument('--exploration-rule', metavar='JSON',
                        help='Frozen cross-task calibration for primary exploration forecasts')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-review', action='store_true',
                       help='Test proposed rules on later exploration outcomes')
    modes.add_argument('--active-self-review', dest='self_review', action='store_const', const='active',
                       help='Also value experiments that can distinguish two proposed rules')
    modes.add_argument('--protected-self-review', dest='self_review', action='store_const', const='protected',
                       help='Preserve the selected model and use spare fits for a matched challenger when affordable')


def feedback_mode(args):
    opening=getattr(args,'task_analysis',None)
    return 'opening-'+opening if opening else getattr(args,'feedback_reflection',None)


def review_flags(mode):
    if mode == 'protected':
        return ['--protected-self-review']
    if mode == 'active':
        return ['--active-self-review']
    if mode is True:
        return ['--self-review']
    if mode is False:
        return []
    raise ValueError('Unknown planner self-review mode')


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

    def seek_evidence(self, ranked, remaining, capabilities):
        """Value a useful observation only if the agent can still test, corroborate and deliver."""
        if self.pending is None:
            return ranked
        missing = 2 - len(self.pending['evaluation'])
        calibration_error = float(np.mean([abs(o['score'] - o['parent_forecast']) for o in self.observations]))

        def fit_cost(action):
            return capabilities[action['members'][0]]['classifier_fits']

        def with_refit(action):
            return fit_cost(action) + action['evidence']['reserved_refits']

        experiments = sorted((a for a in ranked if a['action'] == 'explore'), key=with_refit)
        confirmation = min((a for a in ranked if a['action'] == 'confirm'), key=with_refit, default=None)
        updated = []
        for action in ranked:
            if action['action'] != 'explore':
                updated.append(action)
                continue
            # Each prospective comparison needs a distinct experiment. Preserve a real follow-up
            # after the verdict, and the most expensive refit reserved along this potential path.
            extra_count = missing - 1 + int(confirmation is None)
            others = [a for a in experiments if a['members'] != action['members']][:extra_count]
            if len(others) < extra_count:
                updated.append(action)
                continue
            path = [action, *others]
            if confirmation is not None:
                path.append(confirmation)
            required = sum(fit_cost(a) for a in path) + max(a['evidence']['reserved_refits'] for a in path)
            evidence = action['evidence']
            parent = forecast(self.weights, evidence['expected'], evidence['novelty'])
            child = forecast(self.pending['weights'], evidence['expected'], evidence['novelty'])
            value = min(abs(child - parent), calibration_error) / (missing * fit_cost(action))
            credit = value if required <= remaining else 0.
            information = dict(proposal=self.pending['version'], missing_observations=missing,
                               parent_forecast=parent, child_forecast=child, calibration_error=calibration_error,
                               required_fits_including_followup_and_refit=required,
                               followup_action=path[-1]['action'], remaining_fits=remaining, credit=credit)
            updated.append(dict(action, utility=action['utility'] + credit,
                                evidence=dict(evidence, information_value=information)))
        return sorted(updated, key=lambda row: -row['utility'])

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
