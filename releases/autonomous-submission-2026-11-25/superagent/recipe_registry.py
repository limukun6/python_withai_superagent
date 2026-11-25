"""Feedback-driven bounded composition, persisted before any execution."""
import json
from pathlib import Path
from .metrics import score_of, completed
from .recipe_models import materialize_recipe


def resolve_recipe_registry(kind, candidate_id, document):
    from .models import search_candidates
    if set(document) != {'schema_version', 'kind', 'portfolio', 'entries'} or document['schema_version'] != 2 or document['kind'] != kind:
        raise ValueError('Invalid composition registry')
    if kind not in ('single_text', 'pair_text') or document['portfolio'] not in ('original', 'expanded') or not isinstance(document['entries'], dict) or len(document['entries']) > 4:
        raise ValueError('Unsupported composition registry bounds')
    candidates = {}
    for key, entry in document['entries'].items():
        if set(entry) != {'spec', 'params', 'origin'} or entry['spec'].get('kind') != kind:
            raise ValueError('Invalid recipe entry')
        candidate = materialize_recipe(entry['spec'])
        if candidate.family != key or candidate.params != entry['params']:
            raise ValueError('Recipe ID/params changed')
        candidates[key] = candidate
    if candidate_id in candidates:
        return candidates[candidate_id]
    candidate = next((c for c in search_candidates(kind, 'all') if c.family == candidate_id), None)
    if candidate is None:
        raise ValueError('Unknown composed candidate')
    return candidate


class RecipeRegistry:
    def __init__(self, kind, portfolio, path):
        if kind not in ('single_text', 'pair_text') or portfolio not in ('original', 'expanded'):
            raise ValueError('Composition currently supports text tasks and original/expanded')
        self.kind, self.path = kind, Path(path)
        if self.path.exists():
            raise FileExistsError('Never overwrite a recipe registry')
        self.document = {'schema_version': 2, 'kind': kind, 'portfolio': portfolio, 'entries': {}}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def propose(self, history):
        successful = [r for r in history if completed(r)]
        if len(successful) < 2 or len(self.document['entries']) >= 4:
            return []
        best = min(successful, key=lambda r: (-score_of(r), r['trial']))
        slices = (best.get('error_slices') or {}).get('slices', [])
        worst = min(slices, key=lambda r: r['delta_from_overall_accuracy']) if slices else None
        rare = bool(worst and worst['slice_id'] == 'unseen_token_fraction_high' and worst['delta_from_overall_accuracy'] < -.03)
        analyzer = 'char_wb' if rare else 'word'
        gap = best.get('score_gap', best.get('generalization_gap', 0.))
        classifier = 'linear_svm' if gap > .15 else 'logistic'
        c = float(min(10., max(.05, best['params'].get('C', .3))))
        added = []
        for representation, algorithm in [('lexical', classifier), ('hybrid', 'logistic'), ('latent', classifier), ('hybrid', 'linear_svm')]:
            spec = {'kind': self.kind, 'analyzer': analyzer, 'representation': representation,
                    'classifier': algorithm, 'C': c, 'dimensions': 32}
            candidate = materialize_recipe(spec)
            if candidate.family in self.document['entries']:
                continue
            origin = {'rule': 'bounded_text_recipe_from_observed_best', 'after_trial': history[-1]['trial'],
                      'reference_trial': best['trial'], 'reference_score': score_of(best), 'observed_gap': gap,
                      'rare_token_slice_trigger': rare, 'worst_observed_slice': worst,
                      'hypothesis': 'A different feature/classifier combination may address the observed weakness; must be tested',
                      'revisit_previous_if': 'Keep previous configurations and prefer confirmed prior model if this loses'}
            entry = {'spec': spec, 'params': candidate.params, 'origin': origin}
            self.document['entries'][candidate.family] = entry
            added.append({'id': candidate.family, **entry})
            if len(added) == 2 or len(self.document['entries']) == 4:
                break
        if added:
            self.save()
        return added

    def description(self):
        return {'enabled': True, 'mode': 'compose', 'maximum_generated_total': 4,
                'maximum_generated_per_step': 2, 'generated_candidates': len(self.document['entries']),
                'adds_training_attempts': False, 'adds_api_requests': False,
                'generator': 'deterministic feedback-driven bounded composition; not arbitrary LLM code generation'}
