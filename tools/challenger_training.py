"""Execute the bounded challenger strategy on an anonymous preference package."""
import time

import joblib
import numpy as np

from scripts.unified_encoder_portfolio import candidate_inventory
from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from superagent.ab_run import sha
from superagent.ab_sentence import SentenceFeatures, reuse_frozen_vectors
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.crossfit_challenger import AgreementVote, nominate, residual_directions, select, training_roles
from tools.portfolio_encoders import PortfolioFeatures
from tools.portfolio_training import REFERENCE, configure, fit_one, write_selection
from unified_agent import save, train as common_train


def rows_at(task, indices):
    return [task.X_train[i] for i in indices]


def score_members(models, rows):
    return np.column_stack([canonical_probabilities(m,rows)[1][:,1] for m in models])


def ties(rows):
    return np.asarray([a<b for a,b in rows],dtype=bool)


def screen_vectors(name, rows):
    method = SentenceFeatures("pair_text") if name==REFERENCE[2] else PortfolioFeatures(name)
    texts = [s for pair in rows for s in pair]
    blocks,_ = method.semantic_blocks(rows,texts)
    return blocks[0].toarray()


def train(package, directory, weights, cache=None):
    task = load_training_task(package)
    inventory = {"candidates":[],"reason":"weight_directory_not_configured","excluded":{}}
    if weights is not None:
        configure(weights, cache)
        inventory = candidate_inventory(task, weights,l6_as_challenger=True)
    names = inventory["candidates"]
    roles = None
    if names:
        groups = sample_groups(task,"shared_text")
        roles = training_roles(groups,task.y_train)
        if roles is None:
            inventory["reason"] = "insufficient_disjoint_training_groups"
    if not names or roles is None:
        code = common_train(package,directory)
        save(directory/"challenger_availability.json",inventory)
        return code

    directory.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    package_digest = sha(package)
    members = list(REFERENCE[:2])
    record = {"strategy":"crossfit-challenger","inventory":inventory,"package":package_digest,
              "rows":len(task.y_train),"maximum_classifier_fits":9,"model_api_limit":0,
              "roles":{k:{"rows":len(v),"ids":fingerprint(v.tolist()),"groups":len(set(groups[v]))} for k,v in roles.items()}}
    save(directory/"challenger_protocol.json",record)
    log = []
    def fit(name, stage, ix):
        folder = directory/stage/name
        before = time.monotonic()
        metadata = fit_one(folder,name,rows_at(task,ix),task.y_train[ix],
                           {"package":package_digest,"ids":fingerprint(ix.tolist())})
        log.append({"action":"fit","stage":stage,"candidate":name,"model":metadata["model"],
                    "ledger":metadata["ledger"],"rows":len(ix),"seconds":round(time.monotonic()-before,4)})
        print(stage,name,"fitted",flush=True)
        return joblib.load(folder/"model.joblib")
    with reuse_frozen_vectors():
        a,b,s = (roles[k] for k in ("a","b","nominate"))
        first = [fit(n,"inner_a",a) for n in members]
        screen_rows = rows_at(task,s)
        screen_probability = score_members(first,screen_rows)
        residual = task.y_train[s]-screen_probability.mean(1)
        gradients = {n:residual_directions(screen_vectors(n,screen_rows),residual,groups[s]) for n in names[2:]}
        values = {n:float(v[0] @ v[1]) for n,v in gradients.items()}
        challenger = nominate(values)
        np.savez_compressed(directory/"nomination.npz",row_ids=s,anchors=screen_probability,**gradients)
        save(directory/"nomination.json",{"candidate":challenger,"scores":values,"inputs":sha(directory/"nomination.npz")})
        log.append({"action":"nominate","candidate":challenger,"candidates_compared":len(values)})
        if challenger is not None:
            members.append(challenger)
            first.append(fit(challenger,"inner_a",a))
            second = [fit(n,"inner_b",b) for n in members]
            fb, packed = {}, {}
            for role,ids,models in (("a",a,second),("b",b,first)):
                rows = rows_at(task,ids)
                p = score_members(models,rows)
                fb[role] = dict(scores=p,ties=ties(rows),labels=task.y_train[ids],groups=groups[ids])
                packed[role],packed[role+"_ids"] = p,ids
            np.savez_compressed(directory/"oof.npz",**packed)
            decision = select(fb)
            decision["oof"] = sha(directory/"oof.npz")
        else:
            decision = dict(chosen="baseline",reason="no_positive_residual_alignment")
        decision.update(members=members,nomination=sha(directory/"nomination.json"))
        save(directory/"decision.json",decision)
        log.append({"action":"select","chosen":decision["chosen"]})
        chosen = decision["chosen"]
        used = members if chosen=="agreement_weight" else members[:2]
        final = [fit(n,"refit",np.arange(len(task.y_train))) for n in used]
        model = (AgreementVote(final,decision["weights"]) if chosen=="agreement_weight" else
                 PreferenceProbabilityVote("pair_text",final,np.array([0,1])))
        write_selection(directory/"agent",model,used,task.spec,chosen,"crossfit_challenger_preference")
    actual = sum(e["action"]=="fit" for e in log)
    elapsed = time.monotonic()-started
    save(directory/"challenger_completion.json",dict(actions=log, classifier_fits=actual, strategy=chosen,
         weight_fits=2 if challenger is not None else 0, nomination_screens=1,model_api_calls=0,
         elapsed_seconds=round(elapsed,4), decision_sha256=sha(directory/"decision.json"),
         model_sha256=sha(directory/"agent/selected_model.joblib")))
    return 0
