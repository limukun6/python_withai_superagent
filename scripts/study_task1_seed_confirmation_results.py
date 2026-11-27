"""Export only aggregate scores, method decisions, fit counts and evidence identities."""
import argparse
from pathlib import Path

from scripts.unified_released_study import ROOT, read, require, save, sha

REGISTRY = ROOT / "research/task1_feedback_extension_1127"


def export(out):
    dev = out / "development"
    paths = dict(historical_seed_replay=out / "seed_replication/result.json",
                 prospective_seed_policy=out / "seed_policy/result.json",
                 signed_transfer_nomination=dev / "crosscontext_election/locked.json",
                 signed_transfer=dev / "crosscontext_election/results.json",
                 original_tree_transfer=dev / "crosscontext_election/anchors.json",
                 transfer_outer=out / "cross_context_outer/result.json",
                 full_retrieval_nomination=dev / "tabretrieve_election/locked.json",
                 full_retrieval=dev / "tabretrieve_election/results.json")
    records = {k:read(v) for k,v in paths.items()}
    historical, prospective = records["historical_seed_replay"], records["prospective_seed_policy"]
    release = bool(prospective["advance"])
    counts = dict(historical_seed_classifier_fits=historical["new_fits"],
                  new_cold_solutions=sum(a["fits"] for r in prospective["folds"] for a in r["methods"].values()),
                  full_retrieval_first_context=read(dev / "tabdpt_neighbours/fitted.json")["foundation_context_fits"],
                  full_retrieval_completed_context=read(dev / "tabdpt_neighbours_02/fitted.json")["foundation_context_fits"],
                  development_rate_memory=read(dev / "cross_context_history/sealed.json")["rate_memory_fits"],
                  development_logistic_heads=sum(read(p)["supervised_offset_fits"] for p in dev.glob("crosscontext_*_*/sealed.json")),
                  outer_rate_memories=sum(f["memory_fits"] for f in records["transfer_outer"]["folds"]),
                  outer_logistic_heads=sum(f["offset_fits"] for f in records["transfer_outer"]["folds"]))
    require(counts["development_logistic_heads"] == 4 and len(prospective["folds"]) == 3
            and not prospective["official_test_opened"], "Incomplete stage")
    if release:
        paths["designated"] = out / "designated_result.json"
        records["designated"] = read(paths["designated"])
        counts["qualified_raw_run"] = read(out / "release/result.json")["actual_fits"]
    retrieval = {str(k):read(dev / f"tabretrieve_k{k}/sealed.json") for k in (128,512)}
    prior = REGISTRY / "catalogue_encoder_results.json"
    before = read(prior)["cumulative"]
    value = dict(previous=sha(prior), previous_cumulative=before, results=records,
                 receipts={k:sha(v) for k,v in paths.items()}, fits=counts, stage_total=sum(counts.values()),
                 cumulative=before + sum(counts.values()), retrieval=retrieval,
                 retrieval_runtime_resolution="The initial context fit completed. Nearest-neighbour lookup with PyTorch in the same process stalled on macOS. A separate process calculated the exact FAISS nearest rows. The publisher classifier used them after row-order verification.",
                 model_api_requests=0, new_currency_rmb=0, designated_validation_opened=release, official_test_opened=False, h2_opened=False)
    save(REGISTRY / "seed_confirmation_results.json", value)
    print(counts, value["stage_total"], value["cumulative"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    export(parser.parse_args().out)
