"""Audit the A19 launch on original course archives and stage training-only copies."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from scripts import run_b1_ab_acceptance as rec, run_b_c1_matrix as matrix
from scripts.materialize_b_c1_rows import stored_zip
from superagent import ab_run
from superagent.ab_sentence import verified_directory
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest


class ReservedMember(Exception):
    """Raised before an unapproved member is opened."""


class ReachedPartitions(BaseException):
    """Purposefully stop the CLI before any partitioning or training."""


def probe(path, directory, allowed):
    original = zipfile.ZipFile.open
    opened = []
    def guarded(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if name not in allowed:
            raise ReservedMember(name)
        opened.append(name)
        return original(archive, member, *args, **kwargs)
    def before_training(task, *args):
        if task.X_val is not None or task.y_val is not None:
            raise ValueError('Validation rows reached the development runner')
        raise ReachedPartitions()
    cli = build_parser().parse_args(['run',str(path),'--scheduler','ab-v1','--portfolio','ab-v1',
                                     '--planner','local','--catalog-scope','full','--run-dir',str(directory)])
    with patch.object(zipfile.ZipFile,'open',guarded), patch.object(ab_run,'freeze_partitions',before_training):
        try:
            cli.handler(cli)
        except ReservedMember as error:
            return dict(stage='blocked_before_member_open', denied_member=str(error), opened_training=opened)
        except ReachedPartitions:
            return dict(stage='reached_partitions', opened_training=opened)
    raise RuntimeError('The instrumented CLI did not reach a registered stopping point')


def audit_course_task(path, expected, output):
    output.mkdir()
    if ab_run.sha(path) != expected['task_archive_sha256']:
        raise ValueError('Original archive differs from the A19 record')
    allowed = set(expected['opened_members'])
    valid_names = {'c_train.npy','y_train.npy','train.csv','readme.md'}
    if not allowed or any(Path(n).name.casefold() not in valid_names for n in allowed):
        raise ValueError('Unrecognized member in the published training-only allowlist')
    raw = probe(path,output/'original_probe',allowed)
    if raw['stage']!='blocked_before_member_open' or 'val' not in Path(raw['denied_member']).name.casefold():
        raise ValueError('Original course entry differs from the registered validation access check')
    staged = output/'train_only.zip'
    with zipfile.ZipFile(path) as archive:
        stored_zip(staged, [(n,archive.read(n)) for n in sorted(allowed)])
    safe = probe(staged,output/'staged_probe',allowed)
    if safe['stage']!='reached_partitions':
        raise ValueError('Training-only archive did not reach the official partition step')
    task = load_training_task(staged)
    contract = TaskSpec.from_task(task).record()
    manifest = rec.read(output/'staged_probe/run_manifest.json')
    if (contract != expected['task_spec'] or manifest['task_contract'] != contract
            or task.spec['task_description_source'] != expected['task_description_source']
            or len(task.y_train)!=expected['rows']):
        raise ValueError('The staged course training contract, README or training size changed')
    with patch.dict(os.environ,{'SUPERAGENT_MINILM_WEIGHTS':''}):
        minimal = ab_run.registered_capabilities(contract,len(task.y_train),'full')
    full = ab_run.registered_capabilities(contract,len(task.y_train),'full')
    if (minimal!=expected['capabilities'] or list(minimal)!=list(expected['capabilities'])
            or full!=manifest['capabilities'] or list(full)!=list(manifest['capabilities'])):
        raise ValueError('Candidate availability differs beyond the registered encoder condition')
    (output/'splits').mkdir()
    _, splits = ab_run.freeze_partitions(task, output/'splits', 707, MetricSpec.for_task(task))
    return dict(raw_sha256=expected['task_archive_sha256'], staged_sha256=ab_run.sha(staged),
                opened_members=sorted(allowed), original_cli=raw, staged_cli=safe,
                same_task_contract_and_declared_readme=True, n_train=len(task.y_train),
                training_content_sha256=content_digest(task.X_train,task.y_train),
                metric=contract['metric']['name'], budget_id=expected['task_budget_id'],
                counts=dict(A=len(expected['capabilities']), here_without_encoder=len(minimal), here_with_encoder=len(full)),
                added_by_local_weights=[k for k in full if k not in minimal],
                same_local_row_ids_as_A={s:v['row_ids_sha256']==expected['splits'][s]['row_ids_sha256'] for s,v in splits.items()},
                same_local_labels_as_A={s:v['label_sha256']==expected['splits'][s]['label_sha256'] for s,v in splits.items()})


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    os.environ['SUPERAGENT_MINILM_WEIGHTS'] = str(args.weights.resolve())
    a_file = rec.ROOT / 'research/a17_a19_progress_final/a18_c1_readiness_current/readiness.json'
    a = {t['task']:t for t in rec.read(a_file)}
    plan = rec.ROOT/'docs/batches/BA19_02_released_entry_plan.md'
    rec.write(output/'protocol.json',dict(batch='BA19-02', A_record_sha256=ab_run.sha(a_file),
        plan_sha256=ab_run.sha(plan), runner_sha256=ab_run.sha(Path(__file__)),
        source=rec.digest(ab_run.sources()), fit_cap=0, api_cap=0, scoring_cap=0))
    try:
        if verified_directory() is None:
            raise ValueError('Expected verified frozen encoder')
        published = {name:audit_course_task(path,a[name],output/name)
                     for name,path in [('coupon',args.coupon),('preference',args.headlines)]}
        protocol = matrix.freeze_inputs(args.portable)
        previous = rec.read(matrix.REFERENCE/'results.json')['tasks']
        differences = {}
        for name, a_name in [('headlines','preference'),('bean','bean')]:
            frozen, remote = protocol['tasks'][name], a[a_name]
            if frozen['task_budget_id']!=remote['task_budget_id'] or frozen['catalog_scope']!='full':
                raise ValueError('Task budget or C1 catalogue scope differs from A19')
            sample = remote['commands'][0]['command']
            differences[name] = dict(raw_A_rows=remote['rows'], registered_B_rows=previous[name]['training_rows'],
                B_held_rows=previous[name]['outer_rows'], A_full_count=len(remote['capabilities']),
                B_full_count=len(previous[name]['candidate_ids']), A_time_cap=int(sample[sample.index('--time-limit')+1]),
                B_time_cap=protocol['wall_seconds'], A_worker_cap=int(sample[sample.index('--trial-timeout')+1]),
                B_worker_cap=protocol['trial_seconds'], same_task_budget_id=frozen['task_budget_id'],
                same_actual_inner_row_ids={s:row['row_ids_sha256']==remote['splits'][s]['row_ids_sha256']
                                          for s,row in frozen['splits'].items()})
        denied_ids = []
        for first, second in [('task2-dev','dry_bean'),('dry_bean','task2'),('task2','task2')]:
            with patch.object(matrix.ab_flash,'preflight',side_effect=AssertionError('Touched funding before scope check')):
                try:
                    matrix.run_matrix(SimpleNamespace(headlines_id=first,bean_id=second),protocol)
                except ValueError as error:
                    if 'registered cumulative task IDs' not in str(error):
                        raise
                    denied_ids.append([first,second])
                else:
                    raise AssertionError('A wrong budget ID reached the matrix')
        if rec.trace(output):
            raise ValueError('Unexpected classifier fit during input auditing')
        rec.write(output/'results.json',dict(course_inputs=published, C1_freeze_differences=differences,
            wrong_scope_pairs_blocked_before_ledger=denied_ids))
        rec.write(output/'completion.json',dict(status='passed', official_CLI_probes=4,
            registered_c1_arms=12, fits=0, model_api_requests=0, course_rmb=0, scores=0,
            reserved_archive_members_read=0, formal_c1_paths_executed=0))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed', error=str(error),
            fits=len(rec.trace(output)), model_api_requests=0))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('coupon','headlines','portable','weights','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    run(parser.parse_args())
