"""Publish a verified feature-only prediction independently of the training process."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading

from .events import RunLogger
from .predict import create_predictions


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def release_training():
    """The private pipe carries one fixed byte and no prediction data."""
    fd = sys.stdin.fileno()
    if os.read(fd, 1) != b'G':
        raise RuntimeError('Solve controller ended before the first delivery check')

    def stop_if_abandoned():
        # solve owns this process group on POSIX. Its workers inherit the group.
        if os.read(fd, 1) == b'':
            if os.name == 'posix' and os.getpgrp() == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGTERM)
            else:
                os._exit(130)

    threading.Thread(target=stop_if_abandoned, daemon=True).start()


def guard_output(output, run, inputs):
    reserved = (run/'agent', run/'first_delivery')
    if (run/'first_delivery.json').is_file():
        reserved = (*reserved, run)
    if (output in inputs or output.name in ('prediction_manifest.json', 'prediction_attempt.json')
            or any(output.is_relative_to(directory) for directory in reserved)):
        raise ValueError('Prediction output collides with the input, receipt, or frozen model storage')


def checked_receipt(source, specification=None):
    receipt = read(source.parent/'prediction_manifest.json')
    if receipt.get('test_labels_read') is not False or receipt['prediction_sha256'] != digest(source):
        raise ValueError('Published prediction failed its integrity check')
    contract = receipt.get('output_contract')
    recipe = (specification or {}).get('output_recipe')
    if contract is not None or recipe is not None:
        from .output_recipe import needs_scores, validate_recipe
        if not isinstance(contract, dict) or not isinstance(recipe, dict):
            raise ValueError('Custom export must match the frozen submission recipe')
        recipe = validate_recipe(recipe, specification['metric_spec']['labels'])
        bound = ('version', 'columns', 'order', 'encoding', 'delimiter', 'header')
        checks = contract.get('checks', {})
        required = ['header_checked', 'row_count_checked', 'model_outputs_checked']
        if any(c['kind'] == 'input_id' for c in recipe['columns']):
            required.append('original_id_checked')
        if needs_scores(recipe):
            required.append('probability_columns_checked')
        if (any(contract.get(k) != recipe[k] for k in bound)
                or any(checks.get(k) is not True for k in required)
                or type(receipt.get('prediction_count')) is not int or receipt['prediction_count'] < 1):
            raise ValueError('Custom export was not verified against original rows and model outputs')
        # The native writer verified every serialized cell before atomically publishing these bytes.
        return receipt
    with source.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        labels = receipt.get('score_class_order') or []
        if reader.fieldnames != ['row_id', 'prediction', *[f'score_{i}' for i in range(len(labels))]]:
            raise ValueError('Incorrect prediction columns for the scoring rule')
        count = 0
        for row in reader:
            if row['row_id'] != str(count) or not row['prediction'] or None in row or None in row.values():
                raise ValueError('Incomplete prediction or nonconsecutive row IDs')
            count += 1
    if not count or count != receipt.get('prediction_count'):
        raise ValueError('Incomplete prediction file')
    return receipt


def publish(source, output, *, phase, specification, reason=None):
    receipt = checked_receipt(source, specification)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Readers can always verify the hash; the separate first-delivery directory is immutable.
    descriptor, filename = tempfile.mkstemp(dir=output.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, 'wb') as dest, source.open('rb') as incoming:
            shutil.copyfileobj(incoming, dest)
            dest.flush()
            os.fsync(dest.fileno())
        if digest(temporary) != receipt['prediction_sha256']:
            raise ValueError('Prediction changed during publication')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt.update(prediction_file=str(output), delivery_phase=phase)
    if reason:
        receipt['delivery_issue'] = reason
    log = RunLogger(output.parent)
    log.write_json('prediction_manifest.json', receipt)
    log.write_json('prediction_attempt.json', dict(delivery_status=receipt.get('delivery_status','completed'),
                   generated_by_this_attempt=True, prediction_count=receipt['prediction_count'],
                   prediction_sha256=receipt['prediction_sha256'], delivery_phase=phase))
    return output


def first_prediction(features, run, output):
    agent = run/'agent'
    frozen = run/'first_delivery'
    selected = read(agent/'prior_selection.json')
    if (selected.get('model_format') != 'class_prior_v1' or selected.get('model_file') != 'partial_model.json'
            or selected.get('test_accessed') is not False or selected.get('delivery_status') != 'partial'):
        raise ValueError('First prediction requires a frozen training-label prior')
    with tempfile.TemporaryDirectory(dir=run, prefix='.first-') as temporary:
        stage = Path(temporary)/'snapshot'
        model = stage/'agent'
        model.mkdir(parents=True)
        for name, key in [('partial_model.json','model_sha256'), ('task_spec.json','task_spec_sha256')]:
            shutil.copyfile(agent/name, model/name)
            if digest(model/name) != selected[key]:
                raise ValueError('First prediction checkpoint changed')
        RunLogger(model).write_json('final_selection.json', selected)
        specification = read(model/'task_spec.json')
        source = create_predictions(str(features), model, stage/'predictions.csv')
        receipt = checked_receipt(source, specification)
        receipt.update(prediction_file=str(frozen/'predictions.csv'), input_source=str(features),
                       delivery_phase='first', task_spec_sha256=selected['task_spec_sha256'])
        snapshot = RunLogger(stage)
        snapshot.write_json('prediction_manifest.json', receipt)
        snapshot.write_json('first_delivery.json', dict(model_sha256=selected['model_sha256'],
                            prediction_sha256=receipt['prediction_sha256']))
        os.rename(stage, frozen)
    return publish(frozen/'predictions.csv', output, phase='first', specification=specification)


def final_prediction(features, run, output):
    """Verify the new export before replacing the early one. An early replay uses today's supplied inputs."""
    from .format_intake import DECODE_ERRORS
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.final-') as temporary:
        stage = Path(temporary)/'predictions.csv'
        try:
            create_predictions(str(features), run/'agent', stage)
            return publish(stage, output, phase='selected', specification=read(run/'agent/task_spec.json'))
        except DECODE_ERRORS + (RuntimeError,) as error:
            if not (run/'first_delivery'/'prediction_manifest.json').is_file():
                raise
            issue = type(error).__name__ + ': ' + str(error)[:300]
            create_predictions(str(features), run/'first_delivery'/'agent', stage)
            return publish(stage, output, phase='recovered_first', reason=issue,
                           specification=read(run/'first_delivery/agent/task_spec.json'))


if __name__ == '__main__':
    features, run, output = [Path(p).resolve() for p in sys.argv[1:]]
    try:
        first_prediction(features, run, output)
    except Exception as error:
        RunLogger(run).write_json('first_delivery_issue.json', dict(error=type(error).__name__,
                                  reason=str(error)[:350]))
        raise SystemExit(2)
