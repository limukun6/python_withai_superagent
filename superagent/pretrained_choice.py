"""Optional, feedback-triggered local text transfer. No dataset identity enters the gate."""
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .pricing import resolve_pricing
from .task_budget import BudgetStateError, LIMIT, SHARED, TaskBudgetLedger, encoded, now
from .usage_audit import money_units

MODEL = 'qwen3-0.6b'
CANDIDATE = 'pretrained_qwen3_context_C0.3'
PARAMS = {'model': MODEL, 'C': 0.3, 'pooling': 'mean_final_hidden_l2', 'max_tokens': 64}
FILES = {
    'config.json': ('0ce186d90797c4c5cfefb0d3695517b37d02dd3a', '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd', 726),
    'tokenizer.json': ('0ce186d90797c4c5cfefb0d3695517b37d02dd3a', 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4', 11422654),
    'model.safetensors': ('d033240f3f8fbdda767f6c43ba6d856b6871eaa6', 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b', 1503300328),
}


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def acquire(directory, repository='Qwen/Qwen3-0.6B', files=None):
    import requests
    files = FILES if files is None else files
    directory.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, (revision, sha, size) in files.items():
        dest = directory / name
        if dest.is_file() and dest.stat().st_size == size and digest(dest) == sha:
            continue
        if os.environ.get('SUPERAGENT_OFFLINE') == '1':
            raise ValueError(f'Offline run requires the verified local file {dest.name}')
        url = f'https://modelscope.cn/models/{repository}/resolve/{revision}/{name}'
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            tmp = Path(stream.name)
            try:
                with requests.get(url, stream=True, timeout=(12, 40)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        if stream.tell() > size:
                            raise ValueError('Model file exceeds the pinned size')
                stream.flush()
                if tmp.stat().st_size != size or digest(tmp) != sha:
                    raise ValueError('Model file does not match official SHA-256')
                os.replace(tmp, dest)
                fetched.append(name)
            finally:
                tmp.unlink(missing_ok=True)
    return fetched


def budget(config, tokens, reserve=False, model_id=MODEL):
    """Atomically check both limits against the existing project ledger."""
    ledger = TaskBudgetLedger(config['ledger'])
    price = resolve_pricing(model_id)
    units = money_units(price.course_cost(tokens, 0))
    scope, run = config['scope'], config['run']
    ledger._scope(scope)
    cap = config['cap_rmb']
    if not math.isfinite(cap) or not 0 < cap <= 20:
        raise ValueError('Invalid local model allowance')
    with ledger._db() as db:
        db.execute('BEGIN IMMEDIATE')
        rows = db.execute('SELECT id,status,course_units FROM requests WHERE scope IN (?,?)', (scope, SHARED)).fetchall()
        unknown = {r['id'] for r in rows if r['status'] == 'unknown'}
        if any(r['status'] == 'reserved' for r in rows) or not unknown <= set(config.get('deferred_ids', [])):
            raise BudgetStateError('An unaccounted request is outside the declared historical deferral')
        used = db.execute("SELECT COALESCE(SUM(course_units),0) FROM requests WHERE scope=? AND json_extract(payload,'$.run_id')=?", (scope, run)).fetchone()[0]
        if 'campaign_cap_rmb' in config:
            campaign_cap = config['campaign_cap_rmb']
            if not math.isfinite(campaign_cap) or not 0 < campaign_cap <= 20:
                raise ValueError('Invalid cumulative experiment allowance')
            runs = set(config['previous_run_ids']) | {run}
            roots=[Path(p).resolve() for p in config.get('campaign_run_roots',[])]
            if roots:
                peers=db.execute("SELECT DISTINCT json_extract(payload,'$.run_id') FROM requests WHERE scope=?",(scope,))
                runs.update(name for (name,) in peers if name and Path(name).is_absolute()
                            and any(Path(name).resolve().is_relative_to(root) for root in roots))
            placeholders = ','.join('?' for _ in runs)
            campaign = db.execute(f"SELECT COALESCE(SUM(course_units),0) FROM requests WHERE scope=? AND json_extract(payload,'$.run_id') IN ({placeholders})", (scope,*runs)).fetchone()[0]
            if campaign + units > money_units(campaign_cap):
                raise BudgetStateError('Cumulative experiment token allowance exhausted')
        if used + units > money_units(cap):
            raise BudgetStateError('Local run token allowance exhausted')
        if sum(r['course_units'] or 0 for r in rows) + len(unknown) * 3_000_000_000 + units > LIMIT:
            raise BudgetStateError('Task capacity including historical deferral buffer is insufficient')
        if not reserve:
            return
        key = str(uuid.uuid4())
        payload = encoded(dict(run_id=run, requested_model=model_id, reserved_at=now(), input_tokens=tokens,
                               deferred_ids=sorted(unknown), historical_buffer_rmb=3*len(unknown)))
        db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)', (key, scope, 'reserved', units, None, payload))
        db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',
                   (key, 'reserved', now(), payload))
        db.commit()
        return key


def consider(kind, history, directory, labels, config, train_features=None, search_features=None):
    report = {'kind': kind, 'trial': len(history)+1, 'decision': 'skip', 'inner_rows': len(labels)}
    tabular = kind == 'categorical_tabular' and bool(config.get('tabular_weights'))
    if not tabular and kind not in ('single_text', 'pair_text'):
        return dict(report, reason='no_supported_text_input')
    if not tabular and not config.get('weights') and config.get('tabular_weights'):
        return dict(report, reason='text_weights_not_configured')
    successes = [r for r in history if r['status'] == 'completed']
    if len(successes) < 2:
        return dict(report, reason='insufficient_completed_comparisons')
    common = np.ones(len(labels), dtype=bool)
    for r in successes:
        predicted = joblib.load(directory / f"trial_{r['trial']}.joblib")['search_predictions']
        if np.shape(predicted) != np.shape(labels):
            raise ValueError('Search predictions do not align with labels')
        common &= np.asarray(predicted) != labels
    threshold = max(8, math.ceil(.05 * len(labels)))
    report.update(common_errors=int(common.sum()), required_errors=threshold,
                  observations=[{'candidate': r['candidate'], 'score': r['search_score']} for r in successes])
    if common.sum() < threshold:
        return dict(report, reason='few_shared_errors')
    if tabular:
        from .tabular_encoder import decide_level_transfer
        return decide_level_transfer(report, history, common, train_features, search_features, config)
    try:
        budget(config, 1)
    except BudgetStateError as exc:
        return dict(report, reason=str(exc))
    return dict(report, decision='try', reason='text_routes_share_errors; test a frozen contextual representation')


class Encoder:
    model_id = MODEL
    repository = 'Qwen/Qwen3-0.6B'
    files = FILES
    weight_key = 'weights'
    cache_name = 'context_vectors.sqlite'
    last_token = False

    def __init__(self, runtime):
        self.runtime = Path(os.environ.get('SUPERAGENT_PRETRAINED_RUNTIME') or runtime)
        self.config = json.loads(self.runtime.read_text())
        self.folder = self.runtime.parent
        self.db_path = self.folder / self.cache_name
        self.tokenizer = self.model = None

    def load(self):
        import torch
        from tokenizers import Tokenizer
        from transformers import Qwen3Model
        directory = Path(self.config[self.weight_key])
        new_files = acquire(directory, self.repository, self.files)
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        if self.last_token:
            self.tokenizer.enable_truncation(max_length=64)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        torch.set_num_threads(min(6, torch.get_num_threads()))
        dtype = torch.float16 if self.device == 'mps' else torch.float32
        self.model = Qwen3Model.from_pretrained(str(directory), local_files_only=True,
                    use_safetensors=True, torch_dtype=dtype, attn_implementation='sdpa').eval().to(self.device)
        with (self.folder / 'model_loads.jsonl').open('a') as log:
            log.write(json.dumps(dict(model=self.model_id, files=self.files, downloaded=new_files, device=self.device, utc=now()))+'\n')

    def forward(self, ids):
        import torch
        n = sum(map(len, ids))
        width = max(map(len, ids))
        matrix = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros_like(matrix)
        for i, seq in enumerate(ids):
            matrix[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        ledger = TaskBudgetLedger(self.config['ledger'])
        key = budget(self.config, n, reserve=True, model_id=self.model_id)
        try:
            with torch.inference_mode():
                mask = mask.to(self.device)
                hidden = self.model(input_ids=matrix.to(self.device), attention_mask=mask, use_cache=False).last_hidden_state
                if self.last_token:
                    pooled = hidden[torch.arange(len(ids), device=self.device), mask.sum(1)-1].float()
                else:
                    pooled = (hidden.float()*mask.unsqueeze(-1)).sum(1) / mask.sum(1,keepdim=True)
                result = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError('Non-finite representation')
        except BaseException:
            ledger.mark_unknown(key, 'Local forward interrupted', input_token_count_before_forward=n)
            raise
        ledger.settle(key, self.model_id, n, 0)
        from .events import RunLogger
        RunLogger(self.folder).usage(event='local_forward', model=self.model_id, request_id=key, input_tokens=n,
            output_tokens=0, padded_positions=int(matrix.numel()-n), course_cost_rmb=resolve_pricing(self.model_id).course_cost(n,0),
            actual_cost_rmb=0., usage_known=True, sequences=len(ids))
        return result

    def token_ids(self, texts):
        encoded_texts = self.tokenizer.encode_batch(texts, add_special_tokens=self.last_token)
        return [entry.ids[:64] or [151643] for entry in encoded_texts]

    def check_tokens(self, count):
        budget(self.config, count, model_id=self.model_id)

    def encode(self, texts):
        texts = list(texts)
        unique = dict.fromkeys(texts)
        keys = {t: hashlib.sha256(t.encode()).hexdigest() for t in unique}
        vectors = {}
        with sqlite3.connect(self.db_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY, v BLOB NOT NULL)')
            for t,k in keys.items():
                row = db.execute('SELECT v FROM vectors WHERE id=?',(k,)).fetchone()
                if row:
                    vectors[t] = np.frombuffer(row[0], dtype='<f4')
            missing = [t for t in unique if t not in vectors]
            if missing:
                if self.model is None:
                    self.load()
                entries = list(zip(missing, self.token_ids(missing), strict=True))
                self.check_tokens(sum(len(ids) for _, ids in entries))
                entries.sort(key=lambda pair: len(pair[1]))
                for off in range(0, len(entries), 32):
                    chunk = entries[off:off+32]
                    values = self.forward([s for _,s in chunk])
                    for (t,_),v in zip(chunk, values):
                        vectors[t] = v
                        db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)',(keys[t],v.astype('<f4').tobytes()))
                    db.commit()
        return np.stack([vectors[t] for t in texts])


class ContextClassifier:
    encoder_class = Encoder

    def __init__(self, kind, runtime):
        self.kind, self.runtime = kind, str(runtime)
        self.encoder = None

    def __getstate__(self):
        return {**self.__dict__, 'encoder': None}

    def dense_features(self, rows):
        if self.encoder is None:
            self.encoder = self.encoder_class(self.runtime)
        if self.kind == 'pair_text':
            combined = self.encoder.encode([text for pair in rows for text in pair])
            return combined[::2]-combined[1::2]
        return self.encoder.encode(rows)

    def features(self, rows, fit=False):
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from .models import _difference_features, style_differences
        dense = self.dense_features(rows)
        if fit:
            self.word = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=40_000)
            self.char = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5), min_df=2, sublinear_tf=True, max_features=40_000)
        if self.kind == 'pair_text':
            styles = style_differences(rows)
            if fit:
                self.scaler = StandardScaler(with_mean=False).fit(np.vstack([styles, -styles]))
            matrices = [_difference_features(v, rows, fit) for v in (self.word,self.char)]
            matrices.append(csr_matrix(self.scaler.transform(styles)/np.sqrt(styles.shape[1])))
        else:
            matrices = [(v.fit_transform(rows) if fit else v.transform(rows)) for v in (self.word,self.char)]
        return hstack([*matrices, csr_matrix(dense)]).tocsr()

    def fit(self, rows, labels):
        from scipy.sparse import vstack
        features = self.features(rows, fit=True)
        if self.kind == 'pair_text':
            features = vstack([features,-features])
            labels = np.concatenate([labels,1-labels])
        self.classifier = LogisticRegression(C=.3, max_iter=2000, random_state=42).fit(features,labels)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, rows):
        return self.classifier.predict(self.features(rows))

    def predict_proba(self, rows):
        return self.classifier.predict_proba(self.features(rows))


def candidate(kind, runtime):
    from .models import Candidate
    def fit(train, labels, search):
        model = ContextClassifier(kind, runtime).fit(train, labels)
        return model, model.predict(search)
    if kind not in ('pair_text','single_text'):
        raise ValueError('Context transfer requires text input')
    return Candidate(CANDIDATE, PARAMS, fit)
