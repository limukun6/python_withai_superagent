"""Frozen Sentence Transformers encoder for downstream features. No generation API."""
import json
from pathlib import Path
import time

import numpy as np

from .pretrained_choice import ContextClassifier, Encoder, acquire
from .task_budget import now

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CANDIDATE = 'minilm_category_levels'
REVISION = '524b52f1d626e9a5a16462e141fc9dffc25d5aeb'
FILES = {
    'config.json': (REVISION, '953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41', 612),
    'tokenizer.json': (REVISION, 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037', 466247),
    'model.safetensors': (REVISION, '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db', 90868376),
}
DEEP_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'
DEEP_REVISION = '52f9d2dd711ffc3434e6cff17284b13a11629a90'
DEEP_FILES = {
    'config.json': (DEEP_REVISION,'bc451f333af67312ba0de5018ef1c9ba663cb18549443e568f0bd35262dc1c48',615),
    'tokenizer.json': (DEEP_REVISION,'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037',466247),
    'model.safetensors': (DEEP_REVISION,'d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542',133466304),
}


class MiniLMEncoder(Encoder):
    model_id = MODEL
    files = FILES
    weight_key = 'minilm_weights'
    max_length = 256
    cache_name = 'minilm_category_vectors.sqlite'

    def load(self):
        import torch
        from transformers import BertModel
        from tokenizers import Tokenizer
        directory = Path(self.config[self.weight_key])
        started = time.monotonic()
        downloaded = acquire(directory, self.model_id, self.files)
        self.tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_length)
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.model, info = BertModel.from_pretrained(str(directory), use_safetensors=True, local_files_only=True,
                                                     add_pooling_layer=False, output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
            raise ValueError('The pinned checkpoint did not initialize the complete MiniLM encoder')
        self.model.eval().requires_grad_(False)
        with (self.folder/'model_loads.jsonl').open('a') as log:
            log.write(json.dumps(dict(model=self.model_id, role='downstream_features',files=self.files, downloaded=downloaded,
                                      device='cpu', seconds=time.monotonic()-started, utc=now()))+'\n')

    def token_ids(self, texts):
        return [item.ids for item in self.tokenizer.encode_batch(texts,add_special_tokens=True)]

    def check_tokens(self, count):
        # This local feature transformer has no language-model API allowance.
        if count<=0:raise ValueError('Expected nonempty MiniLM batch')

    def forward(self, ids):
        import torch
        start = time.monotonic()
        matrix = torch.zeros((len(ids),max(map(len,ids))),dtype=torch.long)
        mask = torch.zeros_like(matrix)
        for i, seq in enumerate(ids):
            matrix[i,:len(seq)] = torch.tensor(seq)
            mask[i,:len(seq)] = 1
        with torch.inference_mode():
            hidden = self.model(input_ids=matrix,attention_mask=mask).last_hidden_state
            pooled = (hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            vectors = torch.nn.functional.normalize(pooled,dim=1).numpy()
        if vectors.shape!=(len(ids),384) or not np.isfinite(vectors).all():
            raise ValueError('Invalid MiniLM features')
        n=sum(map(len,ids))
        with (self.folder/'downstream_usage.jsonl').open('a') as log:
            log.write(json.dumps(dict(event='frozen_feature_forward',model=self.model_id,utc=now(),sequences=len(ids),
                input_tokens=n,output_tokens=0,padded_positions=matrix.numel()-n,device='cpu',seconds=time.monotonic()-start,
                remote_calls=0,api_rmb=0.0))+'\n')
        return vectors


class DeeperMiniLMEncoder(MiniLMEncoder):
    model_id = DEEP_MODEL
    files = DEEP_FILES
    weight_key = 'deep_minilm_weights'
    max_length = 128
    cache_name = 'minilm_l12_vectors.sqlite'


class MiniLMTextClassifier(ContextClassifier):
    encoder_class = MiniLMEncoder

    def __init__(self, kind, runtime, dense_only):
        super().__init__(kind, runtime)
        self.dense_only = dense_only

    def features(self, rows, fit=False):
        if self.dense_only:
            from scipy.sparse import csr_matrix
            return csr_matrix(self.dense_features(rows))
        return super().features(rows, fit)

    @property
    def weight_setting(self):
        return getattr(self.encoder_class,'weight_key','minilm_weights')

    def bind_prediction_weights(self, weights, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        runtime = directory / 'pretrained_runtime.json'
        settings = {self.weight_setting:str(Path(weights).resolve())}
        if runtime.exists() and json.loads(runtime.read_text()) != settings:
            raise ValueError('Prediction cache belongs to a different encoder configuration')
        if not runtime.exists():
            runtime.write_text(json.dumps(settings))
        self.runtime = str(runtime)
        self.encoder = None


class DeeperMiniLMTextClassifier(MiniLMTextClassifier):
    encoder_class = DeeperMiniLMEncoder


def feature_usage(directory):
    log = Path(directory) / 'downstream_usage.jsonl'
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if any(e['model'] not in (MODEL,DEEP_MODEL) or e['remote_calls'] != 0 or e['api_rmb'] != 0 for e in events):
        raise ValueError('Unexpected event in frozen feature ledger')
    return dict(batches=len(events), input_tokens=sum(e['input_tokens'] for e in events),
                output_tokens=0, api_rmb=0, remote_calls=0)
