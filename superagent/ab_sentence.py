"""Pinned local sentence encoders for supervised lexical classifiers."""
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import style_differences


MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# A82 verified all three existing byte hashes at this publicly resolvable revision.
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
KERNEL_DIMENSIONS = 512
KERNEL_SEED = 12271
KERNEL_SAMPLE_LIMIT = 128
FILES = {
    'config.json':(612,'953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41'),
    'tokenizer.json':(466247,'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
    'model.safetensors':(90868376,'53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'),
}
DEEP_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'
# A113R verified all B-declared bytes at this publicly resolvable author revision.
DEEP_REVISION = 'a50ef00143b4d5391434df20ae11632588ac25be'
DEEP_ENV = 'SUPERAGENT_DEEP_MINILM_WEIGHTS'
DEEP_FILES = {
    'config.json':(615,'bc451f333af67312ba0de5018ef1c9ba663cb18549443e568f0bd35262dc1c48'),
    'tokenizer.json':FILES['tokenizer.json'],
    'model.safetensors':(133466304,'d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542'),
}
_ALT_TOKENIZER=(711396,'d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66')
ALTERNATIVES={
    'e5':dict(model='intfloat/e5-small-v2',revision='ffb93f3bd4047442299a41ebb6fa998a38507c52',
        env='SUPERAGENT_E5_SMALL_WEIGHTS',prefix='query: ',pool='mean',
        files={'config.json':(615,'5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f'),
               'tokenizer.json':_ALT_TOKENIZER,
               'model.safetensors':(133466304,'45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1')}),
    'bge':dict(model='BAAI/bge-small-en-v1.5',revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
        env='SUPERAGENT_BGE_SMALL_WEIGHTS',prefix='',pool='cls',
        files={'config.json':(743,'094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750'),
               'tokenizer.json':_ALT_TOKENIZER,
               'model.safetensors':(133466304,'3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad')}),
}


@lru_cache(maxsize=16)
def _checksum(path, identity):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verified_directory(*, deep=False):
    return verified_files(DEEP_ENV if deep else 'SUPERAGENT_MINILM_WEIGHTS',DEEP_FILES if deep else FILES)


def verified_files(env,files):
    folder = os.environ.get(env)
    if not folder or any(find_spec(p) is None for p in ('torch','transformers','tokenizers','safetensors')):
        return None
    root = Path(folder).resolve()
    for name,(size,sha) in files.items():
        path = root/name
        if not path.is_file():
            return None
        s = path.stat()
        if s.st_size!=size or _checksum(path,(s.st_size,s.st_ino,s.st_mtime_ns,s.st_ctime_ns))!=sha:
            return None
    return root


@lru_cache(maxsize=2)
def _encoder(root,max_length=256):
    import torch
    from tokenizers import Tokenizer
    from transformers import BertModel
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=max_length)
    model,info=BertModel.from_pretrained(root,local_files_only=True,use_safetensors=True,
                                         add_pooling_layer=False,output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
        raise ValueError('Incomplete MiniLM checkpoint')
    torch.set_num_threads(min(3,torch.get_num_threads()))
    return tokenizer, model.eval().requires_grad_(False)


def encode(texts, *, deep=False):
    root=verified_directory(deep=True) if deep else verified_directory()
    if root is None:
        key = DEEP_ENV if deep else 'SUPERAGENT_MINILM_WEIGHTS'
        raise ValueError(f'Set {key} to the directory of the pinned local files')
    tokenizer,model=_encoder(root,128) if deep else _encoder(root)
    return encode_frozen(texts,tokenizer,model)


def encode_alternative(texts,variant):
    spec=ALTERNATIVES[variant]
    root=verified_files(spec['env'],spec['files'])
    if root is None:
        raise ValueError('Missing verified local weights: '+spec['env'])
    tokenizer,model=_encoder(root,512)
    return encode_frozen(texts,tokenizer,model,prefix=spec['prefix'],pool=spec['pool'])


def encode_frozen(texts,tokenizer,model,*,prefix='',pool='mean'):
    import torch
    unique=list(dict.fromkeys(texts))
    if not unique:
        return np.empty((0,model.config.hidden_size),dtype=np.float32),0
    encoded=tokenizer.encode_batch([prefix+s for s in unique] if prefix else unique)
    lengths=np.array([len(x.ids) for x in encoded])
    order=np.argsort(lengths,kind='stable')
    vectors=np.empty((len(unique),model.config.hidden_size),dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0,len(order),48):
            idx=order[offset:offset+48]
            tokens=torch.zeros((len(idx),int(lengths[idx].max())),dtype=torch.long)
            mask=torch.zeros_like(tokens)
            for row,j in enumerate(idx):
                tokens[row,:lengths[j]]=torch.tensor(encoded[j].ids)
                mask[row,:lengths[j]]=1
            hidden=model(input_ids=tokens,attention_mask=mask).last_hidden_state
            pooled=hidden[:,0] if pool=='cls' else (hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
            vectors[idx]=torch.nn.functional.normalize(pooled,dim=1).numpy()
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite frozen features')
    indices={s:i for i,s in enumerate(unique)}
    return vectors[[indices[s] for s in texts]],int(lengths.sum())


class SentenceFeatures(TransformerMixin,BaseEstimator):
    encoder_model, encoder_revision = MODEL, REVISION
    def __init__(self,kind):
        self.kind=kind

    def fit(self,rows,y=None):
        texts=[s for pair in rows for s in pair] if self.kind=='pair_text' else list(rows)
        options=dict(min_df=2,sublinear_tf=True,max_features=40000)
        self.word_=TfidfVectorizer(ngram_range=(1,2),**options).fit(texts)
        self.char_=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),**options).fit(texts)
        if self.kind=='pair_text':
            raw=style_differences(rows)
            self.scaler_=StandardScaler(with_mean=False).fit(np.vstack([raw,-raw]))
        self.usage_={'input_tokens':0,'feature_transform_calls':0,'api_calls':0,'api_rmb':0}
        return self

    def embedding_features(self,dense):
        values = dense[::2]-dense[1::2] if self.kind=='pair_text' else dense
        return [csr_matrix(values)]

    def semantic_blocks(self,rows,texts):
        dense,tokens=encode(texts)
        return self.embedding_features(dense),tokens

    def transform(self,rows):
        if self.kind=='pair_text':
            texts=[s for pair in rows for s in pair]
            left, right = zip(*rows)
            matrices=[v.transform(left)-v.transform(right) for v in (self.word_,self.char_)]
            raw=self.scaler_.transform(style_differences(rows))
            matrices.append(csr_matrix(raw/np.sqrt(raw.shape[1])))
        else:
            texts=list(rows)
            matrices=[v.transform(rows) for v in (self.word_,self.char_)]
        blocks,n=self.semantic_blocks(rows,texts)
        self.usage_['input_tokens']+=n
        self.usage_['feature_transform_calls']+=1
        return hstack([*matrices,*blocks],format='csr')


class DeepPreferenceFeatures(SentenceFeatures):
    encoder_model, encoder_revision = DEEP_MODEL, DEEP_REVISION

    def __init__(self,kind,edit_context=False):
        super().__init__(kind)
        self.edit_context=edit_context

    def semantic_blocks(self,rows,texts):
        from .ab_edit_views import edit_view
        if self.kind!='pair_text':
            raise ValueError('A preference head requires text pairs')
        indices,contexts=[],[]
        if self.edit_context:
            for i,(left,right) in enumerate(rows):
                pair,changed=edit_view(left,right)
                if changed:
                    indices.append(i)
                    contexts.extend(pair)
        vectors,tokens=encode([*texts,*contexts],deep=True)
        dense=vectors[:len(texts)]
        full=dense[::2]-dense[1::2]
        blocks=[csr_matrix(full)]
        if self.edit_context:
            residual=np.zeros_like(full)
            edited=vectors[len(texts):]
            if indices:
                residual[indices]=edited[::2]-edited[1::2]-full[indices]
            blocks.append(csr_matrix(residual))
        return blocks,tokens


class FamiliarityPreferenceFeatures(DeepPreferenceFeatures):
    def __init__(self, kind):
        super().__init__(kind)

    def fit(self, rows, y=None):
        from .word_familiarity import WordFamiliarity

        super().fit(rows, y)
        self.corpus_ = WordFamiliarity().fit(text for pair in rows for text in pair)
        differences = self.corpus_.differences(rows)
        self.familiarity_scale_ = float(np.sqrt(np.mean(differences ** 2))) or 1.0
        return self

    def semantic_blocks(self, rows, texts):
        blocks, tokens = super().semantic_blocks(rows, texts)
        delta = self.corpus_.differences(rows) / self.familiarity_scale_
        return [*blocks, csr_matrix(delta[:, None])], tokens


class AlternativeSentenceFeatures(SentenceFeatures):
    def __init__(self,kind,variant):
        super().__init__(kind)
        self.variant=variant

    @property
    def encoder_model(self):
        return ALTERNATIVES[self.variant]['model']

    @property
    def encoder_revision(self):
        return ALTERNATIVES[self.variant]['revision']

    def semantic_blocks(self,rows,texts):
        values,tokens=encode_alternative(texts,self.variant)
        return self.embedding_features(values),tokens


class KernelSentenceFeatures(SentenceFeatures):
    """Add a fixed-size nonlinear utility for each individual sentence."""
    def fit(self,rows,y=None):
        super().fit(rows,y)
        texts = (text for pair in rows for text in pair) if self.kind=='pair_text' else iter(rows)
        def sample_order(text):
            return hashlib.sha256(text.encode('utf-8')).digest(),text
        sample = sorted(set(texts),key=sample_order)[:KERNEL_SAMPLE_LIMIT]
        vectors,tokens = encode(sample)
        distances = pdist(vectors,metric='sqeuclidean')
        positive = distances[distances>0]
        self.median_squared_distance_ = float(np.median(positive)) if len(positive) else None
        gamma = np.log(2.)/self.median_squared_distance_ if self.median_squared_distance_ is not None else 1.
        self.kernel_ = RBFSampler(gamma=gamma,n_components=KERNEL_DIMENSIONS,random_state=KERNEL_SEED).fit(vectors)
        self.usage_.update(bandwidth_sample_texts=len(sample),bandwidth_input_tokens=tokens)
        self.usage_['input_tokens'] += tokens
        return self

    def embedding_features(self,dense):
        original = super().embedding_features(dense)
        nonlinear = self.kernel_.transform(dense)
        extra = nonlinear[::2]-nonlinear[1::2] if self.kind=='pair_text' else nonlinear
        return [*original,csr_matrix(extra)]


def fit_sentence(rows,labels,query,kind,nonlinear=False,deep=False,edit_context=False,alternative=None,familiarity=False):
    if familiarity:
        features=FamiliarityPreferenceFeatures(kind)
    elif alternative:
        features=AlternativeSentenceFeatures(kind,alternative)
    elif deep:
        features=DeepPreferenceFeatures(kind,edit_context)
    else:
        features=KernelSentenceFeatures(kind) if nonlinear else SentenceFeatures(kind)
    features.fit(rows)
    x=features.transform(rows)
    if kind=='pair_text':
        if not np.array_equal(np.unique(labels),[0,1]):
            raise ValueError('Preference labels must encode the declared binary sides')
        x=vstack([x,-x])
        labels=np.r_[labels,1-np.asarray(labels)]
    classifier=LogisticRegression(C=.3,max_iter=2000,random_state=42).fit(x,labels)
    model=Pipeline([('features',features),('classifier',classifier)])
    model.training_metadata={'frozen_downstream_encoder':features.encoder_model,
                             'revision':features.encoder_revision,'usage':features.usage_}
    return model, model.predict(query)
