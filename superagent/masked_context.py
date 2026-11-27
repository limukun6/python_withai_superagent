"""Frozen masked-token surprisal summaries for text, independent of task labels and responses."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from .paired_encoder import text_key
from .ab_sentence import ALTERNATIVES, pinned_folder

SPEC=ALTERNATIVES["masked_roberta"]
REVISION=SPEC["revision"]
CACHE_NAME=f"masked-roberta-{REVISION}-four-v1.npz"
ENV=SPEC["env"]


@lru_cache(maxsize=1)
def load_model(root):
    import torch
    from tokenizers import Tokenizer
    from transformers import RobertaForMaskedLM
    tokenizer=Tokenizer.from_file(str(root/"tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    model,info=RobertaForMaskedLM.from_pretrained(root,local_files_only=True,use_safetensors=True,output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError("The masked encoder or its prediction head was not fully loaded")
    torch.set_num_threads(min(3,torch.get_num_threads()))
    return tokenizer,model.eval().requires_grad_(False)


def content_positions(encoded):
    plain=[i for i,special in enumerate(encoded.special_tokens_mask) if not special]
    if len(plain)<=4:
        return plain
    return [plain[i] for i in np.rint(np.linspace(0,len(plain)-1,4)).astype(int)]


def encode(texts,root):
    import torch
    tokenizer,model=load_model(root)
    encoded=tokenizer.encode_batch(list(texts))
    cases=[(i,j) for i,seq in enumerate(encoded) for j in content_positions(seq)]
    cases.sort(key=lambda case:len(encoded[case[0]].ids))
    result=[[] for _ in encoded]
    mask_id=tokenizer.token_to_id("<mask>")
    if mask_id is None:
        raise ValueError("Verified RoBERTa tokenizer lacks a mask token")
    with torch.inference_mode():
        for start in range(0,len(cases),48):
            batch=cases[start:start+48]
            width=max(len(encoded[i].ids) for i,_ in batch)
            ids=torch.full((len(batch),width),model.config.pad_token_id,dtype=torch.long)
            attention=torch.zeros_like(ids)
            positions=torch.as_tensor([j for _,j in batch])
            actual=torch.as_tensor([encoded[i].ids[j] for i,j in batch])
            indices=torch.arange(len(batch))
            for k,(i,j) in enumerate(batch):
                sequence=encoded[i].ids
                ids[k,:len(sequence)]=torch.as_tensor(sequence)
                ids[k,j]=mask_id
                attention[k,:len(sequence)]=1
            states=model.roberta(input_ids=ids,attention_mask=attention).last_hidden_state[indices,positions]
            losses=torch.nn.functional.cross_entropy(model.lm_head(states),actual,reduction="none").numpy()
            for (i,j),loss in zip(batch,losses):
                result[i].append((j,float(loss)))
    values=np.zeros((len(encoded),5),dtype=np.float32)
    for i,entries in enumerate(result):
        if entries:
            scores=np.asarray([value for _,value in sorted(entries)])
            values[i]=[scores.mean(),scores.max(),scores.std(),scores[0],scores[-1]]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite masked context features")
    return values


@lru_cache(maxsize=1)
def read_bank(path,size,mtime):
    with np.load(path,allow_pickle=False) as archive:
        keys,values=archive["keys"],archive["vectors"]
    if values.shape!=(len(keys),5) or len(set(keys.tolist()))!=len(keys) or not np.isfinite(values).all():
        raise ValueError("Invalid masked context feature bank")
    return dict(zip(keys.tolist(),values))


def vectors(texts):
    out={}
    unique=list(dict.fromkeys(texts))
    directory=os.environ.get("SUPERAGENT_PAIRED_FEATURE_CACHE")
    if directory:
        path=Path(directory)/CACHE_NAME
        if path.is_file():
            stat=path.stat()
            bank=read_bank(path,stat.st_size,stat.st_mtime_ns)
            out={s:bank[key] for s in unique if (key:=text_key(s)) in bank}
    missing=[s for s in unique if s not in out]
    if missing:
        root=pinned_folder(Path(os.environ[ENV]),SPEC["files"])
        if root is None:
            raise ValueError("Missing verified local masked encoder")
        out.update(zip(missing,encode(missing,root)))
    return np.asarray([out[s] for s in texts],dtype=np.float32).reshape(len(texts),5)


from .paired_encoder import PairedEncoderFeatures


class MaskedPairFeatures(PairedEncoderFeatures):
    additional_frozen_sentence_variants = ("masked_roberta",)

    def fit(self, rows, y=None):
        super().fit(rows,y)
        differences=self._differences(rows)
        self.masked_scale_=np.maximum(np.sqrt(np.mean(differences**2,axis=0)),1e-6)
        return self

    @staticmethod
    def _differences(rows):
        dense=vectors([text for pair in rows for text in pair])
        return dense[::2]-dense[1::2]

    def semantic_blocks(self, rows, texts):
        from scipy.sparse import csr_matrix
        blocks,tokens=super().semantic_blocks(rows,texts)
        mask_features=np.clip(self._differences(rows)/self.masked_scale_,-8.,8.)/np.sqrt(5.)
        return [*blocks,csr_matrix(mask_features)],tokens


def materialize(spec):
    from functools import partial
    from .open_methods import digest
    from .models import Candidate
    from .paired_encoder import candidates, fit_ranker, NAMES
    if (set(spec)!={"kind","operator","template"} or spec["kind"]!="pair_text" or
        spec["operator"]!="add_masked_context" or NAMES.get(spec["template"],(None,None))[1]!="linear"):
        raise ValueError("Unsupported masked-context proposal")
    base=next(c for c in candidates() if c.family==spec["template"])
    variant,head=NAMES[base.family]
    params=dict(base.params,parent_template=base.family,context_encoder=SPEC["model"],context_revision=REVISION,
                context_representation="masked_token_surprisal",masked_positions=4)
    return Candidate("masked_context_"+digest(spec)[:20],params,
                     partial(fit_ranker,variant=variant,head=head,feature_type=MaskedPairFeatures))
