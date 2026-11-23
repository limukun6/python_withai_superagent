"""Construct an antisymmetric feature from an identified frozen BERT encoder's two text segments."""
import numpy as np
import torch

from superagent.ab_sentence import ALTERNATIVES, _encoder, verified_files


def segment_differences(pairs):
    spec = ALTERNATIVES['e5']
    root = verified_files(spec['env'], spec['files'])
    if root is None:
        raise ValueError('Paired context needs the verified local E5 files')
    tokenizer, model = _encoder(root, 512)
    if model.config.type_vocab_size < 2:
        raise ValueError('This encoder lacks two segment embeddings')
    pref = spec['prefix']
    codes = tokenizer.encode_batch([(pref+left,pref+right) for left,right in pairs])
    lengths = np.array([len(c.ids) for c in codes])
    order = np.argsort(lengths,kind='stable')
    scores = np.empty((len(pairs),model.config.hidden_size),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(order),32):
            ix = order[start:start+32]
            shape = (len(ix),int(lengths[ix].max()))
            tokens = torch.zeros(shape,dtype=torch.long)
            mask = torch.zeros_like(tokens)
            types = torch.zeros_like(tokens)
            segment = torch.full_like(tokens,-1)
            for i,j in enumerate(ix):
                length = lengths[j]
                code = codes[j]
                tokens[i,:length] = torch.tensor(code.ids)
                mask[i,:length] = 1
                types[i,:length] = torch.tensor(code.type_ids)
                segment[i,:length] = torch.tensor([s if s is not None else -1 for s in code.sequence_ids])
            h = model(input_ids=tokens,attention_mask=mask,token_type_ids=types).last_hidden_state
            pooled = []
            for which in (0,1):
                keep = segment.eq(which)
                size = keep.sum(dim=1,keepdim=True)
                if (size==0).any() or not torch.equal(types[keep], torch.full_like(types[keep],which)):
                    raise ValueError('Tokenizer failed to preserve both content segments')
                mean = (h*keep[:,:,None]).sum(dim=1)/size
                pooled.append(torch.nn.functional.normalize(mean,dim=1))
            scores[ix] = (pooled[0]-pooled[1]).numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite contextual feature')
    return scores, int(lengths.sum()), int(lengths.max()) if len(lengths) else 0


def contextual_differences(rows):
    canon = [tuple(sorted(row)) for row in rows]
    unique = list(dict.fromkeys(canon))
    ordered = [order for l,r in unique for order in ((l,r),(r,l))]
    raw,tokens,max_length = segment_differences(ordered)
    vectors = (raw[::2]-raw[1::2])*.5
    ix = {key:i for i,key in enumerate(unique)}
    orientation = np.array([1 if l<r else -1 if l>r else 0 for l,r in rows],dtype=np.float32)
    return vectors[[ix[k] for k in canon]]*orientation[:,None], dict(
        unique_pairs=len(unique),paired_forward_sequences=len(ordered),local_input_tokens=tokens,max_input_tokens=max_length)


def norm_scale(baseline, residual):
    base_rms = np.sqrt(np.mean(np.einsum('ij,ij->i',baseline,baseline)))
    residual_rms = np.sqrt(np.mean(np.einsum('ij,ij->i',residual,residual)))
    return float(base_rms/residual_rms) if residual_rms>1e-12 else 0.
