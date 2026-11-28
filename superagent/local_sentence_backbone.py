"""Read local Sentence Transformers module definitions and apply their declared pooling."""
from functools import lru_cache
import hashlib
import inspect
import json

import numpy as np

from .ab_sentence import _checksum

MODULE = 'sentence_transformers.models.'
ENCODER_FAMILIES = {'albert','bert','deberta','deberta-v2','distilbert','electra','mpnet','roberta','xlm-roberta'}


def metadata(path):
    if path.stat().st_size > 1024*1024:
        raise ValueError('Oversized local encoder metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def describe(root):
    """Return a portable descriptor for supported one-pool packages, without importing their code."""
    parts = metadata(root/'modules.json')
    stages = [part['type'] for part in parts]
    if stages not in ([MODULE+'Transformer',MODULE+'Pooling'],
                      [MODULE+'Transformer',MODULE+'Pooling',MODULE+'Normalize']):
        raise ValueError('Unsupported sentence encoder stages')
    if (parts[0]['path'] or any(p.get('idx')!=i or p.get('kwargs') for i,p in enumerate(parts))
            or any(set(p)-{'idx','type','name','path','kwargs'} for p in parts)):
        raise ValueError('Unsupported sentence encoder module layout')
    pools = parts[1]['path']
    if not pools or not pools.replace('_','').isalnum():
        raise ValueError('Unsupported pooling folder')
    pool_path = pools+'/config.json'
    pool = metadata(root/pool_path)
    flags = [k for k,v in pool.items() if k.startswith('pooling_mode_') and v]
    legacy_modes = {'pooling_mode_mean_tokens':'mean','pooling_mode_cls_token':'cls','pooling_mode_max_tokens':'max'}
    if 'pooling_mode' in pool and not flags:
        pooling=pool['pooling_mode']
    else:
        pooling=legacy_modes.get(flags[0]) if len(flags)==1 else None
    if pooling not in ('mean','cls','max'):
        raise ValueError('Unsupported or ambiguous pooling operation')
    config = metadata(root/'config.json')
    width = config.get('hidden_size',config.get('dim'))
    topology = config.get('model_type')
    if (topology not in ENCODER_FAMILIES or config.get('auto_map') or config.get('is_encoder_decoder')
            or config.get('is_decoder') or type(width) is not int or width<=0
            or pool.get('word_embedding_dimension',pool.get('embedding_dimension'))!=width):
        raise ValueError('Unsupported local text backbone')
    settings = metadata(root/'sentence_bert_config.json')
    length = settings.get('max_seq_length')
    if type(length) is not int or not 1<=length<=config.get('max_position_embeddings',0):
        raise ValueError('Invalid declared sentence length')
    if set(settings)-{'max_seq_length','do_lower_case','model_args'} or type(settings.get('do_lower_case',False)) is not bool:
        raise ValueError('Unsupported tokenizer settings')
    model_args=settings.get('model_args',{})
    if not isinstance(model_args,dict) or set(model_args)-{'add_pooling_layer'} or model_args.get('add_pooling_layer',False) is not False:
        raise ValueError('Unsupported encoder initialization options')
    files = ['modules.json',pool_path,'sentence_bert_config.json','config.json','tokenizer.json','model.safetensors']
    for optional in ('config_sentence_transformers.json','tokenizer_config.json'):
        if (root/optional).is_file():
            item=metadata(root/optional)
            if item.get('default_prompt_name') or item.get('auto_map') or item.get('truncation_side','right')!='right':
                raise ValueError('Unsupported default prompt or tokenizer behavior')
            files.append(optional)
    records={}
    for name in files:
        path=root/name
        stat=path.stat()
        records[name]=dict(bytes=stat.st_size,sha256=_checksum(path,(stat.st_size,stat.st_ino,stat.st_mtime_ns,stat.st_ctime_ns)))
    signature=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(signature=signature,environment_variable='SUPERAGENT_SENTENCE_ASSET_'+signature.upper()+'_WEIGHTS',width=width,
                family=topology,pooling=pooling,normalize=len(stages)==3,max_length=length,
                lower_case=settings.get('do_lower_case',False),files=records)


@lru_cache(maxsize=2)
def backbone(root,length,identity):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModel

    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=length)
    conf=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    model_cls=AutoModel._model_mapping[type(conf)]
    options={'add_pooling_layer':False} if 'add_pooling_layer' in inspect.signature(model_cls).parameters else {}
    if metadata(root/'sentence_bert_config.json').get('model_args') and not options:
        raise ValueError('The installed backbone cannot apply the declared encoder option')
    model,info=AutoModel.from_pretrained(root,config=conf,local_files_only=True,trust_remote_code=False,
                                         use_safetensors=True,output_loading_info=True,**options)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Incomplete local sentence encoder')
    torch.set_num_threads(min(3,torch.get_num_threads()))
    return tokenizer,model.eval().requires_grad_(False)


def encode(root,spec,texts):
    import torch
    if describe(root)!=spec:
        raise ValueError('Local sentence encoder bytes or behavior have changed')
    tokenizer,model=backbone(root,spec['max_length'],spec['signature'])
    chunks=tokenizer.encode_batch([s.strip().lower() if spec['lower_case'] else s.strip() for s in texts])
    order=sorted(range(len(chunks)),key=lambda i:len(chunks[i].ids))
    result=np.zeros((len(texts),spec['width']),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(order),40):
            indices=order[start:start+40]
            size=len(chunks[indices[-1]].ids)
            ids=torch.full((len(indices),size),model.config.pad_token_id,dtype=torch.long)
            mask=torch.zeros_like(ids)
            for row,i in enumerate(indices):
                tokens=chunks[i].ids
                ids[row,:len(tokens)]=torch.as_tensor(tokens)
                mask[row,:len(tokens)]=1
            hidden=model(input_ids=ids,attention_mask=mask).last_hidden_state
            if spec['pooling']=='cls':
                pooled=hidden[:,0,:]
            elif spec['pooling']=='max':
                pooled=hidden.masked_fill(~mask.bool().unsqueeze(-1),float('-inf')).amax(1)
            else:
                pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
            if spec['normalize']:
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
            result[indices]=pooled.numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite local sentence representation')
    return result
