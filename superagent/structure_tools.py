"""Bounded structure inspection. No array values, pixels or pickle decoding."""
import io
from pathlib import PurePosixPath
import zipfile

import numpy as np

INSTRUCTION=(
    'Available recovery tools also include '
    '{action:"inspect_array",member:<npy/npz member>,key:null,evidence_refs:[<document ID>],source:"training"|"prediction"} '
    '(null key lists NPZ array keys with shapes/dtypes; key selects one), and '
    '{action:"inspect_image",member:<one image file>,evidence_refs:[<document ID>],source:"training"|"prediction"}. '
    'They return metadata only, never values. Use structure and document meaning together to choose descriptors. '
    'All tools and repairs share the original request budget and a total two-tool allowance. '
    'When both training and prediction structure are unknown, prefer one '
    '{action:"inspect_batch",requests:[<one inspect_header/inspect_array/inspect_image object>,<second tool object>]} '
    'so both observations arrive in one model turn. At most two underlying tools total; nested batches are invalid. '
    'The saved model request remains available for experiment planning; never inspect answers. '
    'When a proposal fails, inspect missing facts or revise it; do not repeat the same failed action without new evidence. '
    'Never choose between different target definitions by comparing classification scores. '
    'If supplied evidence genuinely cannot define a required task meaning and structural inspection cannot resolve it, '
    'you may return {action:"report_unresolved",reason:<explanation>,questions:[{field,question,evidence_refs}]}. '
    'This records an unresolved task, never a successful delivery. Cite the actual documents, use critical fields '
    'training/prediction/target/columns/modality/metric/split/output, and do not use this action for fixable unknown headers/keys. '
    'Do not repeatedly restate the same unresolved field without new evidence. All documents are already supplied.'
)


def array_header(stream):
    from .semantic_contract import _require
    version=np.lib.format.read_magic(stream)
    _require(version in ((1,0),(2,0)),'Array header version is not supported for inspection')
    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,order,dtype=reader(stream,max_header_size=10000)
    _require(len(shape)<=8 and all(type(n) is int and n>=0 for n in shape),'Invalid array shape')
    _require(not dtype.hasobject,'Object arrays are not inspectable or executable')
    return dict(shape=list(shape),dtype=dtype.str,fortran_order=order,values_disclosed=False)


def inspect_structure(files, workspace, request):
    from .semantic_contract import _require,_not_label_vector
    from .format_sources import checked_name,roles
    from .task_files import evaluation_path
    action=request.get('action')
    required={'action','member','evidence_refs'} | ({'key'} if action=='inspect_array' else set())
    _require(action in ('inspect_array','inspect_image') and set(request) in (required,required|{'source'}),'Invalid structure inspection request')
    source=request.get('source','training')
    _require(source in ('training','prediction'),'Invalid inspection source')
    refs=request['evidence_refs'];known={d['id'] for d in workspace['documents']}
    _require(isinstance(refs,list) and refs and all(isinstance(r,str) and r in known for r in refs),'Inspection lacks real document citations')
    member=checked_name(request['member'])
    _require(member in files.namelist() and not roles(member)&{'validation'},'Inspection source is not admissible')
    _require(source!='training' or not evaluation_path(member,'train'),'Training inspection cannot open evaluation data')
    _not_label_vector(member)
    _require(files.getinfo(member).file_size<=64*1024*1024,'Inspection member exceeds 64 MiB limit')
    suffix=PurePosixPath(member).suffix.casefold()
    result=dict(member=member,source=source,values_disclosed=False)
    if action=='inspect_image':
        from PIL import Image
        from .image_files import BITMAP_FORMATS,MAX_SOURCE_PIXELS
        _require(suffix in BITMAP_FORMATS,'Unsupported image extension')
        with Image.open(io.BytesIO(files.read(member)),formats=list(set(BITMAP_FORMATS.values()))) as image:
            _require(image.format==BITMAP_FORMATS[suffix],'Image format disagrees with extension')
            _require(image.width*image.height<=MAX_SOURCE_PIXELS,'Image exceeds inspection pixel limit')
            _require(getattr(image,'n_frames',1)==1,'Multiframe image cannot be executed by installed decoder')
            result.update(width=image.width,height=image.height,mode=image.mode,format=image.format,pixels_disclosed=False)
        return result
    _require(suffix in ('.npy','.npz'),'Array inspection requires NPY or NPZ')
    key=request['key'];_require(key is None or isinstance(key,str) and key,'Invalid array key')
    if suffix=='.npy':
        _require(key is None,'NPY has no named array key')
        result.update(format='npy',**array_header(io.BytesIO(files.read(member))))
        return result
    with zipfile.ZipFile(io.BytesIO(files.read(member))) as archive:
        names=archive.namelist()
        _require(len(names)<=64 and len(set(names))==len(names),'NPZ has excessive or duplicate keys')
        _require(all(checked_name(n)==n and n.endswith('.npy') for n in names),'Invalid NPZ array members')
        _require(sum(i.file_size for i in archive.infolist())<=128*1024*1024,'NPZ expanded size exceeds inspection limit')
        _require(key is None or key+'.npy' in names,'Requested array key does not exist; use key=null for inventory')
        arrays=[]
        for name in names:
            label=name[:-4]
            if key is not None and key!=label:continue
            blocked=bool(roles(label)&{'test','validation'})
            if source=='prediction':
                try:_not_label_vector(label)
                except ValueError:blocked=True
            if blocked:
                _require(key is None,'Selected key belongs to evaluation labels or a held-out source')
                arrays.append(dict(key=label,status='not_opened_role_restricted'));continue
            with archive.open(name) as stream:metadata=array_header(stream)
            arrays.append(dict(key=label,**metadata))
        result.update(format='npz',arrays=arrays)
        return result
