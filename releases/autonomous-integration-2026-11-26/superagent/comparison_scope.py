"""Reject explicit evidence-scope conflicts before interpreting numerical deltas."""


def comparison_scope(reference,result,*,primary_metric=False):
    keys=['split_id','evaluation_payload_sha256','label_sha256','row_ids_sha256']
    if primary_metric:keys+=['primary_metric','metric_spec']
    conflicts=[key for key in keys if (key in reference or key in result) and reference.get(key)!=result.get(key)]
    supplied=[key for key in keys if key in reference and key in result and reference[key] is not None and reference[key]==result[key]]
    def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    return dict(comparable=not conflicts,conflicts=conflicts,matched_fields=supplied,
        provenance_complete=isinstance(reference.get('split_id'),str) and bool(reference.get('split_id')) and
            all(key in supplied for key in ('split_id','evaluation_payload_sha256','label_sha256')) and
            all(digest(reference.get(key)) for key in ('evaluation_payload_sha256','label_sha256')),
        interpretation='Explicit conflicts or one-sided identities prevent comparison; absent legacy provenance is not independently verified.')
