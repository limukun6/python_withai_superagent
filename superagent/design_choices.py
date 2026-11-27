"""Separate cited task constraints from agent-selected internal decoding parameters."""
from copy import deepcopy

FIELDS=('width','height','mode','resize')


def compile_design_decisions(response,workspace,plan):
    from .semantic_contract import ContractProbeError
    from .image_files import image_contract
    def require(condition,message):
        if not condition:raise ContractProbeError('Design decision: '+message)
    decisions=response['design_decisions']
    require(isinstance(decisions,dict),'design_decisions must be an object')
    if plan['modality']!='image':
        require(decisions=={},'only image_preprocessing is an agent design field in this version')
        return {}
    require(set(decisions)=={'image_preprocessing'},'image task must record its preprocessing decision')
    entry=decisions['image_preprocessing']
    require(isinstance(entry,dict) and set(entry)=={'task_constraints','constraint_evidence','reason','evidence_refs'},'invalid image decision fields')
    chosen=image_contract(plan['image_preprocessing']);constraints=entry['task_constraints'];citations=entry['constraint_evidence']
    require(isinstance(constraints,dict) and set(constraints)<=set(FIELDS),'unknown task constraint parameter')
    require(isinstance(citations,dict) and set(citations)==set(constraints),'each declared constraint needs its own citation')
    known={doc['id'] for doc in workspace['documents']}
    def check_refs(refs):
        return isinstance(refs,list) and bool(refs) and all(isinstance(r,str) and r in known for r in refs)
    require(check_refs(entry['evidence_refs']),'design context needs real task document citations')
    require(isinstance(entry['reason'],str) and 0<len(entry['reason'].strip())<=1200,'agent choice needs a bounded reason')
    for key,value in constraints.items():
        require(check_refs(citations[key]),'missing/fabricated constraint citation: '+key)
        require(type(value) is type(chosen[key]) and value==chosen[key],'chosen preprocessing violates declared constraint: '+key)
    return dict(image_preprocessing=dict(version='image-design-decision-v1',chosen=deepcopy(chosen),
        task_constraints=deepcopy(constraints),constraint_evidence=deepcopy(citations),
        parameter_origins={key:'task_requirement' if key in constraints else 'agent_design_choice' for key in FIELDS},
        reason=entry['reason'],evidence_refs=deepcopy(entry['evidence_refs']),
        constraint_values_enforced=True,source_citations_checked=True,semantic_requirement_extraction_certified=False))
