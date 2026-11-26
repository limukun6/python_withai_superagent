"""Compare submission recipes by CSV-visible semantics rather than optional JSON spelling."""
from superagent.output_recipe import validate_recipe


def canonical_csv_recipe(recipe,labels):
    if recipe is None:return None
    result=validate_recipe(recipe,labels)
    for column in result['columns']:
        if column['kind']=='input_id':column['unique']=column.get('unique',True)
        elif column['kind']=='label':
            mapping=column.get('mapping')
            rendered={str(label):str(mapping[str(label)] if mapping is not None else label) for label in labels}
            if len(set(rendered.values()))!=len(labels):
                raise ValueError('Label mapping collapses classes when serialized as CSV')
            column['mapping']=rendered
    return result


def equivalent_csv_recipe(actual,expected,labels):
    try:return canonical_csv_recipe(actual,labels)==canonical_csv_recipe(expected,labels)
    except (ValueError,TypeError,KeyError):return False


def grade_with_csv_semantics(task,query,oracle):
    from scripts.sa7_multiturn_cases import grade
    from superagent.metrics import MetricSpec
    result=grade(task,query,oracle)
    result['raw_output_dictionary_equal']=result['checks']['output']
    result['checks']['output']=equivalent_csv_recipe(task.spec.get('output_recipe'),oracle['expected']['output'],list(MetricSpec.for_task(task).labels))
    result['complete']=all(result['checks'].values())
    result['accepted_wrong_contract']=not result['complete']
    return result
