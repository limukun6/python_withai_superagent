"""Read declared task fields without guessing a target from its position or class count."""
import csv
import io
import re

from .metric_declarations import declared_metric


def heading(archive,path,fmt):
    if fmt=='parquet':
        import pyarrow.parquet as pq
        return pq.ParquetFile(io.BytesIO(archive.read(path))).schema_arrow.names
    rows=csv.reader(io.StringIO(archive.read(path).decode('utf-8-sig')),delimiter='\t' if fmt=='tsv' else ',')
    return next(rows,[])


def table(archive,path,fmt,columns=None,*,row_mask=None):
    if fmt=='parquet':
        import pyarrow.parquet as pq
        parquet=pq.ParquetFile(io.BytesIO(archive.read(path)))
        names=parquet.schema_arrow.names
        use=names if columns is None else columns
        if len(set(names))!=len(names) or not set(use)<=set(names):
            raise ValueError('Missing or duplicate Parquet columns')
        data=parquet.read(columns=use)
        if row_mask is not None:
            import pyarrow as pa
            if len(row_mask)!=data.num_rows:
                raise ValueError('Row mask differs from the declared table')
            data=data.filter(pa.array(row_mask,type=pa.bool_()))
        def scalar(value):
            if value is None:
                return ''
            if isinstance(value,(list,dict,tuple,bytes)):
                raise ValueError('Nested/binary table field has no declared scalar representation')
            return str(value)
        return use,([scalar(v) for v in row] for row in zip(*(c.to_pylist() for c in data.columns)))
    reader=csv.reader(io.StringIO(archive.read(path).decode('utf-8-sig')),delimiter='\t' if fmt=='tsv' else ',')
    names=next(reader,[])
    if not names or len(set(names))!=len(names):
        raise ValueError('Expected distinct nonempty table headers')
    use=names if columns is None else columns
    if not set(use)<=set(names):
        raise ValueError('A declared column is missing from the table')
    indexes=[names.index(n) for n in use]
    def rows():
        total=0
        for total,row in enumerate(reader,start=1):
            if row_mask is not None:
                if total>len(row_mask):
                    raise ValueError('Row mask is shorter than the declared table')
                if not row_mask[total-1]:
                    continue
            if len(row)!=len(names):
                raise ValueError('Table row width differs from header')
            yield [row[i] for i in indexes]
        if row_mask is not None and total!=len(row_mask):
            raise ValueError('Row mask is longer than the declared table')
    return use,rows()


def target_column(text,header):
    explicit=set()
    intro=r'(?:\b(?:target|label|response|outcome)\b(?:\s+(?:column|field|variable))?\s*(?:(?:is|named|called)\s+|[:=]\s*)?|(?:标签列|目标列|目标变量)\s*(?:为|是|[:：=])?\s*)'
    for name in header:
        token=r'(?<![\w])'+re.escape(name)+r'(?![\w])'
        if re.search(intro+r'[\"\'‘“]?'+token,text,re.I) or re.search(token+r'[\"\'’”]?\s+is\s+the\s+(?:target|label)\s+(?:column|field)',text,re.I):
            explicit.add(name)
        if re.search(r'(?:^|[.!?。]\s*|\n)\s*(?:predict|forecast|submit)\s+(?:the\s+)?[\"\x27]?'
                     +token+r'[\"\x27]?\s*(?:[.!?。]|$)',text,re.I):
            explicit.add(name)
    if len(explicit)>1:
        raise ValueError('Conflicting target declarations')
    # A declaration that names an absent field must not silently activate a conventional label feature.
    if explicit:
        return explicit.pop()
    if re.search(r'\b(?:label|target|response)\s+(?:column|field)\s*(?:is|named|called|[:=])\s|(?:目标列|标签列)\s*(?:为|是|[:：=])',text,re.I):
        raise ValueError('Declared target is absent from the table')
    conventional=[n for n in header if n.strip().lower() in ('target','label','y','class')]
    if len(conventional)!=1:
        raise ValueError('A unique declared target column is unavailable')
    return conventional[0]
