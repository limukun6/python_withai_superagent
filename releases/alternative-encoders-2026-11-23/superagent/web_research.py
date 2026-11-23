"""Bounded public literature retrieval, without model keys or project data.

One arXiv query, then at most one Crossref fallback. This retrieves metadata and
abstracts, not arbitrary websites or full papers. Responses are untrusted data.
"""
from datetime import datetime,timezone
import hashlib
import re
import time
import xml.etree.ElementTree as ET

import requests


MAX_BYTES=256*1024
MAX_RESULTS=3
STOP_WORDS={'a','an','the','and','or','for','with','from','using','of','in','on','to','generic','method','methods','improve','improvement'}


def clean(value,maximum):return ' '.join(str(value or '').split())[:maximum]


def source_record(identifier,title,summary,url,published,provider):
    identity=hashlib.sha256(str(identifier).encode()).hexdigest()[:16]
    return {'source_id':provider+'_'+identity,'title':clean(title,240),'summary':clean(summary,1600),
            'url':str(url),'published':clean(published,40),'provider':provider,'untrusted_content':True}


class LiteratureSearch:
    def __init__(self,cache_dir=None):
        self.records=[];self.cache_dir=cache_dir

    def _fetch(self,url,params):
        started=time.monotonic()
        # Separate session: never reuse the school's model Authorization header.
        with requests.Session() as session:
            session.trust_env=False
            response=session.get(url,params=params,headers={'User-Agent':'AIAA3102-SuperAgent/0.40 (bounded academic metadata research)'},
                timeout=(8,15),allow_redirects=False,stream=True)
            response.raise_for_status()
            chunks=[];size=0
            for chunk in response.iter_content(8192):
                size+=len(chunk)
                if size>MAX_BYTES:raise ValueError('Research response exceeds256KiB')
                chunks.append(chunk)
                if time.monotonic()-started>25:raise TimeoutError('Research download exceeded25seconds')
            body=b''.join(chunks)
            body_hash=hashlib.sha256(body).hexdigest();saved=None
            if self.cache_dir is not None:
                from pathlib import Path
                directory=Path(self.cache_dir);directory.mkdir(parents=True,exist_ok=True)
                saved=directory/(body_hash+'.body')
                if not saved.exists():
                    with saved.open('xb') as stream:stream.write(body)
            return body,{'url':response.url,'http_status':response.status_code,'bytes':size,
                'sha256':body_hash,'raw_body_path':str(saved) if saved is not None else None,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':time.monotonic()-started,'authorization_header_sent':False}

    def search(self,query):
        if not isinstance(query,str) or not 8<=len(query)<=240 or any(c in query for c in '\n\r\x00'):
            raise ValueError('Research query must be one short generic-method question')
        terms=[]
        for word in re.findall(r'\w+(?:-\w+)*',query.casefold()):
            if word not in STOP_WORDS and word not in terms:terms.append(word)
        terms=terms[:4]
        if not terms:raise ValueError('Research query has no method keywords')
        arxiv_query=' AND '.join('all:'+word for word in terms)
        attempts=[];sources=[]
        for provider in ('arxiv','crossref'):
            try:
                if provider=='arxiv':
                    body,receipt=self._fetch('https://export.arxiv.org/api/query',{'search_query':arxiv_query,'start':0,'max_results':MAX_RESULTS})
                    root=ET.fromstring(body);ns={'a':'http://www.w3.org/2005/Atom'}
                    for entry in root.findall('a:entry',ns):
                        identifier=entry.findtext('a:id',default='',namespaces=ns)
                        if '/abs/' not in identifier:continue
                        sources.append(source_record(identifier,entry.findtext('a:title',default='',namespaces=ns),
                            entry.findtext('a:summary',default='',namespaces=ns),identifier.replace('http://','https://'),
                            entry.findtext('a:published',default='',namespaces=ns),provider))
                else:
                    import json
                    body,receipt=self._fetch('https://api.crossref.org/works',{'query.bibliographic':query,'rows':MAX_RESULTS,
                        'select':'DOI,title,abstract,URL,published'})
                    data=json.loads(body)
                    for item in data.get('message',{}).get('items',[])[:MAX_RESULTS]:
                        identifier=item.get('DOI');title=item.get('title') or []
                        if not identifier or not title:continue
                        abstract=re.sub(r'<[^>]*>',' ',str(item.get('abstract','')))
                        published=str(item.get('published',{}).get('date-parts',[]))
                        sources.append(source_record(identifier,title[0],abstract,item.get('URL','https://doi.org/'+identifier),published,provider))
                attempts.append({'provider':provider,'status':'completed',**receipt,'results':len(sources)})
                if sources:break
            except Exception as error:
                attempts.append({'provider':provider,'status':'failed','error_type':type(error).__name__,
                                 'message':clean(str(error),300),'authorization_header_sent':False})
        result={'query':query,'keywords_used':terms,'arxiv_query':arxiv_query,'sources':sources[:MAX_RESULTS],'attempts':attempts,
                'network_requests':len(attempts),'maximum_network_requests':2,'mock':False,
                'scope':'Public metadata/abstracts only, no full-paper reading or general web browsing',
                'project_rows_sent':False,'model_credentials_used':False}
        self.records.append(result)
        if not sources:raise RuntimeError('No literature evidence retrieved; do not invent sources')
        return result
