"""Notebook adapter for Kaggle and Colab; no account credentials embedded."""
from pathlib import Path
import hashlib, zipfile
from src.cloud_learning import write_json
from src.processing import process

def run(config):
    kaggle=Path('/kaggle/working').is_dir()
    if not kaggle and not Path('/content').is_dir():
        raise RuntimeError('Run this notebook on Kaggle or Colab.')
    inputs=Path('/kaggle/input') if kaggle else Path('/content/avc-input')
    root=Path('/kaggle/working') if kaggle else Path('/content')
    output=root/('avc-'+config['job_id'])
    sources=[]
    for v in config['videos']:
        matches=[p for p in inputs.rglob('*') if p.is_file() and p.name==v['name'] and p.stat().st_size==v['bytes']]
        if len(matches)!=1:raise ValueError('Upload exactly one matching input: '+v['name'])
        sources.append(matches[0])
    settings=dict(config['settings'])
    results=process(sources,output,settings,lambda t:print(t,flush=True))
    for result,video in zip(results,config['videos']):
        (output/result['folder']).rename(output/video['id']);result['id']=video['id']
    files=[]
    for video in config['videos']:
        for p in (output/video['id']).rglob('*'):
            if p.is_file():
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                files.append(dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
    write_json(output/'learning-report.json',dict(schema=config['schema'],job_id=config['job_id'],
               settings=config['settings'],results=results,files=files))
    archive=root/'learning-results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(output/'learning-report.json','learning-report.json')
        for item in files:z.write(output/item['path'],item['path'])
    print('DOWNLOAD:',archive,flush=True)
