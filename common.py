"""Shared paths, data integrity and portable output formats."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from asl_data import load_dataset
ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ('random_forest', 'knn', 'decision_tree')
PARTITIONS = ('train', 'validation', 'test')

def table(path, rows):
    rows=list(rows)
    if not rows: raise ValueError('Cannot write empty table')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fingerprint(X,y,ids,t,groups,train,validation,test,config):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for a in (X,y,ids,t,groups,train,validation,test):
        h.update(str((a.dtype.str,a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def load_prepared(prepared=ROOT/'prepared'):
    prepared=Path(prepared)
    meta=json.loads((prepared/'metadata.json').read_text(encoding='utf-8'))
    X,y,ids,t=load_dataset(ROOT/'data/landmarks.npz')
    with np.load(prepared/'split.npz',allow_pickle=False) as d:
        groups,keep,train,val,test=[d[k] for k in ('groups','keep','train','validation','test')]
    digest=fingerprint(X,y,ids,t,groups,train,val,test,meta['config'])
    if digest != meta['fingerprint']:
        raise ValueError('Data or split changed. Run preprocessing, then all three models again.')
    expected=np.sort(np.concatenate([train,val,test]))
    if not np.array_equal(expected,keep) or len(np.unique(expected)) != len(expected):
        raise ValueError('Partitions must cover each retained sample exactly once.')
    if any(len(i)==0 for i in (train,val,test)):raise ValueError('Empty partition')
    if any(set(y[i]) != set(y[keep]) for i in (train,val,test)):
        raise ValueError('Every retained class must occur in each partition.')
    return X,y,ids,t,groups,keep,train,val,test,meta
