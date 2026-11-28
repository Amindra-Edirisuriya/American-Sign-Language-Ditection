"""Verify the prepared data, all three trained models and saved predictions."""
import argparse
import csv
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from common import ROOT,MODEL_NAMES,PARTITIONS,load_prepared,write_json
from audit import audit_arrays
from asl_data import load_dataset


def verify(raw=None):
    X,y,ids,t,g,k,tr,va,te,meta=load_prepared()
    checks=[]
    if raw:
        original=load_dataset(raw)
        for a,b in zip(original,(X,y,ids,t)):np.testing.assert_array_equal(a,b)
        checks.append('Every input vector, label, ID and timestamp matches original source')
    audit=audit_arrays(X,y,t,g,k,tr,va,te,meta['config']['gap_ms'])
    checks.append('Independent overlap audit passed for all three partition pairs')
    for name in MODEL_NAMES:
        directory=ROOT/'results'/name
        m=json.loads((directory/'metrics.json').read_text())
        if m['fingerprint']!=meta['fingerprint']:raise ValueError('Mismatched fingerprint')
        model=joblib.load(directory/'pipeline.joblib')
        if int(model[0].n_samples_seen_)!=len(tr):raise ValueError('Scaler sample count mismatch')
        np.testing.assert_allclose(model[0].mean_,X[tr].mean(axis=0),rtol=1e-12,atol=1e-14)
        for part,idx in zip(PARTITIONS,(tr,va,te)):
            with (directory/(part+'_predictions.csv')).open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            if len(rows)!=len(idx):raise ValueError('Prediction count mismatch')
            if [r['sample_id'] for r in rows]!=ids[idx].tolist():raise ValueError('Sample ID mismatch')
            if [r['actual'] for r in rows]!=y[idx].tolist():raise ValueError('Actual labels mismatch')
            errors=sum(r['actual']!=r['predicted'] for r in rows)
            if errors!=m['scores'][part]['errors']:raise ValueError('Error total mismatch')
            for r in rows:
                if (r['correct']=='True')!=(r['actual']==r['predicted']):raise ValueError('Correctness flag mismatch')
            count=min(30,len(idx))
            with threadpool_limits(limits=2):pred=model.predict(X[idx[:count]])
            if list(pred)!=[r['predicted'] for r in rows[:count]]:raise ValueError('Serialized pipeline mismatch')
        checks.append(name+': training-only scaler, all partition prediction IDs/labels/error totals, and saved-pipeline predictions passed')
    summary=json.loads((ROOT/'results/comparison.json').read_text())
    if [r['model'] for r in summary['rows']]!=list(MODEL_NAMES):raise ValueError('Wrong model comparison')
    expected=max(summary['rows'],key=lambda r:r['validation_accuracy_pct'])['model']
    if summary['selected_by_validation']!=expected:raise ValueError('Selection rule mismatch')
    checks.append('Comparison includes exactly three models and uses validation accuracy')
    result=dict(passed=True,fingerprint=meta['fingerprint'],checks=checks,audit=audit)
    write_json(ROOT/'results/verification.json',result);print(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw');verify(p.parse_args().raw)
