"""Shared training and evaluation: no model-specific data preparation."""
import argparse
import importlib.metadata
import platform
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
from common import ROOT,PARTITIONS,load_prepared,table,write_json
from audit import audit_arrays


def run_model(name,estimator,prepared=ROOT/'prepared',output=ROOT/'results'):
    X,y,ids,t,g,k,tr,va,te,meta=load_prepared(prepared)
    audit_arrays(X,y,t,g,k,tr,va,te,meta['config']['gap_ms'])
    output=Path(output)/name;output.mkdir(parents=True,exist_ok=True)
    pipeline=make_pipeline(StandardScaler(),estimator)
    scores={};predictions={};tick=time.perf_counter()
    with threadpool_limits(limits=2):
        pipeline.fit(X[tr],y[tr]);fit_seconds=time.perf_counter()-tick
        for part,idx in zip(PARTITIONS,(tr,va,te)):
            start=time.perf_counter();pred=pipeline.predict(X[idx]);seconds=time.perf_counter()-start
            predictions[part]=pred
            scores[part]=dict(samples=len(idx),accuracy_pct=100*accuracy_score(y[idx],pred),
                errors=int(np.sum(y[idx]!=pred)),predict_seconds=seconds)
    labels=meta['classes'];pred=predictions['test']
    report=classification_report(y[te],pred,labels=labels,output_dict=True,zero_division=0)
    write_json(output/'classification_report.json',report)
    table(output/'classification_report.csv',[dict(label=label,**report[label]) for label in labels])
    matrix=confusion_matrix(y[te],pred,labels=labels)
    table(output/'confusion_matrix.csv',[dict(actual=l,**dict(zip(labels,row.tolist()))) for l,row in zip(labels,matrix)])
    for part,idx in zip(PARTITIONS,(tr,va,te)):
        table(output/(part+'_predictions.csv'),[dict(sample_id=ids[i],actual=y[i],predicted=p,
            correct=bool(y[i]==p),group_id=int(g[i]),timestamp_known=bool(t[i]>=0)) for i,p in zip(idx,predictions[part])])
    subgroup=[]
    for status,mask in [('known_timestamp',t[te]>=0),('unknown_timestamp',t[te]<0)]:
        n=int(mask.sum());subgroup.append(dict(group=status,samples=n,errors=int(np.sum(y[te][mask]!=pred[mask])),
            accuracy_pct=100*accuracy_score(y[te][mask],pred[mask]) if n else None))
    table(output/'timestamp_status_scores.csv',subgroup)
    # Give each proxy group equal weight as an additional descriptive metric.
    group_scores=[]
    for group in np.unique(g[te]):
        mask=g[te]==group
        group_scores.append(dict(group_id=int(group),samples=int(mask.sum()),accuracy_pct=100*accuracy_score(y[te][mask],pred[mask])))
    table(output/'test_group_scores.csv',group_scores)
    versions={pkg:importlib.metadata.version(pkg) for pkg in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']}
    result=dict(model=name,protocol=meta['config']['protocol'],fingerprint=meta['fingerprint'],scores=scores,
        train_test_gap_pp=scores['train']['accuracy_pct']-scores['test']['accuracy_pct'],
        macro_f1=report['macro avg']['f1-score'],weighted_f1=report['weighted avg']['f1-score'],
        mean_group_accuracy_pct=float(np.mean([r['accuracy_pct'] for r in group_scores])),
        test_groups=len(group_scores),fit_seconds=fit_seconds,estimator_params=estimator.get_params(),
        timestamp_status_scores=subgroup,versions=versions,python=platform.python_version())
    joblib.dump(pipeline,output/'pipeline.joblib',compress=3)
    write_json(output/'metrics.json',result)
    print(name,scores,flush=True);return result


def cli(name,estimator):
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=ROOT/'prepared')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    a=p.parse_args();run_model(name,estimator,a.prepared,a.output)
