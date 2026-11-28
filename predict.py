"""Predict one raw landmark CSV using the validation-selected pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from asl_data import read_sample
from common import ROOT,load_prepared


def predict(csv_path,model_name=None):
    *_,meta=load_prepared()
    selection=json.loads((ROOT/'results/selection.json').read_text())
    if selection['fingerprint']!=meta['fingerprint']:raise ValueError('Selection belongs to an older split')
    name=model_name or selection['model']
    metrics=json.loads((ROOT/'results'/name/'metrics.json').read_text())
    if metrics['fingerprint']!=meta['fingerprint']:raise ValueError('Model belongs to an older split')
    model=joblib.load(ROOT/'results'/name/'pipeline.joblib')
    with Path(csv_path).open('rb') as f:x=read_sample(f,str(csv_path))
    probabilities=model.predict_proba(x.reshape(1,-1))[0]
    result=dict(model=name,predicted=str(model.predict(x.reshape(1,-1))[0]),
        top_three=[dict(label=str(model.classes_[i]),score=float(probabilities[i])) for i in np.argsort(-probabilities)[:3]],
        note='Scores are model estimates, not a guarantee of correctness. This script does not capture webcam images.')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path)
    p.add_argument('--model',choices=['random_forest','knn','decision_tree'])
    a=p.parse_args();predict(a.csv,a.model)
