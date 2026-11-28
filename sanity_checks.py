"""Descriptive controls: majority-class baseline and one shuffled-training-label tree.
Run after freezing settings. These controls are not additional candidate models.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import ROOT,load_prepared,write_json


def run():
    X,y,ids,t,g,k,tr,va,te,meta=load_prepared()
    dummy=DummyClassifier(strategy='most_frequent').fit(X[tr],y[tr])
    shuffled=np.random.default_rng(42).permutation(y[tr])
    model=make_pipeline(StandardScaler(),DecisionTreeClassifier(random_state=42))
    with threadpool_limits(limits=2):
        model.fit(X[tr],shuffled)
        shuffled_accuracy=100*accuracy_score(y[te],model.predict(X[te]))
    result=dict(fingerprint=meta['fingerprint'],majority_baseline_test_accuracy_pct=100*accuracy_score(y[te],dummy.predict(X[te])),
        shuffled_training_labels_tree_test_accuracy_pct=shuffled_accuracy,
        explanation='One negative control: permute only training labels, fit a fresh tree, compare with unchanged true test labels. Low performance supports a functioning evaluation path but cannot prove all leakage is absent. Not an inferential permutation test.')
    write_json(ROOT/'results/sanity_checks.json',result);print(result);return result

if __name__=='__main__':run()
