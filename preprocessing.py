"""Member 1: randomized grouped holdout with exact deduplication.
The partition is frozen before any model fits. No earlier/later assignment policy.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from asl_data import load_dataset
from grouping import make_groups
from common import ROOT,PARTITIONS,fingerprint,write_json,table
from audit import audit_arrays


def prepare(data=ROOT/'data/landmarks.npz',output=ROOT/'prepared',gap_ms=1000):
    X,y,ids,t=load_dataset(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups,keep,reason,rep,stats=make_groups(X,y,t,gap_ms)
    config=dict(protocol='randomized_group_holdout_v1',gap_ms=gap_ms,outer_seed=42,inner_seed=43,
        outer_folds=5,inner_folds=4,test_fold=0,validation_fold=0,
        group_rule='Connected components of exact-vector matches and same-class adjacent timestamp links <= gap_ms, computed on ALL original rows.',
        deduplication='Keep first sample in stable class/filename order for each single-label exact vector; exclude all rows of conflicting-label exact vectors.',
        assignment='First fold of shuffled StratifiedGroupKFold(5,42) is test. First fold of shuffled StratifiedGroupKFold(4,43) on remaining groups is validation. No score-dependent search.')
    classes=np.unique(y[keep])
    coverage=[]
    for label in classes:
        idx=keep[y[keep]==label]
        coverage.append(dict(label=label,original_samples=int(np.sum(y==label)),retained_samples=len(idx),groups=len(np.unique(groups[idx])),
                             timestamped_retained=int(np.sum(t[idx]>=0)),unknown_time_retained=int(np.sum(t[idx]<0))))
    table(output/'group_coverage.csv',coverage)
    if set(classes)!=set(y) or any(r['groups']<3 for r in coverage):
        raise ValueError('Cannot retain every class in three independent group partitions. See group_coverage.csv. Obtain more groups; do not split linked groups.')
    trainval_idx,test_idx=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X[keep],y[keep],groups[keep]))
    trainval=keep[trainval_idx];test=keep[test_idx]
    train_idx,val_idx=next(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=43).split(X[trainval],y[trainval],groups[trainval]))
    train,val=trainval[train_idx],trainval[val_idx]
    parts=dict(zip(PARTITIONS,(train,val,test)))
    if any(set(y[idx])!=set(classes) for idx in parts.values()):
        raise ValueError('Fixed split does not cover every class in every partition. No alternative seed is silently selected.')
    audit=audit_arrays(X,y,t,groups,keep,train,val,test,gap_ms)
    digest=fingerprint(X,y,ids,t,groups,train,val,test,config)
    audit['fingerprint']=digest
    assignment=np.full(len(y),'excluded',dtype='U16');group_owner={}
    for name,idx in parts.items():
        assignment[idx]=name
        for g in np.unique(groups[idx]):group_owner[int(g)]=name
    counts={name:len(idx) for name,idx in parts.items()}
    meta=dict(config=config,fingerprint=digest,features=63,classes=classes.tolist(),stats=stats,
        counts=counts,actual_percentages={name:100*n/len(keep) for name,n in counts.items()},
        target_percentages=dict(train=60,validation=20,test=20),
        note='Group integrity takes priority over exact proportions. Groups are timestamp proxies, not signer/session identities.')
    # Cache canonical input so every member has the exact data used to prepare this split.
    if Path(data).resolve()!=(ROOT/'data/landmarks.npz').resolve():
        (ROOT/'data').mkdir(exist_ok=True)
        np.savez_compressed(ROOT/'data/landmarks.npz',X=X,y=y,ids=ids,t=t)
    np.savez_compressed(output/'split.npz',groups=groups,keep=keep,train=train,validation=val,test=test)
    table(output/'split_manifest.csv',[
        dict(original_index=i,sample_id=ids[i],label=y[i],timestamp_ms=int(t[i]),group_id=int(groups[i]),
             retained=bool(reason[i]=='retained'),reason=reason[i],partition=assignment[i],
             group_partition=group_owner.get(int(groups[i]),'excluded'),representative_sample_id=ids[rep[i]])
        for i in range(len(y))])
    class_rows=[]
    for row in coverage:
        label=row['label'];row=dict(row)
        for name,idx in parts.items():row[name]=int(np.sum(y[idx]==label))
        class_rows.append(row)
    table(output/'class_counts.csv',class_rows)
    write_json(output/'metadata.json',meta);write_json(output/'overlap_audit.json',audit)
    print('PREPROCESSING COMPLETE',stats,counts,flush=True)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/landmarks.npz')
    p.add_argument('--output',type=Path,default=ROOT/'prepared')
    a=p.parse_args();prepare(a.data,a.output)
