"""Independent overlap audit. Uses original timestamps as well as retained rows."""
import argparse
from itertools import combinations
import numpy as np
from common import ROOT,PARTITIONS,load_prepared,write_json


def audit_arrays(X,y,t,groups,keep,train,val,test,gap_ms=1000):
    parts=dict(zip(PARTITIONS,(train,val,test)));assignment=np.full(len(y),-1,dtype=int)
    group_owner={}
    for owner,(name,idx) in enumerate(parts.items()):
        for g in np.unique(groups[idx]):
            if int(g) in group_owner:raise ValueError('A group crosses partitions')
            group_owner[int(g)]=owner
    # Give removed original rows their retained group's owner for the timestamp audit.
    for i,g in enumerate(groups):assignment[i]=group_owner.get(int(g),-1)
    canonical=np.asarray(X,dtype=np.float64).copy();canonical[canonical==0]=0
    seen={};shared=0
    for owner,idx in enumerate(parts.values()):
        for i in idx:
            key=canonical[i].tobytes()
            if key in seen and seen[key]!=owner:shared+=1
            seen[key]=owner
    temporal_edges=cross_edges=0
    for label in np.unique(y):
        idx=np.flatnonzero((y==label)&(t>=0)&(assignment>=0))
        idx=idx[np.argsort(t[idx],kind='stable')]
        for a,b in zip(idx[:-1],idx[1:]):
            if int(t[b])-int(t[a])<=gap_ms:
                temporal_edges+=1;cross_edges+=int(assignment[a]!=assignment[b])
    nearest_counts={}
    for left,right in combinations(PARTITIONS,2):
        a,b=parts[left],parts[right];count=0
        for label in np.unique(y[b]):
            earlier=np.sort(t[a[(y[a]==label)&(t[a]>=0)]])
            target=t[b[(y[b]==label)&(t[b]>=0)]]
            if not len(earlier) or not len(target):continue
            k=np.searchsorted(earlier,target)
            distances=np.minimum(abs(target-earlier[np.clip(k,0,len(earlier)-1)]),
                                 abs(target-earlier[np.clip(k-1,0,len(earlier)-1)]))
            count+=int(np.sum(distances<=gap_ms))
        nearest_counts[left+'_vs_'+right]=count
    unique=len({canonical[i].tobytes() for i in keep})==len(keep)
    disjoint=all(not set(parts[a])&set(parts[b]) for a,b in combinations(PARTITIONS,2))
    passed=shared==0 and cross_edges==0 and not any(nearest_counts.values()) and unique and disjoint
    result=dict(passed=passed,retained_rows_unique=unique,partition_indices_disjoint=disjoint,
        exact_vectors_shared_between_partitions=shared,
        original_timestamp_edges_checked=temporal_edges,original_timestamp_edges_crossing_partitions=cross_edges,
        retained_timestamp_neighbour_counts=nearest_counts,
        missing_timestamp_samples_original=int(np.sum(t<0)),missing_timestamp_samples_retained=int(np.sum(t[keep]<0)),
        retained_timestamp_status={name:dict(known=int(np.sum(t[idx]>=0)),unknown=int(np.sum(t[idx]<0))) for name,idx in parts.items()},
        limitation='Zero overlap refers to exact vectors and known SAME-CLASS filename timestamps within the chosen gap. Unknown timestamp, near-vector, signer and session overlap is not ruled out.')
    if not passed:raise ValueError('Overlap audit failed: '+str(result))
    return result


def run(prepared=ROOT/'prepared',output=ROOT/'prepared/overlap_audit.json'):
    X,y,ids,t,g,k,tr,va,te,meta=load_prepared(prepared)
    result=audit_arrays(X,y,t,g,k,tr,va,te,meta['config']['gap_ms'])
    result['fingerprint']=meta['fingerprint'];write_json(output,result)
    print(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=str,default=str(ROOT/'prepared'))
    p.add_argument('--output',type=str,default=str(ROOT/'prepared/overlap_audit.json'))
    a=p.parse_args();run(a.prepared,a.output)
